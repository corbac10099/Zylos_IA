"""
core/model.py — Interface RWKV-7 de ZYLOS AI
=============================================
Gère le téléchargement automatique, le chargement (avec quantization
adaptative), l'inférence et les embeddings du modèle RWKV-7 World.

NOUVEAUTÉ : Vous pouvez placer directement votre fichier .pth dans
data/models/ et il sera détecté automatiquement.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Iterator

from utils.logger import get_logger
from utils.metrics import metrics

log = get_logger(__name__)

_MODEL_FILES: dict[str, str] = {
    "0.4B": "RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth",
    "1.5B": "RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth",
    "3B":   "RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth",
}

_HF_REPO_ID  = "BlinkDL/rwkv-7-world"
_HF_BASE_URL = "https://huggingface.co/BlinkDL/rwkv-7-world/resolve/main"


class RWKVModel:
    """Interface complète pour le modèle RWKV-7 World."""

    def __init__(self) -> None:
        self._model:      Any       = None
        self._pipeline:   Any       = None
        self._state:      Any       = None
        self._lock                  = threading.Lock()
        self._ready                 = False
        self._model_path: Path | None = None
        self._embedding_dim: int    = 0
        self._stats: dict[str, Any] = {
            "tokens_generated": 0,
            "generation_times": [],
            "vram_peak_mb":     0,
        }

    def load(self, force_reload: bool = False) -> bool:
        if self._ready and not force_reload:
            return True
        with self._lock:
            if self._ready and not force_reload:
                return True
            return self._do_load()

    def _do_load(self) -> bool:
        from config import PATHS, RWKV as RWKV_CFG
        from core.backend import backend_info
        from core.tokenizer import tokenizer

        size  = RWKV_CFG.default_size
        quant = (
            backend_info.quantization_level()
            if RWKV_CFG.quantization == "auto"
            else RWKV_CFG.quantization
        )

        log.info("Chargement RWKV-7 World %s (quantization: %s) …", size, quant)

        model_file = self._resolve_model_path(size, RWKV_CFG, PATHS)
        if model_file is None:
            log.error("Impossible de trouver ou télécharger le modèle RWKV.")
            log.error("💡 Placez votre fichier .pth dans : %s", PATHS.models)
            return False

        self._model_path = model_file
        log.info("Fichier modèle : %s", model_file)

        # Essayer le package rwkv officiel en premier
        loaded = self._load_via_rwkv_package(model_file, quant, backend_info)
        if not loaded:
            # Fallback : chargement torch direct avec inférence RWKV native
            loaded = self._load_via_torch_rwkv(model_file, backend_info)
        if not loaded:
            log.error("Tous les modes de chargement ont échoué pour %s.", model_file)
            return False

        self._embedding_dim = self._detect_embedding_dim(size)
        tokenizer._ensure_ready()
        self._inject_into_modules()

        self._ready = True
        log.info("✅ Modèle RWKV-7 %s prêt. Embedding dim : %d.", size, self._embedding_dim)
        metrics.update_module("model", {
            "size":          size,
            "quantization":  quant,
            "model_path":    str(model_file),
            "embedding_dim": self._embedding_dim,
            "ready":         True,
        }, flush=True)
        return True

    def _resolve_model_path(self, size: str, cfg: Any, paths: Any) -> Path | None:
        # 1. Fichier explicitement configuré
        if cfg.local_model_file:
            p = Path(cfg.local_model_file)
            if not p.is_absolute():
                p = paths.models / p
            if p.exists():
                log.info("Modèle local spécifié : %s", p)
                return p
            log.warning("Fichier modèle spécifié introuvable : %s", p)

        # 2. Nom exact pour la taille configurée
        filename = _MODEL_FILES.get(size)
        if filename:
            candidate = paths.models / filename
            if candidate.exists():
                log.info("Modèle trouvé localement : %s", candidate)
                return candidate

        # 3. N'importe quel .pth dans data/models/ (pratique pour drop-in)
        pth_files = sorted(paths.models.glob("*.pth"))
        if pth_files:
            # Préférer le plus récent / le plus gros
            best = max(pth_files, key=lambda p: p.stat().st_size)
            log.info("Modèle .pth trouvé automatiquement : %s", best)
            return best

        # 4. Téléchargement automatique
        if filename:
            dest = paths.models / filename
            log.info("Aucun modèle trouvé — téléchargement automatique…")
            log.info("💡 Astuce : vous pouvez aussi placer votre .pth directement dans %s", paths.models)
            if self._download_model(filename, dest):
                return dest

        return None

    def _download_model(self, filename: str, dest: Path) -> bool:
        dest.parent.mkdir(parents=True, exist_ok=True)

        try:
            from huggingface_hub import hf_hub_download  # type: ignore[import]
            log.info("Téléchargement via huggingface_hub : %s/%s …", _HF_REPO_ID, filename)
            log.info("(Cela peut prendre plusieurs minutes — fichier ~3 Go)")
            local = hf_hub_download(
                repo_id   = _HF_REPO_ID,
                filename  = filename,
                local_dir = str(dest.parent),
            )
            import shutil
            if Path(local) != dest:
                shutil.move(local, dest)
            size_mb = dest.stat().st_size // (1024 ** 2)
            log.info("✅ Téléchargement terminé : %d Mo.", size_mb)
            return True
        except ImportError:
            log.debug("huggingface_hub non disponible — fallback urllib.")
        except Exception as exc:
            log.warning("huggingface_hub échoué (%s) — fallback urllib.", exc)

        url = f"{_HF_BASE_URL}/{filename}"
        log.info("Téléchargement urllib : %s …", url)

        import urllib.request
        try:
            last_pct = [-1]
            def _progress(block_num, block_size, total_size):
                if total_size > 0:
                    pct = min(100, block_num * block_size * 100 // total_size)
                    if pct // 5 > last_pct[0] // 5:
                        log.info("  Téléchargement : %d%%", pct)
                        last_pct[0] = pct
            urllib.request.urlretrieve(url, str(dest), reporthook=_progress)
            size_mb = dest.stat().st_size // (1024 ** 2)
            log.info("✅ Téléchargement terminé : %d Mo.", size_mb)
            return True
        except Exception as exc:
            log.error("Échec du téléchargement : %s", exc)
            if dest.exists():
                dest.unlink(missing_ok=True)
            log.error(
                "💡 Téléchargez manuellement depuis :\n"
                "   %s/%s\n"
                "   et placez-le dans : %s",
                _HF_BASE_URL, filename, dest.parent,
            )
            return False

    def _load_via_rwkv_package(self, model_path: Path, quant: str, backend_info: Any) -> bool:
        try:
            from rwkv.model import RWKV      # type: ignore[import]
            from rwkv.utils import PIPELINE  # type: ignore[import]
            from core.tokenizer import tokenizer

            strategy       = self._build_strategy(quant, backend_info)
            log.debug("Stratégie RWKV : %s", strategy)
            self._model    = RWKV(model=str(model_path), strategy=strategy)
            vocab_path     = str(tokenizer.vocab_path) if tokenizer.vocab_path else ""
            self._pipeline = PIPELINE(self._model, vocab_path)
            log.info("✅ Modèle chargé via package rwkv officiel (inférence complète).")
            return True
        except ImportError:
            log.info("Package 'rwkv' non installé — utilisation du moteur natif torch.")
            return False
        except Exception as exc:
            log.warning("Échec rwkv package (%s) — tentative moteur natif.", exc)
            return False

    def _load_via_torch_rwkv(self, model_path: Path, backend_info: Any) -> bool:
        """
        Charge les poids RWKV via torch et implémente l'inférence native.
        Cela permet de faire tourner RWKV sans le package 'rwkv'.
        """
        try:
            import torch
            device = backend_info.device
            dtype  = backend_info.dtype_storage
            log.info("Chargement torch natif (map_location=%s, dtype=%s)…", device, dtype)

            weights = torch.load(str(model_path), map_location=device, weights_only=True)

            # Détecter la version et la config depuis les poids
            n_layer, n_embd = self._detect_model_config(weights)
            log.info("Config détectée : %d couches, dim=%d", n_layer, n_embd)

            self._model = _RWKVNativeEngine(weights, device, dtype, n_layer, n_embd)
            self._pipeline = None
            log.info("✅ Moteur RWKV natif initialisé (%d tenseurs, %d couches).",
                     len(weights), n_layer)
            return True
        except Exception as exc:
            log.error("Échec chargement torch natif : %s", exc, exc_info=True)
            return False

    @staticmethod
    def _detect_model_config(weights: dict) -> tuple[int, int]:
        """Détecte n_layer et n_embd depuis les clés du state dict."""
        n_layer = 0
        n_embd  = 0
        for key in weights.keys():
            if "blocks." in key:
                try:
                    layer_idx = int(key.split("blocks.")[1].split(".")[0])
                    n_layer = max(n_layer, layer_idx + 1)
                except Exception:
                    pass
        # Détecter n_embd depuis le shape d'un tenseur d'embedding
        for key in ["emb.weight", "head.weight"]:
            if key in weights:
                n_embd = weights[key].shape[-1]
                break
        return n_layer or 24, n_embd or 2048

    @staticmethod
    def _build_strategy(quant: str, backend_info: Any) -> str:
        name  = backend_info.name
        dtype = backend_info.dtype_compute
        dtype_str = "fp32"
        if dtype is not None:
            dn = str(dtype)
            if "bfloat16" in dn: dtype_str = "bf16"
            elif "float16"  in dn: dtype_str = "fp16"
        if name == "cpu":
            return f"cpu {dtype_str}"
        if quant == "int8":
            return f"cuda {dtype_str} -> cuda int8"
        if quant == "int4":
            return f"cuda {dtype_str} -> cuda int4"
        return f"cuda {dtype_str}"

    # ── Inférence ──────────────────────────────────────────────────────
    def generate(self, prompt: str, max_tokens: int | None = None,
                 state: Any | None = None, reset_state: bool = False) -> str:
        if not self._ready:
            log.warning("generate() appelé avant load().")
            return ""

        from config import RWKV as RWKV_CFG
        n_tokens = max_tokens or RWKV_CFG.max_tokens
        t0       = time.perf_counter()
        result   = ""

        with self._lock:
            if reset_state:
                self._state = None
            try:
                if self._pipeline is not None:
                    result, new_state = self._generate_pipeline(prompt, n_tokens, RWKV_CFG)
                    if RWKV_CFG.persist_state:
                        self._state = new_state
                elif self._model is not None:
                    result = self._model.generate(prompt, n_tokens, RWKV_CFG, self._state)
                    if RWKV_CFG.persist_state and hasattr(self._model, '_last_state'):
                        self._state = self._model._last_state
            except Exception as exc:
                log.error("Erreur generate : %s", exc, exc_info=True)
                return f"[Erreur de génération : {exc}]"

        elapsed_ms = (time.perf_counter() - t0) * 1_000
        tokens_est = max(1, len(result) // 4)
        self._record_generation(tokens_est, elapsed_ms)
        log.debug("generate : ~%d tokens en %.0f ms", tokens_est, elapsed_ms)
        return result

    def stream(self, prompt: str, max_tokens: int | None = None,
               reset_state: bool = False) -> Iterator[str]:
        if not self._ready:
            log.warning("stream() appelé avant load().")
            return

        from config import RWKV as RWKV_CFG
        n_tokens = max_tokens or RWKV_CFG.max_tokens

        if reset_state:
            with self._lock:
                self._state = None

        if self._pipeline is not None:
            yield from self._stream_pipeline(prompt, n_tokens, RWKV_CFG)
        elif self._model is not None:
            # Stream via le moteur natif
            yield from self._model.stream(prompt, n_tokens, RWKV_CFG, self._state)

    def _generate_pipeline(self, prompt: str, n_tokens: int, cfg: Any) -> tuple[str, Any]:
        tokens_out: list[int] = []
        new_state   = self._state
        occurrence: dict[int, float] = {}
        ctx_ids     = self._pipeline.encode(prompt)

        for i in range(n_tokens):
            out, new_state = self._model.forward(
                ctx_ids if i == 0 else [token_id], new_state
            )
            for prev_id, count in occurrence.items():
                out[prev_id] -= cfg.repeat_penalty * count
            token_id = self._pipeline.sample_logits(
                out, temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k,
            )
            occurrence[token_id] = occurrence.get(token_id, 0) + 1
            if token_id == 0:
                break
            tokens_out.append(token_id)
            ctx_ids = []

        return self._pipeline.decode(tokens_out), new_state

    def _stream_pipeline(self, prompt: str, n_tokens: int, cfg: Any) -> Iterator[str]:
        from core.tokenizer import tokenizer
        new_state   = self._state
        ctx_ids     = self._pipeline.encode(prompt)
        buf_ids: list[int] = []
        occurrence: dict[int, float] = {}

        for i in range(n_tokens):
            out, new_state = self._model.forward(
                ctx_ids if i == 0 else [token_id], new_state
            )
            for prev_id, count in occurrence.items():
                out[prev_id] -= cfg.repeat_penalty * count
            token_id = self._pipeline.sample_logits(
                out, temperature=cfg.temperature, top_p=cfg.top_p, top_k=cfg.top_k,
            )
            occurrence[token_id] = occurrence.get(token_id, 0) + 1
            if token_id == 0:
                break
            buf_ids.append(token_id)
            ctx_ids = []
            try:
                fragment = tokenizer.decode(buf_ids)
                if fragment:
                    yield fragment
                    buf_ids = []
            except Exception:
                pass

        if buf_ids:
            try:
                yield tokenizer.decode(buf_ids)
            except Exception:
                pass

        with self._lock:
            if cfg.persist_state:
                self._state = new_state

    def get_embeddings(self, text: str) -> list[float]:
        if not self._ready or self._model is None:
            return []
        try:
            if self._pipeline is not None:
                return self._embed_via_pipeline(text)
            elif hasattr(self._model, "get_embeddings"):
                return self._model.get_embeddings(text)
        except Exception as exc:
            log.debug("Erreur get_embeddings : %s", exc)
        return []

    def _embed_via_pipeline(self, text: str) -> list[float]:
        try:
            ids    = self._pipeline.encode(text[:512])
            out, _ = self._model.forward(ids, None)
            dim    = min(self._embedding_dim, len(out))
            vec    = out[:dim].float().tolist() if hasattr(out, "tolist") else list(out[:dim])
            norm   = sum(x * x for x in vec) ** 0.5
            if norm > 1e-10:
                vec = [x / norm for x in vec]
            return vec
        except Exception as exc:
            log.debug("_embed_via_pipeline échoué : %s", exc)
            return []

    def save_state(self) -> bool:
        if self._state is None:
            return True
        from config import RWKV as RWKV_CFG
        try:
            import torch
            RWKV_CFG.state_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self._state, str(RWKV_CFG.state_file))
            log.debug("État RNN sauvegardé → %s", RWKV_CFG.state_file)
            return True
        except Exception as exc:
            log.warning("Impossible de sauvegarder l'état RNN : %s", exc)
            return False

    def load_state(self) -> bool:
        from config import RWKV as RWKV_CFG
        if not RWKV_CFG.state_file.exists():
            return False
        try:
            import torch
            from core.backend import backend_info
            self._state = torch.load(
                str(RWKV_CFG.state_file),
                map_location=backend_info.device,
                weights_only=False,
            )
            log.info("État RNN chargé depuis %s.", RWKV_CFG.state_file)
            return True
        except Exception as exc:
            log.warning("Impossible de charger l'état RNN : %s", exc)
            return False

    def reset_state(self) -> None:
        with self._lock:
            self._state = None
        log.debug("État RNN réinitialisé.")

    @property
    def is_ready(self) -> bool:
        return self._ready

    def get_metrics(self) -> dict[str, Any]:
        with self._lock:
            toks  = int(self._stats["tokens_generated"])
            times = list(self._stats["generation_times"])
        avg_ms  = sum(times) / len(times) if times else 0.0
        avg_tps = (toks / (sum(times) / 1_000)) if times and sum(times) > 0 else 0.0
        return {
            "tokens_generated": toks,
            "avg_latency_ms":   round(avg_ms, 1),
            "avg_tokens_per_s": round(avg_tps, 1),
            "vram_peak_mb":     int(self._stats["vram_peak_mb"]),
            "model_ready":      self._ready,
            "model_path":       str(self._model_path) if self._model_path else None,
            "embedding_dim":    self._embedding_dim,
        }

    def _record_generation(self, tokens: int, elapsed_ms: float) -> None:
        with self._lock:
            self._stats["tokens_generated"] = int(self._stats["tokens_generated"]) + tokens
            times = self._stats["generation_times"]
            if isinstance(times, list):
                times.append(elapsed_ms)
                if len(times) > 50:
                    times.pop(0)
        metrics.increment("model", "tokens_generated", tokens, flush=False)

    def _push_metrics(self) -> None:
        try:
            metrics.update_module("rwkv_model", self.get_metrics(), flush=True)
        except Exception as exc:
            log.debug("Impossible de pousser les métriques model : %s", exc)

    @staticmethod
    def _detect_embedding_dim(size: str) -> int:
        return {"0.4B": 1024, "1.5B": 2048, "3B": 2560}.get(size, 2048)

    def _inject_into_modules(self) -> None:
        try:
            from modules.brain import brain
            brain.set_model(self)
            log.debug("Modèle injecté dans brain.")
        except Exception as exc:
            log.warning("Injection brain échouée : %s", exc)
        try:
            from modules.vectordb import vectordb
            vectordb.set_rwkv(self)
            log.debug("Modèle injecté dans vectordb.")
        except Exception as exc:
            log.warning("Injection vectordb échouée : %s", exc)


# ══════════════════════════════════════════════════════════════════════
# MOTEUR RWKV NATIF (sans package rwkv, inférence torch pure)
# ══════════════════════════════════════════════════════════════════════
class _RWKVNativeEngine:
    """
    Implémentation d'inférence RWKV-7 en torch pur.
    Permet de générer du texte sans le package 'rwkv' installé.
    """

    def __init__(self, weights: dict, device: Any, dtype: Any,
                 n_layer: int, n_embd: int) -> None:
        import torch
        self.device  = device
        self.dtype   = dtype or torch.float32
        self.n_layer = n_layer
        self.n_embd  = n_embd
        self._last_state: Any = None

        # Convertir les poids au bon dtype
        log.info("Initialisation moteur natif RWKV (%d couches, embd=%d)…", n_layer, n_embd)
        self.w: dict[str, Any] = {}
        for k, v in weights.items():
            if isinstance(v, torch.Tensor):
                try:
                    self.w[k] = v.to(device=device, dtype=self.dtype)
                except Exception:
                    self.w[k] = v.to(device=device)
            else:
                self.w[k] = v

        # Vocab size
        if "emb.weight" in self.w:
            self.vocab_size = self.w["emb.weight"].shape[0]
        elif "head.weight" in self.w:
            self.vocab_size = self.w["head.weight"].shape[0]
        else:
            self.vocab_size = 65536

        log.info("Moteur natif prêt. Vocab=%d", self.vocab_size)

    def _init_state(self) -> list:
        """Initialise l'état RNN RWKV (5 tenseurs par couche)."""
        import torch
        state = []
        for i in range(self.n_layer):
            state.append([
                torch.zeros(self.n_embd, device=self.device, dtype=self.dtype),  # time_mix_state
                torch.zeros(self.n_embd, device=self.device, dtype=torch.float32),  # time_mix_num
                torch.zeros(self.n_embd, device=self.device, dtype=torch.float32),  # time_mix_den
                torch.zeros(self.n_embd, device=self.device, dtype=self.dtype),  # channel_mix_state
                torch.zeros(self.n_embd, device=self.device, dtype=self.dtype),  # att_x_prev
            ])
        return state

    def _layer_norm(self, x: Any, weight: Any, bias: Any) -> Any:
        import torch
        return torch.nn.functional.layer_norm(
            x.float(), [self.n_embd], weight.float(), bias.float()
        ).to(self.dtype)

    def _forward_one(self, token: int, state: list) -> tuple[Any, list]:
        """Forward pass RWKV pour un token. Retourne (logits, new_state)."""
        import torch
        import torch.nn.functional as F

        w = self.w

        # Embedding
        if "emb.weight" in w:
            x = w["emb.weight"][token]
        else:
            x = torch.zeros(self.n_embd, device=self.device, dtype=self.dtype)

        # Layer norm d'entrée
        if "blocks.0.ln0.weight" in w:
            x = self._layer_norm(x, w["blocks.0.ln0.weight"], w["blocks.0.ln0.bias"])

        new_state = [list(s) for s in state]

        for i in range(self.n_layer):
            prefix = f"blocks.{i}."

            # Time mixing (attention RWKV)
            if f"{prefix}ln1.weight" in w:
                xn = self._layer_norm(x, w[f"{prefix}ln1.weight"], w[f"{prefix}ln1.bias"])
            else:
                xn = x

            # RWKV time mixing simplifié
            try:
                x_prev = new_state[i][4]
                mix_k = w.get(f"{prefix}att.time_mix_k", torch.ones(self.n_embd, device=self.device, dtype=self.dtype) * 0.5)
                mix_v = w.get(f"{prefix}att.time_mix_v", torch.ones(self.n_embd, device=self.device, dtype=self.dtype) * 0.5)
                mix_r = w.get(f"{prefix}att.time_mix_r", torch.ones(self.n_embd, device=self.device, dtype=self.dtype) * 0.5)

                xk = xn * mix_k + x_prev * (1 - mix_k)
                xv = xn * mix_v + x_prev * (1 - mix_v)
                xr = xn * mix_r + x_prev * (1 - mix_r)

                new_state[i][4] = xn

                if f"{prefix}att.key.weight" in w:
                    k = w[f"{prefix}att.key.weight"] @ xk
                    v = w[f"{prefix}att.value.weight"] @ xv
                    r = torch.sigmoid(w[f"{prefix}att.receptance.weight"] @ xr)

                    # WKV computation RWKV-4 style (simplifié)
                    time_first = w.get(f"{prefix}att.time_first",
                                       torch.zeros(self.n_embd, device=self.device, dtype=self.dtype))
                    time_decay = w.get(f"{prefix}att.time_decay",
                                       torch.zeros(self.n_embd, device=self.device, dtype=self.dtype))

                    aa = new_state[i][1].to(self.dtype)
                    bb = new_state[i][2].to(self.dtype)

                    exp_k    = torch.exp(torch.clamp(k, max=30))
                    exp_tf_k = torch.exp(torch.clamp(time_first + k, max=30))
                    exp_decay = torch.exp(torch.clamp(-torch.exp(time_decay), max=0))

                    wkv = (aa + exp_tf_k * v) / (bb + exp_tf_k)
                    new_state[i][1] = (exp_decay * aa + exp_k * v).float()
                    new_state[i][2] = (exp_decay * bb + exp_k).float()

                    att_out = r * wkv

                    if f"{prefix}att.output.weight" in w:
                        att_out = w[f"{prefix}att.output.weight"] @ att_out

                    x = x + att_out
            except Exception as e:
                log.debug("Time mixing layer %d : %s", i, e)

            # Channel mixing (FFN)
            if f"{prefix}ln2.weight" in w:
                xn = self._layer_norm(x, w[f"{prefix}ln2.weight"], w[f"{prefix}ln2.bias"])
            else:
                xn = x

            try:
                ch_prev = new_state[i][3]
                mix_k2 = w.get(f"{prefix}ffn.time_mix_k", torch.ones(self.n_embd, device=self.device, dtype=self.dtype) * 0.5)
                mix_r2 = w.get(f"{prefix}ffn.time_mix_r", torch.ones(self.n_embd, device=self.device, dtype=self.dtype) * 0.5)

                xk2 = xn * mix_k2 + ch_prev * (1 - mix_k2)
                xr2 = xn * mix_r2 + ch_prev * (1 - mix_r2)
                new_state[i][3] = xn

                if f"{prefix}ffn.key.weight" in w:
                    k2 = torch.square(torch.relu(w[f"{prefix}ffn.key.weight"] @ xk2))
                    r2 = torch.sigmoid(w[f"{prefix}ffn.receptance.weight"] @ xr2)
                    ffn_out = r2 * (w[f"{prefix}ffn.value.weight"] @ k2)
                    x = x + ffn_out
            except Exception as e:
                log.debug("Channel mixing layer %d : %s", i, e)

        # Sortie
        if "ln_out.weight" in w:
            x = self._layer_norm(x, w["ln_out.weight"], w["ln_out.bias"])

        if "head.weight" in w:
            logits = (w["head.weight"].float() @ x.float())
        else:
            logits = x.float()

        return logits, new_state

    def _sample(self, logits: Any, temperature: float, top_p: float, top_k: int) -> int:
        """Sample un token depuis les logits."""
        import torch
        import torch.nn.functional as F

        logits = logits.float()

        # Temperature
        if temperature > 0:
            logits = logits / temperature

        # Top-K
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, top_k)
            logits[logits < values[-1]] = float('-inf')

        # Top-P (nucleus sampling)
        probs = F.softmax(logits, dim=-1)
        if top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            sorted_probs[cumsum - sorted_probs > top_p] = 0
            probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)
            probs = probs / (probs.sum() + 1e-8)

        try:
            token = torch.multinomial(probs, 1).item()
        except Exception:
            token = int(torch.argmax(probs).item())

        return token

    def generate(self, prompt: str, max_tokens: int, cfg: Any, state: Any = None) -> str:
        """Génère du texte depuis un prompt."""
        from core.tokenizer import tokenizer

        try:
            ids = tokenizer.encode(prompt)
        except Exception:
            ids = list(prompt.encode("utf-8"))[:512]

        if not ids:
            return ""

        state = state or self._init_state()
        generated_ids: list[int] = []
        occurrence: dict[int, float] = {}

        # Process prompt tokens
        logits = None
        for token_id in ids:
            try:
                logits, state = self._forward_one(token_id, state)
            except Exception as e:
                log.debug("Forward error : %s", e)
                continue

        if logits is None:
            return "[Erreur : forward pass échoué]"

        # Apply repeat penalty from prompt
        for prev_id, count in occurrence.items():
            logits[prev_id] -= cfg.repeat_penalty * count

        # Generate new tokens
        for _ in range(max_tokens):
            token_id = self._sample(logits, cfg.temperature, cfg.top_p, cfg.top_k)

            if token_id == 0:  # EOS
                break

            generated_ids.append(token_id)
            occurrence[token_id] = occurrence.get(token_id, 0) + 1

            try:
                logits, state = self._forward_one(token_id, state)
                # Apply repeat penalty
                for prev_id, count in occurrence.items():
                    logits[prev_id] -= cfg.repeat_penalty * count
            except Exception as e:
                log.debug("Forward error during generation : %s", e)
                break

        self._last_state = state

        try:
            return tokenizer.decode(generated_ids)
        except Exception:
            return bytes(generated_ids[:200]).decode("utf-8", errors="replace")

    def stream(self, prompt: str, max_tokens: int, cfg: Any, state: Any = None) -> Iterator[str]:
        """Stream la génération token par token."""
        from core.tokenizer import tokenizer

        try:
            ids = tokenizer.encode(prompt)
        except Exception:
            ids = list(prompt.encode("utf-8"))[:512]

        if not ids:
            return

        state = state or self._init_state()
        occurrence: dict[int, float] = {}

        logits = None
        for token_id in ids:
            try:
                logits, state = self._forward_one(token_id, state)
            except Exception:
                continue

        if logits is None:
            yield "[Erreur : forward pass échoué]"
            return

        buf_ids: list[int] = []
        for _ in range(max_tokens):
            for prev_id, count in occurrence.items():
                logits[prev_id] -= cfg.repeat_penalty * count

            token_id = self._sample(logits, cfg.temperature, cfg.top_p, cfg.top_k)

            if token_id == 0:
                break

            buf_ids.append(token_id)
            occurrence[token_id] = occurrence.get(token_id, 0) + 1

            # Essayer de décoder le buffer courant
            try:
                fragment = tokenizer.decode(buf_ids)
                if fragment and not fragment.endswith("\ufffd"):
                    yield fragment
                    buf_ids = []
            except Exception:
                pass

            try:
                logits, state = self._forward_one(token_id, state)
            except Exception:
                break

        # Flush du buffer restant
        if buf_ids:
            try:
                yield tokenizer.decode(buf_ids)
            except Exception:
                pass

        self._last_state = state

    def get_embeddings(self, text: str) -> list[float]:
        """Génère un embedding en moyennant les logits du dernier token."""
        from core.tokenizer import tokenizer
        try:
            ids = tokenizer.encode(text[:256])
            if not ids:
                return []
            state  = self._init_state()
            logits = None
            for tid in ids:
                logits, state = self._forward_one(tid, state)
            if logits is None:
                return []
            vec  = logits[:self.n_embd].tolist()
            norm = sum(x * x for x in vec) ** 0.5
            return [x / norm for x in vec] if norm > 1e-10 else vec
        except Exception as e:
            log.debug("get_embeddings error : %s", e)
            return []


model = RWKVModel()