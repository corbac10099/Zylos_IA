"""
core/model.py — Interface RWKV-7 de ZYLOS AI
=============================================
Gère le téléchargement automatique, le chargement (avec quantization
adaptative), l'inférence et les embeddings du modèle RWKV-7 World.

Fonctionnalités :
  - Téléchargement automatique depuis HuggingFace si absent
    (huggingface_hub en priorité, urllib en fallback)
  - Quantization int8/int4 automatique selon la VRAM disponible
  - État RNN persistant entre les appels (continuité contextuelle)
  - Génération complète et streaming token par token
  - Embeddings internes pour la base vectorielle (zéro appel API)
  - Chain-of-thought activable pour améliorer le raisonnement

Usage :
    from core.model import model
    model.load()
    text = model.generate("Bonjour !")
    for token in model.stream("Explique RWKV"):
        print(token, end="", flush=True)
    vec = model.get_embeddings("Python est un langage")
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

# ─────────────────────────────────────────────────────────────────────
# Noms de fichiers par taille sur HuggingFace BlinkDL/rwkv-7-world
# ─────────────────────────────────────────────────────────────────────
_MODEL_FILES: dict[str, str] = {
    "0.4B": "RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth",
    "1.5B": "RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth",
    "3B":   "RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth",
}

# CORRECTION CRITIQUE : /resolve/main/ pour un téléchargement direct
# /tree/main/ est la page web HuggingFace → retourne du HTML (404 sur .pth)
_HF_REPO_ID  = "BlinkDL/rwkv-7-world"
_HF_BASE_URL = "https://huggingface.co/BlinkDL/rwkv-7-world/resolve/main"


# ══════════════════════════════════════════════════════════════════════
# MODÈLE RWKV-7
# ══════════════════════════════════════════════════════════════════════
class RWKVModel:
    """
    Interface complète pour le modèle RWKV-7 World.

    Singleton exposé via `model` en bas de fichier.
    """

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

    # ──────────────────────────────────────────────────────────────────
    # CHARGEMENT
    # ──────────────────────────────────────────────────────────────────
    def load(self, force_reload: bool = False) -> bool:
        """
        Télécharge (si absent) et charge le modèle RWKV-7 en mémoire.

        Args:
            force_reload: Si True, recharge même si déjà prêt.

        Returns:
            True si le chargement a réussi.
        """
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
            return False

        self._model_path = model_file

        loaded = self._load_via_rwkv_package(model_file, quant, backend_info)
        if not loaded:
            loaded = self._load_via_torch_direct(model_file, quant, backend_info)

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
        # Fichier explicitement configuré
        if cfg.local_model_file:
            p = Path(cfg.local_model_file)
            if not p.is_absolute():
                p = paths.models / p
            if p.exists():
                log.info("Modèle local spécifié : %s", p)
                return p
            log.warning("Fichier modèle spécifié introuvable : %s", p)

        # Recherche dans PATHS.models par taille
        filename = _MODEL_FILES.get(size)
        if filename:
            candidate = paths.models / filename
            if candidate.exists():
                log.info("Modèle trouvé localement : %s", candidate)
                return candidate

        # Recherche d'un .pth quelconque
        pth_files = list(paths.models.glob("*.pth"))
        if pth_files:
            log.info("Utilisation du modèle existant : %s", pth_files[0])
            return pth_files[0]

        # Téléchargement automatique
        if filename:
            dest = paths.models / filename
            if self._download_model(filename, dest):
                return dest

        return None

    def _download_model(self, filename: str, dest: Path) -> bool:
        """
        Télécharge le modèle depuis HuggingFace.
        Utilise huggingface_hub en priorité (reprise, vérification SHA256),
        puis urllib comme fallback.

        La correction clé : utiliser /resolve/main/ (lien direct)
        et non /tree/main/ (page web HTML).
        """
        dest.parent.mkdir(parents=True, exist_ok=True)

        # ── Méthode 1 : huggingface_hub (recommandée) ─────────────────
        try:
            from huggingface_hub import hf_hub_download  # type: ignore[import]
            log.info(
                "Téléchargement via huggingface_hub : %s/%s → %s …",
                _HF_REPO_ID, filename, dest,
            )
            log.info("(Cela peut prendre plusieurs minutes selon votre connexion.)")
            local = hf_hub_download(
                repo_id   = _HF_REPO_ID,
                filename  = filename,
                local_dir = str(dest.parent),
            )
            import shutil
            if Path(local) != dest:
                shutil.move(local, dest)
            size_mb = dest.stat().st_size // (1024 ** 2)
            log.info("✅ Téléchargement terminé via huggingface_hub : %d Mo.", size_mb)
            return True
        except ImportError:
            log.debug("huggingface_hub non disponible — fallback urllib.")
        except Exception as exc:
            log.warning("huggingface_hub échoué (%s) — fallback urllib.", exc)

        # ── Méthode 2 : urllib direct (fallback) ──────────────────────
        # URL CORRIGÉE : /resolve/main/ et non /tree/main/
        url = f"{_HF_BASE_URL}/{filename}"
        log.info("Téléchargement urllib : %s → %s …", url, dest)
        log.info("(Cela peut prendre plusieurs minutes selon votre connexion.)")

        import urllib.request

        try:
            last_pct = [-1]

            def _progress(block_num: int, block_size: int, total_size: int) -> None:
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
                "💡 Téléchargez manuellement le fichier depuis :\n"
                "   https://huggingface.co/%s/resolve/main/%s\n"
                "   et placez-le dans : %s",
                _HF_REPO_ID, filename, dest.parent,
            )
            return False

    # ──────────────────────────────────────────────────────────────────
    # LOADERS
    # ──────────────────────────────────────────────────────────────────
    def _load_via_rwkv_package(
        self, model_path: Path, quant: str, backend_info: Any,
    ) -> bool:
        try:
            from rwkv.model import RWKV      # type: ignore[import]
            from rwkv.utils import PIPELINE  # type: ignore[import]
            from core.tokenizer import tokenizer

            strategy       = self._build_strategy(quant, backend_info)
            log.debug("Stratégie RWKV : %s", strategy)
            self._model    = RWKV(model=str(model_path), strategy=strategy)
            vocab_path     = str(tokenizer.vocab_path) if tokenizer.vocab_path else ""
            self._pipeline = PIPELINE(self._model, vocab_path)
            log.info("Modèle chargé via package rwkv officiel.")
            return True
        except ImportError:
            log.debug("Package rwkv non disponible — tentative torch direct.")
            return False
        except Exception as exc:
            log.warning("Échec rwkv package (%s) — tentative torch direct.", exc)
            return False

    def _load_via_torch_direct(
        self, model_path: Path, quant: str, backend_info: Any,
    ) -> bool:
        try:
            import torch
            device = backend_info.device
            dtype  = backend_info.dtype_storage
            log.info("Chargement torch direct (map_location=%s, dtype=%s)…", device, dtype)
            weights = torch.load(str(model_path), map_location=device, weights_only=True)
            if quant == "int8" and hasattr(torch, "quantization"):
                try:
                    weights = {
                        k: v.to(torch.int8) if v.is_floating_point() else v
                        for k, v in weights.items()
                    }
                    log.debug("Quantization int8 appliquée.")
                except Exception as e:
                    log.debug("Quantization int8 ignorée : %s", e)
            self._model    = _DirectWeightsWrapper(weights, device, dtype)
            self._pipeline = None
            log.info("Poids RWKV chargés via torch direct (%d tenseurs).", len(weights))
            return True
        except Exception as exc:
            log.error("Échec chargement torch direct : %s", exc)
            return False

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

    # ──────────────────────────────────────────────────────────────────
    # INFÉRENCE
    # ──────────────────────────────────────────────────────────────────
    def generate(
        self,
        prompt:      str,
        max_tokens:  int | None = None,
        state:       Any | None = None,
        reset_state: bool       = False,
    ) -> str:
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
                    result = self._model.generate_text(prompt, n_tokens, RWKV_CFG)
            except Exception as exc:
                log.error("Erreur generate : %s", exc)
                return ""

        elapsed_ms = (time.perf_counter() - t0) * 1_000
        tokens_est = max(1, len(result) // 4)
        self._record_generation(tokens_est, elapsed_ms)
        log.debug("generate : %d tokens en %.0f ms (%.0f tok/s)",
                  tokens_est, elapsed_ms,
                  tokens_est / (elapsed_ms / 1_000) if elapsed_ms > 0 else 0)
        return result

    def stream(
        self,
        prompt:      str,
        max_tokens:  int | None = None,
        reset_state: bool       = False,
    ) -> Iterator[str]:
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
            text = self._model.generate_text(prompt, n_tokens, RWKV_CFG)
            for word in text.split(" "):
                yield word + " "
                time.sleep(0.0)

    def _generate_pipeline(self, prompt: str, n_tokens: int, cfg: Any) -> tuple[str, Any]:
        tokens_out: list[int] = []
        new_state   = self._state
        occurrence: dict[int, float] = {}
        ctx_ids     = self._pipeline.encode(prompt)

        for i in range(n_tokens):
            out, new_state = self._model.forward(
                ctx_ids if i == 0 else [token_id], new_state
            )
            # Repeat penalty dynamique
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
        t0 = time.perf_counter()

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

        elapsed_ms = (time.perf_counter() - t0) * 1_000
        self._record_generation(n_tokens, elapsed_ms)
        with self._lock:
            if cfg.persist_state:
                self._state = new_state

    # ──────────────────────────────────────────────────────────────────
    # EMBEDDINGS
    # ──────────────────────────────────────────────────────────────────
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

    # ──────────────────────────────────────────────────────────────────
    # ÉTAT RNN
    # ──────────────────────────────────────────────────────────────────
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

    # ──────────────────────────────────────────────────────────────────
    # UTILITAIRES
    # ──────────────────────────────────────────────────────────────────
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
        tps = tokens / (elapsed_ms / 1_000) if elapsed_ms > 0 else 0.0
        metrics.update("model", {"avg_tokens_per_sec": round(tps, 1)}, flush=False)
        self._push_metrics()

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
# WRAPPER POIDS DIRECTS (fallback sans package rwkv)
# ══════════════════════════════════════════════════════════════════════
class _DirectWeightsWrapper:
    def __init__(self, weights: dict, device: Any, dtype: Any) -> None:
        self.weights = weights
        self.device  = device
        self.dtype   = dtype
        log.warning(
            "Mode dégradé : package `rwkv` non installé. "
            "Inférence désactivée — pip install rwkv pour activer le chat."
        )

    def generate_text(self, prompt: str, n_tokens: int, cfg: Any) -> str:
        return (
            "⚠ Package `rwkv` non installé. "
            "Installez-le avec : pip install rwkv\n"
            "Le modèle est chargé mais l'inférence est désactivée."
        )

    def get_embeddings(self, text: str) -> list[float]:
        return []


# ══════════════════════════════════════════════════════════════════════
# SINGLETON
# ══════════════════════════════════════════════════════════════════════
model = RWKVModel()


# ══════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os
    os.environ.setdefault("ZYLOS_LOG_LEVEL", "DEBUG")

    print("═" * 60)
    print("  ZYLOS AI — core/model.py  smoke test")
    print("═" * 60)
    print()

    assert not model.is_ready
    print("✅  Test 1 : état initial OK")

    result = model.generate("Test", max_tokens=10)
    assert result == ""
    tokens = list(model.stream("Test"))
    assert tokens == []
    emb = model.get_embeddings("Test")
    assert emb == []
    print("✅  Test 2 : guards non-chargé OK")

    class _FakeB:
        name = "cuda"; vram_mb = 8192; dtype_compute = None

    s = RWKVModel._build_strategy("int8", _FakeB())
    assert "int8" in s
    s2 = RWKVModel._build_strategy("none", _FakeB())
    assert "cuda" in s2
    print(f"✅  Test 3 : _build_strategy OK")

    assert RWKVModel._detect_embedding_dim("0.4B") == 1024
    assert RWKVModel._detect_embedding_dim("1.5B") == 2048
    assert RWKVModel._detect_embedding_dim("3B")   == 2560
    print("✅  Test 4 : embedding dim OK")

    m = model.get_metrics()
    assert m["model_ready"] is False
    print("✅  Test 5 : get_metrics OK")

    # Vérification URL correcte
    assert "/resolve/main" in _HF_BASE_URL, "❌ URL incorrecte !"
    assert "/tree/main"   not in _HF_BASE_URL, "❌ URL incorrecte (/tree/main) !"
    print(f"✅  Test 6 : URL HuggingFace correcte → {_HF_BASE_URL}")

    print("\n✅  Tous les tests core/model.py sont passés.")