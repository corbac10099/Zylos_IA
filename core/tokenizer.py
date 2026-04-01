"""
core/tokenizer.py — Tokenizer RWKV World de ZYLOS AI
=====================================================
Wrappeur autour du WorldTokenizer officiel RWKV.
Gère le téléchargement automatique du fichier de vocabulaire,
le cache en mémoire et les utilitaires d'estimation.
"""

from __future__ import annotations

import os
import threading
import urllib.request
from pathlib import Path
from typing import Any

from utils.logger import get_logger

log = get_logger(__name__)

# URLs alternatives pour le fichier de vocabulaire RWKV World
_VOCAB_URLS = [
    "https://raw.githubusercontent.com/BlinkDL/RWKV-LM/main/RWKV-v4/src/model_run.py",  # placeholder
    "https://huggingface.co/BlinkDL/rwkv-4-world/resolve/main/rwkv_vocab_v20230424.txt",
    "https://raw.githubusercontent.com/BlinkDL/ChatRWKV/main/v2/rwkv_vocab_v20230424.txt",
    "https://github.com/BlinkDL/ChatRWKV/raw/main/v2/rwkv_vocab_v20230424.txt",
]
# URL principale corrigée (le fichier est dans v2, pas dans rwkv_pip_pkg)
_VOCAB_URL = "https://huggingface.co/BlinkDL/rwkv-4-world/resolve/main/rwkv_vocab_v20230424.txt"
_VOCAB_FILENAME = "rwkv_vocab_v20230424.txt"

TOKEN_BOS = 0
TOKEN_EOS = 0
TOKEN_PAD = 1


class _WorldTokenizer:
    """Implémentation du WorldTokenizer RWKV en Python pur."""

    def __init__(self, vocab_path: str | Path) -> None:
        self._idx2token: dict[int, bytes] = {}
        self._token2idx: dict[bytes, int] = {}
        self._trie:      dict             = {}
        self.vocab_size: int              = 0
        self._load(Path(vocab_path))

    def _load(self, path: Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(" ", 2)
            if len(parts) < 2:
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                continue

            raw = parts[1]

            if raw.isdigit():
                token_bytes = bytes([int(raw)])
            elif raw.startswith("b'") or raw.startswith('b"'):
                try:
                    token_bytes = eval(raw)   # noqa: S307
                except Exception:
                    token_bytes = raw.encode("utf-8")
            else:
                token_bytes = raw.encode("utf-8")

            self._idx2token[idx] = token_bytes
            self._token2idx[token_bytes] = idx

        self.vocab_size = max(self._idx2token.keys()) + 1 if self._idx2token else 0

        for token_bytes, idx in self._token2idx.items():
            node = self._trie
            for byte in token_bytes:
                node = node.setdefault(byte, {})
            node[-1] = idx

        log.debug("WorldTokenizer chargé : %d tokens.", self.vocab_size)

    def encode(self, text: str) -> list[int]:
        data  = text.encode("utf-8")
        ids: list[int] = []
        pos   = 0
        n     = len(data)

        while pos < n:
            node   = self._trie
            best_id    = None
            best_len   = 0
            cur_len    = 0

            while pos + cur_len < n:
                byte = data[pos + cur_len]
                if byte not in node:
                    break
                node     = node[byte]
                cur_len += 1
                if -1 in node:
                    best_id  = node[-1]
                    best_len = cur_len

            if best_id is not None:
                ids.append(best_id)
                pos += best_len
            else:
                byte_val = data[pos]
                if byte_val in self._token2idx:
                    ids.append(self._token2idx[bytes([byte_val])])
                else:
                    ids.append(byte_val)
                pos += 1

        return ids

    def decode(self, ids: list[int]) -> str:
        buf = b""
        for i in ids:
            tok = self._idx2token.get(i)
            if tok is not None:
                buf += tok
        return buf.decode("utf-8", errors="replace")


class RWKVTokenizer:
    """Wrappeur thread-safe autour du WorldTokenizer."""

    def __init__(self) -> None:
        self._tokenizer: Any    = None
        self._lock               = threading.Lock()
        self._ready              = False
        self.vocab_size: int     = 0
        self.vocab_path: Path | None = None

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            self._load()
            self._ready = True

    def _load(self) -> None:
        from config import PATHS

        vocab_path = PATHS.models / _VOCAB_FILENAME
        self.vocab_path = vocab_path

        if not vocab_path.exists():
            self._download_vocab(vocab_path)

        # Tentative package rwkv officiel
        try:
            from rwkv.utils import PIPELINE  # type: ignore[import]
            pipeline = PIPELINE(None, str(vocab_path))
            self._tokenizer = pipeline
            self.vocab_size = 65536
            log.info("WorldTokenizer chargé via package rwkv (officiel).")
            return
        except Exception as exc:
            log.debug("Package rwkv non disponible (%s) — implémentation embarquée.", exc)

        # Implémentation pure Python embarquée
        self._tokenizer = _WorldTokenizer(vocab_path)
        self.vocab_size = self._tokenizer.vocab_size
        log.info("WorldTokenizer chargé (implémentation embarquée, %d tokens).",
                 self.vocab_size)

    @staticmethod
    def _download_vocab(dest: Path) -> None:
        """Télécharge le fichier de vocabulaire en essayant plusieurs URLs."""
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Essayer d'abord via huggingface_hub
        try:
            from huggingface_hub import hf_hub_download  # type: ignore[import]
            log.info("Téléchargement vocab via huggingface_hub…")
            local = hf_hub_download(
                repo_id  = "BlinkDL/rwkv-4-world",
                filename = _VOCAB_FILENAME,
                local_dir = str(dest.parent),
            )
            import shutil
            if Path(local) != dest:
                shutil.copy2(local, dest)
            log.info("Vocabulaire téléchargé via huggingface_hub (%d octets).", dest.stat().st_size)
            return
        except Exception as exc:
            log.debug("huggingface_hub vocab échoué : %s", exc)

        # Essayer les URLs alternatives
        urls_to_try = [
            "https://huggingface.co/BlinkDL/rwkv-4-world/resolve/main/rwkv_vocab_v20230424.txt",
            "https://huggingface.co/BlinkDL/rwkv-7-world/resolve/main/rwkv_vocab_v20230424.txt",
            "https://raw.githubusercontent.com/BlinkDL/ChatRWKV/main/v2/rwkv_vocab_v20230424.txt",
        ]

        for url in urls_to_try:
            log.info("Téléchargement vocab depuis %s …", url)
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "ZylosAI/1.0"}
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    content = resp.read()
                dest.write_bytes(content)
                log.info("Vocabulaire téléchargé (%d octets).", dest.stat().st_size)
                return
            except Exception as exc:
                log.warning("Échec URL %s : %s", url, exc)
                if dest.exists():
                    dest.unlink(missing_ok=True)

        raise OSError(
            f"Impossible de télécharger le vocabulaire RWKV.\n"
            f"Placez manuellement '{_VOCAB_FILENAME}' dans data/models/\n"
            f"Téléchargeable sur : https://huggingface.co/BlinkDL/rwkv-4-world"
        )

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        self._ensure_ready()
        try:
            if hasattr(self._tokenizer, "encode"):
                return self._tokenizer.encode(text)
            return self._tokenizer.encode(text)
        except Exception as exc:
            log.error("Erreur encode : %s", exc)
            return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        if not ids:
            return ""
        self._ensure_ready()
        try:
            if hasattr(self._tokenizer, "decode"):
                return self._tokenizer.decode(ids)
            return self._tokenizer.decode(ids)
        except Exception as exc:
            log.error("Erreur decode : %s", exc)
            return ""

    def decode_token(self, token_id: int) -> str:
        return self.decode([token_id])

    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        cjk_chars = sum(
            1 for c in text
            if "\u4e00" <= c <= "\u9fff"
            or "\u3400" <= c <= "\u4dbf"
            or "\uf900" <= c <= "\ufaff"
        )
        latin_chars = len(text) - cjk_chars
        return max(1, round(latin_chars / 4 + cjk_chars / 1.5))

    def fits_in_context(self, text: str, max_tokens: int | None = None) -> bool:
        from config import RWKV
        limit = max_tokens or RWKV.context_length
        return self.estimate_tokens(text) <= limit

    def truncate_to_tokens(self, text: str, max_tokens: int, from_end: bool = False) -> str:
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        if from_end:
            truncated = text[-max_chars:]
            cut = truncated.find(" ")
            return truncated[cut:].lstrip() if cut > 0 else truncated
        else:
            truncated = text[:max_chars]
            cut = truncated.rfind(" ")
            return truncated[:cut] if cut > max_chars // 2 else truncated


tokenizer = RWKVTokenizer()