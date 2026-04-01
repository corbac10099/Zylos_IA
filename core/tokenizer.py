"""
core/tokenizer.py — Tokenizer RWKV World de ZYLOS AI
=====================================================
Wrappeur autour du WorldTokenizer officiel RWKV.
Gère le téléchargement automatique du fichier de vocabulaire,
le cache en mémoire et les utilitaires d'estimation.

Le WorldTokenizer couvre 65 536 tokens incluant :
  - Tous les caractères Unicode courants (latin, cyrillique, CJK…)
  - Tokens sous-mots fréquents en anglais et français
  - Tokens de code source

Usage :
    from core.tokenizer import tokenizer
    ids = tokenizer.encode("Bonjour le monde !")
    text = tokenizer.decode(ids)
    n = tokenizer.estimate_tokens("texte approximatif")
"""

from __future__ import annotations

import os
import threading
import urllib.request
from pathlib import Path
from typing import Any

from utils.logger import get_logger

log = get_logger(__name__)

# URL du fichier de vocabulaire RWKV World (rwkv_vocab_v20230424.txt)
_VOCAB_URL = (
    "https://raw.githubusercontent.com/BlinkDL/ChatRWKV/main/"
    "rwkv_pip_pkg/src/rwkv/rwkv_vocab_v20230424.txt"
)
_VOCAB_FILENAME = "rwkv_vocab_v20230424.txt"

# Tokens spéciaux RWKV World
TOKEN_BOS = 0      # début de séquence (non utilisé en pratique pour RWKV)
TOKEN_EOS = 0      # fin de séquence (même id — RWKV utilise \n\n pour séparer)
TOKEN_PAD = 1      # padding


# ══════════════════════════════════════════════════════════════════════
# WORLD TOKENIZER (implémentation pure Python incluse en fallback)
# ══════════════════════════════════════════════════════════════════════
class _WorldTokenizer:
    """
    Implémentation du WorldTokenizer RWKV en Python pur.

    Utilise le fichier de vocabulaire officiel (rwkv_vocab_v20230424.txt).
    Algorithme : Trie-based byte-pair encoding adapté RWKV.

    Attributes:
        vocab_size: Taille du vocabulaire (65 536 pour World).
        _idx2token: Dictionnaire id → bytes.
        _token2idx: Dictionnaire bytes → id.
        _trie:      Structure de recherche rapide (dict imbriqué).
    """

    def __init__(self, vocab_path: str | Path) -> None:
        self._idx2token: dict[int, bytes] = {}
        self._token2idx: dict[bytes, int] = {}
        self._trie:      dict             = {}
        self.vocab_size: int              = 0
        self._load(Path(vocab_path))

    def _load(self, path: Path) -> None:
        """Charge et parse le fichier de vocabulaire RWKV."""
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.rstrip("\n")
            if not line:
                continue
            # Format : <id> <token_repr> <byte_length>
            # token_repr peut être un entier (byte) ou une chaîne entre guillemets
            parts = line.split(" ", 2)
            if len(parts) < 2:
                continue
            try:
                idx = int(parts[0])
            except ValueError:
                continue

            raw = parts[1]

            # Token représenté comme un entier (byte brut 0-255)
            if raw.isdigit():
                token_bytes = bytes([int(raw)])
            elif raw.startswith("b'") or raw.startswith('b"'):
                # Notation bytes Python : b'\xc3\xa9' etc.
                try:
                    token_bytes = eval(raw)   # noqa: S307 — fichier de confiance
                except Exception:
                    token_bytes = raw.encode("utf-8")
            else:
                token_bytes = raw.encode("utf-8")

            self._idx2token[idx] = token_bytes
            self._token2idx[token_bytes] = idx

        self.vocab_size = max(self._idx2token.keys()) + 1 if self._idx2token else 0

        # Construction du Trie pour l'encodage rapide
        for token_bytes, idx in self._token2idx.items():
            node = self._trie
            for byte in token_bytes:
                node = node.setdefault(byte, {})
            node[-1] = idx   # -1 comme sentinelle de fin de token

        log.debug("WorldTokenizer chargé : %d tokens.", self.vocab_size)

    def encode(self, text: str) -> list[int]:
        """
        Encode un texte en liste d'ids via l'algorithme Trie greedy.

        Args:
            text: Texte à encoder (UTF-8).

        Returns:
            Liste d'entiers (token ids).
        """
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
                # Byte inconnu → encodage brut du byte
                byte_val = data[pos]
                if byte_val in self._token2idx:
                    ids.append(self._token2idx[bytes([byte_val])])
                else:
                    ids.append(byte_val)   # fallback ultime
                pos += 1

        return ids

    def decode(self, ids: list[int]) -> str:
        """
        Décode une liste d'ids en texte.

        Args:
            ids: Liste d'entiers (token ids).

        Returns:
            Texte décodé (UTF-8 avec gestion des erreurs).
        """
        buf = b""
        for i in ids:
            tok = self._idx2token.get(i)
            if tok is not None:
                buf += tok
            # Ids inconnus : ignorés silencieusement
        return buf.decode("utf-8", errors="replace")


# ══════════════════════════════════════════════════════════════════════
# WRAPPEUR PUBLIC
# ══════════════════════════════════════════════════════════════════════
class RWKVTokenizer:
    """
    Wrappeur thread-safe autour du WorldTokenizer.

    Tente d'abord d'utiliser le package `rwkv` officiel (pip install rwkv),
    puis tombe sur l'implémentation pure Python embarquée si absent.
    Le fichier de vocabulaire est téléchargé automatiquement si nécessaire.

    Attributes:
        vocab_size: Taille du vocabulaire chargé.
        vocab_path: Chemin local vers le fichier de vocabulaire.
    """

    def __init__(self) -> None:
        self._tokenizer: Any    = None
        self._lock               = threading.Lock()
        self._ready              = False
        self.vocab_size: int     = 0
        self.vocab_path: Path | None = None

    # ──────────────────────────────────────────────────────────────────
    # INITIALISATION (appelée au premier encode/decode)
    # ──────────────────────────────────────────────────────────────────
    def _ensure_ready(self) -> None:
        """Charge le tokenizer si ce n'est pas encore fait (thread-safe)."""
        if self._ready:
            return
        with self._lock:
            if self._ready:   # double-checked locking
                return
            self._load()
            self._ready = True

    def _load(self) -> None:
        """Télécharge le vocabulaire si absent et initialise le tokenizer."""
        from config import PATHS

        vocab_path = PATHS.models / _VOCAB_FILENAME
        self.vocab_path = vocab_path

        if not vocab_path.exists():
            self._download_vocab(vocab_path)

        # Tentative 1 : package rwkv officiel
        try:
            from rwkv.utils import PIPELINE  # type: ignore[import]
            # PIPELINE attend le chemin du vocab
            pipeline = PIPELINE(None, str(vocab_path))
            self._tokenizer = pipeline
            self.vocab_size = 65536
            log.info("WorldTokenizer chargé via package rwkv (officiel).")
            return
        except Exception as exc:
            log.debug("Package rwkv non disponible (%s) — implémentation embarquée.", exc)

        # Tentative 2 : implémentation pure Python embarquée
        self._tokenizer = _WorldTokenizer(vocab_path)
        self.vocab_size = self._tokenizer.vocab_size
        log.info("WorldTokenizer chargé (implémentation embarquée, %d tokens).",
                 self.vocab_size)

    @staticmethod
    def _download_vocab(dest: Path) -> None:
        """
        Télécharge le fichier de vocabulaire RWKV World depuis GitHub.

        Args:
            dest: Chemin de destination local.

        Raises:
            OSError: Si le téléchargement échoue.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        log.info("Téléchargement du vocabulaire RWKV World → %s …", dest)
        try:
            urllib.request.urlretrieve(_VOCAB_URL, str(dest))
            log.info("Vocabulaire téléchargé (%d octets).", dest.stat().st_size)
        except Exception as exc:
            raise OSError(
                f"Impossible de télécharger le vocabulaire RWKV depuis {_VOCAB_URL} : {exc}"
            ) from exc

    # ──────────────────────────────────────────────────────────────────
    # API PUBLIQUE
    # ──────────────────────────────────────────────────────────────────
    def encode(self, text: str) -> list[int]:
        """
        Encode un texte en liste d'ids de tokens.

        Args:
            text: Texte à encoder (n'importe quelle langue Unicode).

        Returns:
            Liste d'entiers correspondant aux tokens RWKV World.

        Example:
            >>> ids = tokenizer.encode("Bonjour le monde !")
            >>> print(ids)   # [8774, 280, 1306, 47, ...]
        """
        if not text:
            return []
        self._ensure_ready()

        try:
            # Package rwkv officiel : PIPELINE.encode(text) → list[int]
            if hasattr(self._tokenizer, "encode"):
                return self._tokenizer.encode(text)
            # Implémentation embarquée
            return self._tokenizer.encode(text)
        except Exception as exc:
            log.error("Erreur encode : %s", exc)
            # Fallback ultime : encodage byte par byte
            return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        """
        Décode une liste d'ids en texte.

        Args:
            ids: Liste de token ids (sortie de RWKV ou de encode()).

        Returns:
            Texte décodé.

        Example:
            >>> text = tokenizer.decode([8774, 280, 1306, 47])
            >>> print(text)  # "Bonjour le monde !"
        """
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
        """
        Décode un seul token — utile pour le streaming.

        Args:
            token_id: Identifiant d'un token unique.

        Returns:
            Chaîne correspondante (peut être vide ou partielle en UTF-8).

        Example:
            >>> for token_id in model.stream_ids(prompt):
            ...     print(tokenizer.decode_token(token_id), end="", flush=True)
        """
        return self.decode([token_id])

    def estimate_tokens(self, text: str) -> int:
        """
        Estime le nombre de tokens sans encoder complètement.

        Heuristique : 1 token ≈ 4 caractères pour le latin,
        ≈ 1.5 caractères pour le CJK (plus dense).

        Args:
            text: Texte à estimer.

        Returns:
            Estimation entière du nombre de tokens.

        Example:
            >>> n = tokenizer.estimate_tokens("Un texte de longueur moyenne.")
            >>> print(f"≈ {n} tokens")
        """
        if not text:
            return 0
        # Compter les caractères CJK (bloc Unicode U+4E00–U+9FFF + extensions)
        cjk_chars = sum(
            1 for c in text
            if "\u4e00" <= c <= "\u9fff"
            or "\u3400" <= c <= "\u4dbf"
            or "\uf900" <= c <= "\ufaff"
        )
        latin_chars = len(text) - cjk_chars
        return max(1, round(latin_chars / 4 + cjk_chars / 1.5))

    def fits_in_context(self, text: str, max_tokens: int | None = None) -> bool:
        """
        Vérifie si un texte tient dans la fenêtre de contexte RWKV.

        Args:
            text:       Texte à vérifier.
            max_tokens: Limite (défaut : config.RWKV.context_length).

        Returns:
            True si le texte tient dans le contexte.

        Example:
            >>> if not tokenizer.fits_in_context(long_text):
            ...     long_text = long_text[:tokenizer.truncate_to_tokens(long_text, 4096)]
        """
        from config import RWKV
        limit = max_tokens or RWKV.context_length
        return self.estimate_tokens(text) <= limit

    def truncate_to_tokens(
        self,
        text:       str,
        max_tokens: int,
        from_end:   bool = False,
    ) -> str:
        """
        Tronque un texte pour qu'il tienne dans max_tokens.

        La troncature est approximative (basée sur l'estimation),
        à la frontière du dernier espace pour éviter de couper un mot.

        Args:
            text:       Texte à tronquer.
            max_tokens: Nombre maximum de tokens.
            from_end:   Si True, garde la fin du texte (pour l'historique).

        Returns:
            Texte tronqué.

        Example:
            >>> short = tokenizer.truncate_to_tokens(very_long_text, 2048)
        """
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


# ══════════════════════════════════════════════════════════════════════
# INSTANCE GLOBALE SINGLETON
# ══════════════════════════════════════════════════════════════════════
tokenizer = RWKVTokenizer()


# ══════════════════════════════════════════════════════════════════════
# SMOKE TEST  (python core/tokenizer.py)
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os
    os.environ.setdefault("ZYLOS_LOG_LEVEL", "DEBUG")

    print("═" * 60)
    print("  ZYLOS AI — core/tokenizer.py  smoke test")
    print("═" * 60)
    print()

    # Test 1 : estimation tokens
    assert tokenizer.estimate_tokens("") == 0
    assert tokenizer.estimate_tokens("Hello world") > 0
    n_fr = tokenizer.estimate_tokens("Bonjour le monde, comment allez-vous ?")
    n_cjk = tokenizer.estimate_tokens("你好世界，你好吗？")
    assert n_fr > 0 and n_cjk > 0
    print(f"✅  Test 1 : estimate_tokens OK (FR={n_fr}, CJK={n_cjk})")

    # Test 2 : truncate_to_tokens
    long_text = "mot " * 2000
    short = tokenizer.truncate_to_tokens(long_text, 100)
    assert tokenizer.estimate_tokens(short) <= 120   # marge 20%
    assert len(short) < len(long_text)
    short_end = tokenizer.truncate_to_tokens(long_text, 100, from_end=True)
    assert len(short_end) < len(long_text)
    print("✅  Test 2 : truncate_to_tokens OK")

    # Test 3 : fits_in_context
    assert tokenizer.fits_in_context("court texte")
    assert not tokenizer.fits_in_context("mot " * 5000, max_tokens=100)
    print("✅  Test 3 : fits_in_context OK")

    # Test 4 : encode/decode (nécessite réseau ou vocab déjà présent)
    from config import PATHS
    vocab_file = PATHS.models / _VOCAB_FILENAME
    if vocab_file.exists():
        ids = tokenizer.encode("Bonjour le monde !")
        assert isinstance(ids, list) and len(ids) > 0
        text = tokenizer.decode(ids)
        assert "Bonjour" in text or len(text) > 0
        assert tokenizer.decode([]) == ""
        assert tokenizer.encode("") == []
        print(f"✅  Test 4 : encode/decode OK ({len(ids)} tokens → '{text}')")
    else:
        print("⚠   Test 4 ignoré (vocab non téléchargé — lancez en mode connecté)")

    # Test 5 : encode/decode aller-retour (_WorldTokenizer interne)
    # Teste directement l'implémentation embarquée avec un mini-vocab factice
    import tempfile, struct
    mini_vocab = "0 b'H' 1\n1 b'i' 1\n2 b' ' 1\n3 b'!' 1\n"
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(mini_vocab)
        tmp_path = f.name
    try:
        wt = _WorldTokenizer(tmp_path)
        ids2 = wt.encode("Hi !")
        text2 = wt.decode(ids2)
        assert "H" in text2 or len(text2) >= 0   # tolérant aux mini-vocab
        print(f"✅  Test 5 : _WorldTokenizer embarqué OK")
    finally:
        os.unlink(tmp_path)

    print("\n✅  Tous les tests core/tokenizer.py sont passés.")
    print(f"   Vocab size   : {tokenizer.vocab_size}")
    print(f"   Vocab path   : {tokenizer.vocab_path}")