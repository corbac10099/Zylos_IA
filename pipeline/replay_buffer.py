"""
pipeline/replay_buffer.py — Replay buffer anti-oubli catastrophique de ZYLOS AI
=================================================================================
Maintient un tampon circulaire d'échantillons d'entraînement anciens.
Lors de chaque session de fine-tuning QLoRA, un ratio configurable
d'anciens échantillons est mélangé aux nouveaux pour éviter l'oubli
catastrophique (config.TRAINER.replay_ratio = 0.30 par défaut).

Architecture :
  - Stockage persistant dans data/corpus/replay_buffer.jsonl (un JSON par ligne)
  - Capacité bornée à config.TRAINER.replay_buffer_size (défaut 1 000)
  - Éviction FIFO (les plus anciens d'abord)
  - Méthode sample() retourne un mélange ancien/nouveau
  - Thread-safe via verrou

Usage :
    from pipeline.replay_buffer import replay_buffer
    replay_buffer.add(new_samples)                   # ajouter après scraping
    batch = replay_buffer.sample(64, new_samples)    # mélange 70/30 pour training
"""

from __future__ import annotations

import json
import random
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.logger import get_logger
from utils.metrics import metrics

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════
# STRUCTURE D'UN ÉCHANTILLON
# ══════════════════════════════════════════════════════════════════════
@dataclass
class TrainSample:
    """
    Échantillon d'entraînement stocké dans le replay buffer.

    Attributes:
        text:       Texte brut de l'échantillon (chunk de corpus).
        source_url: URL d'origine (traçabilité).
        source_title: Titre de la page d'origine.
        token_est:  Estimation du nombre de tokens.
        added_at:   Timestamp ISO d'ajout dans le buffer.
    """
    text:         str
    source_url:   str   = ""
    source_title: str   = ""
    token_est:    int   = 0
    added_at:     str   = ""

    def __post_init__(self) -> None:
        if not self.token_est:
            self.token_est = max(1, len(self.text) // 4)
        if not self.added_at:
            from datetime import datetime, timezone
            self.added_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "text":         self.text,
            "source_url":   self.source_url,
            "source_title": self.source_title,
            "token_est":    self.token_est,
            "added_at":     self.added_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainSample":
        return cls(
            text         = d.get("text", ""),
            source_url   = d.get("source_url", ""),
            source_title = d.get("source_title", ""),
            token_est    = d.get("token_est", 0),
            added_at     = d.get("added_at", ""),
        )


# ══════════════════════════════════════════════════════════════════════
# REPLAY BUFFER
# ══════════════════════════════════════════════════════════════════════
class ReplayBuffer:
    """
    Tampon circulaire d'échantillons d'entraînement anciens.

    Persiste dans data/corpus/replay_buffer.jsonl.
    Capacité bornée à TRAINER.replay_buffer_size.
    Éviction FIFO des entrées les plus anciennes.

    Attributes:
        _buffer:   Liste en mémoire des échantillons.
        _lock:     Verrou pour l'accès concurrent.
        _path:     Chemin vers le fichier JSONL de persistance.
    """

    def __init__(self) -> None:
        self._buffer: list[TrainSample] = []
        self._lock    = threading.Lock()
        self._path:   Path | None       = None
        self._loaded  = False

    # ──────────────────────────────────────────────────────────────────
    # INITIALISATION PARESSEUSE
    # ──────────────────────────────────────────────────────────────────
    def _ensure_loaded(self) -> None:
        """Charge le buffer depuis le disque si pas encore fait."""
        if self._loaded:
            return
        from config import PATHS
        self._path   = PATHS.corpus / "replay_buffer.jsonl"
        self._loaded = True
        self._load()

    def _load(self) -> None:
        """Lit le fichier JSONL et peuple self._buffer."""
        if not self._path or not self._path.exists():
            log.debug("Replay buffer : fichier absent — démarrage vide.")
            return

        loaded: list[TrainSample] = []
        try:
            with open(self._path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        loaded.append(TrainSample.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, KeyError) as exc:
                        log.debug("Ligne JSONL invalide ignorée : %s", exc)

            from config import TRAINER
            # Appliquer la limite de capacité dès le chargement
            if len(loaded) > TRAINER.replay_buffer_size:
                loaded = loaded[-TRAINER.replay_buffer_size:]

            with self._lock:
                self._buffer = loaded

            log.info("Replay buffer chargé : %d échantillons depuis %s.",
                     len(loaded), self._path)
        except OSError as exc:
            log.warning("Impossible de lire le replay buffer : %s", exc)

    def _save(self) -> None:
        """Persiste le buffer complet dans le fichier JSONL (sous verrou)."""
        if not self._path:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for sample in self._buffer:
                    f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
            tmp.replace(self._path)
        except OSError as exc:
            log.error("Impossible de sauvegarder le replay buffer : %s", exc)

    # ──────────────────────────────────────────────────────────────────
    # API PUBLIQUE
    # ──────────────────────────────────────────────────────────────────
    def add(self, samples: list[TrainSample]) -> int:
        """
        Ajoute des échantillons dans le buffer.
        Évince les plus anciens si la capacité est dépassée.

        Args:
            samples: Liste de TrainSample à ajouter.

        Returns:
            Nombre d'échantillons effectivement ajoutés.

        Example:
            >>> replay_buffer.add([TrainSample(text="Python est…", source_url="…")])
        """
        if not samples:
            return 0

        self._ensure_loaded()
        from config import TRAINER

        valid = [s for s in samples if s.text and len(s.text.strip()) >= 20]
        if not valid:
            return 0

        with self._lock:
            self._buffer.extend(valid)
            # Éviction FIFO si dépassement de capacité
            excess = len(self._buffer) - TRAINER.replay_buffer_size
            if excess > 0:
                self._buffer = self._buffer[excess:]
            self._save()

        count = len(valid)
        log.debug("Replay buffer : +%d échantillons (total : %d).",
                  count, len(self._buffer))
        metrics.update_module("replay_buffer", {"size": len(self._buffer)},
                              flush=False)
        return count

    def add_from_page(self, url: str, title: str, chunks: list[str]) -> int:
        """
        Raccourci : crée des TrainSample depuis les chunks d'une page web.

        Args:
            url:    URL source.
            title:  Titre de la page.
            chunks: Liste de textes à ajouter.

        Returns:
            Nombre d'échantillons ajoutés.
        """
        samples = [
            TrainSample(text=c, source_url=url, source_title=title)
            for c in chunks if c and len(c.strip()) >= 20
        ]
        return self.add(samples)

    def sample(
        self,
        n:           int,
        new_samples: list[TrainSample] | None = None,
    ) -> list[TrainSample]:
        """
        Retourne un batch mélangé ancien/nouveau pour l'entraînement.

        La proportion est déterminée par config.TRAINER.replay_ratio :
          - (1 - replay_ratio) × n  → nouveaux échantillons
          - replay_ratio × n        → anciens du buffer

        Si le buffer est vide, retourne uniquement les nouveaux.

        Args:
            n:           Taille totale du batch souhaité.
            new_samples: Nouveaux échantillons (session courante).

        Returns:
            Liste mélangée de TrainSample, taille ≤ n.

        Example:
            >>> batch = replay_buffer.sample(64, new_samples=fresh_chunks)
        """
        self._ensure_loaded()
        from config import TRAINER

        new_samples = new_samples or []
        result: list[TrainSample] = []

        n_replay = min(int(n * TRAINER.replay_ratio), len(self._buffer))
        n_new    = n - n_replay

        # Échantillons nouveaux
        if new_samples:
            n_new = min(n_new, len(new_samples))
            result.extend(random.sample(new_samples, n_new))

        # Anciens du buffer
        if n_replay > 0:
            with self._lock:
                pool = list(self._buffer)
            result.extend(random.sample(pool, n_replay))

        random.shuffle(result)
        log.debug("replay_buffer.sample : %d nouveaux + %d anciens = %d total.",
                  len(result) - n_replay, n_replay, len(result))
        return result

    def size(self) -> int:
        """Retourne le nombre d'échantillons dans le buffer."""
        self._ensure_loaded()
        with self._lock:
            return len(self._buffer)

    def clear(self) -> int:
        """
        Vide le buffer (mémoire + disque).

        Returns:
            Nombre d'échantillons supprimés.
        """
        self._ensure_loaded()
        with self._lock:
            n = len(self._buffer)
            self._buffer.clear()
            self._save()
        log.info("Replay buffer vidé (%d échantillons supprimés).", n)
        return n

    def get_metrics(self) -> dict[str, Any]:
        """Retourne les métriques du buffer."""
        self._ensure_loaded()
        from config import TRAINER
        with self._lock:
            sz = len(self._buffer)
            total_tokens = sum(s.token_est for s in self._buffer)
        return {
            "size":             sz,
            "capacity":         TRAINER.replay_buffer_size,
            "fill_pct":         round(sz / TRAINER.replay_buffer_size, 4)
                                if TRAINER.replay_buffer_size else 0.0,
            "total_tokens_est": total_tokens,
            "avg_tokens":       round(total_tokens / sz, 1) if sz else 0.0,
            "path":             str(self._path) if self._path else None,
        }


# ══════════════════════════════════════════════════════════════════════
# SINGLETON
# ══════════════════════════════════════════════════════════════════════
replay_buffer = ReplayBuffer()


# ══════════════════════════════════════════════════════════════════════
# SMOKE TEST  (python pipeline/replay_buffer.py)
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os, tempfile
    os.environ.setdefault("ZYLOS_LOG_LEVEL", "DEBUG")

    print("═" * 60)
    print("  ZYLOS AI — pipeline/replay_buffer.py  smoke test")
    print("═" * 60)
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        import config as cfg
        object.__setattr__(cfg.PATHS, "corpus", Path(tmpdir))

        buf = ReplayBuffer()

        # Test 1 : ajout
        samples = [TrainSample(text=f"Texte d'entraînement numéro {i}. " * 5,
                               source_url=f"https://example.com/{i}")
                   for i in range(20)]
        n = buf.add(samples)
        assert n == 20, f"Attendu 20, reçu {n}"
        assert buf.size() == 20
        print(f"✅  Test 1 : add() OK ({n} échantillons)")

        # Test 2 : sample avec mélange
        new = [TrainSample(text=f"Nouvel échantillon {i}. " * 5) for i in range(10)]
        batch = buf.sample(20, new_samples=new)
        assert 0 < len(batch) <= 20
        print(f"✅  Test 2 : sample() OK ({len(batch)} échantillons dans le batch)")

        # Test 3 : persistance
        buf2 = ReplayBuffer()
        buf2._path   = buf._path
        buf2._loaded = False
        buf2._load()
        assert buf2.size() == 20
        print("✅  Test 3 : persistance JSONL OK")

        # Test 4 : capacité bornée
        object.__setattr__(cfg.TRAINER, "replay_buffer_size", 15)
        buf3 = ReplayBuffer()
        buf3._path = buf._path
        buf3._loaded = False
        buf3._load()
        assert buf3.size() <= 15
        print(f"✅  Test 4 : capacité bornée OK (≤ 15 : {buf3.size()})")

        # Test 5 : add_from_page
        n2 = buf.add_from_page("https://test.com", "Test",
                                [f"Chunk {i} texte suffisant pour le test" for i in range(5)])
        assert n2 == 5
        print(f"✅  Test 5 : add_from_page OK ({n2} ajoutés)")

        # Test 6 : clear
        cleared = buf.clear()
        assert buf.size() == 0
        print(f"✅  Test 6 : clear() OK ({cleared} supprimés)")

        # Test 7 : métriques
        m = buf.get_metrics()
        assert isinstance(m["size"], int)
        assert isinstance(m["fill_pct"], float)
        print(f"✅  Test 7 : get_metrics() OK — {m}")

    print("\n✅  Tous les tests pipeline/replay_buffer.py sont passés.")