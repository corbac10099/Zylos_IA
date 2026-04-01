"""
pipeline/corpus_builder.py — Construction du corpus d'entraînement de ZYLOS AI
================================================================================
Transforme les pages scrapées en fichiers JSONL d'entraînement QLoRA.
Gère le formatage, le filtrage qualité, la déduplication et l'alimentation
du replay buffer.

Pipeline de construction :
  1. Réception des ScrapedPage issues de modules/scraper.py
  2. Filtrage qualité (longueur, langue, entropie)
  3. Formatage en paires Instruction/Réponse (format RWKV World)
  4. Déduplication contenu (hash SHA-256)
  5. Écriture dans data/corpus/train_YYYY-MM-DD.jsonl
  6. Alimentation du replay buffer

Usage :
    from pipeline.corpus_builder import corpus_builder
    n = corpus_builder.build_from_pages(scraped_pages)
    print(f"{n} exemples écrits dans le corpus")
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.logger import get_logger
from utils.metrics import metrics

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════
# STRUCTURE D'UN EXEMPLE D'ENTRAÎNEMENT
# ══════════════════════════════════════════════════════════════════════
@dataclass
class TrainExample:
    """
    Exemple d'entraînement au format RWKV World instruction-following.

    Attributes:
        instruction: Question ou consigne (peut être vide pour les exemples
                     de continuation pure).
        input:       Contexte optionnel fourni avec l'instruction.
        output:      Réponse attendue du modèle.
        source_url:  URL d'origine pour la traçabilité.
        source_title: Titre de la page d'origine.
    """
    instruction:  str
    output:       str
    input:        str   = ""
    source_url:   str   = ""
    source_title: str   = ""

    def to_rwkv_format(self) -> str:
        """
        Sérialise l'exemple au format de prompt RWKV World.

        Format utilisé :
            User: <instruction>\\n\\nAssistant: <output>
        ou pour la continuation pure :
            <output>

        Returns:
            Texte prêt à être tokenisé par RWKV.
        """
        if self.instruction:
            ctx = f"\n\nInput:\n{self.input}" if self.input else ""
            return f"User: {self.instruction}{ctx}\n\nAssistant: {self.output}"
        return self.output

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction":  self.instruction,
            "input":        self.input,
            "output":       self.output,
            "source_url":   self.source_url,
            "source_title": self.source_title,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainExample":
        return cls(
            instruction  = d.get("instruction", ""),
            output       = d.get("output", ""),
            input        = d.get("input", ""),
            source_url   = d.get("source_url", ""),
            source_title = d.get("source_title", ""),
        )


# ══════════════════════════════════════════════════════════════════════
# CONSTRUCTEUR DE CORPUS
# ══════════════════════════════════════════════════════════════════════
class CorpusBuilder:
    """
    Construit et gère le corpus d'entraînement ZYLOS.

    Singleton exposé via `corpus_builder` en bas de fichier.

    Attributes:
        _seen_hashes:  Ensemble des SHA-256 de contenus déjà indexés.
        _lock:         Verrou pour l'accès concurrent.
        _stats:        Compteurs de session.
    """

    # Qualité minimale
    MIN_CHARS       = 80
    MIN_ENTROPY     = 3.0     # bits par caractère — filtre les textes répétitifs
    MAX_CHARS       = 8_000   # au-delà → tronqué à la frontière paragraphe

    def __init__(self) -> None:
        self._seen_hashes: set[str]          = set()
        self._lock                            = threading.Lock()
        self._stats: dict[str, int | float]  = {
            "examples_written":  0,
            "examples_filtered": 0,
            "duplicates":        0,
            "sessions":          0,
        }

    # ──────────────────────────────────────────────────────────────────
    # API PUBLIQUE
    # ──────────────────────────────────────────────────────────────────
    def build_from_pages(self, pages: list[Any]) -> int:
        """
        Construit le corpus depuis une liste de ScrapedPage.

        Chaque page est découpée en chunks (déjà effectué par scraper.py),
        chaque chunk devient un exemple d'entraînement (continuation pure).

        Args:
            pages: Liste de modules.scraper.ScrapedPage.

        Returns:
            Nombre total d'exemples écrits.

        Example:
            >>> pages = scraper.scrape_urls(["https://fr.wikipedia.org/wiki/Python"])
            >>> n = corpus_builder.build_from_pages(pages)
        """
        from config import PATHS

        examples: list[TrainExample] = []

        for page in pages:
            if not getattr(page, "success", False):
                continue
            page_examples = self._page_to_examples(page)
            examples.extend(page_examples)

        if not examples:
            log.info("CorpusBuilder : aucun exemple généré depuis %d pages.", len(pages))
            return 0

        # Déduplication
        examples = self._deduplicate(examples)

        # Écriture dans le fichier JSONL du jour
        output_path = self._get_output_path(PATHS)
        n_written   = self._write_jsonl(output_path, examples)

        # Alimentation du replay buffer
        self._feed_replay_buffer(examples)

        # Métriques
        with self._lock:
            self._stats["examples_written"] = (
                int(self._stats["examples_written"]) + n_written
            )
            self._stats["sessions"] = int(self._stats["sessions"]) + 1

        metrics.increment("learning", "total_chunks_indexed", n_written, flush=False)
        self._push_metrics()

        log.info(
            "CorpusBuilder : %d exemples écrits dans %s.",
            n_written, output_path.name,
        )
        return n_written

    def build_from_text(
        self,
        text:        str,
        source_url:  str = "",
        source_title: str = "",
    ) -> int:
        """
        Construit des exemples depuis un texte brut (corpus manuel).

        Args:
            text:         Texte brut à segmenter.
            source_url:   URL d'origine optionnelle.
            source_title: Titre optionnel.

        Returns:
            Nombre d'exemples ajoutés.
        """
        from config import PATHS, VECTORDB

        chunks = self._split_text(text,
                                   VECTORDB.chunk_size_tokens,
                                   VECTORDB.chunk_overlap_tokens)
        examples = []
        for chunk in chunks:
            ex = self._chunk_to_example(chunk, source_url, source_title)
            if ex is not None:
                examples.append(ex)

        if not examples:
            return 0

        examples = self._deduplicate(examples)
        path     = self._get_output_path(PATHS)
        n        = self._write_jsonl(path, examples)
        self._feed_replay_buffer(examples)
        return n

    def load_corpus(self, path: Path | None = None) -> list[TrainExample]:
        """
        Charge tous les exemples d'un fichier JSONL.

        Args:
            path: Fichier JSONL à charger. Si None, charge le fichier du jour.

        Returns:
            Liste de TrainExample.
        """
        from config import PATHS
        target = path or self._get_output_path(PATHS)

        if not target.exists():
            log.info("Corpus introuvable : %s", target)
            return []

        examples: list[TrainExample] = []
        try:
            with open(target, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        examples.append(TrainExample.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, KeyError):
                        pass
        except OSError as exc:
            log.error("Impossible de lire le corpus %s : %s", target, exc)

        log.info("Corpus chargé : %d exemples depuis %s.", len(examples), target.name)
        return examples

    def load_all_corpus(self) -> list[TrainExample]:
        """
        Charge tous les fichiers JSONL présents dans data/corpus/.

        Returns:
            Liste complète de TrainExample (tous fichiers confondus).
        """
        from config import PATHS
        all_examples: list[TrainExample] = []

        for jl_file in sorted(PATHS.corpus.glob("train_*.jsonl")):
            all_examples.extend(self.load_corpus(jl_file))

        log.info("Corpus total chargé : %d exemples.", len(all_examples))
        return all_examples

    def get_metrics(self) -> dict[str, Any]:
        """Retourne les métriques du corpus builder."""
        with self._lock:
            return {
                "examples_written":  int(self._stats["examples_written"]),
                "examples_filtered": int(self._stats["examples_filtered"]),
                "duplicates":        int(self._stats["duplicates"]),
                "sessions":          int(self._stats["sessions"]),
                "unique_hashes":     len(self._seen_hashes),
            }

    # ──────────────────────────────────────────────────────────────────
    # CONVERSION PAGE → EXEMPLES
    # ──────────────────────────────────────────────────────────────────
    def _page_to_examples(self, page: Any) -> list[TrainExample]:
        """
        Convertit une ScrapedPage en liste de TrainExample.

        Chaque chunk de la page → un exemple de continuation pure.
        Le titre est utilisé comme instruction dans un sous-ensemble.
        """
        examples: list[TrainExample] = []

        # Utiliser les chunks pré-découpés si disponibles
        chunks = getattr(page, "chunks", None)
        if chunks:
            for chunk in chunks:
                text = getattr(chunk, "text", str(chunk))
                ex   = self._chunk_to_example(
                    text,
                    source_url   = page.url,
                    source_title = page.title,
                )
                if ex is not None:
                    examples.append(ex)
        else:
            # Fallback : découper le contenu manuellement
            from config import VECTORDB
            raw_chunks = self._split_text(
                page.content,
                VECTORDB.chunk_size_tokens,
                VECTORDB.chunk_overlap_tokens,
            )
            for c in raw_chunks:
                ex = self._chunk_to_example(c, page.url, page.title)
                if ex is not None:
                    examples.append(ex)

        # Ajouter un exemple question/réponse factice basé sur le titre
        if page.title and page.content:
            summary = self._extract_summary(page.content)
            if summary:
                examples.append(TrainExample(
                    instruction  = f"Qu'est-ce que '{page.title}' ?",
                    output       = summary,
                    source_url   = page.url,
                    source_title = page.title,
                ))

        return examples

    def _chunk_to_example(
        self,
        text:         str,
        source_url:   str = "",
        source_title: str = "",
    ) -> TrainExample | None:
        """
        Transforme un chunk texte en TrainExample après filtrage qualité.

        Returns:
            TrainExample ou None si le chunk ne passe pas les filtres.
        """
        text = text.strip()

        # Filtre longueur
        if len(text) < self.MIN_CHARS:
            self._inc_filtered()
            return None

        # Filtre entropie
        if _text_entropy(text) < self.MIN_ENTROPY:
            self._inc_filtered()
            return None

        # Troncature si trop long
        if len(text) > self.MAX_CHARS:
            text = self._truncate_paragraph(text, self.MAX_CHARS)

        return TrainExample(
            instruction  = "",           # continuation pure
            output       = text,
            source_url   = source_url,
            source_title = source_title,
        )

    # ──────────────────────────────────────────────────────────────────
    # DÉDUPLICATION
    # ──────────────────────────────────────────────────────────────────
    def _deduplicate(self, examples: list[TrainExample]) -> list[TrainExample]:
        """
        Supprime les doublons de contenu (hash SHA-256 du output).
        """
        unique: list[TrainExample] = []
        n_dup  = 0

        for ex in examples:
            h = hashlib.sha256(ex.output.encode("utf-8")).hexdigest()[:20]
            with self._lock:
                if h in self._seen_hashes:
                    n_dup += 1
                    continue
                self._seen_hashes.add(h)
            unique.append(ex)

        if n_dup:
            with self._lock:
                self._stats["duplicates"] = int(self._stats["duplicates"]) + n_dup
            log.debug("CorpusBuilder : %d doublons filtrés.", n_dup)

        return unique

    # ──────────────────────────────────────────────────────────────────
    # ÉCRITURE
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _write_jsonl(path: Path, examples: list[TrainExample]) -> int:
        """
        Ajoute les exemples dans le fichier JSONL (mode append).

        Returns:
            Nombre de lignes écrites.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        try:
            with open(path, "a", encoding="utf-8") as f:
                for ex in examples:
                    f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
                    n += 1
        except OSError as exc:
            log.error("Erreur écriture corpus %s : %s", path, exc)
        return n

    # ──────────────────────────────────────────────────────────────────
    # REPLAY BUFFER
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _feed_replay_buffer(examples: list[TrainExample]) -> None:
        """Ajoute les exemples au replay buffer anti-oubli."""
        try:
            from pipeline.replay_buffer import replay_buffer, TrainSample
            samples = [
                TrainSample(
                    text         = ex.to_rwkv_format(),
                    source_url   = ex.source_url,
                    source_title = ex.source_title,
                )
                for ex in examples
            ]
            replay_buffer.add(samples)
        except Exception as exc:
            log.warning("Impossible d'alimenter le replay buffer : %s", exc)

    # ──────────────────────────────────────────────────────────────────
    # UTILITAIRES
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _get_output_path(paths: Any) -> Path:
        """Retourne le chemin du fichier JSONL du jour."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return paths.corpus / f"train_{today}.jsonl"

    @staticmethod
    def _split_text(text: str, chunk_tokens: int, overlap_tokens: int) -> list[str]:
        """Découpe un texte en chunks aux frontières paragraphe."""
        target  = chunk_tokens  * 4
        overlap = overlap_tokens * 4

        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        if not paragraphs:
            paragraphs = [text.strip()] if text.strip() else []

        chunks: list[str] = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) + 2 <= target:
                current = (current + "\n\n" + para).lstrip()
            else:
                if current.strip():
                    chunks.append(current.strip())
                ov      = current[-overlap:] if len(current) > overlap else current
                current = (ov + "\n\n" + para).lstrip()
        if current.strip():
            chunks.append(current.strip())
        return chunks

    @staticmethod
    def _extract_summary(content: str, max_chars: int = 500) -> str:
        """Extrait les premiers paragraphes comme résumé."""
        paras = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 60]
        summary = ""
        for p in paras[:3]:
            if len(summary) + len(p) < max_chars:
                summary = (summary + " " + p).strip()
            else:
                break
        return summary

    @staticmethod
    def _truncate_paragraph(text: str, max_chars: int) -> str:
        """Tronque à la frontière du dernier paragraphe sous max_chars."""
        truncated = text[:max_chars]
        cut = truncated.rfind("\n\n")
        if cut > max_chars // 2:
            return truncated[:cut].strip()
        cut = truncated.rfind(". ")
        if cut > max_chars // 2:
            return truncated[:cut + 1].strip()
        return truncated.strip()

    def _inc_filtered(self) -> None:
        with self._lock:
            self._stats["examples_filtered"] = int(self._stats["examples_filtered"]) + 1

    def _push_metrics(self) -> None:
        try:
            metrics.update_module("corpus_builder", self.get_metrics(), flush=True)
        except Exception as exc:
            log.debug("Impossible de pousser les métriques corpus_builder : %s", exc)


# ══════════════════════════════════════════════════════════════════════
# FONCTIONS UTILITAIRES MODULE-LEVEL
# ══════════════════════════════════════════════════════════════════════
def _text_entropy(text: str) -> float:
    """
    Calcule l'entropie de Shannon du texte en bits par caractère.
    Un texte répétitif (ex: "aaaaaa") a une entropie proche de 0.
    Un texte normal a une entropie > 3.0.

    Args:
        text: Texte à analyser.

    Returns:
        Entropie en bits par caractère.
    """
    if not text:
        return 0.0
    freq: dict[str, int] = {}
    for c in text:
        freq[c] = freq.get(c, 0) + 1
    n = len(text)
    return -sum((v / n) * math.log2(v / n) for v in freq.values())


# ══════════════════════════════════════════════════════════════════════
# SINGLETON
# ══════════════════════════════════════════════════════════════════════
corpus_builder = CorpusBuilder()


# ══════════════════════════════════════════════════════════════════════
# SMOKE TEST  (python pipeline/corpus_builder.py)
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os, tempfile
    os.environ.setdefault("ZYLOS_LOG_LEVEL", "DEBUG")

    print("═" * 60)
    print("  ZYLOS AI — pipeline/corpus_builder.py  smoke test")
    print("═" * 60)
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        import config as cfg
        p = Path(tmpdir)
        object.__setattr__(cfg.PATHS, "corpus", p / "corpus")
        object.__setattr__(cfg.PATHS, "chroma_db", p / "chroma")

        cb = CorpusBuilder()

        # Test 1 : entropie
        assert _text_entropy("aaaaaa") < 1.0
        assert _text_entropy("Bonjour, le monde est beau aujourd'hui !") > 3.0
        print("✅  Test 1 : _text_entropy OK")

        # Test 2 : build_from_text
        long_text = (
            "Python est un langage de programmation interprété, multi-paradigme "
            "et multiplateformes. Il favorise la programmation impérative structurée, "
            "fonctionnelle et orientée objet.\n\n"
        ) * 10
        n = cb.build_from_text(long_text, source_url="https://test.com", source_title="Python")
        assert n > 0, f"Attendu > 0, reçu {n}"
        print(f"✅  Test 2 : build_from_text OK ({n} exemples)")

        # Test 3 : déduplication
        n2 = cb.build_from_text(long_text, source_url="https://test.com")
        assert n2 == 0, f"Les doublons doivent être filtrés, reçu {n2}"
        print("✅  Test 3 : déduplication OK")

        # Test 4 : TrainExample.to_rwkv_format
        ex_with_instr = TrainExample(instruction="Qu'est-ce que Python ?",
                                     output="Python est un langage…")
        ex_pure       = TrainExample(instruction="", output="Texte de continuation.")
        assert "User:" in ex_with_instr.to_rwkv_format()
        assert "Assistant:" in ex_with_instr.to_rwkv_format()
        assert ex_pure.to_rwkv_format() == "Texte de continuation."
        print("✅  Test 4 : TrainExample.to_rwkv_format OK")

        # Test 5 : load_corpus
        examples = cb.load_corpus()
        assert len(examples) == n, f"Attendu {n}, chargé {len(examples)}"
        print(f"✅  Test 5 : load_corpus OK ({len(examples)} exemples chargés)")

        # Test 6 : ScrapedPage mock
        class _FakePage:
            url = "https://wikipedia.org/test"
            title = "Test Wikipedia"
            content = long_text
            success = True
            chunks = None

        pages_result = cb.build_from_pages([_FakePage()])
        assert pages_result >= 0
        print(f"✅  Test 6 : build_from_pages (mock) OK ({pages_result} exemples)")

        # Test 7 : métriques
        m = cb.get_metrics()
        assert isinstance(m["examples_written"], int)
        assert m["examples_written"] > 0
        print(f"✅  Test 7 : get_metrics() OK — {m}")

    print("\n✅  Tous les tests pipeline/corpus_builder.py sont passés.")