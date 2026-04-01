"""
modules/vectordb.py — Mémoire vectorielle infinie de ZYLOS AI
==============================================================
Base de connaissances persistante basée sur ChromaDB local.
Les embeddings sont générés exclusivement par RWKV local
(zéro appel API — cohérent avec l'architecture ZYLOS).

Fonctionnalités :
  - Ajout de pages web scrapées (chunking déjà effectué par scraper.py)
  - Recherche sémantique cosine similarity avec score de pertinence
  - Déduplication automatique des quasi-doublons (similarité > 0.98)
  - Embeddings via RWKV local (fallback hash-TF-IDF si RWKV non chargé)
  - Statistiques complètes exportées vers utils/metrics.py

Usage :
    from modules.vectordb import vectordb
    vectordb.add_page(url, title, content)
    results = vectordb.search("qu'est-ce que Python ?", k=5)
    context = vectordb.format_context(results)
"""

from __future__ import annotations

import hashlib
import math
import re
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from utils.logger import get_logger
from utils.metrics import metrics

log = get_logger(__name__)

try:
    import chromadb                              # type: ignore[import]
    from chromadb.config import Settings        # type: ignore[import]
    _HAS_CHROMA = True
except ImportError:
    _HAS_CHROMA = False
    log.warning("chromadb non installé — pip install chromadb")

try:
    import numpy as np                          # type: ignore[import]
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


# ══════════════════════════════════════════════════════════════════════
# STRUCTURES DE DONNÉES
# ══════════════════════════════════════════════════════════════════════
@dataclass
class MemoryEntry:
    """
    Entrée retournée par une recherche vectorielle.

    Attributes:
        text:      Texte du chunk retrouvé.
        url:       URL source.
        title:     Titre de la page source.
        score:     Score de similarité cosine [0.0, 1.0].
        chunk_idx: Index du chunk dans la page d'origine.
        doc_id:    Identifiant interne ChromaDB.
    """
    text:      str
    url:       str
    title:     str
    score:     float
    chunk_idx: int = 0
    doc_id:    str = ""

    @property
    def is_relevant(self) -> bool:
        """True si le score dépasse le seuil configuré dans config.BRAIN."""
        from config import BRAIN
        return self.score >= BRAIN.rag_min_score


# ══════════════════════════════════════════════════════════════════════
# MOTEUR D'EMBEDDINGS DE SECOURS (sans RWKV)
# ══════════════════════════════════════════════════════════════════════
class _FallbackEmbedder:
    """
    Embeddings TF-IDF sparse projetés en vecteur dense de dimension fixe.
    Utilisé uniquement si RWKV n'est pas encore chargé.
    Zéro dépendance externe.
    """

    DIM = 512

    def embed(self, text: str) -> list[float]:
        """
        Génère un vecteur dense à partir du texte via TF-IDF réduit.

        Args:
            text: Texte à encoder.

        Returns:
            Vecteur flottant normalisé de dimension DIM.
        """
        tokens = re.findall(r"\b[a-zàâäéèêëîïôùûüç]{3,}\b", text.lower())
        if not tokens:
            return [0.0] * self.DIM

        tf: dict[str, float] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0.0) + 1.0
        n = len(tokens)
        tf = {k: v / n for k, v in tf.items()}

        vec = [0.0] * self.DIM
        for word, weight in tf.items():
            h = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.DIM
            vec[h] += weight

        return _normalize(vec)


# ══════════════════════════════════════════════════════════════════════
# BASE VECTORIELLE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════
class VectorDB:
    """
    Mémoire vectorielle persistante de ZYLOS AI (ChromaDB local).

    Singleton exposé via `vectordb` en bas de fichier.
    La collection ChromaDB est créée automatiquement au premier accès.
    Les embeddings sont générés par RWKV local ; un fallback TF-IDF
    est utilisé si RWKV n'est pas encore disponible.
    """

    def __init__(self) -> None:
        self._client:     Any = None
        self._collection: Any = None
        self._embedder        = _FallbackEmbedder()
        self._rwkv:       Any = None
        self._lock            = threading.Lock()
        self._stats: dict[str, Any] = {
            "chunks_added":    0,
            "searches":        0,
            "hits":            0,
            "search_times_ms": [],
            "dedup_removed":   0,
        }

    # ──────────────────────────────────────────────────────────────────
    # INITIALISATION PARESSEUSE
    # ──────────────────────────────────────────────────────────────────
    def _ensure_ready(self) -> bool:
        """
        Initialise ChromaDB et la collection si nécessaire.
        Appelé automatiquement par toutes les méthodes publiques.

        Returns:
            True si ChromaDB est opérationnel.
        """
        if self._collection is not None:
            return True
        if not _HAS_CHROMA:
            log.error("chromadb non disponible — pip install chromadb")
            return False

        from config import PATHS, VECTORDB
        try:
            PATHS.chroma_db.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path     = str(PATHS.chroma_db),
                settings = Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name     = VECTORDB.collection_name,
                metadata = {"hnsw:space": "cosine"},
            )
            count = self._collection.count()
            log.info("ChromaDB prêt : '%s' — %d chunks.", VECTORDB.collection_name, count)
            metrics.update("memory", {"vector_entries": count}, flush=False)
            return True
        except Exception as exc:
            log.error("Impossible d'initialiser ChromaDB : %s", exc)
            return False

    def set_rwkv(self, rwkv_model: Any) -> None:
        """
        Injecte la référence vers le modèle RWKV pour les embeddings.
        Appelé par core/model.py après chargement du modèle.

        Args:
            rwkv_model: Instance exposant get_embeddings(text) → list[float].
        """
        self._rwkv = rwkv_model
        log.info("VectorDB : embeddings RWKV activés.")

    # ──────────────────────────────────────────────────────────────────
    # ÉCRITURE
    # ──────────────────────────────────────────────────────────────────
    def add_page(self, url: str, title: str, content: str) -> int:
        """
        Indexe une page web dans la base vectorielle.

        Découpe le contenu en chunks, génère un embedding par chunk
        et insère (ou met à jour) dans ChromaDB.

        Args:
            url:     URL source de la page.
            title:   Titre de la page.
            content: Texte nettoyé (Markdown ou texte brut).

        Returns:
            Nombre de chunks effectivement indexés.

        Example:
            >>> n = vectordb.add_page("https://example.com", "Exemple", texte)
        """
        if not self._ensure_ready():
            return 0

        from config import VECTORDB
        chunks = _split_chunks(content, VECTORDB.chunk_size_tokens,
                               VECTORDB.chunk_overlap_tokens)
        if not chunks:
            log.warning("add_page : contenu vide pour %s", url)
            return 0

        ids, embeddings, documents, metadatas = [], [], [], []
        for idx, chunk_text in enumerate(chunks):
            ids.append(_chunk_id(url, idx))
            embeddings.append(self._embed(chunk_text))
            documents.append(chunk_text)
            metadatas.append({
                "url":       url,
                "title":     title,
                "chunk_idx": idx,
                "token_est": max(1, len(chunk_text) // 4),
            })

        try:
            with self._lock:
                self._collection.upsert(
                    ids=ids, embeddings=embeddings,
                    documents=documents, metadatas=metadatas,
                )
                self._stats["chunks_added"] = int(self._stats["chunks_added"]) + len(ids)

            total = self._collection.count()
            metrics.update("memory", {"vector_entries": total}, flush=False)
            metrics.increment("learning", "total_chunks_indexed", len(ids), flush=False)
            self._push_metrics()
            log.info("add_page : %d chunks indexés pour %s (total : %d)", len(ids), url, total)
            return len(ids)
        except Exception as exc:
            log.error("Erreur ChromaDB add_page (%s) : %s", url, exc)
            return 0

    def add_chunks_from_page(self, page: Any) -> int:
        """
        Raccourci : indexe une ScrapedPage directement depuis scraper.py.

        Args:
            page: Instance de modules.scraper.ScrapedPage.

        Returns:
            Nombre de chunks ajoutés.
        """
        return self.add_page(page.url, page.title, page.content)

    # ──────────────────────────────────────────────────────────────────
    # RECHERCHE
    # ──────────────────────────────────────────────────────────────────
    def search(self, query: str, k: int | None = None) -> list[MemoryEntry]:
        """
        Recherche les chunks les plus pertinents pour une requête.

        Args:
            query: Texte de la requête.
            k:     Nombre de résultats (défaut : config.VECTORDB.top_k_results).

        Returns:
            Liste de MemoryEntry triée par score décroissant,
            filtrée par config.BRAIN.rag_min_score.

        Example:
            >>> results = vectordb.search("qu'est-ce que Python ?")
        """
        if not self._ensure_ready():
            return []

        from config import VECTORDB, BRAIN
        n_results = k or VECTORDB.top_k_results

        if self._collection.count() == 0:
            return []

        t0 = time.perf_counter()
        try:
            results = self._collection.query(
                query_embeddings = [self._embed(query)],
                n_results        = min(n_results, self._collection.count()),
                include          = ["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            log.error("Erreur ChromaDB search : %s", exc)
            return []

        elapsed_ms = (time.perf_counter() - t0) * 1_000

        docs      = results.get("documents",  [[]])[0]
        metas     = results.get("metadatas",  [[]])[0]
        distances = results.get("distances",  [[]])[0]
        ids_list  = results.get("ids",        [[]])[0]

        entries: list[MemoryEntry] = []
        for doc, meta, dist, doc_id in zip(docs, metas, distances, ids_list):
            score = max(0.0, 1.0 - dist / 2.0)
            if score < BRAIN.rag_min_score:
                continue
            entries.append(MemoryEntry(
                text      = doc,
                url       = meta.get("url", ""),
                title     = meta.get("title", ""),
                score     = round(score, 4),
                chunk_idx = int(meta.get("chunk_idx", 0)),
                doc_id    = doc_id,
            ))

        entries.sort(key=lambda e: e.score, reverse=True)

        with self._lock:
            self._stats["searches"] = int(self._stats["searches"]) + 1
            if entries:
                self._stats["hits"] = int(self._stats["hits"]) + 1
            times = self._stats["search_times_ms"]
            if isinstance(times, list):
                times.append(elapsed_ms)
                if len(times) > 100:
                    times.pop(0)

        self._update_search_metrics()
        log.debug("search('%s…') → %d résultats en %.0f ms",
                  query[:40], len(entries), elapsed_ms)
        return entries

    # ──────────────────────────────────────────────────────────────────
    # FORMATAGE CONTEXTE PROMPT
    # ──────────────────────────────────────────────────────────────────
    def format_context(self, entries: list[MemoryEntry]) -> str:
        """
        Formate les résultats de recherche pour injection dans le prompt RWKV.

        Format par entrée :
            [Source: <title> — <url>]
            <texte du chunk>

        Args:
            entries: Liste de MemoryEntry issue de search().

        Returns:
            Chaîne multi-lignes prête à être insérée dans le contexte.
            Chaîne vide si aucune entrée.

        Example:
            >>> ctx = vectordb.format_context(vectordb.search(query))
            >>> prompt = f"MÉMOIRE:\n{ctx}\n\nQUESTION: {query}"
        """
        if not entries:
            return ""
        parts = []
        for e in entries:
            source = f"[Source: {e.title} — {e.url}]" if e.url else f"[Source: {e.title}]"
            parts.append(f"{source}\n{e.text.strip()}")
        return "\n\n".join(parts)

    # ──────────────────────────────────────────────────────────────────
    # DÉDUPLICATION
    # ──────────────────────────────────────────────────────────────────
    def deduplicate(self) -> int:
        """
        Supprime les chunks quasi-doublons (similarité > seuil configuré).

        Returns:
            Nombre de chunks supprimés.
        """
        if not self._ensure_ready():
            return 0

        from config import VECTORDB
        threshold = VECTORDB.dedup_similarity_threshold
        total     = self._collection.count()
        if total < 2:
            return 0

        log.info("Déduplication (%d chunks, seuil %.2f)…", total, threshold)
        to_delete: set[str] = set()

        try:
            all_data = self._collection.get(include=["embeddings"])
            all_ids  = all_data.get("ids", [])
            all_embs = all_data.get("embeddings", [])

            for i in range(len(all_ids)):
                if all_ids[i] in to_delete:
                    continue
                for j in range(i + 1, min(i + 100, len(all_ids))):
                    if all_ids[j] in to_delete:
                        continue
                    if _cosine_similarity(all_embs[i], all_embs[j]) >= threshold:
                        to_delete.add(all_ids[j])

            if to_delete:
                self._collection.delete(ids=list(to_delete))
                with self._lock:
                    self._stats["dedup_removed"] = (
                        int(self._stats["dedup_removed"]) + len(to_delete)
                    )
                metrics.update("memory", {"vector_entries": self._collection.count()}, flush=True)
                log.info("Déduplication : %d supprimés.", len(to_delete))

        except Exception as exc:
            log.error("Erreur déduplication : %s", exc)

        return len(to_delete)

    # ──────────────────────────────────────────────────────────────────
    # MÉTRIQUES
    # ──────────────────────────────────────────────────────────────────
    def get_metrics(self) -> dict[str, Any]:
        """
        Retourne un snapshot des métriques de la base vectorielle.

        Returns:
            Dictionnaire avec total chunks, hit rate, latence moy., etc.
        """
        with self._lock:
            searches = int(self._stats["searches"])
            hits     = int(self._stats["hits"])
            times    = list(self._stats["search_times_ms"])

        total    = self._collection.count() if self._collection else 0
        hit_rate = hits / searches if searches else 0.0
        avg_ms   = sum(times) / len(times) if times else 0.0

        return {
            "vector_entries":    total,
            "chunks_added":      int(self._stats["chunks_added"]),
            "searches":          searches,
            "rag_hit_rate":      round(hit_rate, 4),
            "avg_search_ms":     round(avg_ms, 1),
            "dedup_removed":     int(self._stats["dedup_removed"]),
            "embedding_backend": "rwkv" if self._rwkv else "fallback_tfidf",
        }

    def count(self) -> int:
        """Retourne le nombre total de chunks indexés."""
        if not self._ensure_ready():
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0

    # ──────────────────────────────────────────────────────────────────
    # EMBEDDINGS
    # ──────────────────────────────────────────────────────────────────
    def _embed(self, text: str) -> list[float]:
        """RWKV local si disponible, sinon fallback TF-IDF."""
        if self._rwkv is not None:
            try:
                vec = self._rwkv.get_embeddings(text)
                if vec:
                    return _normalize(vec)
            except Exception as exc:
                log.debug("Embedding RWKV échoué (%s) — fallback.", exc)
        return self._embedder.embed(text)

    def _update_search_metrics(self) -> None:
        m = self.get_metrics()
        metrics.update("memory", {
            "vector_entries": m["vector_entries"],
            "rag_hit_rate":   m["rag_hit_rate"],
            "avg_search_ms":  m["avg_search_ms"],
        }, flush=False)
        self._push_metrics()

    def _push_metrics(self) -> None:
        try:
            metrics.update_module("vectordb", self.get_metrics(), flush=True)
        except Exception as exc:
            log.debug("Impossible de pousser les métriques vectordb : %s", exc)


# ══════════════════════════════════════════════════════════════════════
# UTILITAIRES MODULE-LEVEL
# ══════════════════════════════════════════════════════════════════════
def _chunk_id(url: str, idx: int) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:12] + f"_{idx:04d}"


def _split_chunks(text: str, chunk_tokens: int, overlap_tokens: int) -> list[str]:
    """Découpe un texte en chunks aux frontières paragraphe/phrase."""
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
            ov = current[-overlap:] if len(current) > overlap else current
            current = (ov + "\n\n" + para).lstrip()

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if len(c) >= 20]


def _normalize(vec: list[float]) -> list[float]:
    """Normalisation L2 d'un vecteur."""
    if _HAS_NUMPY:
        arr  = np.array(vec, dtype=np.float32)
        norm = float(np.linalg.norm(arr))
        return (arr / norm).tolist() if norm > 1e-10 else vec
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 1e-10 else vec


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Similarité cosine entre deux vecteurs."""
    if _HAS_NUMPY:
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
        return float(np.dot(va, vb) / (na * nb)) if na > 1e-10 and nb > 1e-10 else 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na > 1e-10 and nb > 1e-10 else 0.0


# ══════════════════════════════════════════════════════════════════════
# INSTANCE GLOBALE SINGLETON
# ══════════════════════════════════════════════════════════════════════
vectordb = VectorDB()


# ══════════════════════════════════════════════════════════════════════
# SMOKE TEST  (python modules/vectordb.py)
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os, tempfile
    os.environ.setdefault("ZYLOS_LOG_LEVEL", "DEBUG")

    print("═" * 60)
    print("  ZYLOS AI — modules/vectordb.py  smoke test")
    print("═" * 60)
    print()

    # Test 1 : utilitaires
    assert abs(_cosine_similarity([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-6
    assert abs(_cosine_similarity([1.0, 0.0], [0.0, 1.0]) - 0.0) < 1e-6
    vec = _normalize([3.0, 4.0])
    assert abs(sum(x**2 for x in vec) - 1.0) < 1e-5
    print("✅  Test 1 : cosine_similarity / normalize OK")

    # Test 2 : fallback embedder
    emb = _FallbackEmbedder()
    v1  = emb.embed("Python est un langage de programmation")
    v3  = emb.embed("Python est un langage de programmation")
    assert len(v1) == _FallbackEmbedder.DIM
    assert v1 == v3, "Non déterministe"
    print(f"✅  Test 2 : fallback embedder OK (dim={len(v1)})")

    # Test 3 : split_chunks
    long_text = ("Paragraphe de test. " * 20 + "\n\n") * 6
    chunks = _split_chunks(long_text, 100, 10)
    assert len(chunks) > 1
    print(f"✅  Test 3 : _split_chunks OK ({len(chunks)} chunks)")

    # Test 4 : MemoryEntry.is_relevant
    assert MemoryEntry(text="x", url="u", title="t", score=0.85).is_relevant
    assert not MemoryEntry(text="x", url="u", title="t", score=0.20).is_relevant
    print("✅  Test 4 : MemoryEntry.is_relevant OK")

    # Test 5 : ChromaDB end-to-end
    if _HAS_CHROMA:
        with tempfile.TemporaryDirectory() as tmpdir:
            import config as cfg
            object.__setattr__(cfg.PATHS, "chroma_db", Path(tmpdir) / "chroma")
            db = VectorDB()
            n  = db.add_page("https://test.com", "Test",
                             "Python est un langage interprété.\n\n"
                             "Créé par Guido van Rossum en 1991.\n\n"
                             "Très utilisé en data science et IA.")
            assert n > 0
            results = db.search("langage de programmation Python")
            assert isinstance(results, list)
            ctx = db.format_context(results)
            assert isinstance(ctx, str)
            m = db.get_metrics()
            assert m["chunks_added"] == n
        print(f"✅  Test 5 : ChromaDB end-to-end OK ({n} chunks)")
    else:
        print("⚠   Test 5 ignoré (chromadb non installé)")

    print("\n✅  Tous les tests modules/vectordb.py sont passés.")