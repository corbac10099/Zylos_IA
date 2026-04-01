"""
modules/scraper.py — Web scraper robuste de ZYLOS AI
======================================================
Transforme des URLs en contenu Markdown structuré prêt à être indexé
dans la base vectorielle. Fournit :
  - Extraction principale via trafilatura (HTML → texte propre)
  - Fallback BeautifulSoup si trafilatura échoue
  - Throttling configurable + respect optionnel de robots.txt
  - Déduplication par hash URL et hash contenu
  - Chunking intelligent aux frontières phrase/paragraphe
  - Métriques complètes par session

Usage :
    from modules.scraper import scraper
    pages = scraper.scrape_urls(["https://fr.wikipedia.org/wiki/Python"])
    for page in pages:
        print(page.title, len(page.chunks), "chunks")
"""

from __future__ import annotations

import hashlib
import re
import time
import threading
import urllib.request
import urllib.robotparser
import urllib.error
from dataclasses import dataclass, field
from typing import Iterator
from urllib.parse import urlparse

from utils.logger import get_logger
from utils.metrics import metrics

log = get_logger(__name__)

# Imports optionnels — gérés au runtime
try:
    import trafilatura                      # type: ignore[import]
    _HAS_TRAFILATURA = True
except ImportError:
    _HAS_TRAFILATURA = False
    log.warning("trafilatura non installé — utilisation du fallback BeautifulSoup.")

try:
    from bs4 import BeautifulSoup          # type: ignore[import]
    _HAS_BS4 = True
except ImportError:
    _HAS_BS4 = False


# ══════════════════════════════════════════════════════════════════════
# STRUCTURES DE DONNÉES
# ══════════════════════════════════════════════════════════════════════
@dataclass
class PageChunk:
    """
    Fragment de texte extrait d'une page web, prêt pour l'indexation.

    Attributes:
        text:      Contenu textuel du chunk.
        url:       URL source.
        title:     Titre de la page parente.
        chunk_idx: Index du chunk dans la page (0-based).
        token_est: Estimation du nombre de tokens (4 chars ≈ 1 token).
    """
    text:      str
    url:       str
    title:     str
    chunk_idx: int
    token_est: int = field(init=False)

    def __post_init__(self) -> None:
        self.token_est = max(1, len(self.text) // 4)


@dataclass
class ScrapedPage:
    """
    Résultat complet du scraping d'une URL.

    Attributes:
        url:        URL d'origine (canonique).
        title:      Titre extrait (balise <title> ou <h1>).
        content:    Texte Markdown brut, nettoyé.
        chunks:     Fragments découpés pour l'indexation.
        domain:     Domaine de l'URL (ex: "fr.wikipedia.org").
        language:   Langue détectée (code ISO 639-1, ex: "fr").
        url_hash:   SHA-256 tronqué de l'URL (déduplication).
        content_hash: SHA-256 tronqué du contenu (déduplication).
        token_count: Estimation du total de tokens dans le contenu.
        method:     Méthode d'extraction utilisée : "trafilatura" | "bs4" | "fallback".
        error:      Message d'erreur si le scraping a échoué (vide si OK).
    """
    url:          str
    title:        str       = ""
    content:      str       = ""
    chunks:       list[PageChunk] = field(default_factory=list)
    domain:       str       = ""
    language:     str       = "unknown"
    url_hash:     str       = ""
    content_hash: str       = ""
    token_count:  int       = 0
    method:       str       = ""
    error:        str       = ""

    @property
    def success(self) -> bool:
        """True si le scraping a produit du contenu utilisable."""
        return bool(self.content) and not self.error


# ══════════════════════════════════════════════════════════════════════
# SCRAPER PRINCIPAL
# ══════════════════════════════════════════════════════════════════════
class WebScraper:
    """
    Scraper web robuste avec throttling, déduplication et chunking.

    Singleton exposé via `scraper` en bas de fichier.
    Chaque instance maintient un ensemble d'URLs et de contenus
    déjà vus pour éviter les doublons sur toute la durée de vie
    du processus.

    Attributes:
        _seen_urls:     Hashes SHA-256 des URLs déjà scrapées.
        _seen_contents: Hashes SHA-256 des contenus déjà indexés.
        _robots_cache:  Cache des parseurs robots.txt par domaine.
        _lock:          Verrou pour les accès concurrents aux ensembles vus.
        _stats:         Compteurs de session (succès, erreurs, tokens…).
    """

    def __init__(self) -> None:
        self._seen_urls:     set[str]                     = set()
        self._seen_contents: set[str]                     = set()
        self._robots_cache:  dict[str, urllib.robotparser.RobotFileParser] = {}
        self._lock           = threading.Lock()
        self._stats: dict[str, int | float | list] = {
            "success":       0,
            "errors":        0,
            "duplicates":    0,
            "total_tokens":  0,
            "domains_seen":  [],
            "error_types":   {},
        }

    # ──────────────────────────────────────────────────────────────────
    # API PUBLIQUE
    # ──────────────────────────────────────────────────────────────────
    def scrape_urls(
        self,
        urls:        list[str],
        max_pages:   int | None = None,
    ) -> list[ScrapedPage]:
        """
        Scrape une liste d'URLs et retourne les pages extraites.

        Les URLs déjà vues (hash URL ou contenu identique) sont ignorées.
        Le throttling entre chaque requête est respecté automatiquement.

        Args:
            urls:      Liste d'URLs à scraper.
            max_pages: Limite optionnelle (remplace config.SCRAPER.max_pages_per_session).

        Returns:
            Liste de ScrapedPage (uniquement les succès, triées par URL).

        Example:
            >>> pages = scraper.scrape_urls(["https://fr.wikipedia.org/wiki/Python"])
            >>> print(pages[0].title, len(pages[0].chunks), "chunks")
        """
        from config import SCRAPER

        limit   = max_pages or SCRAPER.max_pages_per_session
        results : list[ScrapedPage] = []

        for idx, url in enumerate(urls):
            if len(results) >= limit:
                log.info("Limite de pages atteinte (%d) — scraping interrompu.", limit)
                break

            url = url.strip()
            if not url.startswith(("http://", "https://")):
                log.warning("URL ignorée (schéma invalide) : %s", url)
                continue

            page = self.scrape_one(url)

            if page.success:
                results.append(page)
                # Mise à jour métriques globales
                metrics.increment("learning", "total_pages_scraped", flush=False)
            else:
                log.warning("Échec scraping %s : %s", url, page.error)

            # Throttling entre requêtes (sauf après la dernière)
            if idx < len(urls) - 1 and len(results) < limit:
                time.sleep(SCRAPER.delay_between_requests)

        self._push_metrics()
        log.info(
            "Session scraping terminée : %d/%d succès, %d erreurs, %d doublons.",
            len(results), len(urls),
            self._stats["errors"], self._stats["duplicates"],
        )
        return results

    def scrape_one(self, url: str) -> ScrapedPage:
        """
        Scrape une URL unique et retourne la page extraite.

        Gère la déduplication, robots.txt, l'extraction et le chunking.

        Args:
            url: URL à scraper (doit commencer par http:// ou https://).

        Returns:
            ScrapedPage (success=False si erreur ou doublon).

        Example:
            >>> page = scraper.scrape_one("https://fr.wikipedia.org/wiki/Python")
            >>> print(page.title, page.token_count, "tokens")
        """
        page       = ScrapedPage(url=url)
        page.url_hash = _hash(url)
        page.domain   = urlparse(url).netloc

        # ── Déduplication URL ─────────────────────────────────────────
        with self._lock:
            if page.url_hash in self._seen_urls:
                self._stats["duplicates"] = int(self._stats["duplicates"]) + 1
                page.error = "URL déjà scrapée (doublon)."
                log.debug("Doublon URL ignoré : %s", url)
                return page
            self._seen_urls.add(page.url_hash)

        # ── robots.txt ────────────────────────────────────────────────
        if not self._is_allowed(url):
            page.error = "Bloqué par robots.txt."
            log.info("robots.txt interdit : %s", url)
            with self._lock:
                self._stats["errors"] = int(self._stats["errors"]) + 1
            return page

        # ── Téléchargement du HTML ────────────────────────────────────
        try:
            html = self._fetch_html(url)
        except Exception as exc:
            page.error = f"Erreur réseau : {exc}"
            self._record_error("network")
            log.warning("Erreur réseau %s : %s", url, exc)
            return page

        # ── Extraction du contenu ─────────────────────────────────────
        page = self._extract(html, page)
        if not page.content:
            page.error = "Contenu vide après extraction."
            self._record_error("empty_content")
            return page

        # ── Filtre de langue ─────────────────────────────────────────
        from config import SCRAPER
        if SCRAPER.accepted_languages and page.language not in SCRAPER.accepted_languages:
            page.error = f"Langue non acceptée : {page.language}"
            self._record_error("language_filter")
            log.debug("Langue filtrée (%s) : %s", page.language, url)
            return page

        # ── Déduplication contenu ─────────────────────────────────────
        page.content_hash = _hash(page.content)
        with self._lock:
            if page.content_hash in self._seen_contents:
                self._stats["duplicates"] = int(self._stats["duplicates"]) + 1
                page.error = "Contenu identique à une page déjà indexée."
                log.debug("Doublon contenu ignoré : %s", url)
                return page
            self._seen_contents.add(page.content_hash)

        # ── Troncature tokens ─────────────────────────────────────────
        max_tokens = SCRAPER.max_tokens_per_page
        if page.token_count > max_tokens:
            page.content = _truncate_to_tokens(page.content, max_tokens)
            page.token_count = max_tokens
            log.debug("Contenu tronqué à %d tokens : %s", max_tokens, url)

        # ── Chunking ──────────────────────────────────────────────────
        page.chunks = self._chunk(page)

        # ── Stats session ─────────────────────────────────────────────
        with self._lock:
            self._stats["success"] = int(self._stats["success"]) + 1
            self._stats["total_tokens"] = (
                int(self._stats["total_tokens"]) + page.token_count
            )
            domains = self._stats["domains_seen"]
            if isinstance(domains, list) and page.domain not in domains:
                domains.append(page.domain)

        # Mise à jour domaines dans les métriques globales
        metrics.update(
            "learning",
            {"domains_covered": list(self._stats["domains_seen"])},
            flush=False,
        )

        log.info(
            "✓ %s — %d chunks / %d tokens [%s]",
            url, len(page.chunks), page.token_count, page.method,
        )
        return page

    def iter_scrape(
        self,
        urls: list[str],
    ) -> Iterator[ScrapedPage]:
        """
        Version générateur de scrape_urls pour le traitement en streaming.

        Args:
            urls: Liste d'URLs à scraper.

        Yields:
            ScrapedPage au fur et à mesure (succès uniquement).

        Example:
            >>> for page in scraper.iter_scrape(urls):
            ...     vectordb.add_page(page.url, page.title, page.content)
        """
        from config import SCRAPER
        for idx, url in enumerate(urls):
            url  = url.strip()
            page = self.scrape_one(url)
            if page.success:
                yield page
            if idx < len(urls) - 1:
                time.sleep(SCRAPER.delay_between_requests)

    def get_metrics(self) -> dict:
        """
        Retourne les métriques de scraping de la session courante.

        Returns:
            Dictionnaire avec success_rate, avg_tokens, domaines, etc.

        Example:
            >>> print(scraper.get_metrics())
        """
        with self._lock:
            total    = int(self._stats["success"]) + int(self._stats["errors"])
            success  = int(self._stats["success"])
            avg_tok  = (
                int(self._stats["total_tokens"]) / success if success else 0
            )
            return {
                "success":        success,
                "errors":         int(self._stats["errors"]),
                "duplicates":     int(self._stats["duplicates"]),
                "success_rate":   round(success / total, 4) if total else 0.0,
                "avg_tokens":     round(avg_tok, 1),
                "domains_seen":   list(self._stats["domains_seen"]),
                "error_types":    dict(self._stats["error_types"]),
            }

    def reset_seen(self) -> None:
        """
        Réinitialise les ensembles de déduplication.
        À utiliser entre deux sessions indépendantes si nécessaire.
        """
        with self._lock:
            self._seen_urls.clear()
            self._seen_contents.clear()
        log.debug("Ensembles de déduplication réinitialisés.")

    # ──────────────────────────────────────────────────────────────────
    # TÉLÉCHARGEMENT HTML
    # ──────────────────────────────────────────────────────────────────
    def _fetch_html(self, url: str) -> str:
        """
        Télécharge le HTML brut d'une URL via urllib.

        Args:
            url: URL cible.

        Returns:
            Contenu HTML décodé en UTF-8 (ou latin-1 en fallback).

        Raises:
            OSError: Erreur réseau ou timeout.
            ValueError: HTTP 4xx/5xx.
        """
        from config import SCRAPER

        req = urllib.request.Request(
            url,
            headers={"User-Agent": SCRAPER.user_agent},
        )
        try:
            with urllib.request.urlopen(req, timeout=SCRAPER.request_timeout_sec) as resp:
                if resp.status >= 400:
                    raise ValueError(f"HTTP {resp.status}")
                raw = resp.read()
                # Décodage : UTF-8 en priorité, latin-1 en fallback
                try:
                    return raw.decode("utf-8")
                except UnicodeDecodeError:
                    return raw.decode("latin-1", errors="replace")
        except urllib.error.HTTPError as exc:
            raise ValueError(f"HTTP {exc.code} : {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise OSError(f"URLError : {exc.reason}") from exc

    # ──────────────────────────────────────────────────────────────────
    # EXTRACTION DU CONTENU
    # ──────────────────────────────────────────────────────────────────
    def _extract(self, html: str, page: ScrapedPage) -> ScrapedPage:
        """
        Extrait le contenu textuel du HTML en tentant trafilatura en priorité,
        puis BeautifulSoup, puis un fallback regex minimal.

        Args:
            html: Contenu HTML brut.
            page: ScrapedPage à compléter (url, domain déjà remplis).

        Returns:
            ScrapedPage avec content, title, language, token_count remplis.
        """
        # ── Méthode 1 : trafilatura ───────────────────────────────────
        if _HAS_TRAFILATURA:
            try:
                text = trafilatura.extract(
                    html,
                    include_comments   = False,
                    include_tables     = True,
                    no_fallback        = False,
                    output_format      = "txt",
                )
                if text and len(text.strip()) > 100:
                    page.content     = _clean_text(text)
                    page.title       = self._extract_title_trafilatura(html) or page.domain
                    page.language    = self._detect_language(page.content)
                    page.token_count = max(1, len(page.content) // 4)
                    page.method      = "trafilatura"
                    return page
            except Exception as exc:
                log.debug("trafilatura échec (%s) — fallback BS4.", exc)

        # ── Méthode 2 : BeautifulSoup ─────────────────────────────────
        if _HAS_BS4:
            try:
                soup  = BeautifulSoup(html, "html.parser")
                # Supprimer les balises inutiles
                for tag in soup(["script", "style", "nav", "footer",
                                  "header", "aside", "form"]):
                    tag.decompose()
                # Titre
                title_tag  = soup.find("title")
                h1_tag     = soup.find("h1")
                page.title = (
                    (title_tag.get_text(strip=True) if title_tag else "")
                    or (h1_tag.get_text(strip=True) if h1_tag else "")
                    or page.domain
                )
                # Contenu principal
                main = (
                    soup.find("main")
                    or soup.find("article")
                    or soup.find(id=re.compile(r"content|main|article", re.I))
                    or soup.find("body")
                    or soup
                )
                text = main.get_text(separator="\n", strip=True) if main else ""
                if text and len(text.strip()) > 100:
                    page.content     = _clean_text(text)
                    page.language    = self._detect_language(page.content)
                    page.token_count = max(1, len(page.content) // 4)
                    page.method      = "bs4"
                    return page
            except Exception as exc:
                log.debug("BeautifulSoup échec (%s) — fallback regex.", exc)

        # ── Méthode 3 : fallback regex minimal ───────────────────────
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s{2,}", " ", text).strip()
        if len(text) > 100:
            page.content     = _clean_text(text)
            page.title       = page.domain
            page.language    = self._detect_language(page.content)
            page.token_count = max(1, len(page.content) // 4)
            page.method      = "fallback"

        return page

    @staticmethod
    def _extract_title_trafilatura(html: str) -> str:
        """Extrait le titre via une regex légère sur la balise <title>."""
        m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()
        return ""

    # ──────────────────────────────────────────────────────────────────
    # CHUNKING INTELLIGENT
    # ──────────────────────────────────────────────────────────────────
    def _chunk(self, page: ScrapedPage) -> list[PageChunk]:
        """
        Découpe le contenu en chunks aux frontières paragraphe/phrase.

        Taille cible : config.VECTORDB.chunk_size_tokens avec overlap de
        config.VECTORDB.chunk_overlap_tokens tokens.

        Args:
            page: ScrapedPage avec content rempli.

        Returns:
            Liste de PageChunk prête pour l'indexation.
        """
        from config import VECTORDB

        target_chars   = VECTORDB.chunk_size_tokens   * 4   # 1 token ≈ 4 chars
        overlap_chars  = VECTORDB.chunk_overlap_tokens * 4

        # Découpe le texte en paragraphes (double saut de ligne)
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", page.content) if p.strip()]
        if not paragraphs:
            paragraphs = [page.content]

        chunks : list[PageChunk] = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 1 <= target_chars:
                current = (current + "\n\n" + para).lstrip()
            else:
                if current:
                    chunks.append(PageChunk(
                        text      = current,
                        url       = page.url,
                        title     = page.title,
                        chunk_idx = len(chunks),
                    ))
                # Overlap : reprendre les derniers caractères du chunk précédent
                overlap_text = current[-overlap_chars:] if len(current) > overlap_chars else current
                current      = (overlap_text + "\n\n" + para).lstrip()

        if current.strip():
            chunks.append(PageChunk(
                text      = current.strip(),
                url       = page.url,
                title     = page.title,
                chunk_idx = len(chunks),
            ))

        # Si un paragraphe seul dépasse la taille cible → découpe par phrases
        final_chunks: list[PageChunk] = []
        for chunk in chunks:
            if len(chunk.text) <= target_chars * 1.5:
                final_chunks.append(chunk)
            else:
                final_chunks.extend(
                    self._split_by_sentences(chunk, target_chars, overlap_chars)
                )

        # Renumérotation finale
        for i, c in enumerate(final_chunks):
            c.chunk_idx = i

        return final_chunks

    @staticmethod
    def _split_by_sentences(
        chunk:         PageChunk,
        target_chars:  int,
        overlap_chars: int,
    ) -> list[PageChunk]:
        """
        Découpe un chunk trop long en sous-chunks à la frontière des phrases.

        Args:
            chunk:         Chunk source à subdiviser.
            target_chars:  Taille cible en caractères.
            overlap_chars: Overlap en caractères entre les sous-chunks.

        Returns:
            Liste de PageChunk plus petits.
        """
        sentences = re.split(r"(?<=[.!?])\s+", chunk.text)
        sub_chunks: list[PageChunk] = []
        current = ""

        for sent in sentences:
            if len(current) + len(sent) + 1 <= target_chars:
                current = (current + " " + sent).lstrip()
            else:
                if current:
                    sub_chunks.append(PageChunk(
                        text=current, url=chunk.url,
                        title=chunk.title, chunk_idx=len(sub_chunks),
                    ))
                overlap_text = current[-overlap_chars:] if len(current) > overlap_chars else ""
                current      = (overlap_text + " " + sent).lstrip()

        if current.strip():
            sub_chunks.append(PageChunk(
                text=current.strip(), url=chunk.url,
                title=chunk.title, chunk_idx=len(sub_chunks),
            ))
        return sub_chunks

    # ──────────────────────────────────────────────────────────────────
    # ROBOTS.TXT
    # ──────────────────────────────────────────────────────────────────
    def _is_allowed(self, url: str) -> bool:
        """
        Vérifie si l'URL est autorisée selon robots.txt.
        Retourne True si SCRAPER.respect_robots_txt est False.

        Args:
            url: URL à vérifier.

        Returns:
            True si le scraping est autorisé.
        """
        from config import SCRAPER
        if not SCRAPER.respect_robots_txt:
            return True

        parsed = urlparse(url)
        base   = f"{parsed.scheme}://{parsed.netloc}"

        if base not in self._robots_cache:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(f"{base}/robots.txt")
            try:
                rp.read()
            except Exception as exc:
                log.debug("robots.txt inaccessible pour %s : %s — accès autorisé.", base, exc)
                # En cas d'erreur de lecture, on autorise par défaut
                rp = None
            self._robots_cache[base] = rp

        rp = self._robots_cache.get(base)
        if rp is None:
            return True
        return rp.can_fetch(SCRAPER.user_agent, url)

    # ──────────────────────────────────────────────────────────────────
    # DÉTECTION DE LANGUE (heuristique légère, sans dépendance)
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _detect_language(text: str) -> str:
        """
        Détecte la langue du texte via une heuristique sur les stopwords
        français et anglais. Zéro dépendance externe.

        Args:
            text: Texte à analyser (200 premiers mots suffisent).

        Returns:
            Code langue ISO 639-1 : "fr" | "en" | "unknown".
        """
        sample = " ".join(text.lower().split()[:200])

        fr_words = {"le", "la", "les", "de", "du", "et", "en", "un", "une",
                    "des", "est", "il", "elle", "que", "qui", "dans", "sur",
                    "pour", "avec", "par", "au", "aux", "mais", "ou", "donc"}
        en_words = {"the", "of", "and", "to", "in", "is", "it", "that", "was",
                    "for", "on", "are", "with", "as", "at", "be", "by", "this",
                    "an", "from", "or", "but", "not", "have", "had"}

        words    = set(sample.split())
        fr_score = len(words & fr_words)
        en_score = len(words & en_words)

        if fr_score == 0 and en_score == 0:
            return "unknown"
        return "fr" if fr_score >= en_score else "en"

    # ──────────────────────────────────────────────────────────────────
    # UTILITAIRES INTERNES
    # ──────────────────────────────────────────────────────────────────
    def _record_error(self, error_type: str) -> None:
        with self._lock:
            self._stats["errors"] = int(self._stats["errors"]) + 1
            types = self._stats["error_types"]
            if isinstance(types, dict):
                types[error_type] = types.get(error_type, 0) + 1

    def _push_metrics(self) -> None:
        try:
            metrics.update_module("scraper", self.get_metrics(), flush=True)
        except Exception as exc:
            log.debug("Impossible de pousser les métriques scraper : %s", exc)


# ══════════════════════════════════════════════════════════════════════
# FONCTIONS UTILITAIRES MODULE-LEVEL
# ══════════════════════════════════════════════════════════════════════
def _hash(text: str, length: int = 16) -> str:
    """Retourne un hash SHA-256 hexadécimal tronqué à `length` caractères."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def _clean_text(text: str) -> str:
    """
    Nettoie le texte extrait : supprime les lignes vides en excès,
    les espaces multiples et les caractères de contrôle.

    Args:
        text: Texte brut à nettoyer.

    Returns:
        Texte nettoyé.
    """
    # Supprimer les caractères de contrôle sauf \n et \t
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Normaliser les espaces horizontaux
    text = re.sub(r"[ \t]+", " ", text)
    # Réduire les sauts de ligne multiples (max 2)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    Tronque le texte pour rester sous max_tokens (estimation 4 chars/token).
    La troncature se fait à la frontière du dernier paragraphe complet.

    Args:
        text:       Texte à tronquer.
        max_tokens: Nombre maximum de tokens estimés.

    Returns:
        Texte tronqué.
    """
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    # Chercher le dernier double saut de ligne pour couper proprement
    cut = truncated.rfind("\n\n")
    if cut > max_chars // 2:
        truncated = truncated[:cut]
    return truncated.strip()


# ══════════════════════════════════════════════════════════════════════
# INSTANCE GLOBALE SINGLETON
# ══════════════════════════════════════════════════════════════════════
scraper = WebScraper()


# ══════════════════════════════════════════════════════════════════════
# SMOKE TEST  (python modules/scraper.py)
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os
    os.environ.setdefault("ZYLOS_LOG_LEVEL", "DEBUG")

    print("═" * 60)
    print("  ZYLOS AI — modules/scraper.py  smoke test")
    print("═" * 60)
    print()

    # ── Test 1 : fonctions utilitaires ───────────────────────────────
    assert len(_hash("test")) == 16, "hash longueur KO"
    assert _clean_text("  hello\n\n\n  world  ") == "hello\n\nworld"
    assert _truncate_to_tokens("a" * 10000, 100) is not None
    print("✅  Test 1 : fonctions utilitaires OK")

    # ── Test 2 : détection de langue ─────────────────────────────────
    fr_text = "Le langage Python est un langage de programmation interprété"
    en_text = "Python is a programming language that is widely used"
    assert WebScraper._detect_language(fr_text) == "fr", "Détection FR KO"
    assert WebScraper._detect_language(en_text) == "en", "Détection EN KO"
    print("✅  Test 2 : détection de langue OK")

    # ── Test 3 : chunking ────────────────────────────────────────────
    fake_page          = ScrapedPage(url="https://example.com", title="Test")
    fake_page.content  = ("Paragraph one.\n\n" * 30).strip()
    fake_page.domain   = "example.com"
    fake_page.language = "en"
    chunks = scraper._chunk(fake_page)
    assert len(chunks) > 0, "Chunking n'a produit aucun chunk"
    assert all(c.url == "https://example.com" for c in chunks)
    print(f"✅  Test 3 : chunking OK ({len(chunks)} chunks produits)")

    # ── Test 4 : déduplication URL ────────────────────────────────────
    url_test = "https://example.com/dedupe-test"
    p1 = scraper.scrape_one(url_test)   # peut échouer (réseau), mais hash enregistré
    p2 = scraper.scrape_one(url_test)   # doit être un doublon
    assert p2.error and "doublon" in p2.error.lower(), \
        f"Déduplication URL KO : {p2.error!r}"
    print("✅  Test 4 : déduplication URL OK")

    # ── Test 5 : métriques ────────────────────────────────────────────
    m = scraper.get_metrics()
    assert isinstance(m["success_rate"], float)
    assert isinstance(m["domains_seen"], list)
    print("✅  Test 5 : get_metrics() structure valide")

    print("\n✅  Tous les tests modules/scraper.py sont passés.")
    print(f"   Métriques session : {scraper.get_metrics()}")