"""
utils/metrics.py — Collecte et persistance des métriques de ZYLOS AI
======================================================================
Fournit un registre centralisé thread-safe pour toutes les métriques
du projet. Chaque module y pousse ses données via update_module().
L'état complet est persisté dans data/metrics.json à chaque mise à jour.

Usage dans n'importe quel module :
    from utils.metrics import metrics
    metrics.update_module("scraper", {"pages_scraped": 12, "errors": 0})
    snap = metrics.snapshot()   # dict complet lecture seule
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.logger import get_logger

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════
# STRUCTURE PAR DÉFAUT — miroir exact du Master Prompt
# ══════════════════════════════════════════════════════════════════════
def _default_state() -> dict[str, Any]:
    """Retourne la structure initiale vide des métriques globales."""
    return {
        "session": {
            "start_time":     None,      # ISO datetime, défini au premier init()
            "interactions":   0,         # nb de tours de conversation utilisateur
            "uptime_hours":   0.0,
        },
        "learning": {
            "total_pages_scraped":  0,
            "total_chunks_indexed": 0,
            "domains_covered":      [],  # liste de strings
            "last_scrape":          None,
        },
        "memory": {
            "vector_entries":  0,
            "rag_hit_rate":    0.0,      # ratio requêtes avec au moins 1 chunk utile
            "avg_search_ms":   0.0,
        },
        "model": {
            "tokens_generated":       0,
            "avg_tokens_per_sec":     0.0,
            "rwkv_calls":             0,
            "api_calls":              0,  # appels Mistral (improver uniquement)
            "api_tokens_this_month":  0,
        },
        "training": {
            "sessions_completed":   0,
            "total_samples_trained": 0,
            "best_loss":            None,
            "last_loss_delta":      0.0,
            "rollbacks":            0,
        },
        "improvements": {
            "cycles_run":       0,
            "changes_applied":  0,
            "changes_pending":  0,
            "success_rate":     0.0,
        },
        # Métriques détaillées par module (alimentées par chaque module)
        "_modules": {},
    }


# ══════════════════════════════════════════════════════════════════════
# REGISTRE CENTRALISÉ
# ══════════════════════════════════════════════════════════════════════
class MetricsRegistry:
    """
    Registre thread-safe des métriques de ZYLOS AI.

    Toutes les écritures sont protégées par un verrou.
    La persistance sur disque est déclenchée automatiquement
    à chaque mise à jour (sauf si flush=False est passé).

    Attributes:
        _state:   Dictionnaire complet des métriques en mémoire.
        _lock:    Verrou threading pour l'accès concurrent.
        _path:    Chemin vers metrics.json sur disque.
        _start_ts: Timestamp UNIX du démarrage de session.
    """

    def __init__(self) -> None:
        self._lock:     threading.Lock = threading.Lock()
        self._state:    dict[str, Any] = _default_state()
        self._path:     Path | None    = None
        self._start_ts: float          = 0.0

    # ──────────────────────────────────────────────────────────────────
    # INITIALISATION
    # ──────────────────────────────────────────────────────────────────
    def init(self, metrics_path: Path | None = None) -> None:
        """
        Initialise le registre : charge metrics.json existant ou crée
        une structure vierge, puis marque le début de la session.

        Doit être appelé une seule fois au démarrage de main.py.

        Args:
            metrics_path: Chemin vers metrics.json.
                          Si None, utilise PATHS.metrics_file depuis config.
        """
        from config import PATHS
        self._path     = metrics_path or PATHS.metrics_file
        self._start_ts = time.monotonic()

        with self._lock:
            if self._path.exists():
                try:
                    loaded = json.loads(self._path.read_text(encoding="utf-8"))
                    # Fusionne : les clés existantes écrasent les défauts
                    self._deep_merge(self._state, loaded)
                    log.info("Métriques chargées depuis %s", self._path)
                except (json.JSONDecodeError, OSError) as exc:
                    log.warning("metrics.json illisible (%s) — réinitialisation.", exc)

            # Horodatage de session
            self._state["session"]["start_time"] = (
                datetime.now(timezone.utc).isoformat()
            )

        self._flush()
        log.debug("MetricsRegistry initialisé.")

    # ──────────────────────────────────────────────────────────────────
    # ÉCRITURE
    # ──────────────────────────────────────────────────────────────────
    def update(self, section: str, data: dict[str, Any], flush: bool = True) -> None:
        """
        Met à jour une section de premier niveau (ex: "session", "model").

        Les clés non mentionnées dans data sont conservées telles quelles.
        Les listes sont remplacées (pas concaténées) sauf pour
        "domains_covered" qui est dédupliquée automatiquement.

        Args:
            section: Clé de premier niveau dans la structure globale.
            data:    Dictionnaire partiel des valeurs à mettre à jour.
            flush:   Si True, persiste immédiatement sur disque.

        Example:
            >>> metrics.update("model", {"rwkv_calls": 42, "tokens_generated": 1500})
        """
        with self._lock:
            if section not in self._state:
                self._state[section] = {}
            self._state[section].update(data)

            # Déduplication de la liste des domaines
            if section == "learning" and "domains_covered" in data:
                self._state["learning"]["domains_covered"] = list(
                    dict.fromkeys(self._state["learning"]["domains_covered"])
                )

            # Mise à jour de l'uptime en même temps
            if self._start_ts:
                elapsed = (time.monotonic() - self._start_ts) / 3600
                self._state["session"]["uptime_hours"] = round(elapsed, 4)

        if flush and self._path:
            self._flush()

    def update_module(self, module_name: str, data: dict[str, Any],
                      flush: bool = True) -> None:
        """
        Met à jour les métriques détaillées d'un module spécifique.
        Stockées sous la clé _modules.<module_name>.

        Args:
            module_name: Identifiant du module (ex: "scraper", "trainer").
            data:        Dictionnaire de métriques du module.
            flush:       Si True, persiste sur disque.

        Example:
            >>> metrics.update_module("scraper", {
            ...     "success_rate": 0.94,
            ...     "avg_tokens":   1240,
            ...     "errors":       3,
            ... })
        """
        with self._lock:
            if "_modules" not in self._state:
                self._state["_modules"] = {}
            if module_name not in self._state["_modules"]:
                self._state["_modules"][module_name] = {}
            self._state["_modules"][module_name].update(data)
            self._state["_modules"][module_name]["_updated_at"] = (
                datetime.now(timezone.utc).isoformat()
            )

        if flush and self._path:
            self._flush()

    def increment(self, section: str, key: str, amount: int | float = 1,
                  flush: bool = True) -> None:
        """
        Incrémente atomiquement un compteur numérique.

        Args:
            section: Section de premier niveau.
            key:     Clé du compteur dans la section.
            amount:  Valeur à ajouter (défaut 1).
            flush:   Si True, persiste sur disque.

        Example:
            >>> metrics.increment("session", "interactions")
            >>> metrics.increment("model", "tokens_generated", 128)
        """
        with self._lock:
            if section not in self._state:
                self._state[section] = {}
            current = self._state[section].get(key, 0)
            self._state[section][key] = current + amount

        if flush and self._path:
            self._flush()

    # ──────────────────────────────────────────────────────────────────
    # LECTURE
    # ──────────────────────────────────────────────────────────────────
    def snapshot(self) -> dict[str, Any]:
        """
        Retourne une copie profonde de l'état complet des métriques.
        Sûr à lire depuis n'importe quel thread.

        Returns:
            Dictionnaire complet, indépendant de l'état interne.
        """
        with self._lock:
            return json.loads(json.dumps(self._state))   # deep copy via JSON

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Lit une valeur précise sans copier tout l'état.

        Args:
            section: Section de premier niveau.
            key:     Clé dans la section.
            default: Valeur par défaut si absente.

        Returns:
            Valeur lue ou default.

        Example:
            >>> calls = metrics.get("model", "rwkv_calls", 0)
        """
        with self._lock:
            return self._state.get(section, {}).get(key, default)

    def get_module(self, module_name: str) -> dict[str, Any]:
        """
        Retourne les métriques d'un module spécifique.

        Args:
            module_name: Identifiant du module.

        Returns:
            Dictionnaire des métriques du module (copie), ou {} si absent.
        """
        with self._lock:
            module_data = self._state.get("_modules", {}).get(module_name, {})
            return dict(module_data)

    def format_summary(self) -> str:
        """
        Génère un résumé lisible des métriques principales,
        affiché dans le terminal par `python main.py --stats`.

        Returns:
            Chaîne multi-lignes formatée.
        """
        snap = self.snapshot()
        s    = snap.get("session",     {})
        l    = snap.get("learning",    {})
        m    = snap.get("memory",      {})
        mo   = snap.get("model",       {})
        tr   = snap.get("training",    {})
        im   = snap.get("improvements",{})

        lines = [
            "═" * 55,
            "  ZYLOS AI — Métriques système",
            "═" * 55,
            f"  ⏱  Uptime          : {s.get('uptime_hours', 0):.1f} h",
            f"  💬 Interactions    : {s.get('interactions', 0)}",
            "",
            "  📚 Apprentissage",
            f"     Pages scrapées  : {l.get('total_pages_scraped', 0)}",
            f"     Chunks indexés  : {l.get('total_chunks_indexed', 0)}",
            f"     Domaines couverts: {len(l.get('domains_covered', []))}",
            f"     Dernier scrape  : {l.get('last_scrape') or '—'}",
            "",
            "  🧠 Mémoire vectorielle",
            f"     Entrées total   : {m.get('vector_entries', 0)}",
            f"     RAG hit rate    : {m.get('rag_hit_rate', 0):.1%}",
            f"     Recherche moy.  : {m.get('avg_search_ms', 0):.0f} ms",
            "",
            "  🤖 Modèle RWKV",
            f"     Tokens générés  : {mo.get('tokens_generated', 0):,}",
            f"     Vitesse moy.    : {mo.get('avg_tokens_per_sec', 0):.0f} tok/s",
            f"     Appels RWKV     : {mo.get('rwkv_calls', 0)}",
            f"     Appels API      : {mo.get('api_calls', 0)} (improver only)",
            "",
            "  🎓 Entraînement",
            f"     Sessions        : {tr.get('sessions_completed', 0)}",
            f"     Échantillons    : {tr.get('total_samples_trained', 0):,}",
            f"     Meilleure loss  : {tr.get('best_loss') or '—'}",
            f"     Δ loss dernier  : {tr.get('last_loss_delta', 0):+.4f}",
            f"     Rollbacks       : {tr.get('rollbacks', 0)}",
            "",
            "  🛠  Auto-amélioration",
            f"     Cycles          : {im.get('cycles_run', 0)}",
            f"     Changements     : {im.get('changes_applied', 0)} appliqués"
            f" / {im.get('changes_pending', 0)} en attente",
            f"     Taux succès     : {im.get('success_rate', 0):.1%}",
            "═" * 55,
        ]
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────
    # PERSISTANCE
    # ──────────────────────────────────────────────────────────────────
    def _flush(self) -> None:
        """
        Écrit l'état courant dans metrics.json de manière atomique
        (écriture dans un fichier temporaire puis renommage).
        Appelé automatiquement par update/increment.
        """
        if not self._path:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            with self._lock:
                payload = json.dumps(self._state, ensure_ascii=False, indent=2,
                                     default=str)
            tmp.write_text(payload, encoding="utf-8")
            tmp.replace(self._path)   # atomique sur la plupart des OS
        except OSError as exc:
            log.error("Impossible d'écrire metrics.json : %s", exc)

    # ──────────────────────────────────────────────────────────────────
    # UTILITAIRES INTERNES
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _deep_merge(base: dict, override: dict) -> None:
        """Fusionne override dans base récursivement (in-place)."""
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                MetricsRegistry._deep_merge(base[k], v)
            else:
                base[k] = v


# ══════════════════════════════════════════════════════════════════════
# INSTANCE GLOBALE SINGLETON
# Importée partout par :  from utils.metrics import metrics
# ══════════════════════════════════════════════════════════════════════
metrics = MetricsRegistry()


# ══════════════════════════════════════════════════════════════════════
# SMOKE TEST  (python utils/metrics.py)
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "metrics.json"

        # Simuler l'env minimal attendu par config.py
        os.environ.setdefault("ZYLOS_LOG_LEVEL", "DEBUG")

        metrics.init(metrics_path=test_path)

        # Test update
        metrics.update("model", {"rwkv_calls": 5, "tokens_generated": 300})
        metrics.increment("session", "interactions", 3)
        metrics.update_module("scraper", {"success_rate": 0.95, "pages": 10})
        metrics.update("learning", {
            "total_pages_scraped": 10,
            "domains_covered": ["wikipedia.org", "arxiv.org", "wikipedia.org"],
            "last_scrape": datetime.now(timezone.utc).isoformat(),
        })

        snap = metrics.snapshot()

        # Vérifications
        assert snap["model"]["rwkv_calls"]        == 5,    "rwkv_calls KO"
        assert snap["session"]["interactions"]     == 3,    "interactions KO"
        assert snap["_modules"]["scraper"]["pages"]== 10,   "module scraper KO"
        # Déduplication domaines
        domains = snap["learning"]["domains_covered"]
        assert domains.count("wikipedia.org")      == 1,    "dédup domaines KO"

        # Vérification persistance disque
        assert test_path.exists(), "metrics.json non créé"
        reloaded = json.loads(test_path.read_text())
        assert reloaded["model"]["rwkv_calls"] == 5, "persistance KO"

        print(metrics.format_summary())
        print("\n✅  Tous les tests MetricsRegistry sont passés.")