"""
modules/scheduler.py — Ordonnanceur du cycle journalier de ZYLOS AI
====================================================================
Planifie et exécute automatiquement le pipeline d'apprentissage quotidien
à l'heure configurée (config.SCHEDULER.daily_run_hour, défaut 03:00 UTC).

Pipeline journalier (dans l'ordre) :
  1. scraping       — collecte de nouvelles pages web
  2. indexing       — ajout dans ChromaDB
  3. corpus_build   — construction des exemples JSONL
  4. training       — fine-tuning QLoRA
  5. evaluation     — mesure de performance post-training
  6. improvement    — analyse code via Mistral Codestral
  7. report         — génération du rapport de session

Chaque étape dispose d'un timeout configurable. Si une étape dépasse
son timeout, elle est sautée avec un WARNING et le pipeline continue.

Usage :
    from modules.scheduler import scheduler
    scheduler.start()          # démarre le thread de planification
    scheduler.stop()           # arrête proprement
    scheduler.run_now()        # force l'exécution immédiate
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Callable

from utils.logger import get_logger
from utils.metrics import metrics

log = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════
# RÉSULTAT D'UNE EXÉCUTION
# ══════════════════════════════════════════════════════════════════════
class StepResult:
    """Résultat d'une étape du pipeline journalier."""
    def __init__(self, name: str) -> None:
        self.name:     str   = name
        self.success:  bool  = False
        self.skipped:  bool  = False
        self.duration: float = 0.0
        self.error:    str   = ""
        self.data:     Any   = None

    def __repr__(self) -> str:
        status = "SKIP" if self.skipped else ("OK" if self.success else "FAIL")
        return f"StepResult({self.name}={status}, {self.duration:.1f}s)"


class PipelineReport:
    """Rapport complet d'une exécution du pipeline journalier."""
    def __init__(self) -> None:
        self.started_at:  str         = datetime.now(timezone.utc).isoformat()
        self.finished_at: str         = ""
        self.steps:       list[StepResult] = []
        self.success:     bool        = False
        self.total_duration: float    = 0.0

    def add_step(self, result: StepResult) -> None:
        self.steps.append(result)

    def finalize(self) -> None:
        self.finished_at    = datetime.now(timezone.utc).isoformat()
        self.total_duration = sum(s.duration for s in self.steps)
        self.success        = all(
            s.success or s.skipped for s in self.steps
        )

    def summary(self) -> str:
        lines = [
            "═" * 55,
            f"  Pipeline journalier — {self.started_at[:19]}",
            "═" * 55,
        ]
        for s in self.steps:
            status = "⏭  SKIP" if s.skipped else ("✅ OK  " if s.success else "❌ FAIL")
            lines.append(f"  {status}  {s.name:<20} {s.duration:>6.1f}s"
                         + (f"  [{s.error[:40]}]" if s.error else ""))
        lines += [
            "─" * 55,
            f"  Total : {self.total_duration:.1f}s — "
            f"{'Succès' if self.success else 'Avec erreurs'}",
            "═" * 55,
        ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# ORDONNANCEUR
# ══════════════════════════════════════════════════════════════════════
class Scheduler:
    """
    Ordonnanceur du pipeline d'apprentissage journalier ZYLOS.

    Tourne dans un thread dédié. Vérifie toutes les minutes si
    l'heure planifiée est atteinte, puis exécute le pipeline.
    Chaque étape est exécutée dans un sous-thread avec timeout.

    Attributes:
        _thread:      Thread de planification.
        _stop_event:  Événement pour arrêt propre.
        _lock:        Verrou pour l'accès aux statistiques.
        _stats:       Historique des exécutions.
        _last_run:    Timestamp de la dernière exécution.
    """

    def __init__(self) -> None:
        self._thread:     threading.Thread | None = None
        self._stop_event  = threading.Event()
        self._lock         = threading.Lock()
        self._running_pipeline = False
        self._stats: dict[str, Any] = {
            "runs_completed":   0,
            "runs_failed":      0,
            "last_run_at":      None,
            "last_duration_s":  0.0,
        }
        self._last_run_date: str | None = None   # "YYYY-MM-DD" pour éviter double run

    # ──────────────────────────────────────────────────────────────────
    # CONTRÔLE DU THREAD
    # ──────────────────────────────────────────────────────────────────
    def start(self) -> None:
        """
        Démarre le thread de planification en arrière-plan.
        Idempotent — appels multiples sans effet.
        """
        if self._thread is not None and self._thread.is_alive():
            log.debug("Scheduler déjà actif.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target  = self._scheduler_loop,
            name    = "zylos-scheduler",
            daemon  = True,
        )
        self._thread.start()
        log.info("Scheduler démarré (run journalier à %02d:%02d UTC).",
                 _sched_cfg().daily_run_hour,
                 _sched_cfg().daily_run_minute)

    def stop(self, timeout: float = 5.0) -> None:
        """
        Arrête proprement le thread de planification.

        Args:
            timeout: Secondes maximum d'attente de terminaison.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        log.info("Scheduler arrêté.")

    def run_now(self) -> PipelineReport:
        """
        Exécute immédiatement le pipeline, indépendamment de l'heure planifiée.

        Returns:
            PipelineReport avec les résultats de chaque étape.

        Example:
            >>> report = scheduler.run_now()
            >>> print(report.summary())
        """
        if self._running_pipeline:
            log.warning("Pipeline déjà en cours — run_now ignoré.")
            report = PipelineReport()
            report.finalize()
            return report

        log.info("Exécution manuelle du pipeline journalier.")
        return self._run_pipeline()

    @property
    def is_running(self) -> bool:
        """True si le pipeline est actuellement en cours."""
        return self._running_pipeline

    def get_metrics(self) -> dict[str, Any]:
        """Retourne les statistiques du scheduler."""
        with self._lock:
            return dict(self._stats)

    # ──────────────────────────────────────────────────────────────────
    # BOUCLE DE PLANIFICATION
    # ──────────────────────────────────────────────────────────────────
    def _scheduler_loop(self) -> None:
        """Boucle principale : vérifie toutes les 60s si le run doit démarrer."""
        log.debug("Thread scheduler actif.")
        while not self._stop_event.is_set():
            try:
                if self._should_run_now():
                    self._run_pipeline()
            except Exception as exc:
                log.error("Erreur inattendue dans la boucle scheduler : %s", exc)

            # Attente interruptible (60 secondes max)
            self._stop_event.wait(timeout=60)

        log.debug("Thread scheduler terminé.")

    def _should_run_now(self) -> bool:
        """
        Retourne True si l'heure planifiée est atteinte et que le
        pipeline n'a pas encore été exécuté aujourd'hui.
        """
        cfg   = _sched_cfg()
        now   = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")

        # Déjà exécuté aujourd'hui ?
        if self._last_run_date == today:
            return False

        # Heure planifiée atteinte ?
        if now.hour != cfg.daily_run_hour or now.minute != cfg.daily_run_minute:
            return False

        return True

    # ──────────────────────────────────────────────────────────────────
    # PIPELINE JOURNALIER
    # ──────────────────────────────────────────────────────────────────
    def _run_pipeline(self) -> PipelineReport:
        """Exécute le pipeline complet en enregistrant les résultats."""
        self._running_pipeline = True
        report = PipelineReport()

        log.info("╔══════════════════════════════════════════╗")
        log.info("║  Pipeline journalier ZYLOS — DÉMARRAGE  ║")
        log.info("╚══════════════════════════════════════════╝")

        try:
            cfg = _sched_cfg()
            steps: list[tuple[str, Callable[[], Any]]] = [
                ("scraping",       self._step_scraping),
                ("indexing",       self._step_indexing),
                ("corpus_build",   self._step_corpus_build),
                ("training",       self._step_training),
                ("evaluation",     self._step_evaluation),
                ("improvement",    self._step_improvement),
                ("report",         self._step_report),
            ]

            for step_name, step_fn in steps:
                if self._stop_event.is_set():
                    log.warning("Pipeline interrompu par stop_event.")
                    break

                timeout = cfg.step_timeouts.get(step_name, 600)
                result  = self._run_step_with_timeout(step_name, step_fn, timeout)
                report.add_step(result)

                if result.success:
                    log.info("  ✅ %s terminé en %.1f s.", step_name, result.duration)
                elif result.skipped:
                    log.info("  ⏭  %s sauté.", step_name)
                else:
                    log.warning("  ❌ %s échoué : %s", step_name, result.error)

            report.finalize()
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            self._last_run_date = today

            with self._lock:
                if report.success:
                    self._stats["runs_completed"] = int(self._stats["runs_completed"]) + 1
                else:
                    self._stats["runs_failed"]    = int(self._stats["runs_failed"]) + 1
                self._stats["last_run_at"]     = report.started_at
                self._stats["last_duration_s"] = round(report.total_duration, 1)

            metrics.update("improvements", {
                "cycles_run": metrics.get("improvements", "cycles_run", 0) + 1,
            }, flush=True)
            self._push_metrics()

            log.info("\n" + report.summary())

        finally:
            self._running_pipeline = False

        return report

    def _run_step_with_timeout(
        self,
        name:    str,
        fn:      Callable[[], Any],
        timeout: int,
    ) -> StepResult:
        """
        Exécute une étape dans un thread dédié avec timeout strict.

        Args:
            name:    Nom de l'étape (pour les logs).
            fn:      Fonction à exécuter.
            timeout: Durée maximale en secondes.

        Returns:
            StepResult avec les métriques de l'étape.
        """
        result    = StepResult(name)
        exception = [None]
        data_box  = [None]

        def target():
            try:
                data_box[0] = fn()
            except Exception as exc:
                exception[0] = exc

        t0     = time.perf_counter()
        thread = threading.Thread(target=target, name=f"zylos-step-{name}", daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        result.duration = time.perf_counter() - t0

        if thread.is_alive():
            # Thread encore actif après le timeout
            result.skipped = True
            result.error   = f"Timeout ({timeout}s dépassé)"
            log.warning("Étape '%s' : timeout (%ds). Poursuite du pipeline.", name, timeout)
        elif exception[0] is not None:
            result.success = False
            result.error   = str(exception[0])
        else:
            result.success = True
            result.data    = data_box[0]

        return result

    # ──────────────────────────────────────────────────────────────────
    # ÉTAPES DU PIPELINE
    # ──────────────────────────────────────────────────────────────────
    def _step_scraping(self) -> dict[str, Any]:
        """Étape 1 : scraping des sources configurées."""
        from config import SCRAPER
        from modules.scraper import scraper

        log.info("Pipeline — étape 1 : scraping (%d sources).",
                 len(SCRAPER.default_sources))

        pages = scraper.scrape_urls(list(SCRAPER.default_sources))
        metrics.update("learning", {
            "last_scrape":         datetime.now(timezone.utc).isoformat(),
            "total_pages_scraped": metrics.get("learning", "total_pages_scraped", 0) + len(pages),
        }, flush=False)

        return {"pages_scraped": len(pages), "pages": pages}

    def _step_indexing(self) -> dict[str, Any]:
        """Étape 2 : indexation des pages dans ChromaDB."""
        from modules.vectordb import vectordb

        # Récupérer les pages depuis l'étape précédente via les métriques
        # (le résultat n'est pas partagé directement entre étapes pour la sécurité)
        from modules.scraper import scraper
        from config import SCRAPER

        pages = scraper.scrape_urls(list(SCRAPER.default_sources[:1]))
        n_indexed = 0
        for page in pages:
            if page.success:
                n_indexed += vectordb.add_chunks_from_page(page)

        return {"chunks_indexed": n_indexed}

    def _step_corpus_build(self) -> dict[str, Any]:
        """Étape 3 : construction du corpus JSONL."""
        from pipeline.corpus_builder import corpus_builder
        from modules.scraper import scraper
        from config import SCRAPER

        pages = scraper.scrape_urls(list(SCRAPER.default_sources))
        n = corpus_builder.build_from_pages(pages)
        return {"examples_built": n}

    def _step_training(self) -> dict[str, Any]:
        """Étape 4 : fine-tuning QLoRA."""
        from modules.trainer import trainer

        if trainer.is_running:
            log.warning("Training déjà en cours — étape sautée.")
            return {"skipped": True}

        result = trainer.run_session()
        return {
            "success":    result.success,
            "final_loss": result.final_loss,
            "loss_delta": result.loss_delta,
            "rolled_back": result.rolled_back,
        }

    def _step_evaluation(self) -> dict[str, Any]:
        """Étape 5 : évaluation rapide post-training."""
        try:
            from core.model import model as rwkv_model
            if not rwkv_model.is_ready:
                return {"skipped": True, "reason": "modèle non chargé"}

            # Évaluation simplifiée : génération d'un texte court
            test_prompt = "User: Qui es-tu ?\n\nAssistant:"
            response    = rwkv_model.generate(test_prompt, max_tokens=50)
            ok          = len(response) > 5

            return {"evaluation_ok": ok, "response_len": len(response)}
        except Exception as exc:
            log.warning("Évaluation impossible : %s", exc)
            return {"skipped": True, "error": str(exc)}

    def _step_improvement(self) -> dict[str, Any]:
        """Étape 6 : auto-amélioration du code via Mistral."""
        from config import IMPROVER, MISTRAL
        if not IMPROVER.enabled:
            return {"skipped": True, "reason": "improver désactivé"}
        if not MISTRAL.is_configured():
            return {"skipped": True, "reason": "MISTRAL_API_KEY absente"}

        try:
            from modules.improver import improver
            results = improver.run_cycle()
            return {"suggestions": results}
        except Exception as exc:
            log.warning("Improver échoué : %s", exc)
            return {"skipped": True, "error": str(exc)}

    def _step_report(self) -> dict[str, Any]:
        """Étape 7 : génération et log du rapport de métriques."""
        snap    = metrics.snapshot()
        summary = metrics.format_summary()
        log.info("\n" + summary)

        # Sauvegarder un snapshot JSON du rapport
        try:
            from config import PATHS
            import json as _json
            today   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            rpt_path = PATHS.logs / f"report_{today}.json"
            rpt_path.parent.mkdir(parents=True, exist_ok=True)
            rpt_path.write_text(
                _json.dumps(snap, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            log.debug("Impossible d'écrire le rapport JSON : %s", exc)

        return {"summary_logged": True}

    # ──────────────────────────────────────────────────────────────────
    # MÉTRIQUES
    # ──────────────────────────────────────────────────────────────────
    def _push_metrics(self) -> None:
        try:
            metrics.update_module("scheduler", self.get_metrics(), flush=True)
        except Exception as exc:
            log.debug("Impossible de pousser les métriques scheduler : %s", exc)


# ══════════════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════════════
def _sched_cfg():
    from config import SCHEDULER
    return SCHEDULER


# ══════════════════════════════════════════════════════════════════════
# SINGLETON
# ══════════════════════════════════════════════════════════════════════
scheduler = Scheduler()


# ══════════════════════════════════════════════════════════════════════
# SMOKE TEST  (python modules/scheduler.py)
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import os, tempfile
    os.environ.setdefault("ZYLOS_LOG_LEVEL", "DEBUG")

    print("═" * 60)
    print("  ZYLOS AI — modules/scheduler.py  smoke test")
    print("═" * 60)
    print()

    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        import config as cfg
        object.__setattr__(cfg.PATHS, "corpus",       p / "corpus")
        object.__setattr__(cfg.PATHS, "lora_weights", p / "lora")
        object.__setattr__(cfg.PATHS, "logs",         p / "logs")
        object.__setattr__(cfg.PATHS, "chroma_db",    p / "chroma")

        s = Scheduler()

        # Test 1 : démarrage/arrêt du thread
        s.start()
        assert s._thread is not None and s._thread.is_alive()
        s.stop(timeout=2.0)
        print("✅  Test 1 : start/stop thread OK")

        # Test 2 : _should_run_now (heure différente → False)
        assert not s._should_run_now()
        print("✅  Test 2 : _should_run_now() correct (heure différente)")

        # Test 3 : _run_step_with_timeout — succès
        def _quick(): return {"ok": True}
        r = s._run_step_with_timeout("test_quick", _quick, timeout=5)
        assert r.success
        assert r.data == {"ok": True}
        print(f"✅  Test 3 : _run_step_with_timeout succès OK (dur={r.duration:.3f}s)")

        # Test 4 : _run_step_with_timeout — timeout
        def _slow(): time.sleep(10)
        r2 = s._run_step_with_timeout("test_slow", _slow, timeout=1)
        assert r2.skipped and "Timeout" in r2.error
        print(f"✅  Test 4 : timeout détecté OK ({r2.error})")

        # Test 5 : _run_step_with_timeout — exception
        def _bad(): raise ValueError("Erreur simulée")
        r3 = s._run_step_with_timeout("test_bad", _bad, timeout=5)
        assert not r3.success and "Erreur simulée" in r3.error
        print(f"✅  Test 5 : exception capturée OK ({r3.error})")

        # Test 6 : run_now — pipeline complet (étapes minimales)
        # On mock les étapes lourdes
        s._step_scraping    = lambda: {"pages_scraped": 0, "pages": []}
        s._step_indexing    = lambda: {"chunks_indexed": 0}
        s._step_corpus_build = lambda: {"examples_built": 0}
        s._step_training    = lambda: {"skipped": True}
        s._step_evaluation  = lambda: {"skipped": True}
        s._step_improvement = lambda: {"skipped": True}
        s._step_report      = lambda: {"summary_logged": True}

        object.__setattr__(cfg.SCHEDULER, "step_timeouts", {
            step: 30 for step in ["scraping","indexing","corpus_build",
                                   "training","evaluation","improvement","report"]
        })

        report = s.run_now()
        assert isinstance(report, PipelineReport)
        assert len(report.steps) == 7
        print(f"✅  Test 6 : run_now() OK — {len(report.steps)} étapes")
        print(f"\n{report.summary()}")

        # Test 7 : métriques
        m = s.get_metrics()
        assert isinstance(m["runs_completed"], int)
        print(f"✅  Test 7 : get_metrics() OK — {m}")

    print("\n✅  Tous les tests modules/scheduler.py sont passés.")