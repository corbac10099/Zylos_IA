"""
pipeline/daily_run.py — Point d'entrée du pipeline journalier de ZYLOS AI
==========================================================================
Script autonome exécutable directement pour déclencher le pipeline complet
sans passer par le scheduler automatique. Utile pour les tests, les
exécutions manuelles en CI, ou les tâches cron externes.

Ce script est le seul point d'entrée externe du pipeline ; il :
  1. Initialise les métriques et les chemins
  2. Charge le modèle RWKV si pas déjà fait
  3. Exécute les 7 étapes dans l'ordre via pipeline/daily_run.py lui-même
     (indépendamment du scheduler automatique)
  4. Retourne un code de sortie 0 (succès) ou 1 (échec)

Usage :
    python pipeline/daily_run.py [--dry-run] [--steps scraping,training]
    python pipeline/daily_run.py --help
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Assurer que le répertoire racine du projet est dans sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.logger import get_logger
from utils.metrics import metrics

log = get_logger(__name__)

# Toutes les étapes disponibles dans l'ordre d'exécution
ALL_STEPS = [
    "scraping",
    "indexing",
    "corpus_build",
    "training",
    "evaluation",
    "improvement",
    "report",
]


# ══════════════════════════════════════════════════════════════════════
# EXÉCUTEUR DU PIPELINE JOURNALIER
# ══════════════════════════════════════════════════════════════════════
class DailyRunner:
    """
    Exécute le pipeline journalier de façon autonome.

    Contrairement au Scheduler (qui tourne en continu), DailyRunner
    effectue une seule passe et termine. Conçu pour être invoqué par
    cron, CI/CD, ou manuellement.
    """

    def __init__(self, dry_run: bool = False, steps: list[str] | None = None) -> None:
        self.dry_run = dry_run
        self.steps   = steps or ALL_STEPS
        self._results: dict[str, dict[str, Any]] = {}

    def run(self) -> bool:
        """
        Exécute le pipeline et retourne True si toutes les étapes
        obligatoires ont réussi.

        Returns:
            True si succès global.
        """
        t0 = time.perf_counter()

        log.info("╔════════════════════════════════════════════════════╗")
        log.info("║  ZYLOS AI — Pipeline journalier                    ║")
        log.info("║  %s  %s  ║",
                 datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                 "(DRY RUN)" if self.dry_run else "          ")
        log.info("╚════════════════════════════════════════════════════╝")

        # ── Initialisation ────────────────────────────────────────────
        from config import PATHS
        PATHS.create_all()
        metrics.init()

        if not self.dry_run:
            self._ensure_model_loaded()

        # ── Exécution des étapes ──────────────────────────────────────
        success_count = 0
        fail_count    = 0

        for step_name in self.steps:
            if self.dry_run:
                log.info("  [DRY RUN] %s — sauté", step_name)
                self._results[step_name] = {"dry_run": True}
                success_count += 1
                continue

            log.info("▶  Étape : %s", step_name)
            step_t0 = time.perf_counter()
            try:
                result = self._run_step(step_name)
                duration = time.perf_counter() - step_t0
                self._results[step_name] = {**result, "duration_s": round(duration, 2)}
                log.info("  ✅ %s terminé en %.1f s.", step_name, duration)
                success_count += 1
            except Exception as exc:
                duration = time.perf_counter() - step_t0
                self._results[step_name] = {"error": str(exc), "duration_s": round(duration, 2)}
                log.error("  ❌ %s échoué en %.1f s : %s", step_name, duration, exc)
                fail_count += 1
                # Les étapes non critiques n'interrompent pas le pipeline
                if step_name in ("scraping", "indexing", "corpus_build", "training"):
                    log.warning("  Étape critique — poursuite malgré l'erreur.")

        # ── Rapport final ─────────────────────────────────────────────
        total = time.perf_counter() - t0
        self._print_final_report(total, success_count, fail_count)

        return fail_count == 0

    def _run_step(self, step_name: str) -> dict[str, Any]:
        """Dispatch vers la fonction d'étape correspondante."""
        dispatch: dict[str, Any] = {
            "scraping":     self._do_scraping,
            "indexing":     self._do_indexing,
            "corpus_build": self._do_corpus_build,
            "training":     self._do_training,
            "evaluation":   self._do_evaluation,
            "improvement":  self._do_improvement,
            "report":       self._do_report,
        }
        fn = dispatch.get(step_name)
        if fn is None:
            raise ValueError(f"Étape inconnue : {step_name!r}")
        return fn()

    # ──────────────────────────────────────────────────────────────────
    # IMPLÉMENTATION DES ÉTAPES
    # ──────────────────────────────────────────────────────────────────
    def _do_scraping(self) -> dict[str, Any]:
        """Scrape les sources par défaut configurées."""
        from config import SCRAPER
        from modules.scraper import scraper

        urls  = list(SCRAPER.default_sources)
        pages = scraper.scrape_urls(urls)
        ok    = [p for p in pages if p.success]

        metrics.update("learning", {
            "last_scrape":         datetime.now(timezone.utc).isoformat(),
            "total_pages_scraped": metrics.get("learning", "total_pages_scraped", 0) + len(ok),
        }, flush=False)

        # Stocker les pages dans les résultats pour les étapes suivantes
        self._scraped_pages = ok
        log.info("  Scraping : %d/%d pages récupérées.", len(ok), len(urls))
        return {"pages_ok": len(ok), "pages_total": len(urls)}

    def _do_indexing(self) -> dict[str, Any]:
        """Indexe les pages scrapées dans ChromaDB."""
        from modules.vectordb import vectordb

        pages   = getattr(self, "_scraped_pages", [])
        n_total = 0
        for page in pages:
            n_total += vectordb.add_chunks_from_page(page)

        log.info("  Indexing : %d chunks ajoutés.", n_total)
        return {"chunks_indexed": n_total}

    def _do_corpus_build(self) -> dict[str, Any]:
        """Construit les exemples JSONL d'entraînement."""
        from pipeline.corpus_builder import corpus_builder

        pages   = getattr(self, "_scraped_pages", [])
        n       = corpus_builder.build_from_pages(pages)
        log.info("  Corpus : %d exemples générés.", n)
        return {"examples": n}

    def _do_training(self) -> dict[str, Any]:
        """Lance la session de fine-tuning QLoRA."""
        from modules.trainer import trainer

        result = trainer.run_session()
        if result.success:
            log.info(
                "  Training : loss %.4f → %.4f (Δ %+.4f), %d steps.",
                result.initial_loss, result.final_loss,
                result.loss_delta, result.steps_done,
            )
        else:
            log.warning("  Training : %s", result.error)

        return {
            "success":     result.success,
            "final_loss":  result.final_loss,
            "loss_delta":  result.loss_delta,
            "rolled_back": result.rolled_back,
            "error":       result.error,
        }

    def _do_evaluation(self) -> dict[str, Any]:
        """Évalue rapidement le modèle post-training."""
        try:
            from core.model import model as rwkv_model
            if not rwkv_model.is_ready:
                return {"skipped": True}

            response = rwkv_model.generate(
                "User: Dis bonjour en une phrase.\n\nAssistant:",
                max_tokens=30,
            )
            log.info("  Évaluation : réponse='%s…'", response[:60])
            return {"ok": len(response) > 3, "response_len": len(response)}
        except Exception as exc:
            log.warning("  Évaluation : %s", exc)
            return {"skipped": True, "error": str(exc)}

    def _do_improvement(self) -> dict[str, Any]:
        """Exécute un cycle d'auto-amélioration du code."""
        from config import IMPROVER, MISTRAL

        if not IMPROVER.enabled:
            log.info("  Improver désactivé (IMPROVER.enabled=False).")
            return {"skipped": True}

        if not MISTRAL.is_configured():
            log.info("  Improver : MISTRAL_API_KEY absente — ignoré.")
            return {"skipped": True}

        try:
            from modules.improver import improver
            suggestions = improver.run_cycle()
            log.info("  Improver : %d suggestions générées.", len(suggestions))
            return {"suggestions": len(suggestions)}
        except Exception as exc:
            log.warning("  Improver : %s", exc)
            return {"skipped": True, "error": str(exc)}

    def _do_report(self) -> dict[str, Any]:
        """Génère et sauvegarde le rapport de métriques."""
        import json as _json
        from config import PATHS

        summary = metrics.format_summary()
        log.info("\n" + summary)

        today    = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rpt_path = PATHS.logs / f"report_{today}.json"
        try:
            rpt_path.parent.mkdir(parents=True, exist_ok=True)
            snap = metrics.snapshot()
            snap["pipeline_results"] = self._results
            rpt_path.write_text(
                _json.dumps(snap, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            log.info("  Rapport sauvegardé → %s", rpt_path)
        except Exception as exc:
            log.warning("  Rapport : impossible d'écrire %s : %s", rpt_path, exc)

        return {"report_path": str(rpt_path)}

    # ──────────────────────────────────────────────────────────────────
    # CHARGEMENT DU MODÈLE
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _ensure_model_loaded() -> None:
        """Charge le modèle RWKV si pas encore prêt."""
        try:
            from core.model import model as rwkv_model
            if not rwkv_model.is_ready:
                log.info("Chargement du modèle RWKV…")
                ok = rwkv_model.load()
                if ok:
                    rwkv_model.load_state()
                    log.info("Modèle RWKV prêt.")
                else:
                    log.warning("Modèle RWKV non chargé — certaines étapes seront limitées.")
        except Exception as exc:
            log.warning("Impossible de charger le modèle RWKV : %s", exc)

    # ──────────────────────────────────────────────────────────────────
    # RAPPORT
    # ──────────────────────────────────────────────────────────────────
    def _print_final_report(
        self, total_s: float, ok: int, fail: int
    ) -> None:
        log.info("═" * 55)
        log.info("  Pipeline terminé en %.1f s — %d OK / %d FAIL",
                 total_s, ok, fail)
        for step, result in self._results.items():
            dur = result.get("duration_s", 0)
            err = result.get("error", "")
            skp = result.get("skipped") or result.get("dry_run")
            status = "SKIP" if skp else ("OK  " if not err else "FAIL")
            log.info("  %s  %-20s  %.1fs  %s", status, step, dur, err[:40] if err else "")
        log.info("═" * 55)


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline journalier ZYLOS AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python pipeline/daily_run.py                   # pipeline complet
  python pipeline/daily_run.py --dry-run         # simulation sans exécution
  python pipeline/daily_run.py --steps scraping,indexing,corpus_build
        """
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Simule le pipeline sans exécuter les étapes (debug).",
    )
    parser.add_argument(
        "--steps",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=None,
        metavar="STEP1,STEP2",
        help=f"Étapes à exécuter (défaut : toutes). "
             f"Disponibles : {', '.join(ALL_STEPS)}",
    )
    parser.add_argument(
        "--list-steps", action="store_true",
        help="Affiche les étapes disponibles et quitte.",
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════
def main() -> int:
    """
    Point d'entrée principal.

    Returns:
        Code de sortie POSIX : 0 = succès, 1 = échec.
    """
    args = _parse_args()

    if args.list_steps:
        print("Étapes disponibles :")
        for s in ALL_STEPS:
            print(f"  {s}")
        return 0

    # Validation des étapes demandées
    if args.steps:
        invalid = [s for s in args.steps if s not in ALL_STEPS]
        if invalid:
            print(f"Étapes inconnues : {', '.join(invalid)}", file=sys.stderr)
            print(f"Étapes valides : {', '.join(ALL_STEPS)}", file=sys.stderr)
            return 1

    runner = DailyRunner(dry_run=args.dry_run, steps=args.steps)
    success = runner.run()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())