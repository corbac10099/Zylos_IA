"""
main.py — Point d'entrée principal de ZYLOS AI
===============================================
Démarre l'interface de chat interactive et le scheduler journalier.

Modes de lancement :
  python main.py              # Chat interactif + scheduler automatique
  python main.py --no-sched   # Chat sans scheduler
  python main.py --stats      # Affiche les métriques et quitte
  python main.py --load-only  # Charge le modèle et quitte
  python main.py --once       # Exécute une fois le pipeline et quitte

Architecture de démarrage :
  1. Validation de la configuration
  2. Initialisation des métriques
  3. Création des répertoires de données
  4. Chargement du modèle RWKV (async dans un thread)
  5. Démarrage du scheduler journalier (daemon thread)
  6. Boucle de chat interactive (terminal)

Usage depuis main.py :
    python main.py
    python main.py --stats
    python main.py --once
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

# ── Assurer que le répertoire du projet est dans sys.path ──────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ══════════════════════════════════════════════════════════════════════
# DÉMARRAGE ET INITIALISATION
# ══════════════════════════════════════════════════════════════════════
def _init() -> None:
    """Initialise le projet au démarrage (répertoires, métriques, logs)."""
    from config import PATHS, validate_config
    from utils.metrics import metrics
    from utils.logger import get_logger

    log = get_logger(__name__)

    # Créer tous les répertoires nécessaires
    PATHS.create_all()

    # Initialiser le registre de métriques
    metrics.init()

    # Afficher les avertissements de configuration
    warnings = validate_config()
    for w in warnings:
        log.warning("Config : %s", w)

    log.info("ZYLOS AI — démarrage (répertoire : %s)", PATHS.root)


def _load_model_background(on_ready: "threading.Event | None" = None) -> None:
    """
    Charge le modèle RWKV dans un thread dédié pour ne pas bloquer le démarrage.

    Args:
        on_ready: Événement positionné quand le modèle est prêt.
    """
    from utils.logger import get_logger
    log = get_logger(__name__)

    try:
        from core.model import model
        log.info("Chargement du modèle RWKV en arrière-plan…")
        ok = model.load()
        if ok:
            model.load_state()
            log.info("Modèle RWKV prêt.")
        else:
            log.error("Chargement du modèle RWKV échoué.")
    except Exception as exc:
        log.error("Erreur inattendue lors du chargement du modèle : %s", exc)
    finally:
        if on_ready is not None:
            on_ready.set()


# ══════════════════════════════════════════════════════════════════════
# INTERFACE DE CHAT
# ══════════════════════════════════════════════════════════════════════
def _run_chat(model_ready_event: "threading.Event | None" = None) -> None:
    """
    Lance la boucle de chat interactive dans le terminal.

    Commandes spéciales :
      /quit ou /exit   → quitter
      /reset           → effacer l'historique de conversation
      /stats           → afficher les métriques
      /help            → liste des commandes
    """
    from utils.logger import get_logger
    from modules.brain import brain

    log = get_logger(__name__)

    _print_banner()

    print("\n💬  Chat ZYLOS AI — Tapez votre message (ou /help pour les commandes)\n")

    # Attendre que le modèle soit prêt si demandé
    if model_ready_event is not None and not model_ready_event.is_set():
        print("⏳  Chargement du modèle en cours", end="", flush=True)
        while not model_ready_event.wait(timeout=1.0):
            print(".", end="", flush=True)
        print(" ✅\n")

    while True:
        try:
            user_input = input("Vous : ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nAu revoir !")
            break

        if not user_input:
            continue

        # Commandes spéciales
        cmd = user_input.lower()
        if cmd in ("/quit", "/exit", "/q"):
            print("Au revoir !")
            break
        elif cmd == "/reset":
            brain.reset_history()
            print("Historique réinitialisé.\n")
            continue
        elif cmd == "/stats":
            _print_stats()
            continue
        elif cmd == "/help":
            _print_help()
            continue
        elif cmd.startswith("/"):
            print(f"Commande inconnue : {user_input}. Tapez /help.\n")
            continue

        # Génération de la réponse (streaming)
        print("Zylos : ", end="", flush=True)
        t0          = time.perf_counter()
        full_text   = ""

        try:
            from config import RWKV as RWKV_CFG
            if RWKV_CFG.streaming:
                for token in brain.stream(user_input, use_rag=True):
                    print(token, end="", flush=True)
                    full_text += token
                print()  # saut de ligne après la réponse
            else:
                resp      = brain.chat(user_input, use_rag=True)
                full_text = resp.text
                print(full_text)
        except KeyboardInterrupt:
            print("\n[Interrompu]")
            continue
        except Exception as exc:
            log.error("Erreur de génération : %s", exc)
            print(f"\n[Erreur : {exc}]")
            continue

        elapsed = time.perf_counter() - t0
        tokens  = max(1, len(full_text) // 4)
        tps     = tokens / elapsed if elapsed > 0 else 0
        print(f"\033[2m  [{tokens} tokens, {elapsed:.1f}s, {tps:.0f} tok/s]\033[0m\n")


# ══════════════════════════════════════════════════════════════════════
# UTILITAIRES D'AFFICHAGE
# ══════════════════════════════════════════════════════════════════════
def _print_banner() -> None:
    """Affiche la bannière de démarrage."""
    from config import RWKV as RWKV_CFG, MISTRAL
    from core.backend import backend_info

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║                    ZYLOS  AI                         ║")
    print("║          IA locale non-Transformer                   ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Modèle     : RWKV-7 World {RWKV_CFG.default_size}")
    print(f"  Backend    : {backend_info.name.upper()} — {backend_info.device_name}")
    print(f"  Quant.     : {backend_info.quantization_level()}")
    print(f"  VRAM       : {backend_info.vram_mb:,} Mo" if backend_info.vram_mb else "  VRAM       : CPU")
    print(f"  Mistral    : {'✅ configuré' if MISTRAL.is_configured() else '❌ absent (improver OFF)'}")
    print()


def _print_stats() -> None:
    """Affiche les métriques du système."""
    from utils.metrics import metrics
    print("\n" + metrics.format_summary() + "\n")


def _print_help() -> None:
    """Affiche l'aide des commandes."""
    print()
    print("  Commandes disponibles :")
    print("  /help    → Affiche cette aide")
    print("  /reset   → Efface l'historique de conversation")
    print("  /stats   → Affiche les métriques du système")
    print("  /quit    → Quitte ZYLOS AI")
    print()


# ══════════════════════════════════════════════════════════════════════
# MODES SPÉCIAUX
# ══════════════════════════════════════════════════════════════════════
def _run_once() -> int:
    """
    Exécute le pipeline journalier une fois et retourne le code de sortie.
    """
    from modules.scheduler import scheduler
    from utils.logger import get_logger

    log = get_logger(__name__)
    log.info("Mode --once : exécution du pipeline journalier.")

    report = scheduler.run_now()
    print(report.summary())
    return 0 if report.success else 1


def _run_load_only() -> int:
    """Charge le modèle, affiche les infos et quitte."""
    from utils.logger import get_logger
    log = get_logger(__name__)

    ready = threading.Event()
    t = threading.Thread(target=_load_model_background, args=(ready,), daemon=True)
    t.start()
    ready.wait(timeout=600)

    from core.model import model
    if model.is_ready:
        m = model.get_metrics()
        log.info("Modèle prêt : %s", m)
        print(f"✅  Modèle RWKV chargé ({m.get('model_path', '?')})")
        return 0
    else:
        print("❌  Chargement du modèle échoué.")
        return 1


# ══════════════════════════════════════════════════════════════════════
# ARRÊT PROPRE
# ══════════════════════════════════════════════════════════════════════
def _shutdown() -> None:
    """Sauvegarde l'état et arrête proprement les services."""
    from utils.logger import get_logger
    log = get_logger(__name__)

    log.info("Arrêt de ZYLOS AI…")

    # Arrêter le scheduler
    try:
        from modules.scheduler import scheduler
        scheduler.stop(timeout=3.0)
    except Exception:
        pass

    # Sauvegarder l'état RNN
    try:
        from core.model import model
        if model.is_ready:
            model.save_state()
            log.info("État RNN sauvegardé.")
    except Exception as exc:
        log.debug("Impossible de sauvegarder l'état RNN : %s", exc)

    # Snapshot de clôture
    try:
        from utils.backup import backup
        backup.create_snapshot("shutdown")
        backup.cleanup_old(keep_n=15)
    except Exception as exc:
        log.debug("Backup de clôture : %s", exc)

    log.info("ZYLOS AI arrêté proprement.")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ZYLOS AI — IA locale non-Transformer auto-apprenante",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python main.py                  # Chat interactif avec scheduler
  python main.py --no-sched       # Chat sans scheduler automatique
  python main.py --stats          # Affiche les métriques et quitte
  python main.py --once           # Exécute le pipeline et quitte
  python main.py --load-only      # Charge le modèle et quitte
        """
    )
    parser.add_argument(
        "--no-sched", action="store_true",
        help="Désactive le scheduler automatique journalier.",
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Affiche les métriques et quitte.",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Exécute le pipeline journalier une fois et quitte.",
    )
    parser.add_argument(
        "--load-only", action="store_true",
        help="Charge le modèle, affiche les infos et quitte.",
    )
    parser.add_argument(
        "--no-load", action="store_true",
        help="Ne charge pas le modèle (démarrage rapide pour les tests).",
    )
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════
def main() -> int:
    """
    Point d'entrée principal de ZYLOS AI.

    Returns:
        Code de sortie POSIX (0 = succès).
    """
    args = _parse_args()

    # ── Initialisation commune ────────────────────────────────────────
    _init()

    # ── Mode stats uniquement ─────────────────────────────────────────
    if args.stats:
        _print_stats()
        return 0

    # ── Mode pipeline unique ──────────────────────────────────────────
    if args.once:
        return _run_once()

    # ── Mode chargement seul ──────────────────────────────────────────
    if args.load_only:
        return _run_load_only()

    # ── Mode normal : chat + scheduler ───────────────────────────────
    model_ready = threading.Event()

    # Chargement du modèle en arrière-plan
    if not args.no_load:
        loader_thread = threading.Thread(
            target  = _load_model_background,
            args    = (model_ready,),
            name    = "zylos-model-loader",
            daemon  = True,
        )
        loader_thread.start()
    else:
        model_ready.set()   # marquer comme prêt immédiatement (mode no-load)

    # Démarrage du scheduler journalier
    if not args.no_sched:
        from modules.scheduler import scheduler
        scheduler.start()

    # Boucle de chat principale
    try:
        _run_chat(model_ready_event=model_ready if not args.no_load else None)
    except Exception as exc:
        from utils.logger import get_logger
        get_logger(__name__).critical("Erreur fatale dans la boucle de chat : %s", exc,
                                       exc_info=True)
        return 1
    finally:
        _shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())