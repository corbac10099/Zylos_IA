"""
main.py — Point d'entrée principal de ZYLOS AI
===============================================
Démarre l'interface de chat interactive et le scheduler journalier.

Modes de lancement :
  python main.py              # Chat interactif + scheduler automatique
  python main.py --no-sched   # Chat sans scheduler
  python main.py --web        # Interface web (http://localhost:8080)
  python main.py --web --port 9000  # Interface web sur port 9000
  python main.py --stats      # Affiche les métriques et quitte
  python main.py --once       # Exécute une fois le pipeline et quitte
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _init() -> None:
    from config import PATHS, validate_config
    from utils.metrics import metrics
    from utils.logger import get_logger

    log = get_logger(__name__)
    PATHS.create_all()
    metrics.init()

    warnings = validate_config()
    for w in warnings:
        log.warning("Config : %s", w)

    log.info("ZYLOS AI — démarrage (répertoire : %s)", PATHS.root)


def _load_model_background(on_ready: "threading.Event | None" = None) -> None:
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
            log.error("💡 Conseil : placez votre fichier .pth dans data/models/")
    except Exception as exc:
        log.error("Erreur inattendue lors du chargement du modèle : %s", exc, exc_info=True)
    finally:
        if on_ready is not None:
            on_ready.set()


def _run_web(port: int = 8080, model_ready_event: "threading.Event | None" = None) -> None:
    """Lance le serveur web."""
    from utils.logger import get_logger
    log = get_logger(__name__)

    # Attendre que le modèle soit prêt (max 60s avant d'ouvrir le serveur)
    if model_ready_event:
        log.info("Attente du modèle avant démarrage web (max 60s)…")
        model_ready_event.wait(timeout=60)

    try:
        from api.web_server import run_server
        run_server(host="0.0.0.0", port=port)
    except ImportError as e:
        log.error("Impossible de démarrer le serveur web : %s", e)
        log.error("Vérifiez que api/web_server.py est présent.")


def _run_chat(model_ready_event: "threading.Event | None" = None) -> None:
    from utils.logger import get_logger
    from modules.brain import brain

    log = get_logger(__name__)
    _print_banner()
    print("\n💬  Chat ZYLOS AI — Tapez votre message (ou /help pour les commandes)\n")

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

        print("Zylos : ", end="", flush=True)
        t0        = time.perf_counter()
        full_text = ""

        try:
            from config import RWKV as RWKV_CFG
            if RWKV_CFG.streaming:
                for token in brain.stream(user_input, use_rag=True):
                    print(token, end="", flush=True)
                    full_text += token
                print()
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


def _print_banner() -> None:
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
    if backend_info.vram_mb:
        print(f"  VRAM       : {backend_info.vram_mb:,} Mo")
    else:
        print(f"  VRAM       : CPU (pas de GPU détecté)")
    print(f"  Mistral    : {'✅ configuré' if MISTRAL.is_configured() else '❌ absent (improver OFF)'}")
    print()


def _print_stats() -> None:
    from utils.metrics import metrics
    print("\n" + metrics.format_summary() + "\n")


def _print_help() -> None:
    print()
    print("  Commandes disponibles :")
    print("  /help    → Affiche cette aide")
    print("  /reset   → Efface l'historique de conversation")
    print("  /stats   → Affiche les métriques du système")
    print("  /quit    → Quitte ZYLOS AI")
    print()


def _run_once() -> int:
    from modules.scheduler import scheduler
    from utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Mode --once : exécution du pipeline journalier.")
    report = scheduler.run_now()
    print(report.summary())
    return 0 if report.success else 1


def _run_load_only() -> int:
    ready = threading.Event()
    t = threading.Thread(target=_load_model_background, args=(ready,), daemon=True)
    t.start()
    ready.wait(timeout=600)

    from core.model import model
    if model.is_ready:
        m = model.get_metrics()
        print(f"✅  Modèle RWKV chargé ({m.get('model_path', '?')})")
        return 0
    else:
        print("❌  Chargement du modèle échoué.")
        return 1


def _shutdown() -> None:
    from utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Arrêt de ZYLOS AI…")

    try:
        from modules.scheduler import scheduler
        scheduler.stop(timeout=3.0)
    except Exception:
        pass

    try:
        from core.model import model
        if model.is_ready:
            model.save_state()
    except Exception:
        pass

    try:
        from utils.backup import backup
        backup.create_snapshot("shutdown")
        backup.cleanup_old(keep_n=15)
    except Exception:
        pass

    log.info("ZYLOS AI arrêté proprement.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ZYLOS AI — IA locale non-Transformer auto-apprenante",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python main.py                  # Chat terminal + scheduler
  python main.py --web            # Interface web http://localhost:8080
  python main.py --web --port 9000  # Interface web sur port 9000
  python main.py --no-sched       # Chat sans scheduler
  python main.py --stats          # Métriques et quitte
  python main.py --once           # Pipeline unique et quitte
  python main.py --load-only      # Charge modèle et quitte
        """
    )
    parser.add_argument("--no-sched", action="store_true",
                        help="Désactive le scheduler automatique journalier.")
    parser.add_argument("--stats", action="store_true",
                        help="Affiche les métriques et quitte.")
    parser.add_argument("--once", action="store_true",
                        help="Exécute le pipeline journalier une fois et quitte.")
    parser.add_argument("--load-only", action="store_true",
                        help="Charge le modèle, affiche les infos et quitte.")
    parser.add_argument("--no-load", action="store_true",
                        help="Ne charge pas le modèle (tests).")
    parser.add_argument("--web", action="store_true",
                        help="Lance l'interface web (http://localhost:8080).")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port pour l'interface web (défaut : 8080).")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _init()

    if args.stats:
        _print_stats()
        return 0

    if args.once:
        return _run_once()

    if args.load_only:
        return _run_load_only()

    model_ready = threading.Event()

    if not args.no_load:
        loader_thread = threading.Thread(
            target  = _load_model_background,
            args    = (model_ready,),
            name    = "zylos-model-loader",
            daemon  = True,
        )
        loader_thread.start()
    else:
        model_ready.set()

    if not args.no_sched:
        from modules.scheduler import scheduler
        scheduler.start()

    try:
        if args.web:
            # Mode web : le serveur tourne en foreground
            _print_banner()
            _run_web(port=args.port, model_ready_event=model_ready if not args.no_load else None)
        else:
            # Mode terminal classique
            _run_chat(model_ready_event=model_ready if not args.no_load else None)
    except Exception as exc:
        from utils.logger import get_logger
        get_logger(__name__).critical("Erreur fatale : %s", exc, exc_info=True)
        return 1
    finally:
        _shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())