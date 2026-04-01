"""
utils/logger.py — Système de logs structuré de ZYLOS AI
=========================================================
Fournit un logger nommé par module, avec :
  - Rotation quotidienne automatique (conservation 30 jours)
  - Sortie console colorée selon le niveau
  - Sortie fichier dans data/logs/zylos_YYYY-MM-DD.log
  - Un seul appel : get_logger(__name__)

Usage dans n'importe quel module :
    from utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Message")
    log.warning("Attention")
    log.error("Erreur", exc_info=True)
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

# Import différé pour éviter les imports circulaires au démarrage
def _get_log_config():
    from config import LOG, PATHS
    return LOG, PATHS


# ══════════════════════════════════════════════════════════════════════
# COULEURS ANSI pour la console
# ══════════════════════════════════════════════════════════════════════
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_COLORS = {
    logging.DEBUG:    "\033[36m",    # cyan
    logging.INFO:     "\033[32m",    # vert
    logging.WARNING:  "\033[33m",    # jaune
    logging.ERROR:    "\033[31m",    # rouge
    logging.CRITICAL: "\033[35m",    # magenta
}


class _ColorFormatter(logging.Formatter):
    """Formatter console avec couleurs ANSI selon le niveau de log."""

    def __init__(self, fmt: str, datefmt: str) -> None:
        super().__init__(fmt, datefmt=datefmt)
        self._fmt     = fmt
        self._datefmt = datefmt

    def format(self, record: logging.LogRecord) -> str:
        color  = _COLORS.get(record.levelno, _RESET)
        # Colore uniquement le niveau et le nom du logger
        record.levelname = f"{color}{_BOLD}{record.levelname:<8}{_RESET}"
        record.name      = f"{color}{record.name}{_RESET}"
        return super().format(record)


# ══════════════════════════════════════════════════════════════════════
# INITIALISATION GLOBALE (une seule fois)
# ══════════════════════════════════════════════════════════════════════
_initialized = False

def _init_root_logger() -> None:
    """
    Configure le logger racine 'zylos' une seule fois au premier appel.
    Appelé automatiquement par get_logger().
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    log_cfg, paths = _get_log_config()

    root = logging.getLogger("zylos")
    root.setLevel(log_cfg.level)
    root.propagate = False   # ne pas remonter au logger Python racine

    # ── Formatter fichier (pas de couleur) ────────────────────────────
    file_fmt = logging.Formatter(
        fmt     = log_cfg.format,
        datefmt = log_cfg.date_format,
    )

    # ── Handler fichier avec rotation quotidienne ─────────────────────
    try:
        paths.logs.mkdir(parents=True, exist_ok=True)
        log_file = paths.logs / "zylos.log"

        file_handler = TimedRotatingFileHandler(
            filename    = str(log_file),
            when        = "midnight",
            interval    = 1,
            backupCount = log_cfg.max_log_days,
            encoding    = "utf-8",
            utc         = False,
        )
        # Suffixe des fichiers archivés : zylos.log.2025-07-14
        file_handler.suffix = "%Y-%m-%d"
        file_handler.setFormatter(file_fmt)
        file_handler.setLevel(log_cfg.level)
        root.addHandler(file_handler)

    except OSError as exc:
        # Si le dossier logs est inaccessible, on continue sans fichier
        print(f"[ZYLOS logger] ⚠ Impossible d'ouvrir le fichier de log : {exc}",
              file=sys.stderr)

    # ── Handler console (coloré) ──────────────────────────────────────
    if log_cfg.console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_cfg.level)

        # Couleurs désactivées si stdout n'est pas un terminal (pipe, fichier…)
        if sys.stdout.isatty():
            console_handler.setFormatter(
                _ColorFormatter(
                    fmt     = log_cfg.format,
                    datefmt = log_cfg.date_format,
                )
            )
        else:
            console_handler.setFormatter(file_fmt)

        root.addHandler(console_handler)


# ══════════════════════════════════════════════════════════════════════
# API PUBLIQUE
# ══════════════════════════════════════════════════════════════════════
def get_logger(name: str) -> logging.Logger:
    """
    Retourne un logger nommé, enfant du logger racine 'zylos'.

    Convention de nommage recommandée :
        log = get_logger(__name__)

    Le nom du module sera affiché dans chaque ligne de log,
    ce qui permet d'identifier immédiatement la source du message.

    Args:
        name: Nom du module appelant (en général __name__).

    Returns:
        Instance logging.Logger prête à l'emploi.

    Example:
        >>> log = get_logger("modules.scraper")
        >>> log.info("Page scrapée : https://example.com")
        2025-07-14 03:00:01 [INFO    ] modules.scraper — Page scrapée : ...
    """
    _init_root_logger()

    # Normalise les noms de modules Python en hiérarchie de loggers
    # "modules.scraper" → logger enfant de "zylos"
    child_name = f"zylos.{name.lstrip('zylos.').lstrip('.')}" if not name.startswith("zylos") else name
    return logging.getLogger(child_name)


def set_level(level: str | int) -> None:
    """
    Change dynamiquement le niveau de log de tous les handlers.

    Args:
        level: Niveau sous forme de chaîne ("DEBUG", "INFO", …) ou entier.

    Example:
        >>> set_level("DEBUG")   # Active les logs de débogage
        >>> set_level("WARNING") # Réduit le bruit en production
    """
    _init_root_logger()
    root = logging.getLogger("zylos")
    int_level = logging.getLevelName(level) if isinstance(level, str) else level
    root.setLevel(int_level)
    for handler in root.handlers:
        handler.setLevel(int_level)


# ══════════════════════════════════════════════════════════════════════
# LOGGER SPÉCIALISÉ : sessions d'entraînement
# ══════════════════════════════════════════════════════════════════════
def get_training_logger() -> logging.Logger:
    """
    Logger dédié aux sessions de fine-tuning QLoRA.
    Écrit aussi dans un fichier séparé data/logs/training.log
    pour faciliter l'analyse des courbes de loss.

    Returns:
        Logger nommé 'zylos.training' avec handler fichier dédié.
    """
    _init_root_logger()
    _, paths = _get_log_config()
    log_cfg, _ = _get_log_config()

    logger = logging.getLogger("zylos.training")

    # Ajouter le handler fichier dédié si pas encore présent
    already_has_file = any(
        isinstance(h, (logging.FileHandler, TimedRotatingFileHandler))
        and "training" in getattr(h, "baseFilename", "")
        for h in logger.handlers
    )
    if not already_has_file:
        try:
            training_log = paths.logs / "training.log"
            handler = TimedRotatingFileHandler(
                filename    = str(training_log),
                when        = "midnight",
                backupCount = log_cfg.max_log_days,
                encoding    = "utf-8",
            )
            handler.suffix = "%Y-%m-%d"
            handler.setFormatter(logging.Formatter(
                fmt     = log_cfg.format,
                datefmt = log_cfg.date_format,
            ))
            logger.addHandler(handler)
        except OSError:
            pass   # Silencieux : le logger racine prendra le relais

    return logger


# ══════════════════════════════════════════════════════════════════════
# SMOKE TEST  (python utils/logger.py)
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import tempfile, os

    # Pointer les logs vers un dossier temporaire pour le test
    os.environ.setdefault("ZYLOS_LOG_LEVEL", "DEBUG")

    log = get_logger("test.smoke")

    log.debug   ("Message DEBUG    — visible uniquement en mode DEBUG")
    log.info    ("Message INFO     — démarrage normal")
    log.warning ("Message WARNING  — quelque chose d'inhabituel")
    log.error   ("Message ERROR    — erreur récupérée")
    log.critical("Message CRITICAL — erreur fatale simulée")

    print("\n✅  Logger opérationnel.")
    print(f"   Handlers actifs : {logging.getLogger('zylos').handlers}")