"""File + console logging."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import platformdirs

from ecoutemoi.constants import APP_NAME

_FILE_FMT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_CONSOLE_FMT = "%(levelname)s %(name)s: %(message)s"


def log_dir() -> Path:
    return platformdirs.user_log_path(APP_NAME, appauthor=False)


def setup_logging(verbose: bool = False, directory: Path | None = None) -> Path:
    """Configure root logging: rotating file (DEBUG) + stderr (INFO/DEBUG).

    Returns the log file path.
    """
    d = directory or log_dir()
    d.mkdir(parents=True, exist_ok=True)
    log_file = d / "ecoutemoi.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Idempotent: drop handlers we previously installed (marked with a flag).
    for h in list(root.handlers):
        if getattr(h, "_ecoutemoi", False):
            root.removeHandler(h)
            h.close()

    fh = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(_FILE_FMT))
    fh._ecoutemoi = True  # type: ignore[attr-defined]
    root.addHandler(fh)

    ch = logging.StreamHandler(sys.stderr)
    # Console : avertissements seulement. Le détail va dans le fichier de log, qui
    # est fait pour ça — sur la console il hachait la ligne d'état du mode CLI, et
    # les seules informations qui comptent pour l'opérateur y sont imprimées
    # explicitement. `--verbose` rouvre le robinet en entier.
    ch.setLevel(logging.DEBUG if verbose else logging.WARNING)
    ch.setFormatter(logging.Formatter(_CONSOLE_FMT))
    ch._ecoutemoi = True  # type: ignore[attr-defined]
    root.addHandler(ch)

    logging.getLogger("ecoutemoi").debug("Logging initialised -> %s", log_file)
    return log_file
