"""Lancer un programme de l'HÔTE depuis une application empaquetée.

Un bundle PyInstaller (et plus encore une AppImage) pose `LD_LIBRARY_PATH`,
`PYTHONHOME`, `GIO_MODULE_DIR`… vers ses propres bibliothèques. Tout processus
lancé depuis l'application hérite de cet environnement, charge une libstdc++ ou
un module GIO incompatible, et meurt avant d'avoir rien fait. C'est vrai du
gestionnaire de fichiers (« Ouvrir le dossier ») comme de ffmpeg.

D'où ce module, partagé : il rend une copie de l'environnement débarrassée des
injections du bundle. Les processus qui sont NOUS-MÊMES (le sous-processus
moteur, par exemple) ne doivent surtout pas l'utiliser — eux ont besoin des
bibliothèques embarquées.
"""

from __future__ import annotations

import os
from pathlib import Path

# Variables injectées par PyInstaller / AppRun. Quand le bootloader a sauvegardé
# la valeur d'origine (`<VAR>_ORIG`), on la restaure ; sinon on retire la
# variable.
_BUNDLE_ENV_VARS = (
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "PYTHONHOME",
    "PYTHONPATH",
    "QT_PLUGIN_PATH",
    "QT_QPA_PLATFORM_PLUGIN_PATH",
    "GTK_PATH",
    "GTK_EXE_PREFIX",
    "GTK_DATA_PREFIX",
    "GIO_MODULE_DIR",
    "GDK_PIXBUF_MODULE_FILE",
    "GDK_PIXBUF_MODULEDIR",
    "GSETTINGS_SCHEMA_DIR",
    "XDG_DATA_DIRS",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "FONTCONFIG_FILE",
    "FONTCONFIG_PATH",
)


def in_container() -> bool:
    """Toolbox / distrobox / Flatpak / Docker : le bureau est HORS du conteneur."""
    return (
        Path("/run/.containerenv").exists()
        or Path("/.dockerenv").exists()
        or Path("/.flatpak-info").exists()
        # podman/toolbox/distrobox posent « container=oci » — en minuscules, ce
        # n'est pas une coquille : c'est le nom exact de la variable.
        or bool(os.environ.get("container"))  # noqa: SIM112
    )


def desktop_env() -> dict[str, str]:
    """Copie de l'environnement débarrassée des injections du bundle."""
    env = os.environ.copy()
    for var in _BUNDLE_ENV_VARS:
        original = env.pop(f"{var}_ORIG", None)
        if original:
            env[var] = original
        else:
            env.pop(var, None)
    return env


__all__ = ["desktop_env", "in_container"]
