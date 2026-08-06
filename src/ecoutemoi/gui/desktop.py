"""Intégration bureau : ouvrir un dossier, choisir un dossier.

Sous Linux, `QDesktopServices.openUrl()` (donc `xdg-open`) échoue silencieusement
dans les quatre cas les plus courants pour cette application :

1. **App packagée** — l'AppImage / le bundle PyInstaller pose `LD_LIBRARY_PATH`
   (et `PYTHONHOME`, `GIO_MODULE_DIR`, `GTK_PATH`…) vers ses propres libs. Le
   gestionnaire de fichiers hérite de cet environnement, charge une libstdc++ ou
   un module GIO incompatible et meurt avant d'afficher quoi que ce soit.
2. **`xdg-open` absent** — image minimale, ou conteneur (toolbox / distrobox /
   Docker) sans `xdg-utils`.
3. **Conteneur** — même avec `xdg-open`, il n'y a aucun gestionnaire de fichiers
   ni handler mimetype DANS le conteneur : il faut passer la main à l'hôte
   (`flatpak-spawn --host`, fourni par toolbox et Flatpak).
4. **Pas de portail** — session sans `xdg-desktop-portal`.

D'où une chaîne d'ouvreurs essayés dans l'ordre, avec un environnement assaini,
et surtout une ERREUR REMONTÉE quand aucun n'a fonctionné : un dossier qui ne
s'ouvre pas doit le dire, pas échouer en silence.

Les sélecteurs de fichiers suivent la même logique : sous Linux le dialogue
NATIF (portail xdg / GTK) est court-circuité au profit du dialogue Qt, qui ne
dépend ni du portail ni des modules GTK de l'hôte.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from ecoutemoi.core.hostenv import desktop_env, in_container

log = logging.getLogger(__name__)

# Gestionnaires de fichiers essayés en dernier ressort, quand ni xdg-open ni gio
# ni l'interface FileManager1 ne répondent.
_FILE_MANAGERS = (
    "nautilus",
    "dolphin",
    "nemo",
    "thunar",
    "caja",
    "pcmanfm",
    "pcmanfm-qt",
    "io.elementary.files",
)

# Un ouvreur GUI reste au premier plan : au-delà de ce délai sans sortie, on le
# considère lancé avec succès.
_ALIVE_IS_SUCCESS_S = 1.5


def _spawn(argv: list[str]) -> bool:
    """Lance `argv` détaché. True si le processus a réussi ou tourne toujours."""
    try:
        proc = subprocess.Popen(
            argv,
            env=desktop_env(),
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        log.debug("Ouvreur %s indisponible : %s", argv[0], exc)
        return False
    try:
        code = proc.wait(timeout=_ALIVE_IS_SUCCESS_S)
    except subprocess.TimeoutExpired:
        log.debug("Ouvreur %s toujours actif — considéré comme réussi", argv[0])
        return True  # gestionnaire de fichiers GUI resté au premier plan
    if code == 0:
        return True
    log.debug("Ouvreur %s a rendu le code %d", argv[0], code)
    return False


def _host_prefixes() -> list[list[str]]:
    """Préfixes permettant d'exécuter une commande sur l'HÔTE depuis un conteneur."""
    if not in_container():
        return []
    prefixes = []
    for tool, args in (("flatpak-spawn", ["--host"]), ("host-spawn", []), ("distrobox-host-exec", [])):
        if shutil.which(tool):
            prefixes.append([tool, *args])
    return prefixes


def _show_folders_argv(uri: str) -> list[str]:
    """Appel D-Bus `org.freedesktop.FileManager1.ShowFolders` via gdbus."""
    return [
        "gdbus", "call", "--session",
        "--dest", "org.freedesktop.FileManager1",
        "--object-path", "/org/freedesktop/FileManager1",
        "--method", "org.freedesktop.FileManager1.ShowFolders",
        f"['{uri}']", "",
    ]  # fmt: skip


def _linux_openers(path: Path) -> list[list[str]]:
    """Ouvreurs Linux, du plus standard au plus spécifique."""
    target = str(path)
    uri = path.as_uri()
    generic: list[list[str]] = [
        ["xdg-open", target],
        ["gio", "open", target],
        # Interface freedesktop implémentée par tous les gros gestionnaires de
        # fichiers : fonctionne même sans xdg-utils, et sélectionne le dossier.
        _show_folders_argv(uri),
    ]
    openers: list[list[str]] = []
    # Dans un conteneur, l'hôte d'abord : le conteneur n'a ni handler mimetype ni
    # gestionnaire de fichiers, donc les ouvreurs locaux échouent forcément.
    for prefix in _host_prefixes():
        openers += [[*prefix, *cmd] for cmd in generic]
    openers += generic
    openers += [[manager, target] for manager in _FILE_MANAGERS]
    return openers


def open_path(path: Path) -> str | None:
    """Ouvre `path` dans le gestionnaire de fichiers. None si OK, sinon la raison."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return f"dossier introuvable et non créable : {exc}"

    if sys.platform == "win32":
        try:
            os.startfile(str(path))
            return None
        except OSError as exc:
            log.warning("os.startfile a échoué (%s), repli QDesktopServices", exc)
    elif sys.platform == "darwin":
        if _spawn(["open", str(path)]):
            return None
    else:
        for argv in _linux_openers(path):
            if _spawn(argv):
                log.info("Dossier ouvert via %s", " ".join(argv[:2]))
                return None

    # Dernier recours : Qt. Sous Linux il retombe sur xdg-open (déjà essayé),
    # mais il gère les plateformes exotiques et les futurs backends.
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QDesktopServices

    if QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
        return None
    if sys.platform.startswith("linux") and in_container():
        return (
            "aucun gestionnaire de fichiers joignable — l'application tourne dans "
            "un conteneur (toolbox/distrobox/Flatpak) sans accès au bureau de "
            "l'hôte. Installez flatpak-spawn dans le conteneur, ou ouvrez le "
            "chemin à la main."
        )
    return (
        "aucun gestionnaire de fichiers n'a pu être lancé (xdg-open, gio, "
        "FileManager1 et les gestionnaires courants ont tous échoué)"
    )


def _dialog_options():
    """Options communes aux sélecteurs de fichiers.

    Sous Linux, le dialogue NATIF est désactivé : il passe par le portail xdg ou
    par les modules GTK de l'hôte, qui ne sont ni présents dans un conteneur ni
    compatibles avec les libs embarquées d'un bundle — le dialogue ne s'ouvrait
    alors jamais. Le dialogue Qt, lui, est purement interne.
    """
    from PySide6.QtWidgets import QFileDialog

    options = QFileDialog.Option(0)
    if sys.platform.startswith("linux"):
        options |= QFileDialog.Option.DontUseNativeDialog
    return options


def pick_directory(parent, title: str, start_dir: str = "") -> str:
    """Sélecteur de dossier. Chaîne vide si l'utilisateur annule."""
    from PySide6.QtWidgets import QFileDialog

    return QFileDialog.getExistingDirectory(parent, title, start_dir, _dialog_options())


def pick_files(parent, title: str, name_filter: str, start_dir: str = "") -> list[Path]:
    """Sélecteur multi-fichiers. Liste vide si l'utilisateur annule."""
    from PySide6.QtWidgets import QFileDialog

    paths, _ = QFileDialog.getOpenFileNames(parent, title, start_dir, name_filter, options=_dialog_options())
    return [Path(p) for p in paths]


def pick_save_path(parent, title: str, name_filter: str, start_path: str = "") -> Path | None:
    """Sélecteur d'enregistrement. None si l'utilisateur annule."""
    from PySide6.QtWidgets import QFileDialog

    path, _ = QFileDialog.getSaveFileName(parent, title, start_path, name_filter, options=_dialog_options())
    return Path(path) if path else None


__all__ = [
    "desktop_env",
    "in_container",
    "open_path",
    "pick_directory",
    "pick_files",
    "pick_save_path",
]
