"""Intégration bureau : assainissement de l'environnement et chaîne d'ouvreurs.

Ces fonctions conditionnent « Ouvrir le dossier » sous Linux ; elles se testent
sans Qt et sans bureau.
"""

from __future__ import annotations

from pathlib import Path

from ecoutemoi.gui.desktop import desktop_env


def test_desktop_env_restores_original_ld_library_path(monkeypatch):
    """PyInstaller sauvegarde l'original dans <VAR>_ORIG : c'est LUI qu'il faut passer."""
    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/_MEI123/libs")
    monkeypatch.setenv("LD_LIBRARY_PATH_ORIG", "/usr/lib64")
    env = desktop_env()
    assert env["LD_LIBRARY_PATH"] == "/usr/lib64"
    assert "LD_LIBRARY_PATH_ORIG" not in env


def test_desktop_env_drops_bundle_vars_without_original(monkeypatch):
    """Sans sauvegarde, la variable du bundle doit DISPARAÎTRE, pas être héritée."""
    for var in ("LD_LIBRARY_PATH", "PYTHONHOME", "GIO_MODULE_DIR", "QT_PLUGIN_PATH"):
        monkeypatch.setenv(var, "/tmp/_MEI123")
    monkeypatch.setenv("PATH", "/usr/bin")
    env = desktop_env()
    for var in ("LD_LIBRARY_PATH", "PYTHONHOME", "GIO_MODULE_DIR", "QT_PLUGIN_PATH"):
        assert var not in env
    assert env["PATH"] == "/usr/bin"  # le reste de l'environnement est préservé


def test_linux_openers_cover_the_standard_chain(tmp_path: Path, monkeypatch):
    import ecoutemoi.gui.desktop as desktop

    monkeypatch.setattr(desktop, "in_container", lambda: False)  # hors conteneur
    openers = desktop._linux_openers(tmp_path)
    programs = [argv[0] for argv in openers]
    assert programs[:3] == ["xdg-open", "gio", "gdbus"]
    # un gestionnaire de fichiers concret existe en dernier recours
    assert "nautilus" in programs and "dolphin" in programs
    show_folders = next(argv for argv in openers if argv[0] == "gdbus")
    assert tmp_path.as_uri() in " ".join(show_folders)


def test_linux_openers_prefer_the_host_inside_a_container(tmp_path, monkeypatch):
    """Dans un conteneur, aucun ouvreur LOCAL ne peut aboutir : l'hôte passe devant."""
    import ecoutemoi.gui.desktop as desktop

    monkeypatch.setattr(desktop, "in_container", lambda: True)
    monkeypatch.setattr(desktop.shutil, "which", lambda tool: "/usr/bin/" + tool)
    openers = desktop._linux_openers(tmp_path)
    assert openers[0][:2] == ["flatpak-spawn", "--host"]
    assert openers[0][2] == "xdg-open"
    assert ["xdg-open", str(tmp_path)] in openers  # le chemin local reste tenté ensuite
