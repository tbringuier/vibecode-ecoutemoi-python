# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec unique pour les 3 OS.
# - Windows : onefile — un seul EcouteMoi.exe auto-extractible, Python embarqué,
#   pas de dossier _internal.
# - Linux   : onedir, empaqueté ensuite en AppImage (scripts/make_appimage.sh).
# - macOS   : onedir + BUNDLE EcouteMoi.app (scripts/make_macos_app.sh).
# Embarque l'extension _pywhispercpp + ses libs ggml et la lib native rnnoise,
# mais PAS le loader Vulkan système (vulkan-1.dll appartient au pilote).
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

ROOT = Path(SPECPATH).resolve().parent  # noqa: F821 - SPECPATH is injected
ASSETS = ROOT / "src" / "ecoutemoi" / "assets"

ONEFILE = sys.platform == "win32"

binaries = []
binaries += collect_dynamic_libs("pyrnnoise")  # rnnoise.dll / librnnoise.so / .dylib
binaries += collect_dynamic_libs("pywhispercpp")  # ggml/whisper libs if the wheel ships any

# La wheel moteur compilée localement (non réparée par auditwheel/delocate/
# delvewheel) place les libs ggml/whisper à la RACINE de site-packages, hors de
# tout package : collect_dynamic_libs ne les voit pas. L'analyse binaire suit
# normalement la chaîne NEEDED, mais on les embarque explicitement — ceinture
# indispensable pour libggml-vulkan/metal (le backend GPU).
import sysconfig  # noqa: E402

_site = Path(sysconfig.get_paths()["purelib"])
for _pat in (
    "libggml*.so*", "libwhisper*.so*",
    "libggml*.dylib", "libwhisper*.dylib",
    "ggml*.dll", "whisper*.dll",
):
    for _f in _site.glob(_pat):
        if _f.is_file():
            binaries.append((str(_f), "."))

# Garde-fou anti-wheel-CPU : un venv où `uv sync`/`uv run` a restauré la wheel
# PyPI (CPU pur) produit un binaire qui démarre, transcrit, passe tous les
# smoke tests — et ne fera JAMAIS de Vulkan. On échoue donc AVANT d'empaqueter
# si le backend GPU n'est pas dans les binaires collectés (nom intact ou manglé
# par delvewheel).
# ECOUTEMOI_ALLOW_CPU_BUNDLE=1 pour outrepasser en connaissance de cause.
import os  # noqa: E402

_collected = " ".join(Path(_src).name.lower() for _src, _dest in binaries)
if os.environ.get("ECOUTEMOI_ALLOW_CPU_BUNDLE") != "1":
    if sys.platform in ("win32", "linux") and "ggml-vulkan" not in _collected:
        raise SystemExit(
            "ecoutemoi.spec : aucun ggml-vulkan dans les binaires collectés — le venv "
            "contient la wheel CPU de PyPI, pas la wheel Vulkan (scripts/build_wheel, "
            "puis uv pip install wheelhouse/... --force-reinstall ; ensuite TOUJOURS "
            "`uv run --no-sync`). ECOUTEMOI_ALLOW_CPU_BUNDLE=1 pour forcer un bundle CPU."
        )
    if sys.platform == "darwin" and "ggml-metal" not in _collected and "ggml-blas" not in _collected:
        print("ecoutemoi.spec : AVERTISSEMENT — ni ggml-metal ni ggml-blas collectés, "
              "l'app sera CPU pur (wheel Metal absente du venv ?)")

datas = [(str(ASSETS), "ecoutemoi/assets")]
datas += collect_data_files("pyrnnoise", excludes=["**/*.py"])

# libsndfile, décodeur de fichiers (WAV/FLAC/MP3/OGG/AIFF/CAF). `soundfile` est un
# MODULE, pas un paquet : collect_data_files ne voit donc pas son dossier de
# données, et la bibliothèque native manquerait à l'appel — la transcription de
# fichiers échouerait dans le binaire alors qu'elle marche en développement. Le
# module cherche sa lib dans « <dossier du module>/_soundfile_data », qui est la
# racine du bundle : la destination doit donc être ce nom-là, exactement.
import importlib.util  # noqa: E402

_sf = importlib.util.find_spec("soundfile")
if _sf is not None and _sf.origin:
    _sf_data = Path(_sf.origin).parent / "_soundfile_data"
    _sf_files = [(str(f), "_soundfile_data") for f in _sf_data.glob("*") if f.is_file()]
    if not _sf_files:
        raise SystemExit(f"ecoutemoi.spec : {_sf_data} est vide — libsndfile absente du venv")
    datas += _sf_files
else:
    raise SystemExit("ecoutemoi.spec : le module soundfile est introuvable (uv sync ?)")

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT / "src")],
    hookspath=[str(ROOT / "packaging" / "hooks")],  # webrtcvad-wheels metadata fix
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        "ecoutemoi.gui.main_window",  # lazy-imported from app.main
        "ecoutemoi.cli",
        "_pywhispercpp",
        "soundfile",  # importé à la demande par core/media
    ],
    excludes=[
        # couches hautes de pyrnnoise contournées volontairement (ctypes direct)
        "matplotlib", "audiolab",
        # PyAV embarque tout ffmpeg (~35 Mo) et n'est qu'un décodeur de SECOURS :
        # libsndfile couvre l'essentiel, et un ffmpeg installé fait le reste.
        "av",
        "PIL", "tkinter", "IPython", "pytest",
        "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtWebEngineCore",
    ],
    noarchive=False,
)

# Linux : ne JAMAIS embarquer libstdc++/libgcc_s. Les pilotes Vulkan de l'HÔTE
# (ICD Mesa/NVIDIA) se chargent dans notre processus ; avec LD_LIBRARY_PATH posé
# par le bootloader, un ICD compilé avec un GCC plus récent que celui d'Ubuntu
# 22.04 (conteneur de build) trouve la vieille libstdc++ embarquée et échoue —
# silencieusement : aucune ligne « ggml_vulkan », 0 périphérique, repli CPU.
# Démontré sur Ubuntu 24.04 (Mesa/LLVM 20 : GLIBCXX_3.4.32 > 3.4.30 de jammy).
# L'hôte fournit toujours ces deux libs (plancher glibc 2.35 déjà assumé) ;
# l'excludelist AppImage standard fait le même choix.
if sys.platform.startswith("linux"):
    _hostlibs = ("libstdc++.so", "libgcc_s.so")
    a.binaries = [
        (_dest, _src, _kind)
        for _dest, _src, _kind in a.binaries
        if not Path(_dest).name.startswith(_hostlibs)
    ]

pyz = PYZ(a.pure)

if ONEFILE:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        name="EcouteMoi",
        icon=str(ASSETS / "icon.ico"),
        console=False,  # GUI app; app.main() shims the None std streams
        disable_windowed_traceback=False,
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        exclude_binaries=True,
        name="EcouteMoi",
        icon=None,
        console=False,
        disable_windowed_traceback=False,
    )

    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        name="EcouteMoi",
    )

    if sys.platform == "darwin":
        icns = ROOT / "build" / "ecoutemoi.icns"
        app = BUNDLE(
            coll,
            name="EcouteMoi.app",
            icon=str(icns) if icns.is_file() else None,
            bundle_identifier="io.github.tbringuier.ecoutemoi",
            info_plist={
                # Sans cette clé le micro renvoie silencieusement des zéros sur macOS.
                "NSMicrophoneUsageDescription": (
                    "Écoute Moi capture le micro pour générer les sous-titres en local."
                ),
                "NSHighResolutionCapable": True,
                "LSMinimumSystemVersion": "12.0",
            },
        )
