"""Entry point: argument parsing and dispatch (console CLI or PySide6 GUI)."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys

from ecoutemoi import __version__


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ecoutemoi",
        description=(
            "Écoute Moi — sous-titrage temps réel local pour conférences "
            "(faster-whisper au CPU, whisper.cpp au GPU Vulkan/Metal)."
        ),
    )
    p.add_argument("--version", action="version", version=f"EcouteMoi {__version__}")
    p.add_argument("--cli", action="store_true", help="mode console")
    p.add_argument("--model", default=None, help="clé du modèle (voir --list-models)")
    p.add_argument("--mode", choices=["fr", "translate", "auto"], default=None,
                   help="fr : FR→FR ; translate : FR→EN ; auto : langue détectée → EN")  # fmt: skip
    p.add_argument("--preset", choices=["ultra", "equilibre", "stable", "phrase"], default=None,
                   help="preset de latence")  # fmt: skip
    p.add_argument("--device", type=int, default=None, help="index du périphérique d'entrée")
    p.add_argument("files", nargs="*", default=[], metavar="FICHIER",
                   help="fichiers audio/vidéo à transcrire (ouvre la fenêtre de transcription)")  # fmt: skip
    p.add_argument("--transcribe", nargs="+", metavar="FICHIER", default=None,
                   help="transcrire des fichiers audio/vidéo en console, puis quitter")  # fmt: skip
    p.add_argument("--to", default=None, metavar="FORMATS",
                   help="formats de sortie, séparés par des virgules (voir --list-formats)")  # fmt: skip
    p.add_argument("--out", default=None, metavar="DOSSIER",
                   help="dossier de sortie (défaut : à côté du fichier source)")  # fmt: skip
    p.add_argument("--list-formats", action="store_true",
                   help="lister les formats de sortie texte puis quitter")  # fmt: skip
    p.add_argument("--wav", default=None, help="rejouer un WAV dans le pipeline TEMPS RÉEL")
    p.add_argument("--rate", choices=["realtime", "fast"], default="realtime",
                   help="cadence de lecture du WAV (realtime simule le direct)")  # fmt: skip
    p.add_argument("--duration", type=float, default=None, help="durée maximale de capture (s)")
    p.add_argument("--gain", type=float, default=None, help="gain d'entrée (0.25–4)")
    p.add_argument("--no-denoise", action="store_true", help="désactiver la réduction de bruit RNNoise")
    p.add_argument("--no-highpass", action="store_true", help="désactiver le passe-haut 80 Hz")
    p.add_argument("--backend", choices=["auto", "gpu", "cpu"], default=None,
                   help="auto (GPU si disponible, sinon CPU), gpu (whisper.cpp Vulkan/Metal), "
                        "cpu (faster-whisper)")  # fmt: skip
    p.add_argument("--gpu-device", type=int, default=None, metavar="N",
                   help="index du périphérique GPU en multi-GPU (liste via --diag)")  # fmt: skip
    p.add_argument("--gpu-probe", metavar="OUT", default=None, help=argparse.SUPPRESS)
    p.add_argument("--reprobe-gpu", action="store_true",
                   help="oublier le sondage GPU mémorisé et re-tester la machine, puis quitter")  # fmt: skip
    p.add_argument("--check-engines", action="store_true",
                   help="vérifier que les deux moteurs sont utilisables ici, puis quitter")  # fmt: skip
    p.add_argument("--force-cpu", dest="backend", action="store_const", const="cpu",
                   help=argparse.SUPPRESS)  # alias de compatibilité pour --backend cpu  # fmt: skip
    p.add_argument("--list-devices", action="store_true", help="lister les entrées audio puis quitter")
    p.add_argument("--list-models", action="store_true", help="lister le registre de modèles puis quitter")
    p.add_argument("--diag", action="store_true",
                   help="diagnostic complet (environnement, GPU, modèles) puis quitter")  # fmt: skip
    p.add_argument("--download", metavar="MODEL", default=None,
                   help="télécharger un modèle (ou 'vad', ou 'all') puis quitter")  # fmt: skip
    p.add_argument("--format", dest="model_format", choices=["ggml", "ct2", "both"], default=None,
                   help="format téléchargé par --download : ggml (whisper.cpp/GPU), "
                        "ct2 (faster-whisper/CPU) ou both (défaut : celui du backend)")  # fmt: skip
    p.add_argument("--rtf", metavar="MODELS", default=None,
                   help="mesurer le RTF CPU des modèles donnés (a,b,c) sur --wav puis quitter")  # fmt: skip
    p.add_argument("--bench", action="store_true",
                   help="benchmark guidé (sous-processus) : tableau + JSON puis quitter")  # fmt: skip
    p.add_argument("--wav-fr", default=None, help="WAV de calibration FR pour --bench")
    p.add_argument("--wav-en", default=None, help="WAV de calibration EN pour --bench")
    p.add_argument("--models", default=None, help="liste de modèles pour --bench (défaut : installés)")
    p.add_argument("--bench-worker", nargs=2, metavar=("TASK", "OUT"), default=None,
                   help=argparse.SUPPRESS)  # fmt: skip
    p.add_argument("--engine-worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument(
        "--lexicon",
        default=None,
        help="lexique de la conférence (noms propres, acronymes) biaisant le décodage",
    )
    p.add_argument("--engine-in-process", action="store_true",
                   help="décoder DANS ce processus au lieu d'un sous-processus dédié")  # fmt: skip
    p.add_argument("--import-model", metavar="FICHIER", action="append", default=None,
                   help="installer un .bin ggml fourni à la main, puis quitter")  # fmt: skip
    p.add_argument("--measure-gil", action="store_true",
                   help="mesurer la gigue GIL (thread témoin 100 Hz) pendant l'inférence")  # fmt: skip
    p.add_argument("--save-dir", default=None, help="dossier de session (autosave + exports)")
    p.add_argument("--stats-json", default=None, help="écrire les statistiques de session en JSON")
    p.add_argument("--export", default=None, metavar="FORMATS",
                   help="exporter à l'arrêt : formats séparés par des virgules, ou « all »")  # fmt: skip
    p.add_argument("--verbose", "-v", action="store_true", help="logs détaillés sur la console")
    return p


def _cmd_list_devices() -> int:
    from ecoutemoi.core.audio import list_input_devices

    devices = list_input_devices()
    if not devices:
        print("Aucun périphérique d'entrée détecté.")
        return 1
    for d in devices:
        print(f"[{d.index:3d}] {d.name}  ({d.hostapi}, {d.default_sr} Hz, {d.channels} ch)")
    return 0


def _cmd_list_models() -> int:
    """Le registre, AVEC les deux formats : celui du GPU et celui du CPU.

    Un modèle n'est pas « installé » dans l'absolu en 2.0 — il l'est pour un
    moteur donné. Masquer cette distinction ferait croire qu'un `small` présent
    en ggml suffit à démarrer une session CPU, alors qu'il faudra télécharger
    464 Mo de plus.
    """
    from ecoutemoi.core import models

    ggml = set(models.installed_models(fmt=models.FMT_GGML))
    ct2 = set(models.installed_models(fmt=models.FMT_CT2))
    print(f"{'clé':<22} {'quant':>6} {'ggml':>8} {'ct2':>8} {'trad.':>6} {'RAM':>7}  "
          f"{'installé':<12} rôle")  # fmt: skip
    for key, spec in models.REGISTRY.items():
        marks = [f for f, s in (("ggml", ggml), ("ct2", ct2)) if key in s]
        state = "+".join(marks) if marks else "—"
        ct2_size = f"{spec.ct2_size_mb} Mo" if spec.ct2_repo else "—"
        print(
            f"{key:<22} {spec.quant:>6} {spec.size_mb:>5} Mo {ct2_size:>8} "
            f"{'oui' if spec.translate else 'NON':>6} {spec.ram_gb:>5.1f} Go  {state:<12} {spec.role}"
        )
    print(f"\nDossier des modèles : {models.models_dir()}")
    print("ggml = whisper.cpp (GPU Vulkan/Metal) · ct2 = faster-whisper (CPU).")
    print("Un dossier ct2 est partagé par toutes les quantizations d'une même famille.")
    return 0


def _cmd_list_formats() -> int:
    from ecoutemoi.core.transcript import TRANSCRIPT_FORMATS

    print(f"{'format':<8} {'extensions':<24} description")
    for key, fmt in TRANSCRIPT_FORMATS.items():
        print(f"{key:<8} {' '.join(fmt.extensions):<24} {fmt.label}")
    print(
        "\nToute autre extension est acceptée et reçoit du texte brut (« notes.dat », « conference.rtf »…)."
    )
    return 0


def _cmd_import_models(paths: list[str]) -> int:
    """`--import-model` : installer des .bin fournis à la main."""
    from pathlib import Path

    from ecoutemoi.core import models

    failed = 0
    for raw in paths:
        try:
            key, dest = models.import_model_file(Path(raw))
            print(f"{key} installé : {dest}")
        except (OSError, ValueError) as exc:
            print(f"{raw} : {exc}", file=sys.stderr)
            failed += 1
    return 1 if failed else 0


def _resolve_download_formats(requested: str | None) -> list[str]:
    """Format(s) à télécharger. Par défaut : celui dont le backend a besoin."""
    from ecoutemoi.config import load_settings
    from ecoutemoi.core import models

    if requested == "both":
        return [models.FMT_GGML, models.FMT_CT2]
    if requested in (models.FMT_GGML, models.FMT_CT2):
        return [requested]
    from ecoutemoi.cli import plan_engine

    _choice, fmt = plan_engine(load_settings())
    return [fmt]


def _cmd_download(arg: str, requested_format: str | None = None) -> int:
    from ecoutemoi.core import models

    if arg == "vad":
        models.ensure_vad_model()
        return 0
    formats = _resolve_download_formats(requested_format)
    if arg == "all":
        for fmt in formats:
            models.download_many(list(models.REGISTRY.values()), fmt=fmt)
        if models.FMT_GGML in formats:
            models.ensure_vad_model()
        return 0
    keys = [k.strip() for k in arg.split(",") if k.strip()]
    unknown = [k for k in keys if k not in models.REGISTRY]
    if unknown:
        print(f"Modèle(s) inconnu(s) : {', '.join(unknown)}", file=sys.stderr)
        return 1
    for k in keys:
        for fmt in formats:
            models.ensure_model(k, fmt=fmt)
    if models.FMT_GGML in formats:
        models.ensure_vad_model()  # VAD ggml : whisper.cpp seul en a besoin
    return 0


def _cmd_check_engines() -> int:
    """`--check-engines` : les deux moteurs sont-ils réellement utilisables ici ?

    Contrôle d'EMPAQUETAGE avant tout. Un bundle où faster-whisper ne s'importe
    pas démarre, transcrit et passe tous les autres smoke tests — trois fois plus
    lentement sur CPU, sans que rien ne le signale. C'est exactement ce qui
    arrive quand une exclusion PyInstaller casse une chaîne d'imports : aucune
    erreur au build, aucune erreur au lancement, juste de la lenteur.
    """
    from ecoutemoi.core import engine_fw
    from ecoutemoi.core.engine import gpu_backend_libs

    ok = True
    try:
        import _pywhispercpp  # noqa: F401
        from pywhispercpp.model import Model  # noqa: F401

        libs = gpu_backend_libs()
        print(f"whisper.cpp (GPU)    : OK · libs backend GPU : {', '.join(libs) if libs else 'AUCUNE'}")
    except Exception as exc:
        print(f"whisper.cpp (GPU)    : ABSENT — {type(exc).__name__}: {exc}", file=sys.stderr)
        ok = False
    if engine_fw.available():
        print(f"faster-whisper (CPU) : OK · {engine_fw.versions()} · "
              f"calcul : {', '.join(engine_fw.supported_compute_types())}")  # fmt: skip
    else:
        # `available()` avale l'exception (elle ne doit pas casser une session) :
        # ici, c'est justement le message qu'on veut voir.
        try:
            import ctranslate2  # noqa: F401
            import faster_whisper  # noqa: F401

            reason = "importable mais déclaré indisponible"  # fmt: skip
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
        print(f"faster-whisper (CPU) : ABSENT — {reason}", file=sys.stderr)
        ok = False
    return 0 if ok else 1


def _cmd_reprobe_gpu() -> int:
    """`--reprobe-gpu` : oublie le cache et re-teste la machine."""
    from ecoutemoi.core import gpuprobe

    gpuprobe.clear_cache()
    result = gpuprobe.gpu_candidates(force=True)
    print(f"GPU utilisable : {'oui' if result.gpu else 'non'}")
    print(f"Détail         : {result.summary}")
    if result.backend_libs:
        print(f"Libs backend   : {', '.join(result.backend_libs)}")
    print(f"Mémorisé dans  : {gpuprobe.cache_path()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # PyInstaller --windowed: std streams are None; argparse/print would crash.
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):  # non-tty / redirected streams
            stream.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)

    from ecoutemoi.logging_setup import setup_logging

    setup_logging(verbose=args.verbose)

    # Le worker moteur AVANT tout le reste : son stdout est un tube de protocole,
    # rien d'autre ne doit y écrire.
    if args.engine_worker:
        from ecoutemoi.core.engine_proc import worker_main

        return worker_main()
    if args.gpu_probe:
        from ecoutemoi.core.gpuprobe import worker_main as gpu_probe_main

        return gpu_probe_main(args.gpu_probe)
    if args.check_engines:
        return _cmd_check_engines()
    if args.reprobe_gpu:
        return _cmd_reprobe_gpu()
    if args.import_model:
        return _cmd_import_models(args.import_model)
    if args.diag:
        from ecoutemoi.cli import run_diag

        return run_diag(args)
    if args.bench_worker:
        from ecoutemoi.core.bench import worker_main

        return worker_main(*args.bench_worker)
    if args.bench:
        from ecoutemoi.cli import run_bench

        return run_bench(args)
    if args.list_devices:
        return _cmd_list_devices()
    if args.list_models:
        return _cmd_list_models()
    if args.list_formats:
        return _cmd_list_formats()
    if args.download:
        return _cmd_download(args.download, args.model_format)
    if args.rtf:
        from ecoutemoi.cli import run_rtf_bench

        return run_rtf_bench(args)
    if args.transcribe:
        from ecoutemoi.cli import run_transcribe

        return run_transcribe(args)
    if args.cli or args.wav:
        from ecoutemoi.cli import run_cli

        return run_cli(args)

    from ecoutemoi.gui.main_window import run_gui

    # Fichiers passés en arguments (« Ouvrir avec… » depuis le gestionnaire de
    # fichiers) : la fenêtre de transcription s'ouvre dessus, déjà remplie.
    return run_gui(args.files)
