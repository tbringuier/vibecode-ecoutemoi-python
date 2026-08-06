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
        description="Écoute Moi — sous-titrage temps réel local pour conférences (whisper.cpp).",
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
                   help="moteur : auto (GPU si disponible, repli CPU), gpu (Vulkan/Metal), cpu")  # fmt: skip
    p.add_argument("--gpu-device", type=int, default=None, metavar="N",
                   help="index du périphérique GPU en multi-GPU (liste via --diag)")  # fmt: skip
    p.add_argument("--force-cpu", dest="backend", action="store_const", const="cpu",
                   help=argparse.SUPPRESS)  # alias de compatibilité pour --backend cpu  # fmt: skip
    p.add_argument("--list-devices", action="store_true", help="lister les entrées audio puis quitter")
    p.add_argument("--list-models", action="store_true", help="lister le registre de modèles puis quitter")
    p.add_argument("--diag", action="store_true",
                   help="diagnostic complet (environnement, GPU, modèles) puis quitter")  # fmt: skip
    p.add_argument("--download", metavar="MODEL", default=None,
                   help="télécharger un modèle (ou 'vad', ou 'all') puis quitter")  # fmt: skip
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
    from ecoutemoi.core import models

    installed = set(models.installed_models())
    header = f"{'clé':<22} {'quant':>6} {'taille':>9} {'trad. EN':>9} {'RAM est.':>9}  {'installé':>8}  rôle"
    print(header)
    for key, spec in models.REGISTRY.items():
        mark = "oui" if key in installed else "non"
        trad = "oui" if spec.translate else "NON"
        print(
            f"{key:<22} {spec.quant:>6} {spec.size_mb:>6} Mo {trad:>9} "
            f"{spec.ram_gb:>7.1f} Go  {mark:>8}  {spec.role}"
        )
    print(f"\nDossier des modèles : {models.models_dir()}")
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


def _cmd_download(arg: str) -> int:
    from ecoutemoi.core import models

    if arg == "vad":
        models.ensure_vad_model()
        return 0
    if arg == "all":
        models.download_many(list(models.REGISTRY.values()))
        models.ensure_vad_model()
        return 0
    keys = [k.strip() for k in arg.split(",") if k.strip()]
    unknown = [k for k in keys if k not in models.REGISTRY]
    if unknown:
        print(f"Modèle(s) inconnu(s) : {', '.join(unknown)}", file=sys.stderr)
        return 1
    for k in keys:
        models.ensure_model(k)
    models.ensure_vad_model()
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
        return _cmd_download(args.download)
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
