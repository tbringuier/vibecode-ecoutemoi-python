"""Console front-end: capture -> DSP -> gate -> engine -> ANSI console.

Seul le texte VALIDÉ est imprimé (blanc) : les mots encore en attente de
LocalAgreement ne quittent jamais le streamer. Stats (RTF / lag / latence /
fenêtre) sur la ligne d'état et dans le fichier de log.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import shutil
import statistics
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

from ecoutemoi.config import load_settings
from ecoutemoi.constants import ENGINE_FASTER_WHISPER, PRESETS, TARGET_SR
from ecoutemoi.core import engines, models
from ecoutemoi.core.dsp import DspChain
from ecoutemoi.core.engine_base import EngineParams, Segment
from ecoutemoi.core.gate import SpeechGate
from ecoutemoi.core.streamer import Streamer, StreamStats
from ecoutemoi.core.transcript import SessionSegment, TranscriptStore, new_session_dir

log = logging.getLogger(__name__)

# Largeur de faisceau du décodeur. En DIRECT, glouton (1) : chaque hypothèse
# supplémentaire coûte un passage de décodeur complet, et la latence se paie
# devant la salle. Sur FICHIER, plus rien ne presse et la qualité prime — voir
# FILE_BEAM_SIZE. (Sans effet sur whisper.cpp, dont le décodage reste glouton.)
STREAM_BEAM_SIZE = 1
FILE_BEAM_SIZE = 5

_WHITE = "\x1b[97m"
_GRAY = "\x1b[90m"
_DIM = "\x1b[2m"
_RESET = "\x1b[0m"
_CLEAR = "\r\x1b[K"


def _enable_vt() -> None:
    if os.name == "nt":
        os.system("")  # enables VT100 processing in the Windows console


class ConsoleUI:
    """Single status line for partial text; finalized segments become permanent lines."""

    def __init__(self):
        _enable_vt()
        self._last_stats: StreamStats | None = None
        self._lock = threading.Lock()

    def _width(self) -> int:
        return shutil.get_terminal_size((100, 20)).columns

    def partial(self, committed: str) -> None:
        with self._lock:
            suffix = ""
            if self._last_stats is not None:
                s = self._last_stats
                suffix = f"  {_DIM}[RTF {s.rtf:.1f} · lag {s.lag:.2f} · fen {s.window_s:.1f}s]{_RESET}"
                suffix_len = len(f"  [RTF {s.rtf:.1f} · lag {s.lag:.2f} · fen {s.window_s:.1f}s]")
            else:
                suffix_len = 0
            budget = max(10, self._width() - 1 - suffix_len)
            committed_part = committed
            if len(committed_part) > budget:
                # keep the tail: most recent words matter on a status line
                committed_part = "…" + committed_part[len(committed_part) - budget + 1 :]
            line = f"{_WHITE}{committed_part}{_RESET}{suffix}"
            sys.stdout.write(_CLEAR + line)
            sys.stdout.flush()

    def finalized(self, seg: SessionSegment) -> None:
        with self._lock:
            sys.stdout.write(_CLEAR + f"{_WHITE}{seg.text}{_RESET}\n")
            sys.stdout.flush()

    def stats(self, s: StreamStats) -> None:
        self._last_stats = s

    def notice(self, msg: str) -> None:
        with self._lock:
            sys.stdout.write(_CLEAR + f"{_DIM}ℹ {msg}{_RESET}\n")
            sys.stdout.flush()

    def line(self, msg: str) -> None:
        with self._lock:
            sys.stdout.write(_CLEAR + msg + "\n")
            sys.stdout.flush()


class GilProbe(threading.Thread):
    """100 Hz witness thread: sleep(10 ms) overshoot = scheduler + GIL jitter."""

    def __init__(self, interval_ms: float = 10.0):
        super().__init__(daemon=True, name="gil-probe")
        self.interval = interval_ms / 1000.0
        self.samples: list[float] = []
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            t0 = time.perf_counter()
            time.sleep(self.interval)
            overshoot = (time.perf_counter() - t0 - self.interval) * 1000.0
            self.samples.append(max(0.0, overshoot))

    def stop(self) -> None:
        self._halt.set()

    def report(self) -> dict:
        if not self.samples:
            return {}
        xs = sorted(self.samples)

        def pct(p: float) -> float:
            return xs[min(len(xs) - 1, int(p * len(xs)))]

        return {
            "samples": len(xs),
            "p50_ms": round(pct(0.50), 2),
            "p95_ms": round(pct(0.95), 2),
            "p99_ms": round(pct(0.99), 2),
            "max_ms": round(xs[-1], 2),
            "over_50ms": sum(1 for x in xs if x > 50.0),
        }


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load a PCM WAV as float32 mono in [-1, 1] + sample rate."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sw == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"WAV {sw * 8} bits non géré (utiliser PCM 16 bits)")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return np.ascontiguousarray(x, dtype=np.float32), sr


def resample_to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return x
    import soxr

    return soxr.resample(x, sr, TARGET_SR, quality="HQ").astype(np.float32)


def resolve_mode(mode: str, spec: models.ModelSpec) -> tuple[str, bool]:
    """Map an operator mode to whisper (language, translate).

    - fr        : FR -> FR (language=fr, pas de traduction)
    - translate : FR -> EN (language=fr, task=translate)
    - auto      : AUTO -> EN (language=auto : langue source re-détectée en continu,
                  sortie toujours traduite en anglais)
    """
    if mode not in ("fr", "translate", "auto"):
        raise ValueError(f"Mode inconnu : {mode}")
    translate = mode in ("translate", "auto")
    if translate and not spec.translate:
        raise ValueError(
            f"Le modèle {spec.key} ne sait pas traduire (il ressortirait la langue "
            f"source). Choisissez un autre modèle pour les modes FR→EN et Auto→EN."
        )
    language = {"fr": "fr", "translate": "fr", "auto": "auto"}[mode]
    return language, translate


def plan_engine(settings, backend: str | None = None, engine: str | None = None) -> tuple[str, str]:
    """(moteur retenu, format de modèle INDISPENSABLE) pour ce backend.

    C'est ici que se joue la répartition de la 2.0 : whisper.cpp au GPU,
    faster-whisper au CPU. En backend « auto », il faut trancher AVANT de
    télécharger — sinon on ferait payer les deux formats (665 Mo pour `small` au
    lieu de 181 ou 464) à tout le monde. Le sondage GPU (`core/gpuprobe`) tient
    ce rôle : il n'ouvre aucun modèle et son résultat est mis en cache.

    « auto » RESTE possible en sortie : des périphériques sont énumérés, mais
    seul le chargement réel dira s'ils s'initialisent — `core/engines.py` garde
    donc son repli. Le format indispensable est celui du moteur retenu ; l'autre
    n'est branché que s'il est DÉJÀ sur disque.
    """
    backend = backend or settings.backend
    cpu_engine = getattr(settings, "cpu_engine", ENGINE_FASTER_WHISPER)
    choice = engine or engines.engine_for_backend(backend, cpu_engine)
    if choice == "auto":
        from ecoutemoi.core import gpuprobe

        probe = gpuprobe.gpu_candidates()
        if not probe.gpu:
            log.info("Sondage GPU : aucun périphérique (%s) — moteur CPU.", probe.summary)
            choice = cpu_engine
    fmt = models.FMT_CT2 if choice == ENGINE_FASTER_WHISPER else models.FMT_GGML
    return choice, fmt


def resolve_model_paths(spec, choice: str) -> tuple[Path | None, Path | None]:
    """(chemin ggml, dossier CTranslate2) — télécharge le format indispensable.

    L'autre format n'est jamais téléchargé ici : il n'est branché que s'il se
    trouve déjà sur disque, où il sert de repli gratuit (moteur CPU absent de
    l'environnement, GPU énuméré mais qui refuse de s'initialiser).
    """
    if choice == ENGINE_FASTER_WHISPER and spec.ct2_repo:
        ct2 = models.ensure_model(spec.key, fmt=models.FMT_CT2)
        ggml = models.model_path(spec) if models.is_installed(spec, fmt=models.FMT_GGML) else None
        return ggml, ct2
    ggml = models.ensure_model(spec.key, fmt=models.FMT_GGML)
    ct2 = models.ct2_dir(spec) if models.is_installed(spec, fmt=models.FMT_CT2) else None
    return ggml, ct2


def make_engine(
    settings,
    model_key: str,
    mode: str,
    *,
    backend: str | None = None,
    gpu_device: int | None = None,
    subprocess: bool | None = None,
    engine: str | None = None,
    beam_size: int | None = None,
):
    """Shared engine factory (CLI + GUI). Raises ValueError on unsupported combos.

    `subprocess=None` suit le réglage `engine_subprocess` ; l'objet retourné
    expose la même surface quel que soit le moteur ET le nombre de processus.
    Les workers de benchmark passent explicitement False : ils SONT déjà des
    sous-processus.
    """
    spec = models.REGISTRY[model_key]
    language, translate = resolve_mode(mode, spec)
    backend = backend or settings.backend
    choice, _fmt = plan_engine(settings, backend=backend, engine=engine)
    model_path, ct2_path = resolve_model_paths(spec, choice)
    vad_path = None
    if model_path is not None:  # VAD ggml : whisper.cpp seul en a besoin
        try:
            vad_path = models.ensure_vad_model()
        except Exception as exc:
            log.warning("VAD Silero indisponible (%s) — repli : gate seul + filtre no_speech", exc)
    if gpu_device is None:
        gpu_device = getattr(settings, "gpu_device", 0)
    wanted_compute = getattr(settings, "cpu_compute_type", "auto")
    params = EngineParams(
        model_path=model_path,
        language=language,
        translate=translate,
        n_threads=settings.n_threads,
        backend=backend,
        flash_attn=settings.flash_attn,
        vad_model_path=vad_path,
        carry_context=settings.carry_context,
        gpu_device=max(0, int(gpu_device or 0)),
        lexicon=getattr(settings, "lexicon", "") or "",
        ct2_path=ct2_path,
        compute_type=spec.compute_type if wanted_compute in ("", "auto") else wanted_compute,
        beam_size=STREAM_BEAM_SIZE if beam_size is None else max(1, int(beam_size)),
        engine=choice,
        trim_audio_ctx=bool(getattr(settings, "trim_audio_ctx", True)),
    )
    if subprocess is None:
        subprocess = bool(getattr(settings, "engine_subprocess", False))
    if subprocess:
        from ecoutemoi.core.engine_proc import SubprocessEngine

        return SubprocessEngine(params)
    return engines.create_engine(params)


def _make_engine(args, settings, model_key: str, mode: str, *, subprocess: bool | None = None,
                 beam_size: int | None = None):  # fmt: skip
    try:
        if subprocess is None and getattr(args, "engine_in_process", False):
            subprocess = False
        return make_engine(
            settings, model_key, mode,
            backend=args.backend, gpu_device=getattr(args, "gpu_device", None),
            subprocess=subprocess, beam_size=beam_size,
        )  # fmt: skip
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def run_cli(args) -> int:
    settings = load_settings()
    if getattr(args, "lexicon", None):
        settings = dataclasses.replace(settings, lexicon=args.lexicon)
    model_key = args.model or settings.model
    mode = args.mode or settings.mode
    preset = PRESETS[args.preset or settings.preset]
    denoise = settings.denoise and not args.no_denoise
    highpass = settings.highpass and not args.no_highpass
    realtime = args.wav is None or args.rate == "realtime"

    if model_key not in models.REGISTRY:
        print(f"Modèle inconnu : {model_key} (voir --list-models)", file=sys.stderr)
        return 1

    engine = _make_engine(args, settings, model_key, mode)
    session_dir = Path(args.save_dir) if args.save_dir else new_session_dir()
    store = TranscriptStore(session_dir, autosave=settings.autosave)
    ui = ConsoleUI()
    all_stats: list[StreamStats] = []

    def on_stats(s: StreamStats) -> None:
        ui.stats(s)
        all_stats.append(s)
        log.debug(
            "stats: decode=%.0fms window=%.1fs rtf=%.2f lag=%.2f latency=%.0fms",
            s.decode_ms, s.window_s, s.rtf, s.lag, s.latency_ms,
        )  # fmt: skip

    streamer = Streamer(
        engine,
        store,
        preset,
        mode=mode,
        min_interval_ms=settings.min_update_interval_ms,
        keep_back=settings.keep_back,
        window_max_s=settings.window_max_s,
        no_speech_prob_max=settings.no_speech_prob_max,
        hallucination_filter=settings.hallucination_filter,
        realtime=realtime,
        on_partial=ui.partial,
        on_finalized=ui.finalized,
        on_stats=on_stats,
        on_notice=ui.notice,
    )

    ui.line(f"Modèle : {model_key} · mode {mode} · preset {preset.label}")
    ui.line(f"Backend : {engine.backend_info()} · VAD interne : {'oui' if engine.vad_active else 'non'}")
    gpus = engine.gpu_devices()
    if len(gpus) >= 2:  # le choix du périphérique devient pertinent
        active = engine.active_gpu_index()
        listing = " ; ".join(f"{i} — {n}" for i, n in gpus)
        ui.line(f"GPU Vulkan : {listing}"
                + (f" · actif : {active}" if active is not None else "")
                + " (choix : --gpu-device N)")  # fmt: skip
    requested = args.backend or settings.backend
    if requested != "cpu" and not engine.gpu_active():
        prefix = "Backend GPU demandé mais indisponible" if requested == "gpu" else "GPU indisponible"
        reason = getattr(engine, "fallback_reason", None) or engine.gpu_diagnostic()
        ui.notice(f"{prefix} — décodage CPU ({engine.backend_info()}) : {reason}")
    ui.line(f"Session : {session_dir}")
    ui.line("Ctrl+C pour arrêter.")

    probe: GilProbe | None = None
    baseline: dict = {}
    if args.measure_gil:
        b = GilProbe()
        b.start()
        time.sleep(2.0)
        b.stop()
        b.join()
        baseline = b.report()
        probe = GilProbe()
        probe.start()

    gate = SpeechGate(silence_ms=settings.silence_ms or preset.silence_ms, denoise_fusion=denoise)
    streamer.start()
    exit_code = 0
    try:
        if args.wav:
            _feed_wav(Path(args.wav), gate, streamer, denoise, highpass, realtime, args.duration)
        else:
            _feed_mic(args, settings, gate, streamer, denoise, highpass, args.duration, ui)
    except KeyboardInterrupt:
        ui.line("Arrêt demandé…")
    except Exception as exc:
        log.exception("Erreur pipeline")
        ui.line(f"Erreur : {exc}")
        exit_code = 1
    finally:
        streamer.stop()
        if probe is not None:
            probe.stop()
            probe.join()
        store.close()
        engine.close()  # désinstalle le callback de log whisper (sortie propre)

    _summary(ui, store, all_stats, session_dir, baseline, probe)
    _exports(args, settings, store, session_dir, ui)
    if getattr(args, "stats_json", None):
        _dump_stats_json(Path(args.stats_json), all_stats, baseline, probe, engine, preset)
    return exit_code


def _feed_wav(path, gate, streamer, denoise, highpass, realtime, duration) -> None:
    x, sr = load_wav(path)
    block = max(1, int(sr * 0.01))
    dsp = DspChain(sr, denoise=denoise, highpass=highpass)
    # trailing silence so the gate closes the last utterance naturally
    x = np.concatenate([x, np.zeros(int(sr * 1.2), dtype=np.float32)])
    if duration is not None:
        x = x[: int(duration * sr)]
    t_start = time.monotonic()
    for i in range(0, len(x), block):
        chunk = x[i : i + block]
        for b16, prob in dsp.process(chunk, last=(i + block >= len(x))):
            for ev in gate.feed(b16, prob):
                streamer.on_gate_event(ev)
        if realtime:
            target = t_start + (i + block) / sr
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)


def _feed_mic(args, settings, gate, streamer, denoise, highpass, duration, ui) -> None:
    from ecoutemoi.core.audio import AudioCapture

    cap = AudioCapture(device=args.device if args.device is not None else settings.device_index,
                       gain=args.gain if args.gain is not None else settings.gain)  # fmt: skip
    dsp = DspChain(cap.sr, denoise=denoise, highpass=highpass)
    cap.start()
    t0 = time.monotonic()
    silent_since: float | None = None
    try:
        while True:
            blockn = cap.read(timeout=0.5)
            if blockn is None:
                continue
            for b16, prob in dsp.process(blockn):
                for ev in gate.feed(b16, prob):
                    streamer.on_gate_event(ev)
            if duration is not None and time.monotonic() - t0 >= duration:
                return
            # Flat zero signal for 5 s -> probably a mic permission issue
            if cap.rms < 1e-6:
                if silent_since is None:
                    silent_since = time.monotonic()
                elif time.monotonic() - silent_since > 5.0:
                    silent_since = None
                    ui.notice(
                        "Niveau micro nul depuis 5 s — vérifiez la permission micro "
                        "(macOS : Réglages > Confidentialité > Microphone) et le bon périphérique."
                    )
            else:
                silent_since = None
    finally:
        cap.stop()


def _summary(ui, store, all_stats, session_dir, baseline, probe) -> None:
    ui.line("")
    ui.line(f"Segments : {len(store.segments)} · mots : {store.word_count()}")
    if all_stats:
        rtfs = [s.rtf for s in all_stats if s.rtf > 0]
        lats = [s.latency_ms for s in all_stats]
        decs = [s.decode_ms for s in all_stats]
        ui.line(
            f"Décodages : {len(all_stats)} · RTF médian {statistics.median(rtfs):.2f} · "
            f"décodage médian {statistics.median(decs):.0f} ms · "
            f"latence médiane {statistics.median(lats):.0f} ms"
        )
    ui.line(f"Session enregistrée dans : {session_dir}")
    if probe is not None:
        rep = probe.report()
        ui.line(f"Gigue GIL (base 2 s sans moteur) : {baseline}")
        ui.line(f"Gigue GIL (pendant la session)  : {rep}")


def _exports(args, settings, store, session_dir: Path, ui) -> None:
    from ecoutemoi.core.transcript import TRANSCRIPT_FORMATS

    choice = args.export or ("all" if settings.export_on_stop else None)
    if not choice or not store.segments:
        return
    keys = ["txt", "srt", "vtt"] if choice == "all" else _resolve_formats(choice)
    for key in keys:
        fmt = TRANSCRIPT_FORMATS.get(key)
        if fmt is None:
            ui.line(f"Format inconnu ignoré : {key} (voir --list-formats)")
            continue
        # « transcript.txt » est déjà pris par l'autosave : l'export TXT porte un
        # autre nom pour qu'on ne perde jamais le premier au profit du second.
        stem = "transcript_export" if fmt.key == "txt" else "transcript"
        store.export(session_dir / f"{stem}{fmt.extension}", key=key, timestamps=settings.txt_timestamps)
    ui.line(f"Exports écrits dans {session_dir}")


def _dump_stats_json(path: Path, all_stats, baseline, probe, engine, preset) -> None:
    data = {
        "preset": preset.key,
        "backend": engine.backend_info(),
        "vad_active": engine.vad_active,
        "dropped_params": engine.dropped_params,
        "load_variant": engine.load_variant,
        "gil_baseline": baseline,
        "gil_session": probe.report() if probe is not None else {},
        "decodes": [
            {
                "decode_ms": round(s.decode_ms, 1),
                "window_s": round(s.window_s, 2),
                "rtf": round(s.rtf, 2),
                "lag": round(s.lag, 3),
                "latency_ms": round(s.latency_ms, 1),
            }
            for s in all_stats
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_transcribe(args) -> int:
    """`--transcribe FICHIERS` : transcription hors direct, en console.

    Un seul moteur pour tout le lot (le chargement d'un modèle coûte des
    secondes), un fichier fautif n'interrompt pas les autres, et le code de
    retour ne vaut 0 que si TOUT est passé — c'est ce qu'attend un script.
    """
    from ecoutemoi.core import filejob, media
    from ecoutemoi.core.transcript import (
        DEFAULT_FORMATS,
        TRANSCRIPT_FORMATS,
        output_path,
        unique_path,
        write_transcript,
    )

    settings = load_settings()
    if getattr(args, "lexicon", None):
        settings = dataclasses.replace(settings, lexicon=args.lexicon)
    media.set_ffmpeg_path(settings.ffmpeg_path)

    paths = [Path(p) for p in args.transcribe]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        print("Fichier(s) introuvable(s) : " + ", ".join(str(p) for p in missing), file=sys.stderr)
        return 1

    keys = _resolve_formats(args.to or settings.transcribe_formats) or list(DEFAULT_FORMATS)
    unknown = [k for k in keys if k not in TRANSCRIPT_FORMATS]
    if unknown:
        print(f"Format(s) inconnu(s) : {', '.join(unknown)} (voir --list-formats)", file=sys.stderr)
        return 1

    model_key = args.model or settings.model
    if model_key not in models.REGISTRY:
        print(f"Modèle inconnu : {model_key} (voir --list-models)", file=sys.stderr)
        return 1
    mode = args.mode or settings.mode
    out_dir = Path(args.out) if args.out else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    ui = ConsoleUI()
    ui.line(f"Modèle : {model_key} · mode {mode} · formats {', '.join(keys)}")
    # Hors direct, aucune latence à tenir : on paie un faisceau plus large pour
    # un texte meilleur (sans effet sur whisper.cpp, glouton par construction).
    engine = _make_engine(args, settings, model_key, mode, beam_size=FILE_BEAM_SIZE)
    ui.line(f"Backend : {engine.backend_info()}")
    lang = "fr" if mode == "fr" else "en"

    def on_progress(p: filejob.Progress) -> None:
        ratio = p.ratio
        head = f"[{p.index + 1}/{p.count}] {p.path.name} · {p.stage}"
        if ratio is not None:
            head += f" {ratio:.0%}"
        elif p.done_s:
            head += f" {media.fmt_duration(p.done_s)}"
        ui.partial(head)

    failures = 0
    written: set[Path] = set()
    try:
        for result in filejob.transcribe_many(engine, paths, lang=lang,
                                              hallucination_filter=settings.hallucination_filter,
                                              no_speech_prob_max=settings.no_speech_prob_max,
                                              on_progress=on_progress):  # fmt: skip
            if not result.ok:
                failures += 1
                ui.line(f"✗ {result.path.name} : {result.error}")
                continue
            for key in keys:
                fmt = TRANSCRIPT_FORMATS[key]
                target = unique_path(output_path(result.path, fmt, out_dir), written)
                write_transcript(target, result.segments, key=key,
                                 timestamps=settings.txt_timestamps, title=result.path.name)  # fmt: skip
                written.add(target)
                result.outputs.append(target)
            speed = f" · x{result.speed:.1f} temps réel" if result.speed else ""
            ui.line(
                f"✓ {result.path.name} — {result.word_count} mots, {len(result.segments)} segments{speed}"
            )
            for target in result.outputs:
                ui.line(f"    {target}")
    except KeyboardInterrupt:
        ui.line("Arrêt demandé…")
        return 1
    finally:
        engine.close()
    return 1 if failures else 0


def _resolve_formats(raw: str) -> list[str]:
    """« srt, .md » -> ['srt', 'md'].

    Les extensions sont acceptées aussi bien que les noms de format : personne ne
    doit avoir à retenir que WebVTT s'appelle « vtt » dans nos options. Un jeton
    inconnu est rendu TEL QUEL, pour que l'appelant puisse le signaler au lieu de
    le traduire silencieusement en texte brut.
    """
    from ecoutemoi.core.transcript import TRANSCRIPT_FORMATS

    by_extension = {ext: fmt.key for fmt in TRANSCRIPT_FORMATS.values() for ext in fmt.extensions}
    keys: list[str] = []
    for token in raw.replace(";", ",").split(","):
        name = token.strip().lower().lstrip(".")
        if not name:
            continue
        key = name if name in TRANSCRIPT_FORMATS else by_extension.get(f".{name}", name)
        if key not in keys:
            keys.append(key)
    return keys


def diag_report(args, settings) -> tuple[list[str], int]:
    """Rapport de diagnostic (lignes + code retour) — partagé CLI (--diag) / GUI.

    Couvre tout ce qu'il faut pour un rapport de bug : environnement, libs
    backend GPU embarquées, validité des modèles sur disque, chargement réel du
    moteur avec périphériques Vulkan/Metal détectés et backend actif.
    """
    import platform

    from ecoutemoi import __version__
    from ecoutemoi.config import config_path
    from ecoutemoi.core import engine_fw, gpuprobe
    from ecoutemoi.core.engine import gpu_backend_libs
    from ecoutemoi.logging_setup import log_dir

    lines: list[str] = []
    frozen = "binaire PyInstaller" if getattr(sys, "frozen", False) else "environnement Python"
    lines += [
        f"EcouteMoi {__version__} — diagnostic",
        f"OS         : {platform.platform()} ({platform.machine()})",
        f"Python     : {platform.python_version()} · {frozen}",
        f"Exécutable : {sys.executable}",
        f"Config     : {config_path()}",
        f"Log        : {log_dir() / 'ecoutemoi.log'}",
        f"Modèles    : {models.models_dir()}",
    ]
    libs = gpu_backend_libs()
    libs_txt = ", ".join(libs) if libs else "AUCUNE (moteur CPU pur — wheel PyPI ?)"
    lines.append("")
    lines.append("Moteurs :")
    lines.append(f"  whisper.cpp (GPU {'Metal' if sys.platform == 'darwin' else 'Vulkan'}/CPU)")
    lines.append(f"    libs backend GPU : {libs_txt}")
    if engine_fw.available():
        lines.append(f"  faster-whisper (CPU) — {engine_fw.versions()}")
        lines.append(f"    types de calcul CPU : {', '.join(engine_fw.supported_compute_types())}")
    else:
        lines.append("  faster-whisper (CPU) : ABSENT de cet environnement")
    probe = gpuprobe.gpu_candidates()
    lines.append(f"  Sondage GPU (cache {gpuprobe.cache_path().name}) : {probe.summary}")

    from ecoutemoi.core import media

    media.set_ffmpeg_path(settings.ffmpeg_path)
    lines.append("")
    lines.append("Décodeurs de fichiers :")
    lines += [f"  {row}" for row in media.decoder_report()]

    lines.append("")
    lines.append("Modèles installés :")
    installed_any = False
    for key, spec in models.REGISTRY.items():
        for fmt in models.MODEL_FORMATS:
            if not models.is_installed(spec, fmt=fmt):
                continue
            installed_any = True
            path = models.model_location(spec, fmt)
            reason = models.validate_model(spec, fmt)
            state = "OK" if reason is None else f"INVALIDE — {reason}"
            size = (path / "model.bin").stat().st_size if fmt == models.FMT_CT2 else path.stat().st_size
            lines.append(f"  {key:<22} {fmt:<5} {size / 1e6:8.1f} Mo  {state}")
    if not installed_any:
        lines.append("  (aucun — utilisez --download ou « Gérer les modèles »)")

    model_key = getattr(args, "model", None)
    if model_key is not None and model_key not in models.REGISTRY:
        lines.append("")
        lines.append(f"Modèle inconnu : {model_key} (voir --list-models)")
        return lines, 1
    if model_key is None:
        installed = models.installed_models()
        model_key = settings.model if settings.model in installed else (installed[0] if installed else None)
    if model_key is None:
        lines.append("")
        lines.append("Aucun modèle installé : chargement du moteur sauté.")
        return lines, 0

    backend = getattr(args, "backend", None) or settings.backend
    choice, fmt = plan_engine(settings, backend=backend)
    lines.append("")
    lines.append(f"Chargement du moteur : {model_key} · backend {backend} · moteur {choice} ({fmt})…")
    # Diagnostic DANS ce processus : on veut l'état du moteur ici, pas celui d'un
    # enfant qu'il faudrait interroger à travers un tube.
    try:
        engine = make_engine(settings, model_key, "fr", backend=backend,
                             gpu_device=getattr(args, "gpu_device", None), subprocess=False)  # fmt: skip
    except Exception as exc:
        lines.append(f"ÉCHEC du chargement : {exc}")
        return lines, 1
    try:
        for k, v in engine.diagnostics().items():
            if isinstance(v, list):
                v = " ; ".join(str(x) for x in v) if v else "—"
            lines.append(f"  {k:<22}: {v}")
        tail = list(engine.sink.lines)[-12:]
        if tail:
            lines.append("")
            lines.append("Dernières lignes du moteur :")
            lines += [f"  {ln}" for ln in tail]
    finally:
        engine.close()
    return lines, 0


def run_diag(args) -> int:
    """--diag : imprime le rapport de diagnostic complet."""
    report, code = diag_report(args, load_settings())
    print("\n".join(report))
    return code


def calibration_dir() -> Path:
    """Where the GUI wizard stores the operator's 16 k calibration recordings."""
    import platformdirs

    from ecoutemoi.constants import APP_NAME

    return platformdirs.user_data_path(APP_NAME, appauthor=False) / "calibration"


def run_bench(args) -> int:
    """Guided CLI benchmark: table + persisted JSON."""
    from datetime import datetime

    from ecoutemoi.core import bench

    if args.wav_fr or args.wav_en:  # WAVs explicites : pas de repli automatique
        wav_fr = Path(args.wav_fr) if args.wav_fr else None
        wav_en = Path(args.wav_en) if args.wav_en else None
        wav_fr = wav_fr if wav_fr and wav_fr.is_file() else None
        wav_en = wav_en if wav_en and wav_en.is_file() else None
        is_reference = False
    else:
        wav_fr, wav_en, is_reference = bench.resolve_calibration_wavs(
            calibration_dir() / "calibration_fr.wav",
            calibration_dir() / "calibration_en.wav",
        )
    if wav_fr is None and wav_en is None:
        print(
            "Aucune voix disponible (ni enregistrement, ni voix de référence embarquée). "
            "Fournissez --wav-fr/--wav-en ou réinstallez l'application.",
            file=sys.stderr,
        )
        return 1
    if is_reference:
        print("Voix de référence intégrée (enregistrez votre voix via l'assistant GUI "
              "pour un WER personnalisé).")  # fmt: skip

    if args.models:
        keys = [k.strip() for k in args.models.split(",") if k.strip()]
        unknown = [k for k in keys if k not in models.REGISTRY]
        if unknown:
            print(f"Modèle(s) inconnu(s) : {', '.join(unknown)}", file=sys.stderr)
            return 1
    else:
        keys = models.installed_models()
        if not keys:
            print("Aucun modèle installé (voir --download).", file=sys.stderr)
            return 1
    settings = load_settings()
    backend = args.backend or settings.backend
    gpu_device = args.gpu_device if args.gpu_device is not None else settings.gpu_device
    # Le benchmark mesure LE moteur qui tournera vraiment : c'est donc le format
    # du backend choisi qu'il faut avoir sur disque, pas systématiquement ggml.
    choice, fmt = plan_engine(settings, backend=backend)
    for k in keys:
        models.ensure_model(k, fmt=fmt)

    print(f"Benchmark : {', '.join(sorted(keys))} · moteur {choice} ({fmt})")
    print(f"FR : {wav_fr or '—'} · EN : {wav_en or '—'}")
    results = bench.run_benchmark(keys, wav_fr, wav_en, backend=backend,
                                  gpu_device=gpu_device, progress=print)  # fmt: skip

    fmt_pct = lambda v: f"{v:6.1%}" if v is not None else "     —"  # noqa: E731
    print(f"\n{'modèle':<22} {'backend':<20} {'RTF':>6} {'WER FR':>7} {'WER EN':>7} {'trad.':>7}  verdict")
    for r in results:
        rtf = f"{r.rtf:6.1f}" if r.rtf is not None else "     —"
        print(
            f"{r.model_key:<22} {r.backend:<20} {rtf} {fmt_pct(r.wer_fr)} "
            f"{fmt_pct(r.wer_en)} {fmt_pct(r.translate_wer)}  {bench.verdict(r)}"
        )
    failures = [r for r in results if r.error and "sauté" not in r.error]
    if failures:
        print("\nDétail des échecs (aussi dans le fichier de log) :")
        for r in failures:
            print(f"  {r.model_key} : {r.error}")
    reco = bench.recommend(results)
    if reco:
        print(f"\nRecommandation : modèle {reco[0]} + preset {reco[1]}")
    else:
        print("\nAucun modèle utilisable sur cette machine (RTF < 1,3 partout).")
    path = bench.save_results(results, stamp=datetime.now().isoformat(timespec="seconds"))
    print(f"Résultats enregistrés : {path}")
    return 0


def run_rtf_bench(args) -> int:
    """Measure the CPU RTF of the given models on a WAV (--rtf a,b,c --wav f)."""
    if not args.wav:
        print("--rtf nécessite --wav (audio de référence)", file=sys.stderr)
        return 1
    keys = [k.strip() for k in args.rtf.split(",") if k.strip()]
    unknown = [k for k in keys if k not in models.REGISTRY]
    if unknown:
        print(f"Modèle(s) inconnu(s) : {', '.join(unknown)}", file=sys.stderr)
        return 1
    x, sr = load_wav(Path(args.wav))
    audio = resample_to_16k(x, sr)
    audio_s = len(audio) / TARGET_SR
    settings = load_settings()
    mode = args.mode or "fr"
    print(f"Audio : {args.wav} ({audio_s:.1f} s) · mode {mode}")
    results = []
    for key in keys:
        # Mesure du MOTEUR : en direct, sans le coût d'un tube entre deux processus.
        engine = _make_engine(args, settings, key, mode, subprocess=False)
        engine.transcribe(audio[:TARGET_SR])  # warmup (graph/init)
        t0 = time.perf_counter()
        segments = engine.transcribe(audio)
        dt = time.perf_counter() - t0
        rtf = audio_s / dt if dt > 0 else 0.0
        text = " ".join(s.text for s in segments)
        results.append((key, rtf, dt, engine.backend_info(), engine.vad_active, text))
        engine.close()
    print(f"\n{'modèle':<22} {'RTF':>6} {'décodage':>10} {'backend':<24} VAD")
    for key, rtf, dt, backend, vad, _ in results:
        print(f"{key:<22} {rtf:>6.2f} {dt:>8.1f} s  {backend:<24} {'oui' if vad else 'non'}")
    for key, _, _, _, _, text in results:
        print(f"\n[{key}] {text}")
    return 0


__all__ = ["ConsoleUI", "GilProbe", "Segment", "load_wav", "run_cli", "run_diag", "run_rtf_bench"]
