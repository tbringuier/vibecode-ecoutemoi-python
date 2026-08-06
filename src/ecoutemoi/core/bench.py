"""Guided benchmark: every pass runs in a SUBPROCESS with a hard
timeout of 3x the audio duration. A killed/too-slow model aborts itself and
skips every larger model on the same backend. Results persist to JSON.

The worker is this same executable (`ecoutemoi --bench-worker task.json out.json`),
which works both under uv/venv and inside the PyInstaller bundle, where
sys.stdout may be None (windowed) — hence file-based I/O, not pipes.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import platformdirs

from ecoutemoi.config import atomic_write_text
from ecoutemoi.constants import (
    APP_NAME,
    BENCH_FLUID_RTF,
    BENCH_MAX_WER,
    BENCH_MIN_RTF,
    BENCH_STARTUP_ALLOWANCE_S,
    BENCH_TIMEOUT_FACTOR,
    CALIBRATION_TEXT_EN,
    CALIBRATION_TEXT_FR,
)
from ecoutemoi.core import models
from ecoutemoi.core.wer import wer

log = logging.getLogger(__name__)


def reference_wavs() -> tuple[Path | None, Path | None]:
    """Voix de référence EMBARQUÉES (TTS des textes de calibration, 16 kHz).

    Elles rendent le benchmark utilisable en un clic, sans passage obligé par
    l'enregistrement micro.
    """
    import ecoutemoi

    root = Path(ecoutemoi.__file__).resolve().parent / "assets" / "calibration"
    fr = root / "calibration_fr.wav"
    en = root / "calibration_en.wav"
    return (fr if fr.is_file() else None, en if en.is_file() else None)


def resolve_calibration_wavs(
    operator_fr: Path | None, operator_en: Path | None
) -> tuple[Path | None, Path | None, bool]:
    """(wav_fr, wav_en, voix_de_référence?) — enregistrements opérateur d'abord,
    sinon repli automatique sur les voix embarquées."""
    if (operator_fr and operator_fr.is_file()) or (operator_en and operator_en.is_file()):
        fr = operator_fr if operator_fr and operator_fr.is_file() else None
        en = operator_en if operator_en and operator_en.is_file() else None
        return fr, en, False
    fr, en = reference_wavs()
    return fr, en, True


def pass_timeout_s(audio_duration_s: float) -> float:
    """Timeout d'une passe : démarrage (extraction/chargement/shaders) + 3x l'audio."""
    return BENCH_STARTUP_ALLOWANCE_S + max(10.0, BENCH_TIMEOUT_FACTOR * audio_duration_s)


@dataclass
class BenchResult:
    model_key: str
    backend: str  # backend reported by the engine ("CPU (8 threads)", "Vulkan : ...")
    rtf: float | None  # min(RTF fr, RTF en); None => aborted
    wer_fr: float | None
    wer_en: float | None
    translate_wer: float | None = None  # indicative translation quality
    aborted: bool = False
    error: str | None = None


def model_wer(result: BenchResult) -> float:
    """A model's representative WER = mean of the available FR/EN WERs.
    (The mean avoids crowning a model on its easy language only.)"""
    wers = [w for w in (result.wer_fr, result.wer_en) if w is not None]
    return sum(wers) / len(wers) if wers else 1.0


def verdict(result: BenchResult) -> str:
    """Benchmark verdicts, FR labels used verbatim in the UI.

    Un ÉCHEC technique (sous-processus mort, modèle corrompu…) n'est pas
    maquillé en « trop lent » : l'erreur réelle est montrée. Et l'imprécision
    ne relègue pas en « Mode Phrase » — Phrase compense la LENTEUR, pas
    l'imprécision.
    """
    if result.aborted and result.error and "sauté" in result.error:
        return "Sauté (modèle plus petit déjà trop lent)"
    if result.aborted:
        return "Abandonné (trop lent)"
    if result.error is not None:
        return f"Échec : {result.error}"
    if result.rtf is None:
        return "Abandonné (trop lent)"
    if result.rtf < BENCH_MIN_RTF:
        return "Inadapté (RTF < 1,3)"
    if result.rtf < BENCH_FLUID_RTF:
        return "Mode Phrase uniquement"
    if model_wer(result) <= BENCH_MAX_WER:
        return "Recommandé (sous-titres fluides)"
    return "Fluide — précision limitée du modèle"


def recommend(results: list[BenchResult]) -> tuple[str, str] | None:
    """(model_key, preset_key): best WER among RTF >= 3, else best WER among
    RTF >= 1.3 with the Phrase preset. None if nothing qualifies.
    Les passes en échec technique ne sont jamais recommandées."""
    ok = [r for r in results if not r.aborted and r.rtf is not None and r.error is None]
    fluid = [r for r in ok if r.rtf >= BENCH_FLUID_RTF]
    if fluid:
        return min(fluid, key=model_wer).model_key, "equilibre"
    usable = [r for r in ok if r.rtf >= BENCH_MIN_RTF]
    if usable:
        return min(usable, key=model_wer).model_key, "phrase"
    return None


# ------------------------------------------------------------------ subprocess
def _worker_cmd() -> list[str]:
    if getattr(sys, "frozen", False):  # PyInstaller bundle
        return [sys.executable]
    return [sys.executable, "-m", "ecoutemoi"]


def wav_duration_s(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def run_pass(
    model_key: str,
    wav_path: Path,
    *,
    language: str,
    translate: bool = False,
    backend: str = "auto",
    gpu_device: int = 0,
    work_dir: Path | None = None,
) -> dict:
    """One (model, wav, task) pass in a subprocess. Hard timeout = startup + 3x audio.
    Returns the worker dict, or {"ok": False, "aborted": True/False, "error": ...}."""
    duration = wav_duration_s(wav_path)
    timeout = pass_timeout_s(duration)
    spec = models.REGISTRY[model_key]
    task = {
        "model_path": str(models.model_path(spec)),
        "vad_model_path": None,
        "wav_path": str(wav_path),
        "language": language,
        "translate": translate,
        "backend": backend,
        "gpu_device": gpu_device,
    }
    try:
        vad = models.model_path(models.VAD_SPEC)
        if vad.is_file():
            task["vad_model_path"] = str(vad)
    except Exception:
        pass

    tmp_dir = Path(tempfile.mkdtemp(prefix="ecoutemoi-bench-", dir=work_dir))
    task_path = tmp_dir / "task.json"
    out_path = tmp_dir / "out.json"
    task_path.write_text(json.dumps(task), encoding="utf-8")
    cmd = [*_worker_cmd(), "--bench-worker", str(task_path), str(out_path)]
    log.info("Bench pass: %s %s lang=%s translate=%s (timeout %.0f s)",
             model_key, wav_path.name, language, translate, timeout)  # fmt: skip
    try:
        proc = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        log.warning("Bench pass killed after %.0f s: %s", timeout, model_key)
        return {"ok": False, "aborted": True, "error": f"tué après {timeout:.0f} s (démarrage + 3× l'audio)"}
    # Le RÉSULTAT écrit prime sur le code de sortie : un crash natif de
    # teardown APRÈS l'écriture ne doit pas invalider une mesure complète
    # et valide.
    if out_path.is_file():
        try:
            out = json.loads(out_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {"ok": False, "aborted": False, "error": f"résultat illisible : {exc}"}
        if proc.returncode != 0:
            log.warning("Bench worker %s : sortie non propre (code %s) mais résultat valide — accepté.",
                        model_key, proc.returncode)  # fmt: skip
        return out
    lines = (proc.stderr or "").strip().splitlines()
    # La vraie erreur du worker part dans le fichier de log : un échec doit
    # laisser un indice exploitable, pas seulement un verdict.
    if lines:
        log.warning("Bench worker %s en échec (code %s) :\n%s",
                    model_key, proc.returncode, "\n".join(lines[-8:]))  # fmt: skip
    return {"ok": False, "aborted": False,
            "error": pick_error_line(lines) or f"code {proc.returncode}"}  # fmt: skip


def pick_error_line(stderr_lines: list[str]) -> str | None:
    """Dernière ligne PORTEUSE d'erreur : le stderr du worker mélange logs
    INFO/DEBUG et traceback — la dernière ligne brute serait souvent un
    « INFO pywhispercpp… » sans valeur."""
    noise = ("INFO", "DEBUG", "WARNING", "[")
    for line in reversed([ln.strip() for ln in stderr_lines if ln.strip()]):
        if not line.startswith(noise):
            return line
    return stderr_lines[-1].strip() if stderr_lines else None


def run_benchmark(
    model_keys: list[str],
    wav_fr: Path | None,
    wav_en: Path | None,
    *,
    ref_fr: str = CALIBRATION_TEXT_FR,
    ref_en: str = CALIBRATION_TEXT_EN,
    backend: str = "auto",
    gpu_device: int = 0,
    progress=None,  # callable(str) for UI/CLI feedback
) -> list[BenchResult]:
    """Smallest model first; an abort skips every larger model on the backend."""
    notify = progress or (lambda _msg: None)
    ordered = sorted(
        (k for k in model_keys if k in models.REGISTRY),
        key=lambda k: models.REGISTRY[k].size_mb,
    )
    results: list[BenchResult] = []
    skip_larger = False
    for key in ordered:
        if skip_larger:
            results.append(BenchResult(key, "-", None, None, None, aborted=True,
                                       error="sauté (modèle plus petit déjà trop lent)"))  # fmt: skip
            notify(f"{key} : sauté")
            continue

        backend_info = "-"  # backend réellement rapporté par le moteur
        rtfs: list[float] = []
        wers: dict[str, float | None] = {"fr": None, "en": None}
        aborted = False
        error = None
        for lang, wav, ref in (("fr", wav_fr, ref_fr), ("en", wav_en, ref_en)):
            if wav is None:
                continue
            notify(f"{key} : passe {lang.upper()}…")
            out = run_pass(key, wav, language=lang, backend=backend, gpu_device=gpu_device)
            if not out.get("ok"):
                aborted = bool(out.get("aborted"))
                error = out.get("error")
                break
            backend_info = out.get("backend", backend_info)
            rtfs.append(float(out["rtf"]))
            wers[lang] = wer(ref, out.get("text", ""), lang)

        rtf = min(rtfs) if rtfs else None
        translate_wer = None
        if not aborted and error is None and rtf is not None:
            spec = models.REGISTRY[key]
            if spec.translate and wav_fr is not None and rtf >= BENCH_MIN_RTF:
                notify(f"{key} : passe traduction (indicatif)…")
                out = run_pass(key, wav_fr, language="fr", translate=True,
                               backend=backend, gpu_device=gpu_device)  # fmt: skip
                if out.get("ok"):
                    translate_wer = wer(ref_en, out.get("text", ""), "en")

        result = BenchResult(key, backend_info, rtf, wers["fr"], wers["en"],
                             translate_wer=translate_wer, aborted=aborted, error=error)  # fmt: skip
        results.append(result)
        notify(f"{key} : {verdict(result)}")
        if aborted or (rtf is not None and rtf < BENCH_MIN_RTF) or (error is not None):
            skip_larger = True
    return results


# ------------------------------------------------------------------ persistence
def results_path() -> Path:
    return platformdirs.user_data_path(APP_NAME, appauthor=False) / "bench_results.json"


def save_results(results: list[BenchResult], *, stamp: str, path: Path | None = None) -> Path:
    p = path or results_path()
    reco = recommend(results)
    data = {
        "stamp": stamp,
        "results": [asdict(r) for r in results],
        "verdicts": {r.model_key: verdict(r) for r in results},
        "recommendation": {"model": reco[0], "preset": reco[1]} if reco else None,
    }
    atomic_write_text(p, json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    return p


def load_results(path: Path | None = None) -> dict | None:
    p = path or results_path()
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return None


# ------------------------------------------------------------------ worker side
def worker_main(task_path: str, out_path: str) -> int:
    """Runs inside the bench subprocess; file-based I/O (safe when stdout is None)."""
    from ecoutemoi.cli import load_wav, resample_to_16k
    from ecoutemoi.core.engine import EngineParams, WhisperEngine

    task = json.loads(Path(task_path).read_text(encoding="utf-8"))
    x, sr = load_wav(Path(task["wav_path"]))
    audio = resample_to_16k(x, sr)
    params = EngineParams(
        model_path=Path(task["model_path"]),
        language=task["language"],
        translate=bool(task.get("translate", False)),
        backend=str(task.get("backend", "auto")),
        vad_model_path=Path(task["vad_model_path"]) if task.get("vad_model_path") else None,
        gpu_device=max(0, int(task.get("gpu_device", 0) or 0)),
    )
    engine = WhisperEngine(params)
    try:
        engine.transcribe(audio[:16000])  # warmup: graph allocation etc.
        t0 = time.perf_counter()
        segments = engine.transcribe(audio)
        decode_s = time.perf_counter() - t0
        audio_s = len(audio) / 16000
        out = {
            "ok": True,
            "rtf": (audio_s / decode_s) if decode_s > 0 else 0.0,
            "decode_s": decode_s,
            "audio_s": audio_s,
            "text": " ".join(s.text for s in segments),
            "backend": engine.backend_info(),
            "vad_active": engine.vad_active,
            "load_variant": engine.load_variant,
        }
    finally:
        engine.close()  # désinstalle le callback de log : sortie de processus propre
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return 0
