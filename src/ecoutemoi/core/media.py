"""Lire un fichier audio ou vidéo, quel qu'en soit le codec, et le rendre sous la
seule forme que whisper accepte : float32 mono 16 kHz dans [-1, 1].

« N'importe quel codec » n'existe pas en Python pur — il faut un décodeur. Trois
sont essayés, et le premier qui sait ouvrir le fichier l'emporte :

1. **libsndfile** (paquet `soundfile`, dépendance déclarée, sa bibliothèque
   voyage avec la wheel) : WAV, FLAC, MP3, Ogg Vorbis, Ogg Opus, AIFF, CAF, W64,
   AU… Aucun binaire externe, aucun sous-processus. C'est le socle garanti :
   ce que cette liste couvre marchera toujours, sur toute machine.
2. **ffmpeg**, s'il est installé : tout le reste. M4A/AAC, WMA, AMR, et surtout
   les pistes audio des conteneurs vidéo (MP4, MKV, MOV, WebM, TS) — un
   enregistrement de visioconférence, typiquement.
3. **PyAV**, s'il est importable : le même ffmpeg, mais en bibliothèque. Depuis
   la 2.0 il est TOUJOURS là — faster-whisper en dépend, et le bundle l'embarque
   donc pour de bon : plus rien à installer, même pour un MKV. Il reste en
   troisième position parce qu'un ffmpeg système est souvent plus récent.

Le décodage est un FLUX, pas un tableau : une heure d'audio en 16 kHz mono fait
230 Mo, une conférence de trois heures 700 Mo. On rend donc des blocs de quelques
secondes, ce qui borne la mémoire à celle de la fenêtre en cours de décodage,
et donne au passage une progression honnête.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import subprocess
import sys
import wave
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ecoutemoi.constants import MEDIA_BLOCK_S, TARGET_SR
from ecoutemoi.core.hostenv import desktop_env

log = logging.getLogger(__name__)

# Extensions proposées dans le sélecteur de fichiers. La liste n'est PAS un
# contrat : le décodage tente sa chance sur n'importe quel fichier, c'est le
# contenu qui décide. Elle sert seulement à ne pas noyer l'opérateur.
AUDIO_EXTENSIONS = (
    "wav", "flac", "mp3", "m4a", "m4b", "aac", "ogg", "oga", "opus", "wma",
    "aiff", "aif", "aifc", "caf", "au", "amr", "ape", "wv", "w64", "mka",
)  # fmt: skip
VIDEO_EXTENSIONS = ("mp4", "mkv", "mov", "webm", "avi", "ts", "m4v", "mpg", "mpeg", "wmv", "flv")

# Formats que libsndfile sait ouvrir sans aucune aide extérieure. Interrogé à
# l'exécution plutôt que codé en dur : la liste dépend de la version livrée.
_SNDFILE_CACHE: frozenset[str] | None = None

_FFMPEG_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d\d):(\d\d(?:\.\d+)?)")
_FFMPEG_AUDIO_RE = re.compile(r"Stream #.*?: Audio: ([\w.]+)")

_ffmpeg_override = ""
_ffmpeg_cache: str | bool | None = False  # False = pas encore cherché


class MediaError(RuntimeError):
    """Le fichier n'a pu être ni ouvert ni décodé — avec le pourquoi."""


@dataclass(frozen=True)
class MediaInfo:
    """Ce qu'on sait du fichier AVANT de le décoder."""

    path: Path
    backend: str  # "libsndfile" | "ffmpeg" | "pyav" | "wave"
    duration_s: float | None  # None = inconnue (progression indéterminée)
    sample_rate: int | None
    channels: int | None
    codec: str = ""

    @property
    def label(self) -> str:
        bits = [self.codec or self.path.suffix.lstrip(".").upper() or "?"]
        if self.sample_rate:
            bits.append(f"{self.sample_rate / 1000:g} kHz")
        if self.channels:
            bits.append("mono" if self.channels == 1 else f"{self.channels} canaux")
        if self.duration_s:
            bits.append(fmt_duration(self.duration_s))
        return " · ".join(bits)


def fmt_duration(seconds: float) -> str:
    """Durée lisible : 4 min 12 s, 1 h 05 min."""
    total = round(max(0.0, seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    if minutes:
        return f"{minutes} min {secs:02d} s"
    return f"{secs} s"


def file_dialog_filter() -> str:
    """Filtre Qt : audio, vidéo, tout — dans cet ordre d'utilité."""
    audio = " ".join(f"*.{e}" for e in AUDIO_EXTENSIONS)
    video = " ".join(f"*.{e}" for e in VIDEO_EXTENSIONS)
    return f"Fichiers audio ({audio});;Fichiers vidéo — piste audio ({video});;Tous les fichiers (*)"


def looks_like_media(path: Path) -> bool:
    """Extension connue. Sert au glisser-déposer, pas à autoriser le décodage."""
    return path.suffix.lower().lstrip(".") in {*AUDIO_EXTENSIONS, *VIDEO_EXTENSIONS}


# ------------------------------------------------------------------- décodeurs
def _sndfile_formats() -> frozenset[str]:
    global _SNDFILE_CACHE
    if _SNDFILE_CACHE is None:
        try:
            import soundfile as sf

            _SNDFILE_CACHE = frozenset(k.lower() for k in sf.available_formats())
        except Exception as exc:  # libsndfile absente (portage exotique)
            log.warning("libsndfile indisponible : %s", exc)
            _SNDFILE_CACHE = frozenset()
    return _SNDFILE_CACHE


def set_ffmpeg_path(path: str) -> None:
    """Chemin ffmpeg imposé par les réglages (vide = détection automatique)."""
    global _ffmpeg_override, _ffmpeg_cache
    if path != _ffmpeg_override:
        _ffmpeg_override = path
        _ffmpeg_cache = False


def _ffmpeg_candidates() -> Iterator[str]:
    """Où chercher ffmpeg, du plus explicite au plus optimiste."""
    if _ffmpeg_override:
        yield _ffmpeg_override
    if env := os.environ.get("ECOUTEMOI_FFMPEG"):
        yield env
    name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
    # À CÔTÉ de notre exécutable : déposer ffmpeg près de EcouteMoi.exe suffit
    # alors à débloquer les M4A et les MP4, sans installation système.
    for base in (Path(sys.executable).parent, Path(getattr(sys, "_MEIPASS", "") or ".")):
        yield str(base / name)
    # `which` avec l'environnement de l'HÔTE : le PATH d'un bundle peut être
    # réécrit, et c'est le ffmpeg du système qu'on veut.
    if found := shutil.which(name, path=desktop_env().get("PATH")):
        yield found
    yield from (
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/opt/homebrew/bin/ffmpeg",  # macOS Apple Silicon
        "/snap/bin/ffmpeg",
        r"C:\ffmpeg\bin\ffmpeg.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe"),
    )


def ffmpeg_executable() -> str | None:
    """Chemin d'un ffmpeg utilisable, None s'il n'y en a pas. Résultat mémorisé."""
    global _ffmpeg_cache
    if _ffmpeg_cache is not False:
        return _ffmpeg_cache  # type: ignore[return-value]
    _ffmpeg_cache = None
    for candidate in _ffmpeg_candidates():
        if candidate and Path(candidate).is_file():
            _ffmpeg_cache = candidate
            log.info("ffmpeg trouvé : %s", candidate)
            break
    else:
        log.info("Aucun ffmpeg trouvé — décodage limité aux formats de libsndfile.")
    return _ffmpeg_cache


def pyav_available() -> bool:
    """PyAV importable : le même ffmpeg, en bibliothèque."""
    try:
        import av  # noqa: F401
    except Exception:
        return False
    return True


# Formats de libsndfile qui intéressent quelqu'un : ceux qu'on rencontre pour de
# la parole. La bibliothèque en gère une vingtaine d'autres (MAT4, HTK, XI…) qui
# n'apprendraient rien à personne dans une fenêtre de réglages.
_NOTABLE_FORMATS = ("wav", "wavex", "rf64", "w64", "flac", "mp3", "ogg", "aiff", "caf", "au")


def decoder_report() -> list[str]:
    """État des décodeurs, pour le diagnostic et le message d'erreur."""
    available = _sndfile_formats()
    notable = [name.upper() for name in _NOTABLE_FORMATS if name in available]
    others = len(available) - len(notable)
    if notable:
        sndfile = ", ".join(notable) + (f" (+{others} formats anciens)" if others > 0 else "")
    else:
        sndfile = "INDISPONIBLE"
    exe = ffmpeg_executable()
    return [
        f"libsndfile : {sndfile}",
        f"ffmpeg     : {exe or 'absent (installez-le pour M4A/AAC/WMA et les vidéos)'}",
        f"PyAV       : {'présent' if pyav_available() else 'absent'}",
    ]


def _missing_decoder_error(path: Path, reasons: list[str]) -> MediaError:
    install = {
        "win32": "winget install --id Gyan.FFmpeg  (ou déposez ffmpeg.exe à côté d'EcouteMoi.exe)",
        "darwin": "brew install ffmpeg",
    }.get(sys.platform, "installez le paquet ffmpeg de votre distribution")
    detail = "\n".join(f"  · {r}" for r in reasons)
    return MediaError(
        f"« {path.name} » n'a pu être décodé par aucun des décodeurs disponibles.\n"
        f"{detail}\n\n"
        f"Les formats M4A/AAC, WMA et les pistes audio de fichiers vidéo demandent "
        f"ffmpeg : {install}.\n"
        f"Un chemin ffmpeg explicite peut aussi être indiqué dans "
        f"Réglages avancés → Fichiers."
    )


# --------------------------------------------------------------------- ouverture
def probe(path: Path) -> MediaInfo:
    """Choisit le décodeur et rapporte ce qu'il sait du fichier.

    C'est ici que se décide le backend : `stream_16k_mono` s'y conforme, pour que
    l'interface puisse annoncer le décodeur retenu avant de lancer le travail.
    """
    path = Path(path)
    if not path.is_file():
        raise MediaError(f"Fichier introuvable : {path}")
    reasons: list[str] = []

    suffix = path.suffix.lower().lstrip(".")
    if suffix in _sndfile_formats():
        try:
            import soundfile as sf

            info = sf.info(str(path))
            return MediaInfo(
                path=path,
                backend="libsndfile",
                duration_s=float(info.duration) or None,
                sample_rate=int(info.samplerate),
                channels=int(info.channels),
                codec=f"{info.format} {info.subtype}".strip(),
            )
        except Exception as exc:
            reasons.append(f"libsndfile : {exc}")

    if ffmpeg_executable() is not None:
        try:
            return _probe_ffmpeg(path)
        except MediaError as exc:
            reasons.append(f"ffmpeg : {exc}")

    if pyav_available():
        try:
            return _probe_pyav(path)
        except Exception as exc:
            reasons.append(f"PyAV : {exc}")

    if suffix not in _sndfile_formats():
        try:  # dernier recours : le module wave de la bibliothèque standard
            with wave.open(str(path), "rb") as w:
                if w.getsampwidth() in (1, 2, 4):
                    return MediaInfo(
                        path=path,
                        backend="wave",
                        duration_s=w.getnframes() / w.getframerate(),
                        sample_rate=w.getframerate(),
                        channels=w.getnchannels(),
                        codec=f"PCM {w.getsampwidth() * 8} bits",
                    )
        except Exception as exc:
            reasons.append(f"wave : {exc}")

    raise _missing_decoder_error(path, reasons or ["aucun décodeur ne gère cette extension"])


def _run_ffmpeg_probe(path: Path) -> str:
    """Sortie de `ffmpeg -i` (les métadonnées partent sur stderr, par conception)."""
    exe = ffmpeg_executable()
    assert exe is not None
    try:
        proc = subprocess.run(
            [exe, "-nostdin", "-hide_banner", "-i", str(path)],
            capture_output=True,
            text=True,
            errors="replace",
            env=desktop_env(),
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MediaError(f"ffmpeg injoignable : {exc}") from exc
    return proc.stderr or ""


def _probe_ffmpeg(path: Path) -> MediaInfo:
    text = _run_ffmpeg_probe(path)
    audio = _FFMPEG_AUDIO_RE.search(text)
    if audio is None:
        raise MediaError("aucune piste audio dans ce fichier")
    duration = None
    if m := _FFMPEG_DURATION_RE.search(text):
        duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    sr = None
    if m := re.search(r"(\d+) Hz", text):
        sr = int(m.group(1))
    channels = None
    for token, count in (("mono", 1), ("stereo", 2), ("5.1", 6), ("7.1", 8)):
        if token in text:
            channels = count
            break
    return MediaInfo(
        path=path,
        backend="ffmpeg",
        duration_s=duration,
        sample_rate=sr,
        channels=channels,
        codec=audio.group(1),
    )


def _probe_pyav(path: Path) -> MediaInfo:
    import av

    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise MediaError("aucune piste audio dans ce fichier")
        stream = container.streams.audio[0]
        duration = float(container.duration / av.time_base) if container.duration else None
        return MediaInfo(
            path=path,
            backend="pyav",
            duration_s=duration,
            sample_rate=int(stream.rate or 0) or None,
            channels=int(getattr(stream, "channels", 0) or 0) or None,
            codec=stream.codec_context.name,
        )


# ---------------------------------------------------------------------- décodage
def stream_16k_mono(
    path: Path,
    *,
    info: MediaInfo | None = None,
    should_stop=None,
) -> Iterator[np.ndarray]:
    """Rend le fichier en blocs float32 mono 16 kHz, prêts pour whisper.

    `should_stop()` est consulté entre deux blocs : une annulation est immédiate
    à l'échelle humaine, et le sous-processus ffmpeg est tué proprement.
    """
    info = info or probe(path)
    stop = should_stop or (lambda: False)
    reader = {
        "libsndfile": _stream_soundfile,
        "ffmpeg": _stream_ffmpeg,
        "pyav": _stream_pyav,
        "wave": _stream_wave,
    }[info.backend]
    log.info("Décodage de %s via %s (%s)", info.path.name, info.backend, info.label)
    yield from reader(info, stop)


def decode_16k_mono(path: Path, *, info: MediaInfo | None = None, should_stop=None) -> np.ndarray:
    """Tout le fichier d'un coup. Pour les fichiers courts, les tests, la CLI."""
    blocks = list(stream_16k_mono(path, info=info, should_stop=should_stop))
    if not blocks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(blocks)


def _resampler(in_sr: int):
    """Rééchantillonneur en flux vers 16 kHz, ou None si déjà à la bonne cadence."""
    if in_sr == TARGET_SR:
        return None
    from ecoutemoi.core.dsp import StreamResampler

    return StreamResampler(in_sr, TARGET_SR)


def _emit(resampler, block: np.ndarray, *, last: bool = False) -> np.ndarray:
    if resampler is None:
        return block
    return resampler.process(block, last=last)


def _mono(block: np.ndarray) -> np.ndarray:
    """Mixage mono : la moyenne des canaux, en float32."""
    if block.ndim == 1:
        return block.astype(np.float32, copy=False)
    if block.shape[1] == 1:
        return block[:, 0].astype(np.float32, copy=False)
    return block.mean(axis=1, dtype=np.float32)


def _stream_soundfile(info: MediaInfo, stop) -> Iterator[np.ndarray]:
    import soundfile as sf

    with sf.SoundFile(str(info.path)) as snd:
        resampler = _resampler(snd.samplerate)
        frames = max(1024, int(snd.samplerate * MEDIA_BLOCK_S))
        while not stop():
            block = snd.read(frames, dtype="float32", always_2d=True)
            if not len(block):
                break
            out = _emit(resampler, _mono(block))
            if out.size:
                yield out
        tail = _emit(resampler, np.zeros(0, dtype=np.float32), last=True)
        if tail.size:
            yield tail


def _stream_ffmpeg(info: MediaInfo, stop) -> Iterator[np.ndarray]:
    """ffmpeg rééchantillonne et mixe lui-même : on lit du float32 mono 16 kHz.

    stderr part dans un fichier temporaire et non dans un tube : un fichier
    abîmé peut produire des milliers de lignes d'erreur, de quoi remplir un tube
    de 64 Ko et bloquer ffmpeg pendant qu'on attend son stdout — l'interblocage
    classique.
    """
    import tempfile

    exe = ffmpeg_executable()
    if exe is None:
        raise MediaError("ffmpeg a disparu entre la détection et le décodage")
    argv = [
        exe, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(info.path),
        "-vn", "-sn", "-dn",  # ni vidéo, ni sous-titres, ni données
        "-map", "0:a:0",  # première piste audio, même dans un conteneur vidéo
        "-ac", "1", "-ar", str(TARGET_SR), "-f", "f32le", "-",
    ]  # fmt: skip
    errors = tempfile.TemporaryFile()  # noqa: SIM115 — fermé dans le finally
    try:
        proc = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=errors, env=desktop_env()
        )
    except OSError as exc:
        errors.close()
        raise MediaError(f"ffmpeg n'a pas pu être lancé : {exc}") from exc

    chunk = 4 * int(TARGET_SR * MEDIA_BLOCK_S)
    remainder = b""
    try:
        assert proc.stdout is not None
        while True:
            if stop():
                proc.kill()
                return
            raw = proc.stdout.read(chunk)
            if not raw:
                break
            raw = remainder + raw
            usable = len(raw) - len(raw) % 4  # jamais un float32 coupé en deux
            remainder = raw[usable:]
            if usable:
                yield np.frombuffer(raw[:usable], dtype=np.float32)
        code = proc.wait()
        if code != 0:
            errors.seek(0)
            message = errors.read().decode("utf-8", errors="replace").strip()
            last = message.splitlines()[-1] if message else "—"
            raise MediaError(f"ffmpeg a échoué (code {code}) : {last}")
    finally:
        with contextlib.suppress(Exception):
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)
        with contextlib.suppress(Exception):
            if proc.stdout is not None:
                proc.stdout.close()
        errors.close()


def _stream_pyav(info: MediaInfo, stop) -> Iterator[np.ndarray]:
    import av

    with av.open(str(info.path)) as container:
        stream = container.streams.audio[0]
        stream.thread_type = "AUTO"
        resampler = av.AudioResampler(format="flt", layout="mono", rate=TARGET_SR)
        pending: list[np.ndarray] = []
        pending_size = 0
        target = int(TARGET_SR * MEDIA_BLOCK_S)
        for frame in container.decode(stream):
            if stop():
                return
            for out in resampler.resample(frame):
                samples = out.to_ndarray().reshape(-1).astype(np.float32, copy=False)
                pending.append(samples)
                pending_size += samples.size
            if pending_size >= target:
                yield np.concatenate(pending)
                pending, pending_size = [], 0
        for out in resampler.resample(None):  # purge du rééchantillonneur
            pending.append(out.to_ndarray().reshape(-1).astype(np.float32, copy=False))
        if pending:
            merged = np.concatenate(pending)
            if merged.size:
                yield merged


def _stream_wave(info: MediaInfo, stop) -> Iterator[np.ndarray]:
    """PCM par le module standard : le filet de sécurité, sans aucune dépendance."""
    scale = {1: None, 2: 32768.0, 4: 2147483648.0}
    with wave.open(str(info.path), "rb") as w:
        width = w.getsampwidth()
        channels = w.getnchannels()
        sr = w.getframerate()
        if width not in scale:
            raise MediaError(f"WAV {width * 8} bits non géré par le lecteur de secours")
        resampler = _resampler(sr)
        frames = max(1024, int(sr * MEDIA_BLOCK_S))
        while not stop():
            raw = w.readframes(frames)
            if not raw:
                break
            if width == 1:  # PCM 8 bits : non signé, centré sur 128
                x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            else:
                dtype = np.int16 if width == 2 else np.int32
                x = np.frombuffer(raw, dtype=dtype).astype(np.float32) / scale[width]
            if channels > 1:
                x = x.reshape(-1, channels).mean(axis=1, dtype=np.float32)
            out = _emit(resampler, x)
            if out.size:
                yield out
        tail = _emit(resampler, np.zeros(0, dtype=np.float32), last=True)
        if tail.size:
            yield tail


__all__ = [
    "AUDIO_EXTENSIONS",
    "VIDEO_EXTENSIONS",
    "MediaError",
    "MediaInfo",
    "decode_16k_mono",
    "decoder_report",
    "ffmpeg_executable",
    "file_dialog_filter",
    "fmt_duration",
    "looks_like_media",
    "probe",
    "pyav_available",
    "set_ffmpeg_path",
    "stream_16k_mono",
]
