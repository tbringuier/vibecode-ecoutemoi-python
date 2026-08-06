"""DSP chain: 80 Hz high-pass -> RNNoise (48 kHz/480) -> soxr -> 16 kHz mono.

Output contract: float32 in [-1, 1], 16 kHz mono, 10 ms blocks (160 samples),
each paired with the RNNoise speech probability of the matching frame (or None).
"""

from __future__ import annotations

import contextlib
import logging
import math
from collections import deque

import numpy as np

from ecoutemoi.constants import (
    HIGHPASS_CUTOFF_HZ,
    RNNOISE_FRAME,
    RNNOISE_SR,
    SILENCE_ABS_RMS,
    SILENCE_FRAME_MS,
    SILENCE_REL_RMS,
    TARGET_SR,
)

log = logging.getLogger(__name__)

BLOCK_16K = TARGET_SR // 100  # 160 samples = 10 ms


# --------------------------------------------------------------------- silences
# Deux besoins distincts, un même outil : ne pas donner à whisper une fenêtre qui
# finit par une seconde de silence (source classique de bégaiement du décodeur),
# et couper un long fichier là où personne ne parle.


def frame_rms(x: np.ndarray, frame: int) -> np.ndarray:
    """RMS par tranche de `frame` échantillons (la queue incomplète est ignorée)."""
    n = x.size // frame
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    blocks = x[: n * frame].reshape(n, frame).astype(np.float32, copy=False)
    return np.sqrt(np.mean(np.square(blocks), axis=1, dtype=np.float32))


def silence_floor(levels: np.ndarray) -> float:
    """Seuil « ici personne ne parle », relatif au niveau du passage lui-même.

    Un seuil absolu ne peut pas convenir : le même plancher servirait pour un
    micro-cravate saturé et pour un enregistrement à −30 dB. On prend donc une
    fraction du niveau le plus fort observé, avec un plancher absolu pour ne pas
    déclarer « parole » du bruit de fond dans un passage entièrement muet.
    """
    if levels.size == 0:
        return SILENCE_ABS_RMS
    return max(SILENCE_ABS_RMS, float(levels.max()) * SILENCE_REL_RMS)


def trim_trailing_silence(x: np.ndarray, sr: int = TARGET_SR, keep_ms: int = 200) -> np.ndarray:
    """Retire le silence de FIN, en gardant `keep_ms` de marge.

    Whisper décode une fenêtre entière : s'y trouve une longue traîne muette et
    il la remplit — en répétant la dernière phrase, ou en inventant un « merci
    d'avoir regardé ». On la coupe donc avant le décodage, tout en gardant de
    quoi laisser respirer la dernière syllabe.
    """
    frame = max(1, sr * SILENCE_FRAME_MS // 1000)
    levels = frame_rms(x, frame)
    if levels.size == 0:
        return x
    floor = silence_floor(levels)
    voiced = np.flatnonzero(levels > floor)
    if voiced.size == 0:
        return x[:0]  # rien de sonore : l'appelant décidera de ne pas décoder
    end = (int(voiced[-1]) + 1) * frame + sr * keep_ms // 1000
    return x[: min(x.size, end)]


def quietest_cut(x: np.ndarray, lo: int, hi: int, sr: int = TARGET_SR) -> int:
    """Position la plus silencieuse dans [lo, hi] — où couper un long fichier.

    Couper au milieu d'un mot coûte ce mot dans les deux morceaux (whisper devine
    différemment de chaque côté). Chercher le creux le plus profond de la zone
    autorisée ne coûte, lui, qu'un balayage de RMS.
    """
    lo = max(0, min(lo, x.size))
    hi = max(lo, min(hi, x.size))
    if hi <= lo:
        return hi
    frame = max(1, sr * SILENCE_FRAME_MS // 1000)
    levels = frame_rms(x[lo:hi], frame)
    if levels.size == 0:
        return hi
    return lo + (int(np.argmin(levels)) + 1) * frame


class BiquadHighpass:
    """2nd-order Butterworth high-pass (RBJ cookbook), numpy only, DF2T state."""

    def __init__(self, sr: int, fc: float = HIGHPASS_CUTOFF_HZ):
        w0 = 2.0 * math.pi * fc / sr
        cosw, sinw = math.cos(w0), math.sin(w0)
        q = 1.0 / math.sqrt(2.0)  # Butterworth
        alpha = sinw / (2.0 * q)
        a0 = 1.0 + alpha
        self.b0 = (1.0 + cosw) / 2.0 / a0
        self.b1 = -(1.0 + cosw) / a0
        self.b2 = (1.0 + cosw) / 2.0 / a0
        self.a1 = (-2.0 * cosw) / a0
        self.a2 = (1.0 - alpha) / a0
        self._z1 = 0.0
        self._z2 = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x, dtype=np.float32)
        z1, z2 = self._z1, self._z2
        b0, b1, b2, a1, a2 = self.b0, self.b1, self.b2, self.a1, self.a2
        for i, xi in enumerate(x.astype(np.float64, copy=False)):
            yi = b0 * xi + z1
            z1 = b1 * xi - a1 * yi + z2
            z2 = b2 * xi - a2 * yi
            y[i] = yi
        self._z1, self._z2 = z1, z2
        return y


class _RnnoiseLib:
    """Minimal ctypes binding to the rnnoise library SHIPPED BY pyrnnoise.

    We deliberately bypass the Python layers of pyrnnoise: the high-level
    wrapper breaks with recent PyAV (audiolab), and the low-level .py source
    is unreachable inside a PyInstaller bundle (PYZ archive). Only the native
    library is needed; same call convention as upstream (float32 buffers at
    int16 scale, in-place processing, returns the speech probability).
    """

    def __init__(self):
        import ctypes

        path = self._find_lib()
        lib = ctypes.CDLL(str(path))
        lib.rnnoise_create.argtypes = [ctypes.c_void_p]
        lib.rnnoise_create.restype = ctypes.c_void_p
        lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
        lib.rnnoise_process_frame.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        lib.rnnoise_process_frame.restype = ctypes.c_float
        lib.rnnoise_get_frame_size.restype = ctypes.c_int
        self.frame_size = int(lib.rnnoise_get_frame_size())
        self._ctypes = ctypes
        self._lib = lib
        log.debug("rnnoise library loaded: %s (frame=%d)", path, self.frame_size)

    @staticmethod
    def _find_lib():
        import importlib.util
        import sys
        from pathlib import Path

        name = {"win32": "rnnoise.dll", "darwin": "librnnoise.dylib"}.get(sys.platform, "librnnoise.so")
        candidates: list[Path] = []
        if getattr(sys, "frozen", False):  # PyInstaller: data files live on disk
            candidates.append(Path(getattr(sys, "_MEIPASS", ".")) / "pyrnnoise" / name)
        spec = importlib.util.find_spec("pyrnnoise")
        if spec is not None:
            if spec.origin:
                candidates.append(Path(spec.origin).parent / name)
            for loc in spec.submodule_search_locations or []:
                candidates.append(Path(loc) / name)
        for c in candidates:
            if c.is_file():
                return c
        raise FileNotFoundError(f"{name} introuvable (candidats : {candidates})")

    def create(self):
        return self._lib.rnnoise_create(None)

    def destroy(self, state) -> None:
        self._lib.rnnoise_destroy(state)

    def process(self, state, frame_f32: np.ndarray) -> tuple[float, np.ndarray]:
        ct = self._ctypes
        pcm = np.clip(frame_f32 * 32767.0, -32768.0, 32767.0).astype(np.float32)
        ptr = pcm.ctypes.data_as(ct.POINTER(ct.c_float))
        prob = self._lib.rnnoise_process_frame(state, ptr, ptr)  # in-place
        return float(prob), pcm / 32768.0


class Denoiser:
    """RNNoise: 480-sample frames @ 48 kHz -> (speech_prob, denoised frame)."""

    def __init__(self):
        self._rn = _RnnoiseLib()
        if self._rn.frame_size != RNNOISE_FRAME:
            raise RuntimeError(f"RNNoise frame size inattendu: {self._rn.frame_size}")
        self._state = self._rn.create()

    def process_frame(self, frame_f32: np.ndarray) -> tuple[float, np.ndarray]:
        """Denoise one 480-sample float32 frame in [-1, 1]; returns (prob, float32 frame)."""
        return self._rn.process(self._state, frame_f32)

    def close(self) -> None:
        if getattr(self, "_state", None):
            self._rn.destroy(self._state)
            self._state = None

    def __del__(self):  # pragma: no cover
        with contextlib.suppress(Exception):
            self.close()


class StreamResampler:
    """Streaming soxr resampler, HQ quality, float32 mono."""

    def __init__(self, in_sr: int, out_sr: int):
        import soxr

        self.in_sr, self.out_sr = in_sr, out_sr
        self._rs = soxr.ResampleStream(in_sr, out_sr, 1, dtype="float32", quality="HQ")

    def process(self, x: np.ndarray, last: bool = False) -> np.ndarray:
        return self._rs.resample_chunk(x, last=last)


class _Fifo:
    """Simple float32 sample FIFO emitting fixed-size blocks."""

    def __init__(self):
        self._chunks: deque[np.ndarray] = deque()
        self._size = 0

    def push(self, x: np.ndarray) -> None:
        if x.size:
            self._chunks.append(x)
            self._size += x.size

    def pull(self, n: int) -> np.ndarray | None:
        if self._size < n:
            return None
        out = np.empty(n, dtype=np.float32)
        filled = 0
        while filled < n:
            head = self._chunks[0]
            take = min(n - filled, head.size)
            out[filled : filled + take] = head[:take]
            if take == head.size:
                self._chunks.popleft()
            else:
                self._chunks[0] = head[take:]
            filled += take
        self._size -= n
        return out


class DspChain:
    """Native-SR 10 ms blocks in -> list of (16 kHz 10 ms block, speech_prob|None) out."""

    def __init__(self, in_sr: int, denoise: bool = True, highpass: bool = True):
        self.in_sr = in_sr
        self.denoise = denoise
        self._hp = BiquadHighpass(in_sr) if highpass else None
        self._probs: deque[float] = deque()
        if denoise:
            self._den = Denoiser()
            self._to48 = StreamResampler(in_sr, RNNOISE_SR) if in_sr != RNNOISE_SR else None
            self._to16 = StreamResampler(RNNOISE_SR, TARGET_SR)
            self._fifo48 = _Fifo()
        else:
            self._den = None
            self._to48 = None
            self._to16 = StreamResampler(in_sr, TARGET_SR) if in_sr != TARGET_SR else None
            self._fifo48 = None
        self._fifo16 = _Fifo()

    def process(self, block: np.ndarray, last: bool = False) -> list[tuple[np.ndarray, float | None]]:
        x = block.astype(np.float32, copy=False)
        if self._hp is not None:
            x = self._hp.process(x)
        if self.denoise:
            assert self._fifo48 is not None and self._den is not None
            x48 = self._to48.process(x, last=last) if self._to48 is not None else x
            self._fifo48.push(x48)
            while (frame := self._fifo48.pull(RNNOISE_FRAME)) is not None:
                prob, den = self._den.process_frame(frame)
                self._probs.append(prob)
                self._fifo16.push(self._to16.process(den, last=False))
            if last:
                self._fifo16.push(self._to16.process(np.empty(0, dtype=np.float32), last=True))
        else:
            y = self._to16.process(x, last=last) if self._to16 is not None else x
            self._fifo16.push(y)

        out: list[tuple[np.ndarray, float | None]] = []
        while (b16 := self._fifo16.pull(BLOCK_16K)) is not None:
            prob = self._probs.popleft() if self._probs else None
            out.append((b16, prob))
        return out
