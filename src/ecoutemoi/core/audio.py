"""Microphone capture via sounddevice.

The PortAudio callback does ONLY: gain, mono mixdown, RMS update, queue push.
No heavy allocation, no long lock, no inference.
"""

from __future__ import annotations

import logging
import queue
from dataclasses import dataclass

import numpy as np

from ecoutemoi.constants import CAPTURE_BLOCK_MS, PREFERRED_CAPTURE_SR

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    hostapi: str
    default_sr: int
    channels: int


def list_input_devices() -> list[InputDevice]:
    import sounddevice as sd

    apis = sd.query_hostapis()
    out: list[InputDevice] = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) <= 0:
            continue
        api = apis[dev["hostapi"]]["name"] if 0 <= dev["hostapi"] < len(apis) else "?"
        out.append(
            InputDevice(
                index=idx,
                name=dev["name"],
                hostapi=api,
                default_sr=int(dev.get("default_samplerate") or 0),
                channels=int(dev["max_input_channels"]),
            )
        )
    return out


def default_input_index() -> int | None:
    import sounddevice as sd

    try:
        idx = sd.default.device[0]
        return int(idx) if idx is not None and int(idx) >= 0 else None
    except Exception:
        return None


class AudioCapture:
    """Opens the device at its native rate (48 kHz preferred), float32, 10 ms blocks."""

    def __init__(
        self,
        device: int | None = None,
        gain: float = 1.0,
        prefer_sr: int = PREFERRED_CAPTURE_SR,
        block_ms: int = CAPTURE_BLOCK_MS,
    ):
        import sounddevice as sd

        self._sd = sd
        self.device = device if device is not None else default_input_index()
        self.gain = float(np.clip(gain, 0.25, 4.0))
        info = sd.query_devices(self.device, "input")
        self.sr = self._pick_samplerate(sd, info, prefer_sr)
        self.channels = min(2, int(info["max_input_channels"])) or 1
        self.blocksize = int(self.sr * block_ms / 1000)
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=2048)  # ~20 s of 10 ms blocks
        self.dropped_blocks = 0
        self.callback_errors = 0
        self._rms = 0.0
        self._stream = sd.InputStream(
            device=self.device,
            samplerate=self.sr,
            channels=self.channels,
            dtype="float32",
            blocksize=self.blocksize,
            callback=self._callback,
        )
        log.info(
            "AudioCapture: device=%s sr=%d ch=%d block=%d samples",
            self.device, self.sr, self.channels, self.blocksize,
        )  # fmt: skip

    @staticmethod
    def _pick_samplerate(sd, info, prefer_sr: int) -> int:
        try:
            sd.check_input_settings(device=info["index"], samplerate=prefer_sr, dtype="float32")
            return prefer_sr
        except Exception:
            return int(info["default_samplerate"])

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            self.callback_errors += 1
        x = indata[:, 0] if indata.shape[1] == 1 else indata.mean(axis=1, dtype=np.float32)
        if self.gain != 1.0:
            x = x * self.gain
        x = np.clip(x, -1.0, 1.0).astype(np.float32, copy=False)
        self._rms = float(np.sqrt(np.mean(np.square(x)))) if frames else 0.0
        try:
            self._queue.put_nowait(x.copy())
        except queue.Full:
            self.dropped_blocks += 1

    @property
    def rms(self) -> float:
        return self._rms

    def start(self) -> None:
        self._stream.start()

    def read(self, timeout: float = 0.5) -> np.ndarray | None:
        """Blocking read of one 10 ms block (float32 mono, native SR)."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass
