"""SpeechGate: webrtcvad + RNNoise-probability fusion + state machine + pre-roll.

All timing is sample-count based (not wall-clock) so behaviour is identical for
live capture and for file playback in tests.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

from ecoutemoi.constants import (
    PRE_ROLL_MS,
    RING_BUFFER_S,
    RNNOISE_SPEECH_PROB,
    RNNOISE_START_LOOKBACK_MS,
    SPEECH_START_MIN_FRAMES,
    SPEECH_START_WINDOW,
    TARGET_SR,
    VAD_FRAME_MS,
    VAD_MODE,
)

log = logging.getLogger(__name__)

FRAME_SAMPLES = TARGET_SR * VAD_FRAME_MS // 1000  # 320 samples per 20 ms VAD frame
_START_LOOKBACK_FRAMES = RNNOISE_START_LOOKBACK_MS // VAD_FRAME_MS


@dataclass(frozen=True)
class GateEvent:
    kind: str  # "speech_start" | "frame" | "speech_end"
    audio: np.ndarray | None  # pre-roll for speech_start, 20 ms frame for frame
    sample_pos: int  # samples fed so far (16 kHz) when the event fired


class SpeechGate:
    """Feeds on 10 ms 16 kHz blocks (+ optional RNNoise prob), emits gate events.

    idle -> (>=4 speech frames in the last 6, plus RNNoise fusion) -> speech
    speech -> (continuous silence >= silence_ms) -> idle (speech_end)
    In idle no inference happens upstream; the pre-roll is replayed from
    the ring buffer so the first syllable is never clipped.
    """

    def __init__(
        self,
        silence_ms: int = 500,
        denoise_fusion: bool = True,
        vad=None,  # injectable for tests
    ):
        if vad is None:
            import webrtcvad

            vad = webrtcvad.Vad(VAD_MODE)
        self._vad = vad
        self.silence_ms = silence_ms
        self.denoise_fusion = denoise_fusion

        ring_blocks = int(RING_BUFFER_S * 1000 / VAD_FRAME_MS)
        self._ring: deque[np.ndarray] = deque(maxlen=ring_blocks)  # 20 ms frames, ~5 s
        self._pending: list[np.ndarray] = []  # 10 ms blocks awaiting frame assembly
        self._pending_probs: list[float] = []
        self._webrtc_flags: deque[bool] = deque(maxlen=SPEECH_START_WINDOW)
        self._recent_probs: deque[float] = deque(maxlen=max(1, _START_LOOKBACK_FRAMES))
        self.state = "idle"
        self._silence_acc_ms = 0
        self._samples_fed = 0

    def feed(self, block: np.ndarray, prob: float | None = None) -> list[GateEvent]:
        """Feed one 10 ms block @ 16 kHz; returns the gate events it triggered."""
        self._samples_fed += block.size
        self._pending.append(block)
        if prob is not None:
            self._pending_probs.append(prob)
        if sum(b.size for b in self._pending) < FRAME_SAMPLES:
            return []
        frame = np.concatenate(self._pending)[:FRAME_SAMPLES]
        frame_prob = max(self._pending_probs) if self._pending_probs else None
        self._pending = []
        self._pending_probs = []
        return self._on_frame(frame, frame_prob)

    def _on_frame(self, frame: np.ndarray, prob: float | None) -> list[GateEvent]:
        self._ring.append(frame)
        if prob is not None:
            self._recent_probs.append(prob)

        pcm = np.clip(frame * 32767.0, -32768, 32767).astype(np.int16).tobytes()
        webrtc = bool(self._vad.is_speech(pcm, TARGET_SR))
        self._webrtc_flags.append(webrtc)

        fused = webrtc or (self.denoise_fusion and prob is not None and prob >= RNNOISE_SPEECH_PROB)

        events: list[GateEvent] = []
        if self.state == "idle":
            if self._start_condition():
                self.state = "speech"
                self._silence_acc_ms = 0
                events.append(GateEvent("speech_start", self._preroll(), self._samples_fed))
        else:
            events.append(GateEvent("frame", frame, self._samples_fed))
            if fused:
                self._silence_acc_ms = 0
            else:
                self._silence_acc_ms += VAD_FRAME_MS
                if self._silence_acc_ms >= self.silence_ms:
                    self.state = "idle"
                    self._silence_acc_ms = 0
                    self._webrtc_flags.clear()
                    events.append(GateEvent("speech_end", None, self._samples_fed))
        return events

    def _start_condition(self) -> bool:
        """>= 4 webrtcvad speech frames among the last 6; the start
        additionally requires max(RNNoise prob over 200 ms) >= 0.5 when fusion is on."""
        if sum(self._webrtc_flags) < SPEECH_START_MIN_FRAMES:
            return False
        if self.denoise_fusion and self._recent_probs:
            return max(self._recent_probs) >= RNNOISE_SPEECH_PROB
        return True

    def _preroll(self) -> np.ndarray:
        """Pre-roll: last 300 ms from the ring buffer (includes the triggering frames)."""
        need = PRE_ROLL_MS * TARGET_SR // 1000
        frames: list[np.ndarray] = []
        total = 0
        for f in reversed(self._ring):
            frames.append(f)
            total += f.size
            if total >= need:
                break
        if not frames:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(list(reversed(frames)))
        return audio[-need:] if audio.size > need else audio

    @property
    def samples_fed(self) -> int:
        return self._samples_fed
