"""Streaming integration tests:
- auto-degradation (bascule Phrase) fires and is logged when the model is too slow;
- 10 minutes of looped speech: no window drift, no text loops, no memory leak.

Run with: pytest -m integration
"""

from __future__ import annotations

import logging
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import pytest

import ecoutemoi.core.streamer as streamer_mod
from ecoutemoi.constants import PRESETS, TARGET_SR
from ecoutemoi.core.engine import Segment
from ecoutemoi.core.gate import SpeechGate
from ecoutemoi.core.streamer import Streamer
from ecoutemoi.core.transcript import TranscriptStore

pytestmark = pytest.mark.integration

JFK_URL = "https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav"


class SlowFakeEngine:
    """Engine stub: decoding takes `factor` x the window duration (a too-big
    model behaves proportionally — that is what makes a machine diverge)."""

    def __init__(self, factor: float):
        self.factor = factor
        self.last_decode_ms = 0.0

    def transcribe(self, audio: np.ndarray) -> list[Segment]:
        window_s = len(audio) / TARGET_SR
        time.sleep(self.factor * window_s)
        self.last_decode_ms = self.factor * window_s * 1000
        ms = int(window_s * 1000)
        return [Segment(0, ms, "un exemple de phrase produite lentement")]

    def detected_language(self):
        return "fr"

    def set_language(self, lang):
        pass


class FakeVadAlwaysSpeech:
    def is_speech(self, pcm: bytes, sr: int) -> bool:
        return True


def test_auto_degradation_triggers_and_reverts(monkeypatch, caplog):
    """Feed real-time-paced audio to a deliberately slow engine: lag median > 2.0
    sustained -> auto Phrase mode + log; restore_partials() reverts."""
    monkeypatch.setattr(streamer_mod, "LAG_PHRASE_SWITCH_S", 1.2)
    engine = SlowFakeEngine(factor=2.5)  # decode takes 2.5x the audio: diverges
    store = TranscriptStore(None, autosave=False)
    degraded_events: list[bool] = []
    st = Streamer(
        engine,
        store,
        PRESETS["equilibre"],
        window_max_s=1.5,  # keep the fake decodes short so the test stays quick
        realtime=True,
        on_degraded=degraded_events.append,
    )
    gate = SpeechGate(silence_ms=500, denoise_fusion=False, vad=FakeVadAlwaysSpeech())
    st.start()
    block = np.full(160, 0.3, dtype=np.float32)
    stop = threading.Event()

    def feed():
        # 10 ms blocks paced at real time: continuous speech, engine can't keep up
        while not stop.is_set():
            for ev in gate.feed(block):
                st.on_gate_event(ev)
            time.sleep(0.01)

    t = threading.Thread(target=feed, daemon=True)
    with caplog.at_level(logging.WARNING, logger="ecoutemoi.core.streamer"):
        t.start()
        deadline = time.monotonic() + 25.0
        while time.monotonic() < deadline and not degraded_events:
            time.sleep(0.1)
        stop.set()
        t.join(timeout=2)
        st.stop()
    assert degraded_events and degraded_events[0] is True, "auto Phrase switch did not fire"
    assert st.partials is False
    assert any("auto Phrase mode" in r.message for r in caplog.records)  # journalisée
    st.restore_partials()
    assert st.partials is True
    assert degraded_events[-1] is False


@pytest.fixture(scope="module")
def jfk_wav(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("audio") / "jfk.wav"
    urllib.request.urlretrieve(JFK_URL, path)
    return path


def test_10_minutes_looped_speech(jfk_wav):
    """10 min of speech (looped wav) without window drift, text loops
    or memory leak — real tiny-q5_1 engine + full DSP chain (fast feed)."""
    import psutil

    from ecoutemoi.cli import load_wav
    from ecoutemoi.core import models
    from ecoutemoi.core.dsp import DspChain
    from ecoutemoi.core.engine import EngineParams, WhisperEngine

    model_path = models.ensure_model("tiny-q5_1")
    vad_path = models.ensure_vad_model()
    x, sr = load_wav(jfk_wav)
    assert sr == TARGET_SR  # whisper.cpp sample is 16 kHz mono
    silence = np.zeros(int(sr * 0.8), dtype=np.float32)
    unit = np.concatenate([x, silence])
    loops = int(np.ceil(600 * sr / len(unit)))  # >= 10 minutes

    engine = WhisperEngine(EngineParams(model_path=model_path, language="en", vad_model_path=vad_path))
    engine.transcribe(np.zeros(sr, dtype=np.float32))  # warmup before RSS baseline
    store = TranscriptStore(None, autosave=False)
    max_window_s = 0.0
    notices: list[str] = []

    def on_stats(s):
        nonlocal max_window_s
        max_window_s = max(max_window_s, s.window_s)

    st = Streamer(
        engine, store, PRESETS["equilibre"], mode="auto", realtime=False,
        on_stats=on_stats, on_notice=notices.append,
    )  # fmt: skip
    gate = SpeechGate(silence_ms=500, denoise_fusion=True)
    dsp = DspChain(sr, denoise=True, highpass=True)  # full chain incl. RNNoise 16->48->16
    st.start()

    rss_before = psutil.Process().memory_info().rss
    block = sr // 100
    fed_s = 0.0
    for _ in range(loops):
        for i in range(0, len(unit), block):
            for b16, prob in dsp.process(unit[i : i + block]):
                for ev in gate.feed(b16, prob):
                    st.on_gate_event(ev)
        fed_s += len(unit) / sr
    st.stop(timeout=300)
    rss_after = psutil.Process().memory_info().rss

    assert fed_s >= 600, "should have fed at least 10 minutes"
    # 1) no window drift: bounded by the preset window cap (max 9 s + one frame of slack)
    assert max_window_s <= 9.5, f"window drifted to {max_window_s:.1f} s"
    # 2) no text loops: no immediate repeated 5-gram in the final transcript
    words = [w.lower().strip(".,!?") for s in store.segments for w in s.text.split()]
    k = 5
    repeats = sum(1 for i in range(len(words) - 2 * k) if words[i : i + k] == words[i + k : i + 2 * k])
    assert repeats == 0, f"{repeats} repeated 5-grams found"
    # 3) transcription actually happened (~55 loops of the JFK sentence)
    assert len(store.segments) >= 50
    assert store.word_count() >= 500
    # 4) no memory leak: generous bound, catches linear growth
    growth_mb = (rss_after - rss_before) / 1e6
    assert growth_mb < 250, f"RSS grew by {growth_mb:.0f} MB over 10 min"
