"""SpeechGate state machine with an injected fake VAD (start latency, pre-roll, silence-based end)."""

import numpy as np

from ecoutemoi.core.gate import FRAME_SAMPLES, GateEvent, SpeechGate

BLOCK = 160  # 10 ms @ 16 kHz


class FakeVad:
    """Deterministic VAD: speech iff mean absolute amplitude above threshold."""

    def is_speech(self, pcm: bytes, sr: int) -> bool:
        x = np.frombuffer(pcm, dtype=np.int16)
        return bool(np.abs(x).mean() > 1000)


def speech_block() -> np.ndarray:
    return np.full(BLOCK, 0.3, dtype=np.float32)


def silence_block() -> np.ndarray:
    return np.zeros(BLOCK, dtype=np.float32)


def feed_blocks(gate: SpeechGate, blocks, prob=None) -> list[GateEvent]:
    events = []
    for b in blocks:
        events.extend(gate.feed(b, prob))
    return events


def test_idle_on_silence():
    gate = SpeechGate(silence_ms=500, denoise_fusion=False, vad=FakeVad())
    events = feed_blocks(gate, [silence_block() for _ in range(100)])  # 1 s
    assert events == []
    assert gate.state == "idle"


def test_speech_start_after_4_frames_with_preroll():
    gate = SpeechGate(silence_ms=500, denoise_fusion=False, vad=FakeVad())
    feed_blocks(gate, [silence_block() for _ in range(50)])  # 500 ms of history
    events = []
    n_blocks_to_start = None
    for i in range(20):
        evs = gate.feed(speech_block())
        events.extend(evs)
        if any(e.kind == "speech_start" for e in evs):
            n_blocks_to_start = i + 1
            break
    assert n_blocks_to_start is not None
    # 4 speech frames of 20 ms => 8 blocks of 10 ms (~80-120 ms detection)
    assert 7 <= n_blocks_to_start <= 12
    start = next(e for e in events if e.kind == "speech_start")
    # 300 ms pre-roll replayed from the ring buffer
    assert start.audio is not None
    assert start.audio.size == int(0.3 * 16000)
    assert start.sample_pos == 50 * BLOCK + n_blocks_to_start * BLOCK


def test_frames_emitted_during_speech():
    gate = SpeechGate(silence_ms=500, denoise_fusion=False, vad=FakeVad())
    feed_blocks(gate, [speech_block() for _ in range(10)])  # start fired
    events = feed_blocks(gate, [speech_block() for _ in range(20)])  # 200 ms
    frames = [e for e in events if e.kind == "frame"]
    assert len(frames) == 10  # 20 blocks of 10 ms -> 10 frames of 20 ms
    assert all(f.audio is not None and f.audio.size == FRAME_SAMPLES for f in frames)


def test_speech_end_after_silence_ms():
    gate = SpeechGate(silence_ms=500, denoise_fusion=False, vad=FakeVad())
    feed_blocks(gate, [speech_block() for _ in range(50)])
    assert gate.state == "speech"
    events = feed_blocks(gate, [silence_block() for _ in range(60)])  # 600 ms
    ends = [e for e in events if e.kind == "speech_end"]
    assert len(ends) == 1
    assert gate.state == "idle"
    # end fires after exactly >= 500 ms of continuous silence (25 frames of 20 ms)
    frames_before_end = [e for e in events if e.kind == "frame"]
    assert len(frames_before_end) == 25


def test_short_pause_does_not_end_utterance():
    gate = SpeechGate(silence_ms=500, denoise_fusion=False, vad=FakeVad())
    feed_blocks(gate, [speech_block() for _ in range(50)])
    events = feed_blocks(gate, [silence_block() for _ in range(30)])  # 300 ms pause
    assert not any(e.kind == "speech_end" for e in events)
    events = feed_blocks(gate, [speech_block() for _ in range(30)])
    assert gate.state == "speech"
    assert not any(e.kind == "speech_end" for e in events)


def test_rnnoise_fusion_blocks_start_without_speech_prob():
    """Start requires webrtcvad AND max(RNNoise prob over 200 ms) >= 0.5."""
    gate = SpeechGate(silence_ms=500, denoise_fusion=True, vad=FakeVad())
    events = feed_blocks(gate, [speech_block() for _ in range(30)], prob=0.1)
    assert not any(e.kind == "speech_start" for e in events)  # keyboard click case
    events = feed_blocks(gate, [speech_block() for _ in range(30)], prob=0.9)
    assert any(e.kind == "speech_start" for e in events)


def test_rnnoise_fusion_extends_speech():
    """During speech a frame counts as speech if webrtcvad OR prob >= 0.5."""
    gate = SpeechGate(silence_ms=500, denoise_fusion=True, vad=FakeVad())
    feed_blocks(gate, [speech_block() for _ in range(30)], prob=0.9)
    assert gate.state == "speech"
    # quiet audio (webrtc says silence) but RNNoise still confident -> no end
    events = feed_blocks(gate, [silence_block() for _ in range(60)], prob=0.9)
    assert not any(e.kind == "speech_end" for e in events)
    # both silent -> end fires
    events = feed_blocks(gate, [silence_block() for _ in range(60)], prob=0.0)
    assert any(e.kind == "speech_end" for e in events)


def test_no_events_before_enough_audio_for_frame():
    gate = SpeechGate(silence_ms=500, denoise_fusion=False, vad=FakeVad())
    assert gate.feed(speech_block()) == []  # only 10 ms: half a VAD frame
