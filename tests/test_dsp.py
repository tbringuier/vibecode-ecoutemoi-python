"""Biquad high-pass + FIFO block logic. RNNoise itself is integration scope."""

import numpy as np
import pytest

from ecoutemoi.core.dsp import BLOCK_16K, BiquadHighpass, _Fifo


def test_highpass_kills_dc():
    hp = BiquadHighpass(48000)
    x = np.ones(48000, dtype=np.float32) * 0.5
    y = hp.process(x)
    assert np.abs(y[-4800:]).max() < 1e-3  # DC fully rejected after settling


def test_highpass_passes_1khz():
    hp = BiquadHighpass(48000)
    t = np.arange(48000) / 48000
    x = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
    y = hp.process(x)
    rms_in = np.sqrt(np.mean(x[24000:] ** 2))
    rms_out = np.sqrt(np.mean(y[24000:] ** 2))
    assert rms_out > 0.95 * rms_in  # 1 kHz well above the 80 Hz corner


def test_highpass_attenuates_50hz_hum():
    hp = BiquadHighpass(48000)
    t = np.arange(48000) / 48000
    x = np.sin(2 * np.pi * 50 * t).astype(np.float32)
    y = hp.process(x)
    rms_out = np.sqrt(np.mean(y[24000:] ** 2))
    assert rms_out < 0.4  # 2nd-order Butterworth: ~ -8 dB at 50 Hz


def test_highpass_stateful_across_blocks():
    """Processing block by block must equal processing in one go."""
    hp1 = BiquadHighpass(48000)
    hp2 = BiquadHighpass(48000)
    rng = np.random.default_rng(42)
    x = rng.standard_normal(4800).astype(np.float32) * 0.1
    y_full = hp1.process(x)
    parts = [hp2.process(x[i : i + 480]) for i in range(0, 4800, 480)]
    y_blocks = np.concatenate(parts)
    np.testing.assert_allclose(y_full, y_blocks, atol=1e-6)


def test_fifo_fixed_blocks():
    f = _Fifo()
    f.push(np.arange(100, dtype=np.float32))
    assert f.pull(160) is None
    f.push(np.arange(100, 300, dtype=np.float32))
    b1 = f.pull(160)
    assert b1 is not None
    np.testing.assert_array_equal(b1, np.arange(160, dtype=np.float32))
    b2 = f.pull(140)
    assert b2 is not None
    np.testing.assert_array_equal(b2, np.arange(160, 300, dtype=np.float32))
    assert f.pull(1) is None


def test_block_16k_is_10ms():
    assert BLOCK_16K == 160


def test_dspchain_passthrough_16k_no_denoise_no_highpass():
    from ecoutemoi.core.dsp import DspChain

    chain = DspChain(16000, denoise=False, highpass=False)
    x = np.random.default_rng(0).standard_normal(1600).astype(np.float32) * 0.1
    out = chain.process(x)
    assert len(out) == 10  # 100 ms -> 10 blocs de 10 ms
    got = np.concatenate([b for b, _ in out])
    np.testing.assert_allclose(got, x, atol=1e-7)
    assert all(p is None for _, p in out)


def test_dspchain_resamples_48k_without_denoise():
    from ecoutemoi.core.dsp import DspChain

    chain = DspChain(48000, denoise=False, highpass=True)
    x = np.zeros(9600, dtype=np.float32)  # 200 ms @ 48 kHz
    out = chain.process(x)
    assert out, "the resampler must emit 16 kHz blocks"
    for b, p in out:
        assert b.size == 160
        assert p is None


def test_dspchain_denoise_emits_speech_probs():
    """Chaîne complète avec RNNoise réel : chaque bloc 16 kHz porte une probabilité."""
    from ecoutemoi.core.dsp import DspChain

    chain = DspChain(48000, denoise=True, highpass=True)
    rng = np.random.default_rng(1)
    probs = []
    for _ in range(20):  # 20 x 10 ms @ 48 kHz
        block = (rng.standard_normal(480) * 0.05).astype(np.float32)
        for _b16, prob in chain.process(block):
            probs.append(prob)
    assert probs, "denoised blocks should come out"
    assert any(p is not None for p in probs)
    assert all(p is None or 0.0 <= p <= 1.0 for p in probs)


# --------------------------------------------------------------------- silences
def test_frame_rms_and_floor():
    from ecoutemoi.core.dsp import frame_rms, silence_floor

    x = np.concatenate([np.zeros(320, dtype=np.float32), np.full(320, 0.5, dtype=np.float32)])
    levels = frame_rms(x, 320)
    assert levels.shape == (2,)
    assert levels[0] == 0.0
    assert levels[1] == pytest.approx(0.5, abs=1e-6)
    assert frame_rms(np.zeros(10, dtype=np.float32), 320).size == 0  # queue incomplète ignorée
    # Le seuil suit le niveau du passage, avec un plancher absolu
    assert silence_floor(levels) == pytest.approx(0.5 * 0.06)
    assert silence_floor(np.zeros(0, dtype=np.float32)) > 0.0


def test_trim_trailing_silence_keeps_a_margin():
    from ecoutemoi.constants import TARGET_SR
    from ecoutemoi.core.dsp import trim_trailing_silence

    rng = np.random.default_rng(1)
    voice = (rng.standard_normal(TARGET_SR) * 0.2).astype(np.float32)
    padded = np.concatenate([voice, np.zeros(2 * TARGET_SR, dtype=np.float32)])
    trimmed = trim_trailing_silence(padded, TARGET_SR, keep_ms=200)
    assert TARGET_SR <= trimmed.size <= int(1.3 * TARGET_SR)
    # Un signal sans silence de fin n'est pas touché
    assert trim_trailing_silence(voice, TARGET_SR).size == voice.size
    # Un passage entièrement muet ne rend rien : l'appelant ne le décodera pas
    assert trim_trailing_silence(np.zeros(TARGET_SR, dtype=np.float32), TARGET_SR).size == 0


def test_quietest_cut_finds_the_gap():
    from ecoutemoi.constants import TARGET_SR
    from ecoutemoi.core.dsp import quietest_cut

    rng = np.random.default_rng(2)
    x = (rng.standard_normal(4 * TARGET_SR) * 0.3).astype(np.float32)
    gap_start = int(2.5 * TARGET_SR)
    x[gap_start : gap_start + int(0.2 * TARGET_SR)] = 0.0
    cut = quietest_cut(x, int(2.0 * TARGET_SR), int(3.0 * TARGET_SR), TARGET_SR)
    assert gap_start <= cut <= gap_start + int(0.25 * TARGET_SR)
    # Fenêtre vide : la borne haute est rendue telle quelle
    assert quietest_cut(x, 100, 100) == 100
