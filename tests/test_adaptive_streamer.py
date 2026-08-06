"""Cadence et fenêtre adaptatives : l'intervalle suit le décodage médian
(plancher preset), la fenêtre max respire entre le plancher et le plafond
preset selon la charge — en temps réel uniquement (les feeds fichier rapides
ont un temps mur distordu)."""

import pytest

from ecoutemoi.constants import ADAPT_WINDOW_MIN_S, PRESETS
from ecoutemoi.core.streamer import Streamer
from ecoutemoi.core.transcript import TranscriptStore


def _streamer(realtime: bool = True) -> Streamer:
    store = TranscriptStore(None, autosave=False)
    return Streamer(engine=None, store=store, preset=PRESETS["equilibre"], realtime=realtime)


def test_interval_follows_median_decode_time():
    st = _streamer()
    assert st.effective_interval_s() == st.min_interval_s  # aucun décodage encore
    for ms in (900.0, 1000.0, 1100.0):
        st._decode_times_ms.append(ms)
    assert st.effective_interval_s() == pytest.approx(1.2, rel=0.01)  # 1.2 x médiane 1.0 s


def test_interval_never_below_preset_floor():
    st = _streamer()
    for _ in range(5):
        st._decode_times_ms.append(50.0)  # GPU très rapide
    assert st.effective_interval_s() == st.min_interval_s


def test_interval_static_without_realtime():
    st = _streamer(realtime=False)
    st._decode_times_ms.append(5000.0)
    assert st.effective_interval_s() == st.min_interval_s


def test_window_shrinks_under_load_then_regrows_to_ceiling():
    st = _streamer()
    ceiling = st.window_max_ms
    st._adapt_window(1.5)  # machine surchargée
    shrunk_once = st.window_max_ms
    assert shrunk_once < ceiling
    for _ in range(40):
        st._adapt_window(1.5)
    assert st.window_max_ms >= int(ADAPT_WINDOW_MIN_S * 1000)  # jamais sous le plancher
    for _ in range(60):
        st._adapt_window(0.1)  # machine à l'aise
    assert st.window_max_ms == ceiling  # remonte exactement au plafond du preset


def test_window_static_without_realtime():
    st = _streamer(realtime=False)
    before = st.window_max_ms
    st._adapt_window(5.0)
    assert st.window_max_ms == before


def test_adaptive_opt_out():
    store = TranscriptStore(None, autosave=False)
    st = Streamer(engine=None, store=store, preset=PRESETS["equilibre"], realtime=True, adaptive=False)
    st._decode_times_ms.append(5000.0)
    assert st.effective_interval_s() == st.min_interval_s
    before = st.window_max_ms
    st._adapt_window(5.0)
    assert st.window_max_ms == before
