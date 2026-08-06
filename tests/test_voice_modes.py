"""Modes de gestion de la voix : FR→FR, FR→EN, AUTO→EN (détection continue),
résolution moteur et suivi de la langue source, avec un moteur factice."""

from __future__ import annotations

import time

import numpy as np
import pytest

from ecoutemoi.cli import resolve_mode
from ecoutemoi.constants import PRESETS, TARGET_SR
from ecoutemoi.core.engine import Segment
from ecoutemoi.core.gate import GateEvent
from ecoutemoi.core.models import REGISTRY
from ecoutemoi.core.streamer import Streamer
from ecoutemoi.core.transcript import TranscriptStore


class FakeEngine:
    """Moteur factice : décode instantanément, langue détectée pilotable."""

    def __init__(self, detected: str | None = "en"):
        self.detected = detected
        self.last_decode_ms = 1.0

    def transcribe(self, audio: np.ndarray) -> list[Segment]:
        ms = int(len(audio) * 1000 / TARGET_SR)
        return [Segment(0, ms, "bonjour tout le monde ici présent")]

    def detected_language(self) -> str | None:
        return self.detected


def run_utterance(streamer: Streamer, seconds: float = 2.5) -> None:
    """Injecte un énoncé complet (start + frames + end) puis arrête le streamer."""
    n = int(seconds * TARGET_SR)
    audio = np.full(n, 0.2, dtype=np.float32)
    streamer.start()
    streamer.on_gate_event(GateEvent("speech_start", audio[: TARGET_SR // 2], TARGET_SR // 2))
    streamer.on_gate_event(GateEvent("frame", audio[TARGET_SR // 2 :], n))
    streamer.on_gate_event(GateEvent("speech_end", None, n))
    streamer.stop(timeout=10.0)


def make_streamer(mode: str, engine: FakeEngine, notices: list[str] | None = None) -> Streamer:
    store = TranscriptStore(None, autosave=False)
    return Streamer(
        engine,
        store,
        PRESETS["equilibre"],
        mode=mode,
        realtime=False,
        on_notice=(notices.append if notices is not None else None),
    )


# ----------------------------------------------------------- resolve_mode
def test_resolve_mode_fr():
    spec = REGISTRY["small-q5_1"]
    assert resolve_mode("fr", spec) == ("fr", False)


def test_resolve_mode_translate():
    spec = REGISTRY["small-q5_1"]
    assert resolve_mode("translate", spec) == ("fr", True)


def test_resolve_mode_translate_refused_on_turbo():
    spec = REGISTRY["large-v3-turbo-q5_0"]
    with pytest.raises(ValueError, match="traduire"):
        resolve_mode("translate", spec)


def test_resolve_mode_auto_translates_to_english():
    spec = REGISTRY["tiny-q5_1"]
    assert resolve_mode("auto", spec) == ("auto", True)  # AUTO -> EN : traduction active


def test_resolve_mode_auto_refused_on_turbo():
    spec = REGISTRY["large-v3-turbo-q5_0"]
    with pytest.raises(ValueError, match="traduire"):
        resolve_mode("auto", spec)


def test_resolve_mode_unknown():
    with pytest.raises(ValueError, match="Mode inconnu"):
        resolve_mode("de", REGISTRY["tiny-q5_1"])


# ----------------------------------------------------- suivi de langue (auto)
def test_mode_fr_outputs_french():
    engine = FakeEngine()
    st = make_streamer("fr", engine)
    run_utterance(st)
    assert st.lang == "fr"
    assert all(seg.lang == "fr" for seg in st.store.segments)


def test_mode_translate_outputs_english():
    engine = FakeEngine()
    st = make_streamer("translate", engine)
    run_utterance(st)
    assert st.lang == "en"
    assert all(seg.lang == "en" for seg in st.store.segments)


def test_mode_auto_always_outputs_english():
    engine = FakeEngine(detected="fr")
    notices: list[str] = []
    st = make_streamer("auto", engine, notices)
    assert st.lang == "en"  # la sortie est TOUJOURS anglaise en auto
    run_utterance(st)
    assert st.src_lang == "fr"  # langue source suivie pour l'affichage
    assert any("Langue détectée : fr" in n for n in notices)
    assert all(seg.lang == "en" for seg in st.store.segments)


def test_mode_auto_tracks_language_changes_live():
    engine = FakeEngine(detected="fr")
    notices: list[str] = []
    st = make_streamer("auto", engine, notices)
    st.start()
    n = TARGET_SR  # 1 s par énoncé
    audio = np.full(n, 0.2, dtype=np.float32)
    st.on_gate_event(GateEvent("speech_start", audio, n))
    st.on_gate_event(GateEvent("speech_end", None, n))
    time.sleep(0.3)
    engine.detected = "de"  # l'orateur change de langue en cours de session
    st.on_gate_event(GateEvent("speech_start", audio, 2 * n))
    st.on_gate_event(GateEvent("speech_end", None, 2 * n))
    st.stop(timeout=10.0)
    assert st.src_lang == "de"
    assert any("Langue détectée : fr" in n_ for n_ in notices)
    assert any("Langue détectée : de" in n_ for n_ in notices)


def test_language_tracking_is_noop_outside_auto():
    engine = FakeEngine(detected="de")
    notices: list[str] = []
    st = make_streamer("fr", engine, notices)
    run_utterance(st)
    assert st.src_lang is None
    assert not any("Langue détectée" in n for n in notices)


# ----------------------------------------------------------- phrase preset
def test_preset_phrase_decodes_only_at_end():
    engine = FakeEngine()
    store = TranscriptStore(None, autosave=False)
    partials: list[str] = []
    st = Streamer(
        engine,
        store,
        PRESETS["phrase"],
        mode="fr",
        realtime=False,
        on_partial=partials.append,
    )
    assert st.partials is False
    n = int(2.0 * TARGET_SR)
    audio = np.full(n, 0.2, dtype=np.float32)
    st.start()
    st.on_gate_event(GateEvent("speech_start", audio[: TARGET_SR // 2], TARGET_SR // 2))
    st.on_gate_event(GateEvent("frame", audio[TARGET_SR // 2 :], n))
    time.sleep(0.5)  # laisser tourner la boucle : aucun décodage partiel attendu
    assert partials == []
    st.on_gate_event(GateEvent("speech_end", None, n))
    st.stop(timeout=10.0)
    assert len(partials) == 1  # un seul décodage, final
    assert store.segments
