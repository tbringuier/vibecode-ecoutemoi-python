"""Moteur faster-whisper : conversions d'unités, garde-fous, paramètres de décodage.

faster-whisper compte en SECONDES flottantes là où whisper.cpp compte en
centisecondes : la conversion se teste à la frontière, comme pour l'autre
moteur. Le chargement d'un vrai modèle CTranslate2 est du ressort des tests
d'intégration.
"""

import types
from pathlib import Path

import numpy as np
import pytest

from ecoutemoi.constants import VAD_MIN_SILENCE_MS, VAD_THRESHOLD
from ecoutemoi.core import engine_fw
from ecoutemoi.core.engine_base import EngineParams, _LogSink
from ecoutemoi.core.engine_fw import (
    COMPUTE_TYPES,
    MIN_AUDIO_SAMPLES,
    FasterWhisperEngine,
    resolve_compute_type,
)


def _bare_engine(**kw) -> FasterWhisperEngine:
    """Moteur sans modèle chargé : on teste la logique, pas CTranslate2."""
    e = object.__new__(FasterWhisperEngine)
    e.params = EngineParams(ct2_path=Path("ct2/small"), **{"language": "fr", **kw})
    e.sink = _LogSink()
    e.n_threads = 6
    e.compute_type = "int8"
    e.vad_active = bool(e.params.vad_filter)
    e.flash_attn_active = False
    e.dropped_params = []
    e.load_variant = "cpu-int8"
    e.load_s = 0.4
    e.last_decode_ms = 0.0
    e.warmup_ms = []
    e.fallback_reason = None
    e._detected_language = None
    e._model = None
    return e


class _FakeSegment:
    def __init__(self, start, end, text, no_speech_prob=None):
        self.start, self.end, self.text = start, end, text
        self.no_speech_prob = no_speech_prob


class _FakeModel:
    """Rend un GÉNÉRATEUR, comme faster-whisper : consommer ou ne rien décoder."""

    def __init__(self, segments, language="fr"):
        self._segments = segments
        self._info = types.SimpleNamespace(language=language)
        self.calls: list[dict] = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        return (s for s in self._segments), self._info


# ------------------------------------------------------------------ compute type
def test_compute_types_are_cpu_only():
    # float16 n'existe PAS sur CPU dans CTranslate2 : le proposer ferait échouer
    # le chargement : CTranslate2 ne l'expose pas côté CPU.
    assert "float16" not in COMPUTE_TYPES
    assert COMPUTE_TYPES[0] == "int8"  # du plus compact au plus fidèle


def test_resolve_compute_type_keeps_supported(monkeypatch):
    monkeypatch.setattr(engine_fw, "supported_compute_types", lambda: ["int8", "float32"])
    assert resolve_compute_type("float32") == "float32"


def test_resolve_compute_type_falls_back_to_most_compact(monkeypatch):
    """Un CPU sans le type demandé ne doit pas faire échouer la session."""
    monkeypatch.setattr(engine_fw, "supported_compute_types", lambda: ["float32"])
    assert resolve_compute_type("int8") == "float32"


def test_resolve_compute_type_survives_missing_ctranslate2(monkeypatch):
    monkeypatch.setattr(engine_fw, "supported_compute_types", lambda: [])
    assert resolve_compute_type("int8") == "int8"


# ------------------------------------------------------------------ décodage
def test_seconds_become_milliseconds():
    e = _bare_engine()
    e._model = _FakeModel([_FakeSegment(0.5, 2.5, " bonjour ", 0.12)])
    (seg,) = e.transcribe(np.zeros(16000, dtype=np.float32))
    assert (seg.t0_ms, seg.t1_ms) == (500, 2500)
    assert seg.text == "bonjour"  # espaces de faster-whisper retirés
    assert seg.no_speech_prob == pytest.approx(0.12)


def test_empty_segments_are_dropped():
    e = _bare_engine()
    e._model = _FakeModel([_FakeSegment(0.0, 1.0, "   "), _FakeSegment(1.0, 2.0, "oui")])
    out = e.transcribe(np.zeros(16000, dtype=np.float32))
    assert [s.text for s in out] == ["oui"]


def test_too_short_audio_is_refused_not_crashed():
    """Sous ~100 ms, le banc de filtres mel n'a pas de quoi remplir une trame :
    CTranslate2 lèverait. Le moteur ne doit pas dépendre de son appelant."""
    e = _bare_engine()
    e._model = _FakeModel([_FakeSegment(0.0, 1.0, "jamais décodé")])
    assert e.transcribe(np.zeros(MIN_AUDIO_SAMPLES - 1, dtype=np.float32)) == []
    assert e.last_decode_ms == 0.0
    assert e._model.calls == []  # le modèle n'a même pas été sollicité


def test_decode_time_measures_the_generator_being_consumed():
    e = _bare_engine()
    e._model = _FakeModel([_FakeSegment(0.0, 1.0, "bonjour")])
    e.transcribe(np.zeros(16000, dtype=np.float32))
    assert e.last_decode_ms > 0.0


def test_detected_language_follows_info():
    e = _bare_engine()
    e._model = _FakeModel([_FakeSegment(0.0, 1.0, "hello")], language="en")
    assert e.detected_language() is None
    e.transcribe(np.zeros(16000, dtype=np.float32))
    assert e.detected_language() == "en"


# ------------------------------------------------------------------ paramètres
def test_auto_language_becomes_none():
    """« auto » côté application = détection côté faster-whisper, qui l'exprime
    par `language=None` — passer la chaîne « auto » serait une langue inconnue."""
    assert _bare_engine(language="auto")._decode_kwargs()["language"] is None
    assert _bare_engine(language="fr")._decode_kwargs()["language"] == "fr"


def test_translate_maps_to_task():
    assert _bare_engine(translate=True)._decode_kwargs()["task"] == "translate"
    assert _bare_engine(translate=False)._decode_kwargs()["task"] == "transcribe"


def test_temperature_fallback_is_disabled():
    """La cascade par défaut (0 → 1.0) relance le décodage jusqu'à six fois :
    en direct, ce sursaut de latence est pire que le segment qu'il rattrape."""
    assert _bare_engine()._decode_kwargs()["temperature"] == 0.0


def test_vad_parameters_match_the_other_engine():
    """Même Silero des deux côtés : le découpage de la parole ne doit pas
    changer selon le backend choisi."""
    kw = _bare_engine(vad_filter=True)._decode_kwargs()
    assert kw["vad_filter"] is True
    assert kw["vad_parameters"]["threshold"] == VAD_THRESHOLD
    assert kw["vad_parameters"]["min_silence_duration_ms"] == VAD_MIN_SILENCE_MS
    assert "vad_parameters" not in _bare_engine(vad_filter=False)._decode_kwargs()


def test_lexicon_becomes_initial_prompt_and_is_bounded():
    kw = _bare_engine(lexicon="Ceph\nKubernetes\nOpenStack")._decode_kwargs()
    assert kw["initial_prompt"] == "Ceph, Kubernetes, OpenStack"
    assert "initial_prompt" not in _bare_engine(lexicon="")._decode_kwargs()


def test_beam_size_is_at_least_one():
    assert _bare_engine(beam_size=0)._decode_kwargs()["beam_size"] == 1
    assert _bare_engine(beam_size=5)._decode_kwargs()["beam_size"] == 5


def test_carry_context_maps_to_condition_on_previous_text():
    assert _bare_engine(carry_context=True)._decode_kwargs()["condition_on_previous_text"] is True
    assert _bare_engine(carry_context=False)._decode_kwargs()["condition_on_previous_text"] is False


# ------------------------------------------------------------------ diagnostic
def test_engine_never_claims_a_gpu():
    """CTranslate2 ne parle ni Vulkan ni Metal : prétendre le contraire ferait
    croire à un GPU actif et masquerait la vraie raison du repli."""
    e = _bare_engine()
    assert e.gpu_active() is False
    assert e.gpu_devices() == []
    assert e.active_gpu_index() is None
    assert "CPU" in e.backend_info()
    assert "Vulkan" in e.gpu_diagnostic()


def test_diagnostics_dict_is_json_friendly():
    import json

    d = _bare_engine().diagnostics()
    json.dumps(d)
    assert d["gpu_actif"] is False
    assert d["type_calcul"] == "int8"
    assert d["modele"] == "small"
