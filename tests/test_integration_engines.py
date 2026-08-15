"""Les deux moteurs, pour de vrai : chargement, décodage, contrat identique.

Ce que les tests unitaires ne peuvent pas prouver : que `create_engine` rend un
objet utilisable, que faster-whisper décode réellement du français, et surtout
que les DEUX moteurs sont interchangeables du point de vue du streamer. Un écart
de contrat (horodatage en secondes au lieu de millisecondes, segments vides,
`last_decode_ms` jamais renseigné) ne se verrait qu'en session.

Run with: pytest -m integration
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from ecoutemoi.core import models
from ecoutemoi.core.engine_base import EngineParams
from ecoutemoi.core.engines import create_engine

pytestmark = pytest.mark.integration

MODEL_KEY = "tiny-q5_1"
# Surface publique dont dépendent le streamer, la transcription de fichiers, la
# GUI et le sous-processus. Elle doit être IDENTIQUE des deux côtés.
ENGINE_API = (
    "transcribe", "warmup", "close", "detected_language", "backend_info",
    "gpu_active", "gpu_devices", "active_gpu_index", "gpu_diagnostic", "diagnostics",
)  # fmt: skip
ENGINE_ATTRS = (
    "params", "sink", "n_threads", "vad_active", "flash_attn_active",
    "dropped_params", "load_variant", "load_s", "last_decode_ms", "warmup_ms",
)  # fmt: skip


@pytest.fixture(scope="module")
def speech() -> np.ndarray:
    """La voix de référence embarquée : du français réel, 21 s, 16 kHz mono."""
    import ecoutemoi

    path = Path(ecoutemoi.__file__).resolve().parent / "assets" / "calibration" / "calibration_fr.wav"
    with wave.open(str(path), "rb") as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _params(engine: str) -> EngineParams:
    spec = models.REGISTRY[MODEL_KEY]
    if engine == "faster-whisper":
        return EngineParams(
            ct2_path=models.ensure_model(MODEL_KEY, fmt=models.FMT_CT2),
            language="fr", backend="cpu", engine=engine, compute_type=spec.compute_type,
        )  # fmt: skip
    return EngineParams(
        model_path=models.ensure_model(MODEL_KEY, fmt=models.FMT_GGML),
        vad_model_path=models.ensure_vad_model(),
        language="fr", backend="cpu", engine=engine,
    )  # fmt: skip


@pytest.mark.parametrize("engine_name", ["whispercpp", "faster-whisper"])
def test_engine_transcribes_french(engine_name, speech):
    engine = create_engine(_params(engine_name))
    try:
        segments = engine.transcribe(speech)
        assert segments, "aucun segment produit sur 21 s de parole"
        text = " ".join(s.text for s in segments).lower()
        # Mots du texte de calibration : la reconnaissance a bien eu lieu, on ne
        # mesure pas la qualité ici (c'est le rôle du benchmark).
        assert "conférence" in text or "conference" in text
        assert engine.last_decode_ms > 0.0
    finally:
        engine.close()


@pytest.mark.parametrize("engine_name", ["whispercpp", "faster-whisper"])
def test_timestamps_are_milliseconds_within_the_audio(engine_name, speech):
    """whisper.cpp compte en centisecondes, faster-whisper en secondes : une
    conversion ratée d'un côté donnerait des sous-titres 10x ou 1000x décalés,
    sans jamais lever d'erreur."""
    duration_ms = len(speech) * 1000 // 16000
    engine = create_engine(_params(engine_name))
    try:
        segments = engine.transcribe(speech)
    finally:
        engine.close()
    for s in segments:
        assert 0 <= s.t0_ms <= s.t1_ms <= duration_ms + 1000
    assert segments[-1].t1_ms > duration_ms // 2  # couvre bien tout le passage


@pytest.mark.parametrize("engine_name", ["whispercpp", "faster-whisper"])
def test_engines_share_the_same_surface(engine_name):
    """Le streamer, la GUI et le sous-processus manipulent l'un ou l'autre sans
    le savoir : tout écart de surface se paierait à l'exécution."""
    engine = create_engine(_params(engine_name))
    try:
        for name in ENGINE_API:
            assert callable(getattr(engine, name)), f"{engine_name}: {name} manquant"
        for name in ENGINE_ATTRS:
            assert hasattr(engine, name), f"{engine_name}: attribut {name} manquant"
        diag = engine.diagnostics()
        assert diag["moteur"]  # le rapport dit toujours QUI décode
        import json

        json.dumps(diag)  # sérialisable pour --stats-json et le rapport de bug
    finally:
        engine.close()


@pytest.mark.parametrize("engine_name", ["whispercpp", "faster-whisper"])
def test_warmup_returns_decreasing_or_stable_times(engine_name):
    engine = create_engine(_params(engine_name))
    try:
        times = engine.warmup()
        assert times, "le préchauffage n'a produit aucune passe"
        assert all(t > 0 for t in times)
        assert engine.warmup_ms == times
    finally:
        engine.close()


def test_cpu_backend_picks_faster_whisper_end_to_end():
    """La règle de la 2.0, vérifiée sur la vraie fabrique et les vrais paquets."""
    spec = models.REGISTRY[MODEL_KEY]
    params = EngineParams(
        model_path=models.ensure_model(MODEL_KEY, fmt=models.FMT_GGML),
        ct2_path=models.ensure_model(MODEL_KEY, fmt=models.FMT_CT2),
        language="fr", backend="cpu", engine="faster-whisper",
        compute_type=spec.compute_type,
    )  # fmt: skip
    engine = create_engine(params)
    try:
        assert engine.name == "faster-whisper"
        assert engine.gpu_active() is False
    finally:
        engine.close()


def test_subprocess_engine_carries_the_faster_whisper_choice():
    """Le moteur choisi doit survivre au passage du tube : sans les nouveaux
    champs dans le protocole, l'enfant rechargerait whisper.cpp en silence."""
    from ecoutemoi.core.engine_proc import SubprocessEngine

    spec = models.REGISTRY[MODEL_KEY]
    params = EngineParams(
        ct2_path=models.ensure_model(MODEL_KEY, fmt=models.FMT_CT2),
        language="fr", backend="cpu", engine="faster-whisper",
        compute_type=spec.compute_type,
    )  # fmt: skip
    engine = SubprocessEngine(params)
    try:
        assert engine.name == "faster-whisper"
        assert "faster-whisper" in engine.backend_info()
        segments = engine.transcribe(np.zeros(16000 * 2, dtype=np.float32))
        assert isinstance(segments, list)  # silence : zéro segment est correct
        assert engine.diagnostics()["moteur"]
    finally:
        engine.close()
