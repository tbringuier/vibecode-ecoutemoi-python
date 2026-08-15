"""Quel moteur pour quel backend — la règle de la 2.0, vérifiée sans modèle.

L'invariant à tenir : le GPU va à whisper.cpp (le seul à parler Vulkan/Metal),
le CPU à faster-whisper (~3x plus rapide), et AUCUN repli n'est silencieux.
Les moteurs réels sont remplacés par des doublures : ce qui est testé ici, c'est
la décision, pas l'inférence.
"""

from pathlib import Path

import pytest

from ecoutemoi.constants import ENGINE_FASTER_WHISPER, ENGINE_WHISPERCPP
from ecoutemoi.core import engines
from ecoutemoi.core.engine_base import EngineParams

GGML = Path("models/ggml-small-q5_1.bin")
CT2 = Path("models/ct2/small")


class _FakeWhisperCpp:
    name = ENGINE_WHISPERCPP

    def __init__(self, params, backend, gpu_ok=True):
        self.params = params
        self.backend = backend
        self._gpu_ok = gpu_ok
        self.closed = False
        self.fallback_reason = None

    def gpu_active(self):
        return self._gpu_ok and self.backend in ("gpu", "gpu-only", "auto")

    def gpu_diagnostic(self):
        return None if self.gpu_active() else "aucun périphérique GPU utilisable"

    def close(self):
        self.closed = True


class _FakeFasterWhisper:
    name = ENGINE_FASTER_WHISPER

    def __init__(self, params):
        self.params = params
        self.fallback_reason = None

    def gpu_active(self):
        return False


@pytest.fixture
def wired(monkeypatch):
    """Branche les deux fabriques sur des doublures et retourne le journal."""
    built: list[tuple[str, str | None]] = []

    def make_wcpp(params, backend, gpu_ok=True):
        built.append((ENGINE_WHISPERCPP, backend))
        return _FakeWhisperCpp(params, backend, gpu_ok=state["gpu_ok"])

    def make_fw(params):
        built.append((ENGINE_FASTER_WHISPER, None))
        return _FakeFasterWhisper(params)

    state = {"gpu_ok": True, "fw_available": True}
    monkeypatch.setattr(engines, "_build_whispercpp", make_wcpp)
    monkeypatch.setattr(engines, "_build_faster_whisper", make_fw)
    monkeypatch.setattr(
        engines,
        "_faster_whisper_ready",
        lambda p: (
            (True, None)
            if state["fw_available"] and p.ct2_path is not None
            else (False, "paquet faster-whisper absent de cet environnement")
        ),
    )
    return built, state


def _params(**kw) -> EngineParams:
    return EngineParams(model_path=GGML, ct2_path=CT2, **kw)


# --------------------------------------------------------------- backend -> moteur
def test_engine_for_backend_is_the_whole_rule():
    assert engines.engine_for_backend("gpu") == ENGINE_WHISPERCPP
    assert engines.engine_for_backend("cpu") == ENGINE_FASTER_WHISPER
    assert engines.engine_for_backend("auto") == "auto"
    # Le repli CPU reste configurable pour les machines où faster-whisper coince.
    assert engines.engine_for_backend("cpu", ENGINE_WHISPERCPP) == ENGINE_WHISPERCPP


def test_cpu_backend_never_touches_the_gpu(wired):
    built, _ = wired
    engine = engines.create_engine(_params(backend="cpu", engine=ENGINE_FASTER_WHISPER))
    assert engine.name == ENGINE_FASTER_WHISPER
    assert built == [(ENGINE_FASTER_WHISPER, None)]


def test_gpu_backend_uses_whispercpp_with_its_own_ladder(wired):
    """Moteur demandé explicitement : l'échelle interne de whisper.cpp (GPU,
    GPU sans flash, CPU) reste le bon comportement — c'est un choix opérateur."""
    built, _ = wired
    engine = engines.create_engine(_params(backend="gpu", engine=ENGINE_WHISPERCPP))
    assert engine.name == ENGINE_WHISPERCPP
    assert built == [(ENGINE_WHISPERCPP, "gpu")]


# ------------------------------------------------------------------------- auto
def test_auto_prefers_the_gpu_when_it_really_answers(wired):
    built, _ = wired
    engine = engines.create_engine(_params(backend="auto", engine="auto"))
    assert engine.name == ENGINE_WHISPERCPP
    # « gpu-only » : SANS barreau CPU, sinon le repli serait le CPU lent de
    # whisper.cpp au lieu de faster-whisper.
    assert built == [(ENGINE_WHISPERCPP, "gpu-only")]


def test_auto_falls_back_to_faster_whisper_when_gpu_is_inactive(wired):
    built, state = wired
    state["gpu_ok"] = False
    engine = engines.create_engine(_params(backend="auto", engine="auto"))
    assert engine.name == ENGINE_FASTER_WHISPER
    assert built == [(ENGINE_WHISPERCPP, "gpu-only"), (ENGINE_FASTER_WHISPER, None)]


def test_auto_records_why_it_fell_back(wired):
    """Un repli muet est pire qu'un repli : l'opérateur doit pouvoir lire la
    raison dans la barre d'état, pas seulement constater la lenteur."""
    _built, state = wired
    state["gpu_ok"] = False
    engine = engines.create_engine(_params(backend="auto", engine="auto"))
    assert engine.fallback_reason
    assert "GPU" in engine.fallback_reason


def test_auto_survives_a_gpu_engine_that_raises(wired, monkeypatch):
    """Pilote Vulkan qui explose au chargement : la session doit démarrer quand
    même, sur l'autre moteur."""
    built, _ = wired
    original = engines._build_whispercpp

    def boom(params, backend, **kw):
        original(params, backend, **kw)
        raise RuntimeError("pilote Vulkan inchargeable")

    monkeypatch.setattr(engines, "_build_whispercpp", boom)
    engine = engines.create_engine(_params(backend="auto", engine="auto"))
    assert engine.name == ENGINE_FASTER_WHISPER
    assert "pilote Vulkan inchargeable" in engine.fallback_reason
    assert built[0] == (ENGINE_WHISPERCPP, "gpu-only")


def test_auto_without_ggml_goes_straight_to_cpu(wired):
    built, _ = wired
    engine = engines.create_engine(EngineParams(model_path=None, ct2_path=CT2, backend="auto", engine="auto"))
    assert engine.name == ENGINE_FASTER_WHISPER
    assert built == [(ENGINE_FASTER_WHISPER, None)]


# ------------------------------------------------------------- moteur manquant
def test_missing_faster_whisper_falls_back_to_whispercpp_cpu(wired):
    """Environnement sans faster-whisper (wheel absente) : on ne refuse pas de
    démarrer, on prévient et on décode plus lentement."""
    built, state = wired
    state["fw_available"] = False
    engine = engines.create_engine(_params(backend="cpu", engine=ENGINE_FASTER_WHISPER))
    assert engine.name == ENGINE_WHISPERCPP
    assert built[-1] == (ENGINE_WHISPERCPP, "cpu")
    assert "faster-whisper indisponible" in engine.fallback_reason


def test_no_engine_at_all_fails_loudly(wired):
    _built, state = wired
    state["fw_available"] = False
    with pytest.raises(RuntimeError, match="Aucun moteur CPU"):
        engines.create_engine(EngineParams(model_path=None, ct2_path=None, backend="cpu"))
