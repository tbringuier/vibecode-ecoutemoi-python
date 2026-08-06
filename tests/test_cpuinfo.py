"""Topologie CPU (P-cores / cœurs physiques) et profilage du modèle par défaut."""

from ecoutemoi.core import cpuinfo, models
from ecoutemoi.core.cpuinfo import _parse_cpu_list


def test_parse_cpu_list():
    assert _parse_cpu_list("0-3") == [0, 1, 2, 3]
    assert _parse_cpu_list("0-1,4-5") == [0, 1, 4, 5]
    assert _parse_cpu_list("7") == [7]
    assert _parse_cpu_list("") == []


def test_best_n_threads_prefers_p_cores(monkeypatch):
    monkeypatch.setattr(cpuinfo, "p_core_count", lambda: 6)
    monkeypatch.setattr(cpuinfo, "physical_cores", lambda: 16)
    assert cpuinfo.best_n_threads() == 6


def test_best_n_threads_falls_back_to_physical(monkeypatch):
    monkeypatch.setattr(cpuinfo, "p_core_count", lambda: None)
    monkeypatch.setattr(cpuinfo, "physical_cores", lambda: 12)
    assert cpuinfo.best_n_threads() == 12


def test_p_core_count_returns_none_or_positive():
    n = cpuinfo.p_core_count()
    assert n is None or n >= 1


class _Mem:
    def __init__(self, gb: float):
        self.total = gb * 1e9


def _profile(monkeypatch, cores: int, ram_gb: float) -> str:
    import psutil

    monkeypatch.setattr(cpuinfo, "best_n_threads", lambda: cores)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: _Mem(ram_gb))
    return models.recommended_default_model()


def test_recommended_model_big_machine(monkeypatch):
    assert _profile(monkeypatch, 8, 16) == "small-q5_1"


def test_recommended_model_mid_machine(monkeypatch):
    assert _profile(monkeypatch, 6, 16) == "base-q5_1"
    assert _profile(monkeypatch, 8, 6) == "base-q5_1"


def test_recommended_model_small_machine(monkeypatch):
    assert _profile(monkeypatch, 2, 4) == "tiny-q5_1"
    assert _profile(monkeypatch, 4, 2) == "tiny-q5_1"


def test_recommended_model_is_in_registry():
    assert models.recommended_default_model() in models.REGISTRY
