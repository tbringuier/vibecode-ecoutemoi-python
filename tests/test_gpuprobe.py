"""Sondage GPU : cache, invalidation, et lecture des lignes ggml.

Le sondage sert à trancher AVANT de télécharger : se tromper coûte 464 Mo de
CTranslate2 sur une machine qui a un GPU, ou l'inverse. Il tourne dans un
sous-processus (un ICD Vulkan cassé ne doit pas emporter l'application) et son
résultat est mémorisé — les deux comportements sont vérifiés ici.
"""

import json
import subprocess

import pytest

from ecoutemoi.core import gpuprobe
from ecoutemoi.core.gpuprobe import ProbeResult


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(gpuprobe, "cache_path", lambda: tmp_path / "gpu_probe.json")


def test_cache_roundtrip():
    result = ProbeResult(gpu=True, devices=[(0, "Intel(R) Arc(tm) Graphics (MTL)")],
                         backend_libs=["libggml-vulkan.so"])  # fmt: skip
    result.version, result.platform = gpuprobe._fingerprint()
    gpuprobe.save_cache(result)
    back = gpuprobe.load_cache()
    assert back is not None
    assert back.gpu is True
    assert back.devices == [(0, "Intel(R) Arc(tm) Graphics (MTL)")]


def test_cache_is_ignored_after_a_version_change():
    """Mise à jour de l'app ou de l'OS : le pilote a pu changer, on re-sonde
    plutôt que de servir un verdict périmé."""
    _version, plat = gpuprobe._fingerprint()
    gpuprobe.cache_path().parent.mkdir(parents=True, exist_ok=True)
    gpuprobe.cache_path().write_text(
        json.dumps({"gpu": True, "devices": [], "version": "0.0.0", "platform": plat}),
        encoding="utf-8",
    )
    assert gpuprobe.load_cache() is None


def test_corrupt_cache_is_ignored_not_fatal():
    gpuprobe.cache_path().parent.mkdir(parents=True, exist_ok=True)
    gpuprobe.cache_path().write_text("{ ceci n'est pas du JSON", encoding="utf-8")
    assert gpuprobe.load_cache() is None


def test_gpu_candidates_uses_the_cache_without_spawning(monkeypatch):
    cached = ProbeResult(gpu=True, devices=[(0, "GPU de test")])
    cached.version, cached.platform = gpuprobe._fingerprint()
    gpuprobe.save_cache(cached)

    def forbidden(*_a, **_k):
        raise AssertionError("le sous-processus ne devait pas être lancé")

    monkeypatch.setattr(gpuprobe, "probe", forbidden)
    assert gpuprobe.gpu_candidates().gpu is True


def test_force_reprobes_and_rewrites_the_cache(monkeypatch):
    fresh = ProbeResult(gpu=False, reason="aucun périphérique")
    fresh.version, fresh.platform = gpuprobe._fingerprint()
    monkeypatch.setattr(gpuprobe, "probe", lambda **_k: fresh)
    assert gpuprobe.gpu_candidates(force=True).gpu is False
    assert gpuprobe.load_cache().gpu is False


# ------------------------------------------------------------------- robustesse
def test_dead_worker_means_no_gpu_not_a_crash(monkeypatch):
    """Un pilote qui fait segfaulter l'enfant se lit « pas de GPU » : l'app
    démarre sur le moteur CPU au lieu de mourir au lancement."""

    def dead(*_a, **_k):
        return subprocess.CompletedProcess([], returncode=-11, stdout="", stderr="Segmentation fault")

    monkeypatch.setattr(subprocess, "run", dead)
    result = gpuprobe.probe()
    assert result.gpu is False
    assert "repli CPU" in result.reason


def test_worker_timeout_means_no_gpu(monkeypatch):
    def hang(*_a, **_k):
        raise subprocess.TimeoutExpired(cmd="probe", timeout=1.0)

    monkeypatch.setattr(subprocess, "run", hang)
    result = gpuprobe.probe()
    assert result.gpu is False
    assert result.reason


def test_summary_is_operator_readable():
    assert "Arc" in ProbeResult(gpu=True, devices=[(0, "Intel Arc")]).summary
    assert ProbeResult(gpu=False, reason="pilote absent").summary == "pilote absent"
    assert ProbeResult(gpu=False).summary  # jamais une chaîne vide
