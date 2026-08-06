"""Robustesse des téléchargements de modèles : validation, retries, auto-réparation.

Aucun accès réseau : _download_once / download_model sont doublés ; on teste la
logique (magie ggml, tailles, backoff, suppression des fichiers corrompus,
agrégation des échecs) — le transport réel reste à huggingface_hub.
"""

import types

import pytest

from ecoutemoi.core import models
from ecoutemoi.core.models import (
    GGML_MAGIC,
    ModelSpec,
    download_many,
    download_model,
    ensure_model,
    validate_model_file,
)

# ~1 Mo => plancher de taille (300 ko) au lieu du ±5 %
SPEC = ModelSpec("test-model", "ggml-test.bin", "repo/x", 1, True, 0.1, "test")
SPEC_5MB = ModelSpec("test-5mb", "ggml-test5.bin", "repo/x", 5, True, 0.1, "test")


def _write_valid(path, size: int = 400_000):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(GGML_MAGIC + b"\x00" * size)


# ------------------------------------------------------------------ validation
def test_validate_detects_each_corruption_kind(tmp_path):
    p = tmp_path / "m.bin"
    assert validate_model_file(p, SPEC) == "fichier absent"
    p.write_bytes(b"")
    assert validate_model_file(p, SPEC) == "fichier vide"
    p.write_bytes(b"<html>portail captif</html>" + b"\x00" * 400_000)
    assert "magie ggml absente" in validate_model_file(p, SPEC)
    p.write_bytes(GGML_MAGIC + b"\x00" * 10)  # tronqué sous le plancher
    assert "taille hors tolérance" in validate_model_file(p, SPEC)
    _write_valid(p)
    assert validate_model_file(p, SPEC) is None


def test_validate_size_tolerance_five_percent(tmp_path):
    p = tmp_path / "m5.bin"
    _write_valid(p, size=5 * 1024 * 1024 - 4)  # pile la taille attendue
    assert validate_model_file(p, SPEC_5MB) is None
    _write_valid(p, size=1024)  # très en dessous des 5 Mo ±5 %
    assert "taille hors tolérance" in validate_model_file(p, SPEC_5MB)


# ------------------------------------------------------------------- download
def test_download_retries_then_succeeds(tmp_path, monkeypatch):
    calls = {"n": 0}

    def flaky_once(spec, dest_dir, tqdm_class):
        calls["n"] += 1
        if calls["n"] < 3:
            raise OSError("réseau qui tousse")
        p = dest_dir / spec.filename
        _write_valid(p)
        return p

    monkeypatch.setattr(models, "_download_once", flaky_once)
    monkeypatch.setattr(models.time, "sleep", lambda s: None)  # pas d'attente réelle
    p = download_model(SPEC, base=tmp_path)
    assert calls["n"] == 3
    assert validate_model_file(p, SPEC) is None


def test_download_gives_up_with_actionable_message(tmp_path, monkeypatch):
    def always_fails(spec, dest_dir, tqdm_class):
        raise OSError("proxy récalcitrant")

    monkeypatch.setattr(models, "_download_once", always_fails)
    monkeypatch.setattr(models.time, "sleep", lambda s: None)
    with pytest.raises(OSError, match="3 tentatives") as exc:
        download_model(SPEC, base=tmp_path)
    assert "manuellement" in str(exc.value)  # le message propose l'import manuel


def test_corrupt_download_is_deleted_not_kept(tmp_path, monkeypatch):
    def bad_once(spec, dest_dir, tqdm_class):
        p = dest_dir / spec.filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"<html>page d'erreur du proxy</html>")
        return p

    monkeypatch.setattr(models, "_download_once", bad_once)
    monkeypatch.setattr(models.time, "sleep", lambda s: None)
    with pytest.raises(OSError, match="invalide"):
        download_model(SPEC, base=tmp_path)
    # jamais de résidu corrompu qui passerait pour un modèle installé
    assert not (models.models_dir(tmp_path) / SPEC.filename).exists()


def test_insufficient_disk_space_fails_early(tmp_path, monkeypatch):
    monkeypatch.setattr(
        models.shutil, "disk_usage",
        lambda p: types.SimpleNamespace(total=0, used=0, free=10 * 1024 * 1024),
    )  # fmt: skip

    def must_not_be_called(*a):
        raise AssertionError("le téléchargement ne doit pas démarrer sans espace disque")

    monkeypatch.setattr(models, "_download_once", must_not_be_called)
    with pytest.raises(OSError, match="Espace disque insuffisant"):
        download_model(SPEC, base=tmp_path)


def test_download_many_reports_every_failure(tmp_path, monkeypatch):
    bad = ModelSpec("bad-model", "ggml-bad.bin", "repo/x", 1, True, 0.1, "test")

    def fake_dl(spec, base=None, tqdm_class=None):
        if spec.key == "bad-model":
            raise OSError("boom")
        p = models.model_path(spec, base)
        _write_valid(p)
        return p

    monkeypatch.setattr(models, "download_model", fake_dl)
    with pytest.raises(OSError, match="bad-model: boom"):
        download_many([SPEC, bad], base=tmp_path)
    # le modèle sain a quand même été téléchargé malgré l'échec de l'autre
    assert (models.models_dir(tmp_path) / SPEC.filename).exists()


# ---------------------------------------------------------------- auto-repair
def test_ensure_model_redownloads_corrupt_file(tmp_path, monkeypatch):
    monkeypatch.setitem(models.REGISTRY, SPEC.key, SPEC)
    bad = models.model_path(SPEC, tmp_path)
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"corrompu par une coupure")

    def fake_dl(spec, base=None, tqdm_class=None):
        p = models.model_path(spec, base)
        _write_valid(p)
        return p

    monkeypatch.setattr(models, "download_model", fake_dl)
    p = ensure_model(SPEC.key, base=tmp_path)
    assert p.read_bytes()[:4] == GGML_MAGIC  # réparé, pas réutilisé tel quel


def test_ensure_model_reuses_valid_file_without_download(tmp_path, monkeypatch):
    monkeypatch.setitem(models.REGISTRY, SPEC.key, SPEC)
    _write_valid(models.model_path(SPEC, tmp_path))

    def must_not_download(*a, **k):
        raise AssertionError("fichier sain : aucun téléchargement attendu")

    monkeypatch.setattr(models, "download_model", must_not_download)
    assert ensure_model(SPEC.key, base=tmp_path).is_file()
