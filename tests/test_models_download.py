"""Robustesse des téléchargements de modèles : validation, retries, auto-réparation.

Aucun accès réseau : _download_once / download_model sont doublés ; on teste la
logique (magie ggml, tailles, backoff, suppression des fichiers corrompus,
agrégation des échecs) — le transport réel reste à huggingface_hub.
"""

import dataclasses
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


# --------------------------------------------------- format CTranslate2 (CPU)
# Un modèle CTranslate2 est un ENSEMBLE de fichiers dont aucun n'est facultatif.
# Un snapshot interrompu laisse un dossier à moitié peuplé, et CTranslate2 ne
# s'en plaindrait qu'à l'ouverture de la session — micro déjà branché.
CT2_SPEC = ModelSpec("test-ct2", "ggml-test-ct2.bin", "repo/x", 1, True, 0.1, "test",
                     family="testfam", quant="q5_1", ct2_repo="repo/ct2", ct2_size_mb=1,
                     compute_type="int8")  # fmt: skip


def _write_ct2(d, model_bytes: int = 1024 * 1024, vocab: str = "vocabulary.txt"):
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.bin").write_bytes(b"\x00" * model_bytes)
    (d / "config.json").write_text("{}", encoding="utf-8")
    (d / "tokenizer.json").write_text("{}", encoding="utf-8")
    (d / vocab).write_text("", encoding="utf-8")


def test_validate_ct2_detects_each_missing_piece(tmp_path):
    d = models.ct2_dir(CT2_SPEC, tmp_path)
    assert models.validate_ct2_dir(d, CT2_SPEC) == "dossier absent"
    _write_ct2(d)
    assert models.validate_ct2_dir(d, CT2_SPEC) is None
    (d / "tokenizer.json").unlink()
    assert "tokenizer.json" in models.validate_ct2_dir(d, CT2_SPEC)
    _write_ct2(d)
    (d / "vocabulary.txt").unlink()
    assert "vocabulaire absent" in models.validate_ct2_dir(d, CT2_SPEC)


def test_validate_ct2_accepts_either_vocabulary_flavour(tmp_path):
    """Les dépôts amont publient `vocabulary.txt` (Systran) OU `vocabulary.json`
    (large-v3, turbo) : exiger l'un des deux rendrait la moitié invalide."""
    d = models.ct2_dir(CT2_SPEC, tmp_path)
    _write_ct2(d, vocab="vocabulary.json")
    assert models.validate_ct2_dir(d, CT2_SPEC) is None


def test_validate_ct2_rejects_truncated_weights(tmp_path):
    d = models.ct2_dir(CT2_SPEC, tmp_path)
    _write_ct2(d, model_bytes=1024)  # ~1 Ko au lieu de ~1 Mo
    assert "hors tolérance" in models.validate_ct2_dir(d, CT2_SPEC)


def test_is_installed_is_per_format(tmp_path, monkeypatch):
    monkeypatch.setitem(models.REGISTRY, CT2_SPEC.key, CT2_SPEC)
    assert not models.is_installed(CT2_SPEC, tmp_path)
    _write_ct2(models.ct2_dir(CT2_SPEC, tmp_path))
    assert models.is_installed(CT2_SPEC, tmp_path, models.FMT_CT2)
    assert not models.is_installed(CT2_SPEC, tmp_path, models.FMT_GGML)
    assert models.is_installed(CT2_SPEC, tmp_path)  # fmt=None => l'un OU l'autre
    assert models.installed_formats(CT2_SPEC, tmp_path) == [models.FMT_CT2]
    _write_valid(models.model_path(CT2_SPEC, tmp_path))
    assert models.installed_formats(CT2_SPEC, tmp_path) == [models.FMT_GGML, models.FMT_CT2]


def test_download_many_ct2_dedupes_by_family(tmp_path, monkeypatch):
    """`small-q5_1` et `small-q8_0` désignent le même dossier CTranslate2 : les
    télécharger en parallèle, c'est se marcher dessus pour 464 Mo en double."""
    sibling = dataclasses.replace(CT2_SPEC, key="test-ct2-bis", quant="q8_0",
                                  compute_type="int8_float32")  # fmt: skip
    seen: list[str] = []

    def fake_ct2(spec, base=None, tqdm_class=None):
        seen.append(spec.key)
        d = models.ct2_dir(spec, base)
        _write_ct2(d)
        return d

    monkeypatch.setattr(models, "download_ct2_model", fake_ct2)
    models.download_many([CT2_SPEC, sibling], base=tmp_path, fmt=models.FMT_CT2)
    assert seen == ["test-ct2"]  # une seule fois pour la famille


def test_download_many_ct2_skips_models_without_conversion(tmp_path, monkeypatch):
    def must_not_be_called(*a, **k):
        raise AssertionError("aucune conversion CTranslate2 : rien à télécharger")

    monkeypatch.setattr(models, "download_ct2_model", must_not_be_called)
    assert models.download_many([SPEC], base=tmp_path, fmt=models.FMT_CT2) == []


def test_ensure_model_ct2_refuses_a_family_without_conversion(tmp_path, monkeypatch):
    monkeypatch.setitem(models.REGISTRY, SPEC.key, SPEC)  # ct2_repo vide
    with pytest.raises(ValueError, match="CTranslate2"):
        ensure_model(SPEC.key, base=tmp_path, fmt=models.FMT_CT2)
