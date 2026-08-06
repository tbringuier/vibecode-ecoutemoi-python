"""Import manuel des modèles : le nom identifie, le contenu est validé."""

from __future__ import annotations

from pathlib import Path

import pytest

from ecoutemoi.core import models
from ecoutemoi.core.models import GGML_MAGIC, REGISTRY, VAD_SPEC


def _write_valid(path: Path, spec) -> Path:
    """Fichier de modèle plausible : magie ggml + taille dans la tolérance."""
    path.parent.mkdir(parents=True, exist_ok=True)
    size = spec.size_mb * 1024 * 1024
    path.write_bytes(GGML_MAGIC + b"\0" * (size - len(GGML_MAGIC)))
    return path


def test_import_model_installs_and_returns_key(tmp_path):
    spec = REGISTRY["tiny-q5_1"]
    source = _write_valid(tmp_path / "depuis_usb" / spec.filename, spec)
    key, dest = models.import_model_file(source, base=tmp_path / "data")
    assert key == "tiny-q5_1"
    assert dest == models.models_dir(tmp_path / "data") / spec.filename
    assert models.validate_model_file(dest, spec) is None
    assert models.installed_models(base=tmp_path / "data") == ["tiny-q5_1"]


def test_import_accepts_the_vad_model(tmp_path):
    source = tmp_path / VAD_SPEC.filename
    source.write_bytes(GGML_MAGIC + b"\0" * 900_000)
    key, dest = models.import_model_file(source, base=tmp_path / "data")
    assert key == VAD_SPEC.key
    assert dest.is_file()


def test_import_rejects_unknown_filename(tmp_path):
    source = tmp_path / "mon-modele-perso.bin"
    source.write_bytes(GGML_MAGIC + b"\0" * 1024)
    with pytest.raises(ValueError) as err:
        models.import_model_file(source, base=tmp_path / "data")
    # L'erreur doit LISTER les noms attendus, sinon elle est inexploitable
    assert "ggml-tiny-q5_1.bin" in str(err.value)


def test_import_rejects_corrupt_content(tmp_path):
    spec = REGISTRY["tiny-q5_1"]
    source = tmp_path / spec.filename
    source.write_bytes(b"ceci n'est pas un modele ggml")
    with pytest.raises(ValueError, match="magie ggml"):
        models.import_model_file(source, base=tmp_path / "data")
    assert not (models.models_dir(tmp_path / "data") / spec.filename).exists()


def test_import_rejects_wrong_size(tmp_path):
    """Bonne magie mais mauvaise taille : téléchargement tronqué, à refuser."""
    spec = REGISTRY["small-q5_1"]
    source = tmp_path / spec.filename
    source.write_bytes(GGML_MAGIC + b"\0" * 1024)
    with pytest.raises(ValueError, match="tolérance"):
        models.import_model_file(source, base=tmp_path / "data")


def test_import_is_idempotent_on_the_installed_file(tmp_path):
    spec = REGISTRY["tiny-q5_1"]
    installed = _write_valid(models.models_dir(tmp_path) / spec.filename, spec)
    key, dest = models.import_model_file(installed, base=tmp_path)
    assert (key, dest) == ("tiny-q5_1", installed)
    assert dest.is_file()
