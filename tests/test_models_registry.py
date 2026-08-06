"""Model registry invariants."""

import pytest

from ecoutemoi.constants import DEFAULT_MODEL
from ecoutemoi.core.models import (
    QUANT_NOTES,
    QUANT_ORDER,
    REGISTRY,
    VAD_SPEC,
    check_size,
    model_path,
    models_dir,
)

# Clés historiques : elles vivent dans des settings.json et des bench_results.json
# déjà écrits. Les renommer casserait silencieusement le modèle choisi.
LEGACY_KEYS = {
    "tiny-q5_1",
    "base-q5_1",
    "small-q5_1",
    "small-q8_0",
    "small",
    "medium-q5_0",
    "medium-q8_0",
    "large-v3-turbo-q5_0",
    "large-v3-turbo-q8_0",
}


def test_legacy_keys_still_present():
    assert set(REGISTRY) >= LEGACY_KEYS


# Ce que le dépôt ggml publie RÉELLEMENT et qui tient en temps réel. Assertion
# explicite plutôt que « au moins deux variantes » : elle documente les trous du
# dépôt amont au lieu de les deviner, et signale tout ajout non voulu.
EXPECTED_QUANTS = {
    "tiny": {"q5_1", "q8_0", "f16"},
    "base": {"q5_1", "q8_0", "f16"},
    "small": {"q5_1", "q8_0", "f16"},
    "medium": {"q5_0", "q8_0", "f16"},
    "large-v3-turbo": {"q5_0", "q8_0", "f16"},
    "large-v2": {"q5_0", "q8_0"},  # f16 = 2,9 Go : hors budget temps réel
    "large-v3": {"q5_0"},  # le dépôt amont ne publie pas de q8_0 pour v3
}


def test_registry_covers_every_realtime_family():
    families: dict[str, set[str]] = {}
    for spec in REGISTRY.values():
        families.setdefault(spec.family, set()).add(spec.quant)
    assert families == EXPECTED_QUANTS


def test_registry_keys_and_filenames():
    for key, spec in REGISTRY.items():
        assert spec.filename.startswith("ggml-")
        assert spec.filename.endswith(".bin")
        # La clé et le nom de fichier sont dérivés du couple (famille, quant) :
        # aucune divergence possible entre les deux.
        stem = spec.family if spec.quant == "f16" else f"{spec.family}-{spec.quant}"
        assert key == stem
        assert spec.filename == f"ggml-{stem}.bin"
        assert spec.size_mb > 0
        assert spec.ram_gb > spec.size_mb / 1024  # poids + activations
        assert spec.repo_id == "ggerganov/whisper.cpp"
        assert spec.quant in QUANT_NOTES
        assert spec.role


def test_no_full_precision_large_and_no_english_only():
    """large f16 (2,9 Go) et modèles .en : hors sujet, documenté dans models.py."""
    for key in REGISTRY:
        assert ".en" not in key
    assert "large-v2" not in REGISTRY and "large-v3" not in REGISTRY
    assert "large-v1" not in REGISTRY


def test_quant_notes_cover_every_used_quant():
    used = {spec.quant for spec in REGISTRY.values()}
    assert used <= set(QUANT_ORDER)
    for quant in used:
        short, detail = QUANT_NOTES[quant]
        assert short and detail


@pytest.mark.parametrize("family", ["tiny", "base", "small", "medium"])
def test_bigger_quantization_is_bigger_and_heavier(family):
    """f16 > q8_0 > q5_* en taille : un tableau mal saisi se verrait ici."""
    specs = sorted(
        (s for s in REGISTRY.values() if s.family == family),
        key=lambda s: QUANT_ORDER.index(s.quant),
    )
    sizes = [s.size_mb for s in specs]
    assert sizes == sorted(sizes), f"{family}: tailles non croissantes {sizes}"
    rams = [s.ram_gb for s in specs]
    assert rams == sorted(rams), f"{family}: RAM non croissante {rams}"


def test_turbo_locked_out_of_translation():
    # large-v3-turbo (toutes quantizations) is not trained for task=translate
    turbo = [k for k in REGISTRY if k.startswith("large-v3-turbo")]
    assert len(turbo) == 3 and all(not REGISTRY[k].translate for k in turbo)
    others = [k for k in REGISTRY if not k.startswith("large-v3-turbo")]
    assert all(REGISTRY[k].translate for k in others)


def test_default_model_is_registered():
    assert DEFAULT_MODEL in REGISTRY
    assert DEFAULT_MODEL == "small-q5_1"


def test_vad_spec():
    assert VAD_SPEC.repo_id == "ggml-org/whisper-vad"
    assert "silero" in VAD_SPEC.filename
    assert VAD_SPEC.filename.endswith(".bin")


def test_model_path_under_models_dir(tmp_path):
    spec = REGISTRY["tiny-q5_1"]
    p = model_path(spec, base=tmp_path)
    assert p == models_dir(tmp_path) / "ggml-tiny-q5_1.bin"


def test_check_size_tolerance(tmp_path):
    spec = REGISTRY["tiny-q5_1"]  # 31 MiB
    f = tmp_path / spec.filename
    f.write_bytes(b"\0" * (31 * 1024 * 1024))
    assert check_size(f, spec)
    f.write_bytes(b"\0" * int(31 * 1024 * 1024 * 0.90))  # -10 % -> reject
    assert not check_size(f, spec)
    f.write_bytes(b"\0" * int(31 * 1024 * 1024 * 1.04))  # +4 % -> ok
    assert check_size(f, spec)


def test_check_size_vad_floor(tmp_path):
    f = tmp_path / VAD_SPEC.filename
    f.write_bytes(b"\0" * 900_000)
    assert check_size(f, VAD_SPEC)
    f.write_bytes(b"\0" * 1000)  # truncated download
    assert not check_size(f, VAD_SPEC)
