"""Settings persistence: atomic JSON, tolerant load."""

import json

from ecoutemoi.config import Settings, atomic_write_text, load_settings, save_settings


def test_defaults():
    s = Settings()
    assert s.model == "small-q5_1"
    assert s.mode == "fr"
    assert s.preset == "stable"
    # Nettoyage micro entièrement désactivé par défaut (« Aucun traitement »)
    assert s.denoise is False
    assert s.highpass is False
    assert s.carry_context is False
    assert s.obs_ws_enabled is False and s.web_enabled is False


def test_roundtrip(tmp_path):
    p = tmp_path / "settings.json"
    s = Settings(model="tiny-q5_1", mode="auto", gain=2.0, backend="cpu")
    save_settings(s, p)
    loaded = load_settings(p)
    assert loaded == s


def test_legacy_force_cpu_migrates_to_backend(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"model": "base-q5_1", "force_cpu": True}), encoding="utf-8")
    loaded = load_settings(p)
    assert loaded.backend == "cpu"
    p.write_text(json.dumps({"force_cpu": False}), encoding="utf-8")
    assert load_settings(p).backend == "auto"


def test_invalid_backend_falls_back_to_auto(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"backend": "cuda"}), encoding="utf-8")
    assert load_settings(p).backend == "auto"


def test_overlay_transparent_roundtrip(tmp_path):
    p = tmp_path / "settings.json"
    save_settings(Settings(overlay_transparent=True), p)
    assert load_settings(p).overlay_transparent is True
    assert Settings().overlay_transparent is False  # défaut : chroma


def test_display_style_sanitized_and_roundtrip(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"display_style": "matrix"}), encoding="utf-8")
    assert load_settings(p).display_style == "defilement"
    save_settings(Settings(display_style="fondu"), p)
    assert load_settings(p).display_style == "fondu"


def test_unknown_indexing_values_sanitized(tmp_path):
    """Preset / mode / alignement indexent des widgets au démarrage de la GUI :
    une valeur inconnue doit retomber sur le défaut, pas faire planter la fenêtre."""
    p = tmp_path / "settings.json"
    p.write_text(json.dumps({"preset": "turbo", "mode": "klingon", "align": "haut"}), encoding="utf-8")
    loaded = load_settings(p)
    assert loaded.preset == "stable"
    assert loaded.mode == "fr"
    assert loaded.align == "center"


def test_publish_ports_sanitized(tmp_path):
    p = tmp_path / "settings.json"
    save_settings(Settings(obs_ws_port=4460, web_port=9000), p)
    loaded = load_settings(p)
    assert (loaded.obs_ws_port, loaded.web_port) == (4460, 9000)
    p.write_text(json.dumps({"obs_ws_port": 0, "web_port": "beaucoup"}), encoding="utf-8")
    loaded = load_settings(p)
    assert loaded.obs_ws_port == 4455
    assert loaded.web_port == 8777


def test_clear_settings_removes_file(tmp_path):
    from ecoutemoi.config import clear_settings

    p = tmp_path / "settings.json"
    save_settings(Settings(), p)
    assert clear_settings(p) is True
    assert not p.exists()
    assert clear_settings(p) is False  # déjà effacé : signalé sans erreur


def test_gpu_device_roundtrip_and_sanitized(tmp_path):
    p = tmp_path / "settings.json"
    save_settings(Settings(gpu_device=2), p)
    assert load_settings(p).gpu_device == 2
    # settings.json édité à la main : négatif et non-numérique retombent sur 0
    p.write_text(json.dumps({"gpu_device": -3}), encoding="utf-8")
    assert load_settings(p).gpu_device == 0
    p.write_text(json.dumps({"gpu_device": "beaucoup"}), encoding="utf-8")
    assert load_settings(p).gpu_device == 0


def test_missing_file_gives_defaults_with_profiled_model(tmp_path, monkeypatch):
    import ecoutemoi.core.models as models

    monkeypatch.setattr(models, "recommended_default_model", lambda: "base-q5_1")
    loaded = load_settings(tmp_path / "nope.json")
    assert loaded.model == "base-q5_1"  # premier lancement : modèle profilé machine
    assert loaded == Settings(model="base-q5_1")  # tout le reste = défauts


def test_unknown_keys_ignored(tmp_path):
    p = tmp_path / "settings.json"
    data = {"model": "base-q5_1", "some_future_key": 42, "another": {"x": 1}}
    p.write_text(json.dumps(data), encoding="utf-8")
    loaded = load_settings(p)
    assert loaded.model == "base-q5_1"


def test_corrupted_file_gives_defaults(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text("{ not json", encoding="utf-8")
    assert load_settings(p) == Settings()


def test_atomic_write_no_tmp_left(tmp_path):
    p = tmp_path / "sub" / "settings.json"
    atomic_write_text(p, '{"a": "é"}')
    assert p.read_text(encoding="utf-8") == '{"a": "é"}'
    assert list(p.parent.glob("*.tmp")) == []


def test_save_creates_parent_dirs(tmp_path):
    p = tmp_path / "a" / "b" / "settings.json"
    save_settings(Settings(), p)
    assert p.is_file()
    # UTF-8 with accents readable
    raw = json.loads(p.read_text(encoding="utf-8"))
    assert raw["preset"] == "stable"
