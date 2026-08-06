"""Engine unit conversions, thread policy and GPU-fallback diagnostic.

whisper.cpp timestamps are centiseconds (units of 10 ms): the ms conversion is
tested at the boundary; the real engine is integration scope.
"""

import subprocess
import sys
import types
from pathlib import Path

import pytest

from ecoutemoi.core.engine import (
    Segment,
    WhisperEngine,
    _LogSink,
    backend_attempts,
    centi_to_ms,
    gpu_backend_libs,
    gpu_fallback_reason,
    pick_n_threads,
)


def test_centi_to_ms():
    assert centi_to_ms(0) == 0
    assert centi_to_ms(1) == 10
    assert centi_to_ms(150) == 1500
    assert centi_to_ms(123.4) == 1234


def test_segment_holds_ms():
    s = Segment(t0_ms=centi_to_ms(50), t1_ms=centi_to_ms(250), text="bonjour")
    assert (s.t0_ms, s.t1_ms) == (500, 2500)
    assert s.no_speech_prob is None


def test_pick_n_threads_honours_explicit_request():
    assert pick_n_threads(16) == 16
    assert pick_n_threads(4) == 4
    assert pick_n_threads(1) == 1


def test_pick_n_threads_default_p_cores_then_physical(monkeypatch):
    from ecoutemoi.core import cpuinfo

    monkeypatch.setattr(cpuinfo, "p_core_count", lambda: 6)
    assert pick_n_threads(None) == 6  # CPU hybride : P-cores physiques
    monkeypatch.setattr(cpuinfo, "p_core_count", lambda: None)
    monkeypatch.setattr(cpuinfo, "physical_cores", lambda: 12)
    assert pick_n_threads(None) == 12  # CPU homogène : tous les cœurs physiques


def test_pick_n_threads_default_positive():
    assert pick_n_threads(None) >= 1


def test_backend_cpu_never_touches_gpu():
    attempts = backend_attempts("cpu", flash_attn=True)
    assert [name for name, _ in attempts] == ["cpu"]
    assert all(ctx["use_gpu"] is False for _, ctx in attempts)


def test_backend_gpu_ladder_falls_back_to_cpu():
    attempts = backend_attempts("gpu", flash_attn=True)
    assert [name for name, _ in attempts] == ["full", "no-flash", "cpu"]
    assert attempts[0][1] == {"use_gpu": True, "flash_attn": True, "gpu_device": 0}
    assert attempts[-1][1] == {"use_gpu": False, "flash_attn": False}


def test_backend_ladder_carries_gpu_device_but_not_on_cpu_rung():
    attempts = backend_attempts("gpu", flash_attn=True, gpu_device=1)
    by_name = dict(attempts)
    assert by_name["full"]["gpu_device"] == 1
    assert by_name["no-flash"]["gpu_device"] == 1
    assert "gpu_device" not in by_name["cpu"]  # le rung CPU reste insensible au GPU


def test_backend_auto_same_ladder_as_gpu():
    assert backend_attempts("auto", True) == backend_attempts("gpu", True)


def test_backend_ladder_dedupes_when_flash_off():
    attempts = backend_attempts("auto", flash_attn=False)
    assert [name for name, _ in attempts] == ["full", "cpu"]  # no-flash duplicate dropped


# --------------------------------------------------------- diagnostic GPU
# Un échec de création d'instance Vulkan (ICD hôte inchargeable) est SILENCIEUX :
# aucune ligne « ggml_vulkan » dans le sink. Le diagnostic doit donc s'appuyer
# sur la présence physique de la lib backend, pas seulement sur les logs.


def test_fallback_reason_cpu_wheel_when_no_backend_no_logs():
    reason = gpu_fallback_reason([], backend_shipped=False)
    assert "wheel CPU" in reason


def test_fallback_reason_driver_when_backend_shipped_but_silent():
    # Backend livré mais instance Vulkan morte sans un mot : c'est le pilote.
    reason = gpu_fallback_reason(["whisper_model_load: loading model"], backend_shipped=True)
    assert "pilote" in reason
    assert "wheel CPU" not in reason


def test_fallback_reason_driver_when_logs_mention_vulkan():
    reason = gpu_fallback_reason(["ggml_vulkan: No devices found."], backend_shipped=False)
    assert "pilote" in reason


def test_gpu_backend_libs_detects_shipped_backend(monkeypatch, tmp_path):
    (tmp_path / "ggml-vulkan-0a1b2c3d.dll").write_bytes(b"")  # nom manglé delvewheel
    fake = types.ModuleType("_pywhispercpp")
    fake.__file__ = str(tmp_path / "_pywhispercpp.pyd")
    monkeypatch.setitem(sys.modules, "_pywhispercpp", fake)
    assert gpu_backend_libs() == ["ggml-vulkan-0a1b2c3d.dll"]


def test_gpu_backend_libs_empty_for_cpu_wheel(monkeypatch, tmp_path):
    (tmp_path / "ggml-cpu-329fe868.dll").write_bytes(b"")  # wheel CPU : pas de vulkan/metal
    fake = types.ModuleType("_pywhispercpp")
    fake.__file__ = str(tmp_path / "_pywhispercpp.pyd")
    monkeypatch.setitem(sys.modules, "_pywhispercpp", fake)
    assert gpu_backend_libs() == []


# ------------------------------------------------- détection backend (fenêtré)
# App fenêtrée (double-clic) : les prints natifs de ggml (« ggml_vulkan: 0 = … »)
# n'existent que si fd 2 est capturé ; la ligne « using VulkanN backend » du
# logger whisper sert de signal GPU-actif de secours.


def _bare_engine(lines: list[str], gpu_device: int = 0) -> WhisperEngine:
    from ecoutemoi.core.engine import EngineParams

    e = object.__new__(WhisperEngine)
    e.sink = _LogSink()
    for line in lines:
        e.sink.write(line + "\n")
    e.params = EngineParams(model_path=Path("ggml-test.bin"), gpu_device=gpu_device)
    e.n_threads = 8
    e._gpu_requested = True
    e.load_variant = "full"
    e.load_s = 0.2
    e.flash_attn_active = True
    e.vad_active = True
    e.dropped_params = []
    e.warmup_ms = [820.0, 310.0]
    return e


_TWO_GPUS = [
    "ggml_vulkan: Found 2 Vulkan devices:",
    "ggml_vulkan: 0 = AMD Radeon RX 7900 XTX (AMD proprietary driver) | uma: 0 | fp16: 1",
    "ggml_vulkan: 1 = Intel(R) Arc(tm) Graphics (MTL) (Intel driver) | uma: 1 | fp16: 1",
]


def test_backend_info_generic_vulkan_when_device_line_lost():
    e = _bare_engine(["whisper_backend_init_gpu: using Vulkan0 backend"])
    assert e.backend_info() == "Vulkan : GPU 0"
    assert e.gpu_active()


def test_backend_info_prefers_pretty_device_name():
    e = _bare_engine(
        [
            "whisper_backend_init_gpu: using Vulkan0 backend",
            "ggml_vulkan: 0 = AMD Radeon RX 7900 XTX (RADV) | uma: 0",
        ]
    )
    assert e.backend_info() == "Vulkan : AMD Radeon RX 7900 XTX"


def test_backend_info_cpu_without_gpu_lines():
    e = _bare_engine(["whisper_model_load: loading model"])
    assert e.backend_info() == "CPU (8 threads)"
    assert not e.gpu_active()


def test_enumeration_alone_is_not_gpu_active():
    # ggml énumère les périphériques même quand whisper finit en CPU (index
    # invalide, init ratée) : la liste seule ne doit PAS faire croire au GPU.
    e = _bare_engine([*_TWO_GPUS, "whisper_backend_init_gpu: no GPU found"])
    assert not e.gpu_active()
    assert e.backend_info() == "CPU (8 threads)"
    assert e.gpu_devices() == [
        (0, "AMD Radeon RX 7900 XTX (AMD proprietary driver)"),
        (1, "Intel(R) Arc(tm) Graphics (MTL) (Intel driver)"),
    ]


def test_backend_info_names_the_active_device_in_multi_gpu():
    e = _bare_engine([*_TWO_GPUS, "whisper_backend_init_gpu: using Vulkan1 backend"])
    assert e.active_gpu_index() == 1
    assert e.backend_info() == "Vulkan : Intel(R) Arc(tm) Graphics"  # nom court, coupé au pilote
    assert e.gpu_active()


def test_gpu_diagnostic_reports_invalid_device_index():
    e = _bare_engine([*_TWO_GPUS, "whisper_backend_init_gpu: no GPU found"], gpu_device=5)
    diag = e.gpu_diagnostic()
    assert "index GPU 5 invalide" in diag
    assert "AMD Radeon" in diag  # liste les périphériques valides


def test_supported_context_keys_include_gpu_device():
    from ecoutemoi.core.engine import supported_context_keys

    keys = supported_context_keys()
    assert keys is None or "gpu_device" in keys  # pywhispercpp >= 1.5.0


def test_diagnostics_dict_is_json_friendly():
    import json

    e = _bare_engine([*_TWO_GPUS, "whisper_backend_init_gpu: using Vulkan0 backend"])
    d = e.diagnostics()
    assert d["gpu_actif"] is True
    assert d["gpu_device_actif"] == 0
    assert len(d["gpu_devices"]) == 2
    json.dumps(d)  # sérialisable tel quel (rapport de bug, --stats-json…)


def test_whisper_log_callback_feeds_sink(monkeypatch):
    """Le callback whisper_log_set est LE canal fiable en app fenêtrée : les
    lignes ggml_vulkan ET whisper doivent alimenter le sink sans passer par stderr."""
    installed: list = []
    fake = types.ModuleType("_pywhispercpp")
    fake.whisper_log_set = installed.append
    monkeypatch.setitem(sys.modules, "_pywhispercpp", fake)

    e = _bare_engine([])
    assert e._install_whisper_log_callback() is True
    assert len(installed) == 1
    installed[0](2, "ggml_vulkan: 0 = AMD Radeon RX 7900 XTX (RADV) | uma: 0\n")
    installed[0](2, "whisper_backend_init_gpu: using Vulkan0 backend\n")
    assert e.backend_info() == "Vulkan : AMD Radeon RX 7900 XTX"


def test_whisper_log_callback_absent_binding(monkeypatch):
    fake = types.ModuleType("_pywhispercpp")  # vieux binding sans whisper_log_set
    monkeypatch.setitem(sys.modules, "_pywhispercpp", fake)
    e = _bare_engine([])
    assert e._install_whisper_log_callback() is False


@pytest.mark.skipif(sys.platform != "win32", reason="reproduction app fenêtrée Windows")
def test_stderr_capture_works_without_console(tmp_path):
    """DETACHED_PROCESS + handles std NULL ≈ double-clic sur l'exe fenêtré.
    Selon le recyclage des slots ucrt, fd 2 peut être invalide OU pointer sur
    un fichier arbitraire : dans les deux cas la capture doit rediriger fd 2
    vers le tampon et remplir le sink (ceinture du callback whisper_log_set)."""
    out = tmp_path / "result.txt"
    src = Path(__file__).resolve().parents[1] / "src"
    script = (
        "import os, sys\n"
        f"sys.path.insert(0, {str(src)!r})\n"
        "from ecoutemoi.core.engine import _LogSink, _StderrCapture\n"
        "sink = _LogSink()\n"
        "with _StderrCapture(sink):\n"
        "    os.write(2, b'ggml_vulkan: 0 = TestGPU (drv) | uma: 1\\n')\n"
        "captured = any('TestGPU' in li for li in sink.lines)\n"
        "verdict = 'OK' if captured else f'FAIL lines={list(sink.lines)!r}'\n"
        f"open({str(out)!r}, 'w').write(verdict)\n"
    )
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESTDHANDLES  # handles NULL : pas de stdio hérité
    subprocess.run(
        [sys.executable, "-c", script],
        creationflags=subprocess.DETACHED_PROCESS,  # pas de console
        startupinfo=si,
        timeout=60,
        check=False,
    )
    assert out.read_text() == "OK"
