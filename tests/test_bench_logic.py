"""Benchmark verdicts, recommendation, skip-larger behaviour."""

from pathlib import Path

from ecoutemoi.core import bench
from ecoutemoi.core.bench import BenchResult, model_wer, recommend, verdict


def R(key, rtf, wfr=0.1, wen=0.1, aborted=False, error=None):
    return BenchResult(key, "CPU (8)", rtf, wfr, wen, aborted=aborted, error=error)


def test_verdicts():
    assert verdict(R("a", 5.0, 0.10, 0.10)) == "Recommandé (sous-titres fluides)"
    # Rapide mais imprécis : PAS « Mode Phrase » (contresens — Phrase compense la
    # lenteur, pas l'imprécision du modèle).
    assert verdict(R("a", 5.0, 0.30, 0.30)) == "Fluide — précision limitée du modèle"
    assert verdict(R("a", 2.0, 0.05, 0.05)) == "Mode Phrase uniquement"  # 1.3 <= RTF < 3
    assert verdict(R("a", 1.0)) == "Inadapté (RTF < 1,3)"
    assert verdict(R("a", None, None, None, aborted=True, error="tué après 60 s")) == (
        "Abandonné (trop lent)"
    )
    assert verdict(
        R("a", None, None, None, aborted=True, error="sauté (modèle plus petit déjà trop lent)")
    ) == ("Sauté (modèle plus petit déjà trop lent)")


def test_verdict_shows_real_error_not_too_slow():
    # Un sous-processus mort n'est PAS « trop lent » : l'erreur réelle s'affiche.
    r = R("a", None, None, None, aborted=False, error="ModuleNotFoundError: no module named x")
    assert verdict(r).startswith("Échec : ModuleNotFoundError")


def test_pick_error_line_skips_log_noise():
    lines = [
        "Traceback (most recent call last):",
        '  File "x.py", line 1, in <module>',
        "ValueError: modèle corrompu",
        "INFO pywhispercpp.model: Inference time: 0.238 s",
        "DEBUG httpcore: close.complete",
    ]
    assert bench.pick_error_line(lines) == "ValueError: modèle corrompu"
    assert bench.pick_error_line(["INFO seulement"]) == "INFO seulement"  # repli
    assert bench.pick_error_line([]) is None


def test_recommend_never_picks_an_errored_model():
    results = [
        R("tiny-q5_1", 80.0, 0.10, 0.10),
        BenchResult("base-q5_1", "-", 90.0, 0.01, 0.01, error="crash pendant la passe EN"),
    ]
    assert recommend(results) == ("tiny-q5_1", "equilibre")  # base exclu malgré son WER


def test_pass_timeout_budgets_startup():
    # démarrage (extraction onefile + chargement + shaders) + 3x l'audio
    assert bench.pass_timeout_s(20.0) == bench.BENCH_STARTUP_ALLOWANCE_S + 60.0
    assert bench.pass_timeout_s(0.5) == bench.BENCH_STARTUP_ALLOWANCE_S + 10.0  # plancher décodage


def test_reference_wavs_bundled_and_valid():
    import wave

    fr, en = bench.reference_wavs()
    assert fr is not None and en is not None, "voix de référence absentes des assets"
    for p in (fr, en):
        with wave.open(str(p), "rb") as w:
            assert w.getframerate() == 16000
            assert w.getnchannels() == 1
            assert w.getnframes() / w.getframerate() > 10  # assez long pour un RTF stable


def test_resolve_calibration_prefers_operator_recordings(tmp_path):
    op_fr = tmp_path / "calibration_fr.wav"
    op_fr.write_bytes(b"RIFF")  # existence suffit pour la résolution
    fr, en, is_ref = bench.resolve_calibration_wavs(op_fr, tmp_path / "absent.wav")
    assert (fr, en, is_ref) == (op_fr, None, False)
    # aucun enregistrement opérateur -> repli sur les voix embarquées
    fr, en, is_ref = bench.resolve_calibration_wavs(tmp_path / "a.wav", tmp_path / "b.wav")
    assert is_ref is True and fr is not None and en is not None


def test_model_wer_is_mean_of_languages():
    assert model_wer(R("a", 5.0, 0.20, 0.00)) == 0.10
    assert model_wer(BenchResult("a", "-", 5.0, 0.30, None)) == 0.30
    assert model_wer(BenchResult("a", "-", 5.0, None, None)) == 1.0


def test_recommend_prefers_fluid_mean_wer():
    results = [
        R("tiny-q5_1", 80.0, 0.21, 0.00),  # mean 10.5 %
        R("base-q5_1", 40.0, 0.10, 0.00),  # mean 5 %
        R("small-q5_1", 12.0, 0.08, 0.00),  # mean 4 % -> winner
    ]
    assert recommend(results) == ("small-q5_1", "equilibre")


def test_recommend_falls_back_to_phrase():
    results = [R("tiny-q5_1", 2.0, 0.2, 0.2), R("base-q5_1", 1.5, 0.1, 0.1)]
    assert recommend(results) == ("base-q5_1", "phrase")
    assert recommend([R("a", 1.0), R("b", None, None, None, aborted=True)]) is None


def test_run_benchmark_skips_larger_after_abort(monkeypatch, tmp_path):
    """A killed model skips every larger one."""
    calls = []

    def fake_run_pass(
        model_key, wav_path, *, language, translate=False, backend="auto", gpu_device=0, work_dir=None
    ):
        calls.append((model_key, language, translate))
        if model_key == "base-q5_1":
            return {"ok": False, "aborted": True, "error": "tué après 60 s (3× l'audio)"}
        return {"ok": True, "rtf": 10.0, "text": "bonjour", "backend": "CPU (8)"}

    monkeypatch.setattr(bench, "run_pass", fake_run_pass)
    wav = tmp_path / "x.wav"
    wav.touch()
    results = bench.run_benchmark(["small-q5_1", "tiny-q5_1", "base-q5_1"], wav, None, ref_fr="bonjour")
    keys = [r.model_key for r in results]
    assert keys == ["tiny-q5_1", "base-q5_1", "small-q5_1"]  # sorted by size
    assert not results[0].aborted and results[0].wer_fr == 0.0
    assert results[1].aborted and "tué" in results[1].error
    assert results[2].aborted and "sauté" in results[2].error
    # small was never actually run (skipped before any pass)
    assert all(c[0] != "small-q5_1" for c in calls)


def test_run_benchmark_skips_larger_when_rtf_low(monkeypatch, tmp_path):
    def fake_run_pass(
        model_key, wav_path, *, language, translate=False, backend="auto", gpu_device=0, work_dir=None
    ):
        return {"ok": True, "rtf": 1.1, "text": "bonjour", "backend": "CPU (8)"}

    monkeypatch.setattr(bench, "run_pass", fake_run_pass)
    wav = tmp_path / "x.wav"
    wav.touch()
    results = bench.run_benchmark(["tiny-q5_1", "base-q5_1"], wav, None, ref_fr="bonjour")
    assert verdict(results[0]) == "Inadapté (RTF < 1,3)"
    assert results[1].aborted and "sauté" in results[1].error


def test_save_load_roundtrip(tmp_path):
    p = Path(tmp_path) / "bench.json"
    results = [R("tiny-q5_1", 80.0, 0.2, 0.0)]
    bench.save_results(results, stamp="2026-08-02T12:00:00", path=p)
    data = bench.load_results(p)
    assert data["stamp"] == "2026-08-02T12:00:00"
    assert data["recommendation"] == {"model": "tiny-q5_1", "preset": "equilibre"}
    assert data["results"][0]["model_key"] == "tiny-q5_1"
    assert "Recommandé" in data["verdicts"]["tiny-q5_1"]


def test_translate_pass_only_for_translating_models(monkeypatch, tmp_path):
    calls = []

    def fake_run_pass(
        model_key, wav_path, *, language, translate=False, backend="auto", gpu_device=0, work_dir=None
    ):
        calls.append((model_key, translate))
        return {"ok": True, "rtf": 10.0, "text": "hello", "backend": "CPU (8)"}

    monkeypatch.setattr(bench, "run_pass", fake_run_pass)
    wav = tmp_path / "x.wav"
    wav.touch()
    bench.run_benchmark(["large-v3-turbo-q5_0"], wav, None)
    assert (("large-v3-turbo-q5_0", True)) not in calls  # turbo never translates


def test_run_benchmark_propagates_requested_backend(monkeypatch, tmp_path):
    seen: list[str] = []

    def fake_run_pass(
        model_key, wav_path, *, language, translate=False, backend="auto", gpu_device=0, work_dir=None
    ):
        seen.append(backend)
        return {"ok": True, "rtf": 10.0, "text": "bonjour", "backend": "CPU (8 threads)"}

    monkeypatch.setattr(bench, "run_pass", fake_run_pass)
    wav = tmp_path / "x.wav"
    wav.touch()
    results = bench.run_benchmark(["tiny-q5_1", "base-q5_1"], wav, None, ref_fr="bonjour", backend="cpu")
    # le backend demandé est transmis à CHAQUE passe (le backend rapporté ne l'écrase pas)
    assert seen and set(seen) == {"cpu"}
    assert all(r.backend == "CPU (8 threads)" for r in results)
