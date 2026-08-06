"""Transcription de fichiers : découpage, recollage, progression, annulation.

Tout est piloté avec un moteur factice : ces tests vérifient la mécanique du mode
fichier, pas la qualité de whisper — laquelle se mesure au benchmark.
"""

from pathlib import Path

import numpy as np
import pytest

from ecoutemoi.constants import FILE_CHUNK_S, SILENCE_FRAME_MS, TARGET_SR
from ecoutemoi.core import filejob
from ecoutemoi.core.engine import Segment


class FakeEngine:
    """Un segment par passe, texte donné à l'avance."""

    def __init__(self, texts: list[str]):
        self.texts = list(texts)
        self.calls: list[float] = []

    def transcribe(self, audio):
        self.calls.append(audio.size / TARGET_SR)
        text = self.texts.pop(0) if self.texts else "suite"
        return [Segment(0, max(1, audio.size * 1000 // TARGET_SR), text)]

    def detected_language(self):
        return "fr"


def speech(seconds: float, *, quiet_every: float = 1.0, level: float = 0.2) -> np.ndarray:
    """Bruit ponctué de silences réguliers — de quoi offrir des points de coupe."""
    rng = np.random.default_rng(3)
    audio = (rng.standard_normal(int(seconds * TARGET_SR)) * level).astype(np.float32)
    step = int(quiet_every * TARGET_SR)
    for start in range(step, audio.size, step):
        audio[start - int(0.15 * TARGET_SR) : start] = 0.0
    return audio


def blocks_of(audio: np.ndarray, seconds: float = 8.0):
    step = int(seconds * TARGET_SR)
    return [audio[i : i + step] for i in range(0, audio.size, step)]


def test_plan_cut_lands_on_a_silence():
    audio = speech(40.0)
    cut = filejob.plan_cut(audio)
    # La coupe suit la tranche la plus calme : les 20 ms qui la précèdent sont muettes
    frame = SILENCE_FRAME_MS * TARGET_SR // 1000
    assert float(np.abs(audio[cut - frame : cut]).max()) < 1e-6
    assert abs(cut / TARGET_SR - FILE_CHUNK_S) <= 4.0  # proche de la cible


def test_short_audio_is_decoded_in_one_pass():
    engine = FakeEngine(["une seule passe"])
    segments = filejob.transcribe_blocks(engine, blocks_of(speech(12.0)))
    assert len(engine.calls) == 1
    assert [s.text for s in segments] == ["une seule passe"]


def test_long_audio_is_split_and_timestamps_advance():
    engine = FakeEngine(["première partie", "deuxième partie", "troisième partie"])
    segments = filejob.transcribe_blocks(engine, blocks_of(speech(70.0)))
    assert len(engine.calls) >= 3
    assert [s.t0_ms for s in segments] == sorted(s.t0_ms for s in segments)
    assert segments[0].t0_ms == 0
    assert segments[-1].t1_ms > 40_000  # on a bien couvert la fin du fichier


def test_overlap_between_passes_is_not_duplicated():
    """Le recouvrement existe pour ne pas perdre un mot ; il ne doit pas le dire deux fois."""
    engine = FakeEngine(["bienvenue dans cette conférence", "cette conférence parle de stockage", "et voilà"])
    segments = filejob.transcribe_blocks(engine, blocks_of(speech(70.0)))
    text = " ".join(s.text for s in segments)
    assert text.count("cette conférence") == 1
    assert "parle de stockage" in text


def test_decoder_loops_are_collapsed():
    """Le bégaiement en fin de segment est coupé, le reste du texte est gardé."""
    engine = FakeEngine(["voici la fin merci merci merci"])
    segments = filejob.transcribe_blocks(engine, blocks_of(speech(12.0)))
    assert [s.text for s in segments] == ["voici la fin merci"]


def test_segment_entirely_made_of_a_loop_is_dropped():
    engine = FakeEngine(["merci beaucoup merci beaucoup merci beaucoup"])
    assert filejob.transcribe_blocks(engine, blocks_of(speech(12.0))) == []


def test_hallucinated_segment_is_dropped():
    engine = FakeEngine(["Sous-titres réalisés par la communauté d'Amara.org"])
    segments = filejob.transcribe_blocks(engine, blocks_of(speech(12.0)))
    assert segments == []


def test_silent_file_produces_nothing():
    engine = FakeEngine(["ceci ne doit pas sortir"])
    silence = np.zeros(int(30 * TARGET_SR), dtype=np.float32)
    segments = filejob.transcribe_blocks(engine, blocks_of(silence))
    assert engine.calls == [] and segments == []


def test_progress_is_reported_per_pass():
    engine = FakeEngine([])
    seen: list[float] = []
    filejob.transcribe_blocks(
        engine, blocks_of(speech(70.0)), on_chunk=lambda segments, done_s: seen.append(done_s)
    )
    assert len(seen) >= 3
    assert seen == sorted(seen)  # la progression ne recule jamais
    assert seen[-1] == pytest.approx(70.0, abs=1.0)


def test_cancellation_raises_cancelled():
    engine = FakeEngine([])
    with pytest.raises(filejob.Cancelled):
        filejob.transcribe_blocks(engine, blocks_of(speech(70.0)), should_stop=lambda: True)


def test_transcribe_file_reports_errors_instead_of_raising(tmp_path):
    """Dans un lot de trente fichiers, un fichier abîmé n'emporte pas les autres."""
    broken = tmp_path / "cassé.wav"
    broken.write_bytes(b"pas un wav")
    result = filejob.transcribe_file(FakeEngine([]), broken)
    assert not result.ok
    assert result.error
    assert result.segments == []


def test_transcribe_file_end_to_end(tmp_path):
    sf = pytest.importorskip("soundfile")
    path = tmp_path / "voix.wav"
    sf.write(str(path), speech(30.0), TARGET_SR)
    engine = FakeEngine(["bonjour à tous", "et bienvenue"])
    stages: list[str] = []
    result = filejob.transcribe_file(engine, path, on_progress=lambda p: stages.append(p.stage))
    assert result.ok
    assert result.info is not None and result.info.backend == "libsndfile"
    assert result.word_count == len(" ".join(s.text for s in result.segments).split())
    assert result.speed is not None and result.speed > 0
    assert stages[0] == "ouverture" and "transcription" in stages


def test_transcribe_many_keeps_the_batch_going(tmp_path):
    sf = pytest.importorskip("soundfile")
    good = tmp_path / "bon.wav"
    sf.write(str(good), speech(10.0), TARGET_SR)
    bad = tmp_path / "mauvais.wav"
    bad.write_bytes(b"nope")
    results = list(filejob.transcribe_many(FakeEngine([]), [bad, good, bad]))
    assert [r.ok for r in results] == [False, True, False]
    assert [Path(r.path).name for r in results] == ["mauvais.wav", "bon.wav", "mauvais.wav"]


def test_progress_ratio_is_none_without_duration():
    progress = filejob.Progress(path=Path("a.mp3"), index=0, count=1, stage="transcription")
    assert progress.ratio is None
    assert filejob.Progress(Path("a"), 0, 1, "x", done_s=5.0, total_s=10.0).ratio == 0.5
    assert filejob.Progress(Path("a"), 0, 1, "x", done_s=99.0, total_s=10.0).ratio == 1.0
