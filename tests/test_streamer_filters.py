"""Loop guard and word extraction of the streamer."""

from ecoutemoi.constants import PRESETS
from ecoutemoi.core.engine import Segment
from ecoutemoi.core.streamer import Streamer, Word
from ecoutemoi.core.textguard import norm_token
from ecoutemoi.core.transcript import TranscriptStore


def w(text: str, t0=0, t1=100) -> Word:
    return Word(t0, t1, text, norm_token(text))


def _make_streamer() -> Streamer:
    store = TranscriptStore(None, autosave=False)
    return Streamer(engine=None, store=store, preset=PRESETS["equilibre"], realtime=False)


def test_loop_cut_words():
    words = [w(t) for t in ["a", "b", "c", "d", "e", "a", "b", "c", "d", "e"]]
    cut = Streamer._loop_cut_words(words)
    assert [x.text for x in cut] == ["a", "b", "c", "d", "e"]
    # triple repetition collapses too
    words = [w(t) for t in ("x y z u v " * 3).split()]
    cut = Streamer._loop_cut_words(words)
    assert [x.text for x in cut] == ["x", "y", "z", "u", "v"]
    # no repetition -> untouched
    words = [w(t) for t in ["un", "deux", "trois", "quatre", "cinq", "six"]]
    assert Streamer._loop_cut_words(words) == words


def test_loop_guard_rejects_repeated_commit():
    st = _make_streamer()
    words = [w(t) for t in ["a", "b", "c", "d", "e", "a", "b", "c", "d", "e"]]
    assert st._loop_guard(words, 10) == 0  # nothing committed yet -> reject to 0
    st._committed = words[:5]
    assert st._loop_guard(words, 10) == 5  # keep previous commit level


def test_extract_words_interpolates_times():
    st = _make_streamer()
    st._win_t0_ms = 1000
    segs = [Segment(t0_ms=0, t1_ms=1000, text="un deux trois quatre")]
    words = st._extract_words(segs)
    assert [x.text for x in words] == ["un", "deux", "trois", "quatre"]
    assert words[0].t0_ms == 1000
    assert words[-1].t1_ms == 2000
    assert words[1].t0_ms == 1250  # linear interpolation
    assert all(x.norm == x.text for x in words)


def test_filter_segments_no_speech_prob():
    st = _make_streamer()
    segs = [
        Segment(0, 100, "vraie parole", no_speech_prob=0.1),
        Segment(100, 200, "hallucination probable", no_speech_prob=0.9),
        Segment(200, 300, "sans probabilité", no_speech_prob=None),
    ]
    kept = st._filter_segments(segs)
    assert [s.text for s in kept] == ["vraie parole", "sans probabilité"]


def test_filter_segments_blacklist():
    st = _make_streamer()
    segs = [
        Segment(0, 100, "Bonjour à tous"),
        Segment(100, 200, "Sous-titres réalisés par la communauté"),
    ]
    kept = st._filter_segments(segs)
    assert [s.text for s in kept] == ["Bonjour à tous"]
