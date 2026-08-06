"""Non-régression : le texte ne doit JAMAIS sortir deux fois.

Trois mécanismes du direct rejouent de l'audio déjà décodé ou invitent le modèle
à bégayer. Chacun a produit, en vrai, des sous-titres qui se répétaient :

1. la coupe de fenêtre garde 200 ms de recouvrement ;
2. la pré-amorce du détecteur de parole rejoue 300 ms à chaque reprise de parole,
   APRÈS la remise à zéro de l'énoncé ;
3. une fenêtre qui finit dans le silence fait boucler le décodeur.

Ces tests pilotent le `Streamer` avec un moteur factice : ils vérifient ce qui
sort côté transcript ET côté affichage, sans modèle ni GPU.
"""

import numpy as np

from ecoutemoi.constants import PRESETS, TARGET_SR
from ecoutemoi.core.engine import Segment
from ecoutemoi.core.streamer import Streamer
from ecoutemoi.core.transcript import TranscriptStore


class FakeEngine:
    """Rend les hypothèses fournies, une par appel de `transcribe`."""

    def __init__(self, hypotheses: list[str]):
        self.hypotheses = list(hypotheses)
        self.calls: list[int] = []
        self.last_decode_ms = 10.0

    def transcribe(self, audio):
        self.calls.append(int(audio.size))
        text = self.hypotheses.pop(0) if self.hypotheses else ""
        span = max(1, audio.size * 1000 // TARGET_SR)
        return [Segment(0, span, text)] if text else []

    def detected_language(self):
        return "fr"


def make_streamer(engine, **kwargs):
    store = TranscriptStore(None, autosave=False)
    finalized: list[str] = []
    partials: list[str] = []
    streamer = Streamer(
        engine,
        store,
        PRESETS["equilibre"],
        realtime=False,
        on_finalized=lambda seg: finalized.append(seg.text),
        on_partial=partials.append,
        **kwargs,
    )
    return streamer, store, finalized, partials


def voice(seconds: float, level: float = 0.2) -> np.ndarray:
    """Signal audible (bruit) : de quoi passer les seuils de silence."""
    rng = np.random.default_rng(7)
    return (rng.standard_normal(int(seconds * TARGET_SR)) * level).astype(np.float32)


def test_final_segments_come_from_the_guarded_words():
    """Le transcript passe par les mêmes filtres que l'affichage.

    Régression : les segments bruts de whisper partaient dans le transcript
    pendant que l'affichage recevait la version dédoublonnée — le fichier gardait
    donc les doublons, et l'historique affiché aussi.
    """
    engine = FakeEngine(["bonjour à tous merci merci merci"])
    streamer, store, finalized, partials = make_streamer(engine)
    streamer._win_t0_ms = 0
    streamer._chunks = [voice(3.0)]
    streamer._win_samples = streamer._chunks[0].size
    streamer._active = True
    streamer._decode_final()

    assert finalized == ["bonjour à tous merci"]
    assert [s.text for s in store.segments] == ["bonjour à tous merci"]
    assert partials[-1] == "bonjour à tous merci"


def test_preroll_overlap_is_not_said_twice():
    """La pré-amorce du gate rejoue la fin de l'énoncé précédent."""
    engine = FakeEngine(
        [
            "bienvenue dans cette conférence",
            "cette conférence parle de stockage",  # 2 mots rejoués par la pré-amorce
        ]
    )
    streamer, store, finalized, _ = make_streamer(engine)
    for _ in range(2):
        streamer._active = True
        streamer._chunks = [voice(3.0)]
        streamer._win_samples = streamer._chunks[0].size
        streamer._decode_final()

    assert finalized == ["bienvenue dans cette conférence", "parle de stockage"]
    assert " ".join(s.text for s in store.segments) == ("bienvenue dans cette conférence parle de stockage")


def test_duplicated_segment_is_dropped():
    """Whisper resservant le même énoncé en deux segments."""
    engine = FakeEngine([])
    streamer, _store, _finalized, _partials = make_streamer(engine)
    segments = [
        Segment(0, 1000, "bonjour à tous"),
        Segment(1000, 2000, "Bonjour, à tous !"),  # même texte, autre horodatage
        Segment(2000, 3000, "on commence"),
    ]
    kept = streamer._filter_segments(segments)
    assert [s.text for s in kept] == ["bonjour à tous", "on commence"]


def test_silence_only_window_is_not_decoded():
    """Une fenêtre muette n'est pas donnée au modèle : il la remplirait."""
    engine = FakeEngine(["ceci ne doit jamais sortir"])
    streamer, store, finalized, _ = make_streamer(engine)
    streamer._active = True
    streamer._chunks = [np.zeros(int(2.0 * TARGET_SR), dtype=np.float32)]
    streamer._win_samples = streamer._chunks[0].size
    streamer._decode_final()

    assert engine.calls == []
    assert finalized == [] and store.segments == []


def test_trailing_silence_is_trimmed_before_decoding():
    """La traîne muette qui a déclenché la fin d'énoncé ne part pas au modèle."""
    engine = FakeEngine(["deux mots"])
    streamer, _store, _finalized, _ = make_streamer(engine)
    streamer._active = True
    audio = np.concatenate([voice(1.5), np.zeros(int(2.0 * TARGET_SR), dtype=np.float32)])
    streamer._chunks = [audio]
    streamer._win_samples = audio.size
    streamer._decode_final()

    assert len(engine.calls) == 1
    decoded_s = engine.calls[0] / TARGET_SR
    assert 1.5 <= decoded_s <= 1.9  # parole + marge, pas les 2 s de silence


def test_loop_guard_refuses_a_repeated_commit():
    engine = FakeEngine([])
    streamer, _store, _f, _p = make_streamer(engine)
    words = streamer._extract_words([Segment(0, 1000, "a b c a b c")])
    assert streamer._loop_guard(words, len(words)) == 0  # rien n'était encore validé
    streamer._committed = words[:3]
    assert streamer._loop_guard(words, len(words)) == 3  # on garde l'acquis
