"""LocalAgreement-2: commit prefix of the last two hypotheses, keep_back withheld."""

import inspect
import time

import numpy as np

from ecoutemoi.constants import PRESETS, TARGET_SR
from ecoutemoi.core.engine import Segment
from ecoutemoi.core.gate import GateEvent
from ecoutemoi.core.streamer import LocalAgreement, Streamer
from ecoutemoi.core.transcript import TranscriptStore


def test_first_hypothesis_commits_nothing():
    la = LocalAgreement(keep_back=1)
    assert la.update(["bonjour", "à", "tous"]) == 0


def test_agreement_commits_prefix_minus_keep_back():
    la = LocalAgreement(keep_back=1)
    la.update(["bonjour", "à", "tous"])
    assert la.update(["bonjour", "à", "tous"]) == 2  # prefix 3, keep_back 1


def test_growing_hypothesis():
    la = LocalAgreement(keep_back=1)
    la.update(["bonjour", "à"])
    assert la.update(["bonjour", "à", "tous", "et"]) == 1  # prefix 2 - 1
    assert la.update(["bonjour", "à", "tous", "et", "toutes"]) == 3  # prefix 4 - 1


def test_last_word_revision_not_committed():
    # Rationale: the word being spoken gets revised on the next decode
    la = LocalAgreement(keep_back=1)
    la.update(["il", "était", "bon"])
    n = la.update(["il", "était", "bonjour"])  # last word revised
    assert n == 1  # only "il" committed ("était" is kept back)
    assert la.update(["il", "était", "bonjour"]) == 2


def test_keep_back_two_stable_preset():
    la = LocalAgreement(keep_back=2)
    la.update(["a", "b", "c", "d", "e"])
    assert la.update(["a", "b", "c", "d", "e"]) == 3


def test_monotonic_never_decreases():
    la = LocalAgreement(keep_back=1)
    la.update(["un", "deux", "trois", "quatre"])
    assert la.update(["un", "deux", "trois", "quatre"]) == 3
    # hypothesis suddenly diverges at word 2: committed count must not go down
    assert la.update(["un", "autre", "chemin"]) == 3


def test_reset_with_committed_offset():
    la = LocalAgreement(keep_back=1)
    la.reset(committed=4)
    assert la.committed == 4
    # first update after reset cannot commit more (no previous hypothesis)
    assert la.update(["a", "b", "c", "d", "e", "f"]) == 4


def test_empty_hypothesis():
    la = LocalAgreement(keep_back=1)
    assert la.update([]) == 0
    assert la.update([]) == 0


# ------------------------------------------------- le non validé ne sort JAMAIS
def test_on_partial_takes_a_single_text_argument():
    """Il n'existe AUCUN canal pour transporter des mots non validés hors du
    streamer : `on_partial` reçoit un texte, point. Verrou contre une
    réintroduction accidentelle d'un second argument « pending »."""
    params = list(inspect.signature(Streamer._emit_partial).parameters)
    assert params == ["self", "committed"]


class _GrowingEngine:
    """Hypothèses qui s'allongent, comme un vrai décodage de fenêtre glissante."""

    def __init__(self, hypotheses: list[str]):
        self._hypotheses = hypotheses
        self._calls = 0
        self.last_decode_ms = 1.0

    def transcribe(self, audio: np.ndarray) -> list[Segment]:
        text = self._hypotheses[min(self._calls, len(self._hypotheses) - 1)]
        self._calls += 1
        return [Segment(0, int(len(audio) * 1000 / TARGET_SR), text)]

    def detected_language(self) -> str | None:
        return None


def test_streamer_never_emits_the_words_still_being_revised():
    """Le mot en cours de prononciation est RETENU (keep_back) : il ne doit jamais
    apparaître dans un partiel, sinon le public le verrait se corriger."""
    engine = _GrowingEngine(
        [
            "le protocole",
            "le protocole réseau",
            "le protocole réseau distribué",
        ]
    )
    emitted: list[str] = []
    streamer = Streamer(
        engine,
        TranscriptStore(None, autosave=False),
        PRESETS["equilibre"],  # keep_back = 1
        realtime=False,
        min_interval_ms=1,
        on_partial=emitted.append,
    )
    n = int(3.0 * TARGET_SR)
    audio = np.full(n, 0.2, dtype=np.float32)
    streamer.start()
    streamer.on_gate_event(GateEvent("speech_start", audio[: TARGET_SR // 2], TARGET_SR // 2))
    for step in range(1, 4):
        streamer.on_gate_event(GateEvent("frame", audio[: TARGET_SR // 2], TARGET_SR // 2 * (step + 1)))
        time.sleep(0.08)
    streamer.stop(timeout=10.0)

    partials = [text for text in emitted if text]
    assert partials, "aucun partiel émis : le test ne prouve rien"
    # Chaque partiel est un PRÉFIXE STRICT de l'hypothèse la plus longue, amputé
    # du mot retenu : « distribué » ne peut pas sortir avant d'être confirmé.
    for text in partials[:-1]:
        assert "le protocole réseau distribué".startswith(text)
        assert not text.endswith("distribué")
