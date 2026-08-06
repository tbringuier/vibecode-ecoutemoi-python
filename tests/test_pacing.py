"""Débit de lecture constant : lissage des rafales, plafond de retard, corrections.

L'horloge est injectée : aucun `sleep`, et les assertions portent sur des valeurs
exactes plutôt que sur des marges de tolérance.
"""

from __future__ import annotations

import pytest

from ecoutemoi.core.pacing import CHARS_PER_WORD, TextPacer, chars_per_second


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now


@pytest.fixture
def clock():
    return FakeClock()


def test_chars_per_second_follows_subtitle_convention():
    assert chars_per_second(180) == pytest.approx(180 * CHARS_PER_WORD / 60)
    assert chars_per_second(60) == pytest.approx(6.0)
    # Bornes : une valeur absurde ne doit pas produire un débit absurde
    assert chars_per_second(0) == chars_per_second(60)
    assert chars_per_second(100_000) == chars_per_second(1200)


def test_burst_is_spread_over_time(clock):
    """Huit mots validés d'un coup ne doivent PAS s'afficher d'un coup."""
    pacer = TextPacer(wpm=180, max_lag_s=10.0, clock=clock)
    pacer.set_live("un deux trois quatre cinq six sept huit")
    assert pacer.text() == ""  # rien libéré avant le premier tick

    clock.advance(0.5)  # 18 c/s x 0.5 s = 9 caractères
    pacer.tick()
    first = pacer.text().split()
    assert 0 < len(first) < 8, f"la rafale est sortie d'un bloc : {first}"

    clock.advance(5.0)  # largement de quoi tout libérer
    pacer.tick()
    assert pacer.text() == "un deux trois quatre cinq six sept huit"
    assert pacer.pending_words() == 0


def test_release_order_and_rate_are_exact(clock):
    """Le budget se compte en caractères réels, espace compris."""
    pacer = TextPacer(wpm=60, max_lag_s=100.0, clock=clock)  # 6 c/s
    pacer.set_live("abc de")  # coûts : 4 puis 3 caractères
    clock.advance(0.5)  # 3 caractères : insuffisant pour « abc » (4)
    pacer.tick()
    assert pacer.text() == ""
    clock.advance(0.2)  # cumul 4.2 : « abc » passe
    pacer.tick()
    assert pacer.text() == "abc"
    clock.advance(0.5)  # 0.2 restant + 3 = 3.2 : « de » (3) passe
    pacer.tick()
    assert pacer.text() == "abc de"


def test_lag_is_capped_by_catch_up(clock):
    """Un orateur trop rapide ne doit pas creuser un retard sans fin."""
    pacer = TextPacer(wpm=180, max_lag_s=1.0, clock=clock)
    words = " ".join(f"mot{i:03d}" for i in range(200))  # 7 caractères chacun
    pacer.set_live(words)
    for _ in range(5):
        clock.advance(0.2)
        pacer.tick()
    # Le rattrapage a ramené l'attente sous le plafond (à un mot près)
    assert pacer.lag_s() <= 1.0 + 7 / pacer.rate()
    assert pacer.pending_words() > 0  # …sans tout lâcher d'un coup pour autant


def test_disabled_releases_everything_immediately(clock):
    pacer = TextPacer(wpm=180, enabled=False, clock=clock)
    pacer.set_live("un deux trois quatre cinq")
    assert pacer.tick() is True
    assert pacer.text() == "un deux trois quatre cinq"
    assert pacer.lag_s() == 0.0


def test_toggling_back_on_does_not_dump_backlog(clock):
    pacer = TextPacer(wpm=180, max_lag_s=10.0, enabled=False, clock=clock)
    clock.advance(30.0)  # long moment sans cadencement
    pacer.configure(enabled=True)
    pacer.set_live("un deux trois quatre cinq six sept huit neuf dix")
    clock.advance(0.1)
    pacer.tick()
    # La dette de temps accumulée pendant la pause ne doit pas être encaissée
    assert 0 <= len(pacer.text().split()) <= 2


def test_finalized_text_replaces_live_tail(clock):
    """push_final() remplace la queue en cours sans dupliquer les mots."""
    pacer = TextPacer(wpm=180, enabled=False, clock=clock)
    pacer.set_live("bonjour à tous")
    pacer.tick()
    pacer.push_final("bonjour à tous.")
    pacer.tick()
    assert pacer.text() == "bonjour à tous."
    pacer.set_live("la suite")
    pacer.tick()
    assert pacer.text() == "bonjour à tous. la suite"


def test_shorter_correction_does_not_overrun(clock):
    """Le filtre anti-hallucination peut RACCOURCIR le texte déjà libéré."""
    pacer = TextPacer(wpm=180, enabled=False, clock=clock)
    pacer.set_live("un deux trois quatre cinq")
    pacer.tick()
    assert pacer.text() == "un deux trois quatre cinq"
    pacer.push_final("un deux")  # segment finalisé plus court
    pacer.tick()
    assert pacer.text() == "un deux"
    assert pacer.pending_words() == 0


def test_history_rolls_and_keeps_alignment(clock):
    """L'historique roule ; l'index d'affichage doit suivre, pas dériver."""
    pacer = TextPacer(wpm=180, enabled=False, history_words=10, clock=clock)
    for i in range(20):
        pacer.push_final(f"mot{i:02d}")
        pacer.tick()
    shown = pacer.text().split()
    assert len(shown) == 10
    assert shown[0] == "mot10" and shown[-1] == "mot19"
    assert pacer.pending_words() == 0


def test_clear_resets_everything(clock):
    pacer = TextPacer(wpm=180, clock=clock)
    pacer.set_live("un deux trois")
    clock.advance(5.0)
    pacer.tick()
    pacer.clear()
    assert pacer.text() == ""
    assert pacer.pending_words() == 0
    assert pacer.lag_s() == 0.0


def test_idle_does_not_bank_budget(clock):
    """Sans texte en attente, le budget ne doit pas s'accumuler indéfiniment,
    sinon la rafale suivante sortirait d'un bloc."""
    pacer = TextPacer(wpm=180, max_lag_s=10.0, clock=clock)
    clock.advance(60.0)  # une minute de silence
    pacer.tick()
    pacer.set_live("un deux trois quatre cinq six sept huit neuf dix")
    clock.advance(0.05)
    pacer.tick()
    assert len(pacer.text().split()) <= 2
