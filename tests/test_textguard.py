"""Garde-fous anti-répétition : la seule chose qui distingue un sous-titre
utilisable d'un sous-titre qui bégaie."""

from ecoutemoi.core.textguard import (
    collapse_repeats,
    is_hallucination,
    is_self_repeating,
    norm_token,
    overlap_length,
    repeated_tail_length,
    same_text,
)


def n(text: str) -> list[str]:
    return [norm_token(w) for w in text.split()]


def test_norm_token():
    assert norm_token("Bonjour,") == "bonjour"
    assert norm_token("Écoute !") == "écoute"
    assert norm_token("(rires)") == "rires"
    assert norm_token("—") == ""


def test_repeated_tail_finds_longest_pattern():
    # « a b a b » est « a b » répété, pas « b » répété
    assert repeated_tail_length(n("a b a b")) == 2
    assert repeated_tail_length(n("un deux trois un deux trois")) == 3
    assert repeated_tail_length(n("un deux trois quatre")) == 0


def test_single_word_needs_three_occurrences():
    """« nous nous », « très très » : du français, pas un bégaiement."""
    assert repeated_tail_length(n("nous nous")) == 0
    assert repeated_tail_length(n("il faut que nous nous")) == 0
    assert repeated_tail_length(n("d'accord d'accord d'accord")) == 1


def test_collapse_keeps_one_occurrence():
    assert collapse_repeats(n("x y z x y z x y z")) == 6  # garde un seul « x y z »
    assert collapse_repeats(n("oui oui oui oui")) == 3  # garde un seul « oui »
    assert collapse_repeats(n("bonjour à tous")) == 0
    # Un doublé légitime en fin de phrase est préservé
    assert collapse_repeats(n("il faut que nous nous")) == 0


def test_overlap_length_matches_longest_join():
    previous = n("bienvenue dans cette conférence")
    assert overlap_length(previous, n("cette conférence aujourd'hui nous")) == 2
    assert overlap_length(previous, n("aujourd'hui nous parlerons")) == 0
    # Le recouvrement est borné : une répétition lointaine n'est pas un doublon
    assert overlap_length(n("x a b"), n("a b c"), max_overlap=2) == 2


def test_overlap_ignores_empty_memory():
    assert overlap_length([], n("bonjour à tous")) == 0
    assert overlap_length(n("bonjour"), []) == 0


def test_is_self_repeating():
    assert is_self_repeating("merci merci merci")
    assert is_self_repeating("et voilà et voilà et voilà")
    assert not is_self_repeating("merci beaucoup")
    assert not is_self_repeating("oui oui")  # deux occurrences seulement


def test_is_hallucination_covers_the_three_cases():
    assert is_hallucination("Sous-titres réalisés par la communauté d'Amara.org")
    assert is_hallucination("♪♪♪")
    assert is_hallucination("merci merci merci merci")
    assert not is_hallucination("Bonjour à toutes et à tous.")


def test_same_text_ignores_case_and_punctuation():
    assert same_text("Bonjour, à tous !", "bonjour à tous")
    assert not same_text("Bonjour à tous", "Bonjour à toutes")
