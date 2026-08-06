"""WER normalization + Levenshtein."""

from ecoutemoi.constants import CALIBRATION_TEXT_EN, CALIBRATION_TEXT_FR
from ecoutemoi.core.wer import levenshtein, normalize_words, wer


def test_identical_is_zero():
    assert wer(CALIBRATION_TEXT_FR, CALIBRATION_TEXT_FR, "fr") == 0.0
    assert wer(CALIBRATION_TEXT_EN, CALIBRATION_TEXT_EN, "en") == 0.0


def test_case_and_punctuation_ignored():
    assert wer("Bonjour, à tous !", "bonjour à TOUS", "fr") == 0.0


def test_digits_match_words_fr():
    assert wer("quarante-deux téraoctets", "42 téraoctets", "fr") == 0.0
    assert wer("trois serveurs, douze disques", "3 serveurs 12 disques", "fr") == 0.0


def test_digits_match_words_en():
    assert wer("forty-two terabytes", "42 terabytes", "en") == 0.0
    assert wer("three servers, twelve disks", "3 servers 12 disks", "en") == 0.0


def test_apostrophes_and_hyphens_split():
    assert normalize_words("d'avoir dix-sept", "fr") == ["d", "avoir", "dix", "sept"]
    assert wer("l'idée", "l’idée", "fr") == 0.0  # unicode apostrophe


def test_substitution_counts():
    # 1 substitution over 4 reference words
    assert wer("le chat mange la souris", "le chien mange la souris", "fr") == 1 / 5


def test_levenshtein_basics():
    assert levenshtein([], []) == 0
    assert levenshtein(["a"], []) == 1
    assert levenshtein([], ["a", "b"]) == 2
    assert levenshtein(["a", "b", "c"], ["a", "x", "c"]) == 1
    assert levenshtein(["a", "b"], ["b", "a"]) == 2


def test_empty_reference():
    assert wer("", "", "fr") == 0.0
    assert wer("", "quelque chose", "fr") == 1.0


def test_insertion_and_deletion():
    assert wer("un deux trois", "un deux", "fr") == 1 / 3
    assert wer("un deux", "un deux trois", "fr") == 1 / 2
