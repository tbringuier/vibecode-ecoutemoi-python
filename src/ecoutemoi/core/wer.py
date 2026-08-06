"""Home-grown WER: normalization + word-level Levenshtein. Zero dependencies."""

from __future__ import annotations

import re

_PUNCT_RE = re.compile(r"[^\w\s]|_", re.UNICODE)

# Small digit->words tables (FR/EN) so "42" == "quarante-deux" == "forty-two".
_NUM_FR = {
    "0": "zéro", "1": "un", "2": "deux", "3": "trois", "4": "quatre", "5": "cinq",
    "6": "six", "7": "sept", "8": "huit", "9": "neuf", "10": "dix", "11": "onze",
    "12": "douze", "13": "treize", "14": "quatorze", "15": "quinze", "16": "seize",
    "17": "dix sept", "18": "dix huit", "19": "dix neuf", "20": "vingt",
    "30": "trente", "40": "quarante", "42": "quarante deux", "50": "cinquante",
    "60": "soixante", "100": "cent",
}  # fmt: skip
_NUM_EN = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten", "11": "eleven",
    "12": "twelve", "13": "thirteen", "14": "fourteen", "15": "fifteen", "16": "sixteen",
    "17": "seventeen", "18": "eighteen", "19": "nineteen", "20": "twenty",
    "30": "thirty", "40": "forty", "42": "forty two", "50": "fifty",
    "60": "sixty", "100": "one hundred",
}  # fmt: skip


def normalize_words(text: str, lang: str = "fr") -> list[str]:
    """Lowercase, drop punctuation, split hyphens/apostrophes, map digits to words."""
    text = text.lower()
    for ch in ("-", "‐", "–", "—", "'", "’"):
        text = text.replace(ch, " ")
    text = _PUNCT_RE.sub(" ", text)
    table = _NUM_FR if lang == "fr" else _NUM_EN
    words: list[str] = []
    for w in text.split():
        words.extend(table.get(w, w).split())
    return words


def levenshtein(a: list[str], b: list[str]) -> int:
    """Word-level edit distance, two-row DP."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, wb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (wa != wb))
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str, lang: str = "fr") -> float:
    """Word error rate of `hypothesis` against `reference` (0.0 = perfect)."""
    ref = normalize_words(reference, lang)
    hyp = normalize_words(hypothesis, lang)
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)
