"""Garde-fous anti-répétition, en fonctions pures.

Un sous-titre qui répète la même phrase est pire qu'un sous-titre absent : le
public croit avoir manqué quelque chose et cherche la différence entre les deux
occurrences. Or le flux temps réel a **trois** sources de répétition, de natures
différentes, qu'il faut traiter séparément :

1. **Le recouvrement de fenêtre.** Quand la fenêtre de décodage est coupée, on
   garde 200 ms de l'audio déjà décodé (et la pré-amorce du détecteur de parole
   rejoue 300 ms avant chaque reprise de parole). Whisper redécode donc un bout
   d'audio dont les mots sont DÉJÀ sortis en « finalisé ». Il faut retirer ce
   préfixe — pas un mot, tout le préfixe : 300 ms de français, c'est facilement
   deux ou trois mots.
2. **Le bégaiement du décodeur.** Sur une fenêtre qui finit dans le silence, ou
   sur un signal pauvre, whisper part en boucle : « d'accord d'accord d'accord ».
   La longueur du motif répété n'a rien de fixe — un mot, un groupe de trois,
   une phrase entière.
3. **Le segment servi deux fois.** Le même énoncé ressort en deux segments dont
   le texte est identique (horodatages différents).

Ces fonctions ne connaissent ni Qt, ni whisper, ni le streamer : elles prennent
des mots normalisés et rendent des indices. C'est ce qui les rend testables une
par une, ce qui compte pour du code dont chaque erreur se voit à l'écran.
"""

from __future__ import annotations

import re

from ecoutemoi.constants import HALLUCINATION_BLACKLIST

_NON_WORD_RE = re.compile(r"[\W_]+", re.UNICODE)
_HAS_WORD_RE = re.compile(r"\w", re.UNICODE)

# Longueur maximale d'un motif répété que l'on accepte de détecter. Au-delà de
# huit mots, deux groupes identiques d'affilée ne sont plus un bégaiement de
# décodeur mais peut-être une vraie reprise de l'orateur — on n'y touche pas.
MAX_LOOP_NGRAM = 8

# Nombre de mots déjà finalisés gardés en mémoire pour reconnaître le
# recouvrement. 12 mots couvrent largement les 500 ms de recouvrement possible
# (200 ms de coupe + 300 ms de pré-amorce) même à débit soutenu.
OVERLAP_MEMORY_WORDS = 12


def norm_token(word: str) -> str:
    """Mot réduit à sa forme comparable : minuscules, sans ponctuation."""
    return _NON_WORD_RE.sub("", word.lower())


def words_of(text: str) -> list[str]:
    """Mots normalisés, sans les jetons vidés par la normalisation (« ! », « — »)."""
    return [n for n in (norm_token(w) for w in text.split()) if n]


def repeated_tail_length(
    norms: list[str], max_ngram: int = MAX_LOOP_NGRAM, *, allow_double: bool = False
) -> int:
    """Longueur du motif final immédiatement répété, 0 s'il n'y en a pas.

    On cherche le motif le PLUS LONG d'abord : « a b a b » doit se lire comme
    « a b » répété, pas comme « b » répété (ce qui laisserait « a b a »).

    Cas particulier du mot seul : « nous nous », « très très », « vous vous »
    sont du français, pas un bégaiement de décodeur. Un mot doublé n'est donc
    retenu qu'à partir de TROIS occurrences — sauf `allow_double`, utilisé quand
    on a déjà établi qu'il y avait boucle et qu'on finit de la dérouler.
    """
    for k in range(min(max_ngram, len(norms) // 2), 0, -1):
        if norms[-k:] != norms[-2 * k : -k]:
            continue
        if k == 1 and not allow_double and not (len(norms) >= 3 and norms[-3] == norms[-1]):
            continue
        return k
    return 0


def collapse_repeats(norms: list[str], max_ngram: int = MAX_LOOP_NGRAM) -> int:
    """Nombre de mots à retirer en fin de liste pour tuer les répétitions.

    Appliqué en boucle : « x y x y x y » perd deux fois « x y ». La liste rendue
    par l'appelant garde donc UNE occurrence du motif, celle que l'orateur a
    réellement prononcée.
    """
    end = len(norms)
    relaxed = False
    while True:
        k = repeated_tail_length(norms[:end], max_ngram, allow_double=relaxed)
        if k == 0:
            return len(norms) - end
        # Une coupe a eu lieu : la boucle est établie, on peut la dérouler
        # jusqu'à une seule occurrence — y compris quand il ne reste qu'un mot
        # doublé (« oui oui oui oui » vu comme « oui oui » répété).
        relaxed = True
        end -= k


def overlap_length(previous: list[str], words: list[str], max_overlap: int = OVERLAP_MEMORY_WORDS) -> int:
    """Longueur du préfixe de `words` qui redit la fin de `previous`.

    Autrement dit : le plus long suffixe de `previous` qui est aussi un préfixe
    de `words`. Le plus LONG, car c'est celui qui correspond à la réalité — le
    recouvrement audio est continu, il ne saute pas de mots.
    """
    limit = min(len(previous), len(words), max_overlap)
    for k in range(limit, 0, -1):
        if previous[-k:] == words[:k]:
            return k
    return 0


def is_self_repeating(text: str, min_repeats: int = 3) -> bool:
    """Texte fait d'un même groupe de mots répété au moins `min_repeats` fois.

    Sert au niveau du SEGMENT, avant tout découpage en mots : un segment entier
    du genre « merci merci merci merci » n'a rien à apporter et son horodatage
    est de toute façon faux.
    """
    norms = words_of(text)
    if len(norms) < min_repeats:
        return False
    for k in range(1, len(norms) // min_repeats + 1):
        if len(norms) % k:
            continue
        pattern = norms[:k]
        if all(norms[i : i + k] == pattern for i in range(0, len(norms), k)):
            return True
    return False


def same_text(a: str, b: str) -> bool:
    """Deux textes identiques à la ponctuation et à la casse près."""
    return words_of(a) == words_of(b)


def is_hallucination(text: str) -> bool:
    """Segment à jeter : générique de sous-titrage, bruit typographique, boucle.

    Les modèles Whisper ont appris sur des sous-titres du web : sur du silence ou
    du signal pauvre, ils produisent les formules qui terminaient ces
    sous-titres (« Sous-titres réalisés par… », « Merci d'avoir regardé »). Elles
    n'ont jamais été prononcées.
    """
    flat = " ".join(text.lower().split())
    if not _HAS_WORD_RE.search(flat):
        return True  # « ♪♪♪ », « ... », « — — — » : aucun mot
    if any(b in flat for b in HALLUCINATION_BLACKLIST):
        return True
    # « merci merci merci merci » : un segment fait d'un motif répété n'apporte
    # rien, et son horodatage est de toute façon faux.
    return is_self_repeating(text)


__all__ = [
    "MAX_LOOP_NGRAM",
    "OVERLAP_MEMORY_WORDS",
    "collapse_repeats",
    "is_hallucination",
    "is_self_repeating",
    "norm_token",
    "overlap_length",
    "repeated_tail_length",
    "same_text",
    "words_of",
]
