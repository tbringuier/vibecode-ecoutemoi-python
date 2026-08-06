"""Débit de lecture constant : les normes de sous-titrage appliquées au flot.

Whisper valide les mots par RAFALES. Une fenêtre décodée confirme huit mots d'un
coup, la suivante aucun. À l'écran cela donne deux lignes qui surgissent d'un
bloc, puis un silence, puis un nouveau bloc : le texte est juste, mais le public
n'arrive pas à le lire. C'est le défaut de tous les sous-titres automatiques
« bruts ».

Les normes du métier (BBC Subtitle Guidelines, EBU-TT-D) bornent deux choses :

- la **vitesse de lecture** — 160 à 180 mots/min pour un public qui suit aussi un
  orateur et des slides ; au-delà, il décroche ;
- la **largeur de ligne** — 37 à 42 caractères, au-delà l'œil perd la ligne au
  retour chariot. (Cette seconde borne est appliquée au rendu, voir
  `SubtitleStyle.max_chars_per_line`.)

`TextPacer` applique la première. Il connaît la CIBLE (tout ce que le moteur a
validé) et ne libère les mots qu'à débit borné, ce qui lisse les rafales en un
défilement régulier.

Le retard est PLAFONNÉ : passé `max_lag_s` de texte en attente, le débit
s'accélère juste assez pour revenir sous le plafond. Sans ce garde-fou, un orateur
rapide creuserait un décalage qui grandit sans fin et les sous-titres finiraient
par parler d'autre chose que la diapositive affichée — un pic de vitesse ponctuel
est un moindre mal.

Convention de mesure : un « mot » de sous-titrage vaut 6 caractères (5 lettres +
1 espace), c'est ainsi que les normes convertissent mots/min en caractères/s. On
facture néanmoins chaque mot à sa longueur réelle, pour qu'une suite de mots longs
ne défile pas plus vite que ce que l'œil peut suivre.
"""

from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)

CHARS_PER_WORD = 6.0  # convention de sous-titrage (mots/min -> caractères/s)
MIN_WPM = 60
MAX_WPM = 1200  # au-delà, c'est du « pas de limite » déguisé
DEFAULT_HISTORY_WORDS = 80  # aligné sur SubtitleView.MAX_HISTORY_WORDS


def chars_per_second(wpm: int) -> float:
    return max(MIN_WPM, min(MAX_WPM, int(wpm))) * CHARS_PER_WORD / 60.0


class TextPacer:
    """Lisse le flot validé en un défilement à débit constant.

    Entrée : les mêmes évènements que la vue — `push_final()` pour un segment
    définitif, `set_live()` pour la queue de l'énoncé en cours (monotone, elle
    remplace la précédente). Sortie : `text()`, ce qui doit être affiché MAINTENANT.
    """

    def __init__(
        self,
        wpm: int = 180,
        max_lag_s: float = 2.5,
        *,
        enabled: bool = True,
        history_words: int = DEFAULT_HISTORY_WORDS,
        clock=time.monotonic,
    ):
        self._clock = clock
        self._history_words = max(8, history_words)
        self.enabled = enabled
        self.wpm = wpm
        self.max_lag_s = max_lag_s
        self._final: list[str] = []
        self._live: list[str] = []
        self._target: list[str] = []
        self._shown = 0
        self._budget = 0.0
        self._last_tick = clock()

    # ---------------------------------------------------------------- réglages
    def configure(self, *, wpm: int | None = None, max_lag_s: float | None = None,
                  enabled: bool | None = None) -> None:  # fmt: skip
        if wpm is not None:
            self.wpm = wpm
        if max_lag_s is not None:
            self.max_lag_s = max_lag_s
        if enabled is not None and enabled != self.enabled:
            self.enabled = enabled
            if enabled:  # reprise du cadencement : repartir sans dette de budget
                self._budget = 0.0
                self._last_tick = self._clock()

    def rate(self) -> float:
        """Caractères par seconde autorisés."""
        return chars_per_second(self.wpm)

    # ------------------------------------------------------------------ entrée
    def push_final(self, text: str) -> None:
        """Segment définitif : rejoint l'historique, l'énoncé en cours se vide."""
        words = text.split()
        self._final.extend(words)
        dropped = max(0, len(self._final) - self._history_words)
        if dropped:
            del self._final[:dropped]
            self._shown = max(0, self._shown - dropped)
        self._live = []
        self._resync()

    def set_live(self, text: str) -> None:
        """Queue validée de l'énoncé en cours (remplace la précédente)."""
        self._live = text.split()
        self._resync()

    def clear(self) -> None:
        self._final = []
        self._live = []
        self._target = []
        self._shown = 0
        self._budget = 0.0
        self._last_tick = self._clock()

    def _resync(self) -> None:
        self._target = self._final + self._live
        # La cible peut RÉTRÉCIR : le texte finalisé n'est pas toujours identique
        # à ce que l'énoncé en cours avait déjà validé (filtre anti-hallucination,
        # coupe de boucle). On ne peut pas « désafficher », mais on ne doit pas
        # pointer au-delà de la fin.
        self._shown = min(self._shown, len(self._target))

    # ------------------------------------------------------------------ sortie
    def tick(self, now: float | None = None) -> bool:
        """Avance le débit. True si le texte affiché a changé."""
        moment = self._clock() if now is None else now
        elapsed = max(0.0, moment - self._last_tick)
        self._last_tick = moment
        before = self._shown

        if not self.enabled:
            self._shown = len(self._target)
            return self._shown != before

        rate = self.rate()
        self._budget += elapsed * rate
        # Rattrapage : ramener l'attente sous le plafond, en une seule fois.
        overflow = self.pending_chars() - self.max_lag_s * rate
        if overflow > 0:
            self._budget = max(self._budget, overflow)

        while self._shown < len(self._target):
            cost = len(self._target[self._shown]) + 1
            if self._budget < cost:
                break
            self._budget -= cost
            self._shown += 1
        if self._shown >= len(self._target):
            # Rien en attente : ne pas capitaliser du budget, sinon la prochaine
            # rafale sortirait d'un bloc — exactement ce qu'on veut éviter.
            self._budget = min(self._budget, rate * 0.25)
        return self._shown != before

    def text(self) -> str:
        return " ".join(self._target[: self._shown])

    def pending_words(self) -> int:
        return len(self._target) - self._shown

    def pending_chars(self) -> int:
        return sum(len(w) + 1 for w in self._target[self._shown :])

    def lag_s(self) -> float:
        """Retard introduit par le cadencement (secondes de lecture en attente)."""
        if not self.enabled:
            return 0.0
        return self.pending_chars() / self.rate()


__all__ = ["CHARS_PER_WORD", "TextPacer", "chars_per_second"]
