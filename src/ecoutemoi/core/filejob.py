"""Transcrire un fichier, hors direct.

Rien ici ne ressemble au temps réel, et c'est volontaire. En direct, tout est
contraint par la latence : on redécode une fenêtre glissante, on ne valide un mot
qu'une fois confirmé (LocalAgreement), on rétrécit la fenêtre quand la machine
sature. Sur un fichier, aucune de ces contraintes n'existe — on a l'audio en
entier, personne n'attend, et le seul objectif est la qualité du texte.

D'où une mécanique bien plus simple : découper l'audio en passes d'environ 25 s,
**couper là où personne ne parle**, décoder chaque passe une seule fois, et
recoller. Le recouvrement entre passes existe pour ne pas perdre un mot à cheval
sur la coupe ; les doublons qu'il produit sont retirés par les mêmes garde-fous
que le direct (`core/textguard.py`), parce que le bégaiement du décodeur est un
problème du décodeur, pas du mode d'emploi.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path

import numpy as np

from ecoutemoi.constants import (
    FILE_CHUNK_MIN_S,
    FILE_CHUNK_OVERLAP_S,
    FILE_CHUNK_S,
    FILE_CHUNK_SEARCH_S,
    FINAL_TRIM_KEEP_MS,
    MIN_FINAL_AUDIO_MS,
    NO_SPEECH_PROB_MAX,
    TARGET_SR,
)
from ecoutemoi.core import media
from ecoutemoi.core.dsp import quietest_cut, trim_trailing_silence
from ecoutemoi.core.textguard import (
    OVERLAP_MEMORY_WORDS,
    collapse_repeats,
    is_hallucination,
    norm_token,
    overlap_length,
    same_text,
    words_of,
)
from ecoutemoi.core.transcript import SessionSegment

log = logging.getLogger(__name__)


class Cancelled(RuntimeError):
    """L'opérateur a interrompu le travail."""


@dataclass(frozen=True)
class Progress:
    """Où en est le travail, pour la barre de progression et le journal."""

    path: Path
    index: int  # rang du fichier dans le lot (0-based)
    count: int  # taille du lot
    stage: str  # "ouverture" | "transcription" | "écriture" | "terminé"
    done_s: float = 0.0  # secondes d'audio traitées
    total_s: float | None = None  # durée du fichier, None si inconnue
    text: str = ""  # dernier texte produit (aperçu en direct)

    @property
    def ratio(self) -> float | None:
        """Avancement dans [0, 1], None si la durée est inconnue."""
        if not self.total_s:
            return None
        return max(0.0, min(1.0, self.done_s / self.total_s))


@dataclass
class Transcript:
    """Résultat d'un fichier : le texte, ce qu'il a coûté, ce qui a été écrit."""

    path: Path
    info: media.MediaInfo | None = None
    segments: list[SessionSegment] = field(default_factory=list)
    outputs: list[Path] = field(default_factory=list)
    elapsed_s: float = 0.0
    error: str | None = None
    detected_language: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def word_count(self) -> int:
        return sum(len(s.text.split()) for s in self.segments)

    @property
    def speed(self) -> float | None:
        """Vitesse par rapport au temps réel (2,5 = deux fois et demie plus vite)."""
        if not self.elapsed_s or self.info is None or not self.info.duration_s:
            return None
        return self.info.duration_s / self.elapsed_s


# ------------------------------------------------------------------- découpage
def plan_cut(buffer: np.ndarray, chunk_s: float = FILE_CHUNK_S) -> int:
    """Où couper le tampon : le creux le plus silencieux autour de la cible.

    Couper au milieu d'un mot le fait perdre des deux côtés — whisper devine
    différemment à gauche et à droite de la coupe. On cherche donc le passage le
    plus calme dans une fenêtre de quelques secondes autour de la durée visée.
    """
    target = int(chunk_s * TARGET_SR)
    half = int(FILE_CHUNK_SEARCH_S * TARGET_SR / 2)
    lo = max(int(FILE_CHUNK_MIN_S * TARGET_SR), target - half)
    hi = min(buffer.size, target + half)
    if hi <= lo:
        return min(buffer.size, target)
    return quietest_cut(buffer, lo, hi)


# ------------------------------------------------------------------ assemblage
class _Assembler:
    """Segments whisper d'une passe -> segments définitifs dédoublonnés.

    Volontairement séparé de la mécanique du direct : il n'y a ici ni validation
    progressive ni texte provisoire, seulement des passes successives dont les
    bords se recouvrent. Les fonctions de `textguard`, elles, sont les mêmes —
    un bégaiement de décodeur se reconnaît de la même façon dans les deux modes.
    """

    def __init__(self, lang: str, *, hallucination_filter: bool, no_speech_prob_max: float):
        self.lang = lang
        self.hallucination_filter = hallucination_filter
        self.no_speech_prob_max = no_speech_prob_max
        self._emitted: list[str] = []

    def push(self, raw_segments: list, offset_ms: int) -> list[SessionSegment]:
        kept: list = []
        for s in raw_segments:
            if s.no_speech_prob is not None and s.no_speech_prob > self.no_speech_prob_max:
                log.debug("Segment écarté (no_speech=%.2f) : %r", s.no_speech_prob, s.text)
                continue
            if self.hallucination_filter and is_hallucination(s.text):
                log.info("Hallucination écartée : %r", s.text)
                continue
            if kept and same_text(kept[-1].text, s.text):
                continue
            kept.append(s)

        words = _words(kept, offset_ms)
        words = self._drop_overlap(words)
        drop = collapse_repeats([norm for _t0, _t1, _text, norm, _grp in words])
        if drop:
            log.info("Boucle : %d mot(s) répété(s) retiré(s)", drop)
            words = words[: len(words) - drop]
        out = _group(words, self.lang)
        for seg in out:
            self._emitted += words_of(seg.text)
        del self._emitted[:-OVERLAP_MEMORY_WORDS]
        return out

    def _drop_overlap(self, words: list[tuple]) -> list[tuple]:
        """Retire du début de la passe ce que la passe précédente a déjà dit."""
        if not self._emitted or not words:
            return words
        k = overlap_length(self._emitted, [norm for _t0, _t1, _text, norm, _grp in words])
        if k:
            log.debug("Recouvrement retiré : %s", [w[2] for w in words[:k]])
        return words[k:]


def _words(segments: list, offset_ms: int) -> list[tuple[int, int, str, str, int]]:
    """(t0, t1, mot, forme normalisée, index du segment), horodatage absolu.

    Whisper date les segments, pas les mots : on interpole linéairement à
    l'intérieur du segment. C'est faux au mot près, juste à la phrase près — et
    c'est la phrase qui porte le sous-titre.
    """
    out: list[tuple[int, int, str, str, int]] = []
    for index, s in enumerate(segments):
        tokens = s.text.split()
        if not tokens:
            continue
        t0 = offset_ms + s.t0_ms
        span = max(0, s.t1_ms - s.t0_ms)
        for i, token in enumerate(tokens):
            w0 = t0 + span * i // len(tokens)
            w1 = t0 + span * (i + 1) // len(tokens)
            out.append((w0, w1, token, norm_token(token), index))
    return out


def _group(words: list[tuple], lang: str) -> list[SessionSegment]:
    """Regroupe les mots retenus en segments, un par segment whisper d'origine."""
    out: list[SessionSegment] = []
    for _index, group in groupby(words, key=lambda w: w[4]):
        kept = list(group)
        text = " ".join(w[2] for w in kept).strip()
        if text:
            out.append(SessionSegment(t0_ms=kept[0][0], t1_ms=kept[-1][1], text=text, lang=lang))
    return out


# ---------------------------------------------------------------- transcription
def transcribe_blocks(
    engine,
    blocks: Iterable[np.ndarray],
    *,
    lang: str = "fr",
    chunk_s: float = FILE_CHUNK_S,
    hallucination_filter: bool = True,
    no_speech_prob_max: float | None = None,
    on_chunk=None,  # (segments, done_s) après chaque passe
    should_stop=None,
) -> list[SessionSegment]:
    """Cœur du mode fichier : un flux de blocs 16 kHz -> des segments horodatés.

    Séparé du décodage ET des entrées/sorties : cela se teste avec un moteur
    factice et un tableau numpy, sans fichier, sans modèle et sans GPU.
    """
    stop = should_stop or (lambda: False)
    report = on_chunk or (lambda segments, done_s: None)
    assembler = _Assembler(
        lang,
        hallucination_filter=hallucination_filter,
        no_speech_prob_max=(NO_SPEECH_PROB_MAX if no_speech_prob_max is None else no_speech_prob_max),
    )
    segments: list[SessionSegment] = []
    buffer = np.zeros(0, dtype=np.float32)
    base_ms = 0
    overlap = int(FILE_CHUNK_OVERLAP_S * TARGET_SR)
    # Il faut la zone de recherche du creux ENTIÈRE avant de couper, sinon la
    # coupe tomberait systématiquement sur le bord du tampon.
    ready = int((chunk_s + FILE_CHUNK_SEARCH_S / 2) * TARGET_SR)

    def decode(chunk: np.ndarray, offset_ms: int) -> None:
        chunk = trim_trailing_silence(chunk, TARGET_SR, keep_ms=FINAL_TRIM_KEEP_MS)
        if chunk.size < TARGET_SR * MIN_FINAL_AUDIO_MS // 1000:
            return  # passe muette : la donner à whisper, c'est l'inviter à inventer
        fresh = assembler.push(engine.transcribe(chunk), offset_ms)
        segments.extend(fresh)
        report(fresh, (offset_ms + chunk.size * 1000 // TARGET_SR) / 1000.0)

    for block in blocks:
        if stop():
            raise Cancelled
        buffer = np.concatenate([buffer, block]) if buffer.size else block
        while buffer.size >= ready:
            if stop():
                raise Cancelled
            cut = plan_cut(buffer, chunk_s)
            decode(buffer[:cut], base_ms)
            keep_from = max(0, cut - overlap)
            base_ms += keep_from * 1000 // TARGET_SR
            buffer = buffer[keep_from:].copy()
    if buffer.size:
        decode(buffer, base_ms)
    return segments


def transcribe_file(
    engine,
    path: Path,
    *,
    lang: str = "fr",
    info: media.MediaInfo | None = None,
    hallucination_filter: bool = True,
    no_speech_prob_max: float | None = None,
    on_progress=None,  # (Progress)
    should_stop=None,
    index: int = 0,
    count: int = 1,
) -> Transcript:
    """Un fichier, du décodage aux segments. Les erreurs sont RENDUES, pas levées.

    C'est un choix : dans un lot de trente fichiers, un MP3 tronqué ne doit pas
    emporter les vingt-neuf autres. L'appelant lit `Transcript.error`.
    """
    path = Path(path)
    result = Transcript(path=path)
    started = time.monotonic()
    emit = on_progress or (lambda progress: None)

    def announce(stage: str, done_s: float = 0.0, text: str = "") -> None:
        emit(
            Progress(
                path=path,
                index=index,
                count=count,
                stage=stage,
                done_s=done_s,
                total_s=result.info.duration_s if result.info else None,
                text=text,
            )
        )

    try:
        announce("ouverture")
        result.info = info or media.probe(path)
        announce("transcription")

        def on_chunk(fresh: list[SessionSegment], done_s: float) -> None:
            announce("transcription", done_s, fresh[-1].text if fresh else "")

        result.segments = transcribe_blocks(
            engine,
            media.stream_16k_mono(path, info=result.info, should_stop=should_stop),
            lang=lang,
            hallucination_filter=hallucination_filter,
            no_speech_prob_max=no_speech_prob_max,
            on_chunk=on_chunk,
            should_stop=should_stop,
        )
        with_lang = getattr(engine, "detected_language", None)
        result.detected_language = with_lang() if callable(with_lang) else None
    except Cancelled:
        result.error = "interrompu"
    except Exception as exc:
        log.exception("Transcription de %s en échec", path)
        result.error = str(exc) or type(exc).__name__
    result.elapsed_s = time.monotonic() - started
    if result.ok:
        log.info(
            "%s transcrit : %d segments, %d mots, %.0f s (x%.1f temps réel)",
            path.name, len(result.segments), result.word_count, result.elapsed_s,
            result.speed or 0.0,
        )  # fmt: skip
    return result


def transcribe_many(
    engine,
    paths: Iterable[Path],
    *,
    lang: str = "fr",
    hallucination_filter: bool = True,
    no_speech_prob_max: float | None = None,
    on_progress=None,
    should_stop=None,
) -> Iterator[Transcript]:
    """Le lot, un fichier à la fois, avec UN seul moteur pour tous.

    Générateur : l'appelant voit chaque résultat dès qu'il est prêt et peut
    écrire les fichiers de sortie sans attendre la fin du lot.
    """
    files = list(paths)
    for index, path in enumerate(files):
        yield transcribe_file(
            engine,
            path,
            lang=lang,
            hallucination_filter=hallucination_filter,
            no_speech_prob_max=no_speech_prob_max,
            on_progress=on_progress,
            should_stop=should_stop,
            index=index,
            count=len(files),
        )


__all__ = [
    "Cancelled",
    "Progress",
    "Transcript",
    "plan_cut",
    "transcribe_blocks",
    "transcribe_file",
    "transcribe_many",
]
