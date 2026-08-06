"""Real-time core: adaptive decode cadence, LocalAgreement-2, window bounds,
anti-hallucination filters, telemetry.

Threading contract: gate events arrive from the DSP thread via a queue; a single
decode thread owns the engine. Callbacks fire from the decode
thread and must be cheap/thread-safe.
"""

from __future__ import annotations

import logging
import queue
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass
from itertools import groupby

import numpy as np

from ecoutemoi.constants import (
    ADAPT_DECODE_WINDOW,
    ADAPT_GROW_FACTOR,
    ADAPT_GROW_LAG,
    ADAPT_INTERVAL_FACTOR,
    ADAPT_SHRINK_FACTOR,
    ADAPT_SHRINK_LAG,
    ADAPT_WINDOW_MIN_S,
    FINAL_TRIM_KEEP_MS,
    LAG_MEDIAN_WINDOW,
    LAG_PHRASE_SWITCH,
    LAG_PHRASE_SWITCH_S,
    LAG_WARN_MEDIAN,
    MIN_DECODE_INTERVAL_MS,
    MIN_FINAL_AUDIO_MS,
    NO_SPEECH_PROB_MAX,
    OVERLAP_GUARD_MS,
    SENTENCE_CUT_OVERLAP_MS,
    SENTENCE_END_CHARS,
    TARGET_SR,
    Preset,
)
from ecoutemoi.core.dsp import trim_trailing_silence
from ecoutemoi.core.gate import GateEvent
from ecoutemoi.core.textguard import (
    OVERLAP_MEMORY_WORDS,
    collapse_repeats,
    is_hallucination,
    norm_token,
    overlap_length,
    repeated_tail_length,
    same_text,
    words_of,
)
from ecoutemoi.core.transcript import SessionSegment, TranscriptStore

log = logging.getLogger(__name__)

_STOP = object()
MIN_NEW_AUDIO_MS = 150  # don't re-decode for less than this much fresh audio


@dataclass
class Word:
    t0_ms: int
    t1_ms: int
    text: str
    norm: str
    seg: int = 0  # segment whisper d'origine (regroupement des mots retenus)


class LocalAgreement:
    """LocalAgreement-2: commit the longest common prefix of the last two
    hypotheses, withholding the trailing `keep_back` words. Monotonic."""

    def __init__(self, keep_back: int = 1):
        self.keep_back = keep_back
        self.committed = 0
        self._prev: list[str] | None = None

    def reset(self, committed: int = 0) -> None:
        self.committed = committed
        self._prev = None

    def update(self, norm_words: list[str]) -> int:
        if self._prev is not None:
            n = 0
            for a, b in zip(self._prev, norm_words, strict=False):
                if a != b:
                    break
                n += 1
            allowed = max(0, n - self.keep_back)
            if allowed > self.committed:
                self.committed = allowed
        self._prev = list(norm_words)
        return self.committed


@dataclass
class StreamStats:
    decode_ms: float
    window_s: float
    rtf: float  # window duration / decode duration
    lag: float  # decode time / decoded window duration (1/RTF); > 1 diverges
    latency_ms: float  # decode end - capture time of the newest window sample


def _noop(*_a, **_k) -> None:
    return None


class Streamer:
    """Consumes gate events, drives the engine, emits partial/finalized text."""

    def __init__(
        self,
        engine,
        store: TranscriptStore,
        preset: Preset,
        *,
        mode: str = "fr",  # fr | translate | auto
        keep_back: int | None = None,
        window_max_s: float | None = None,
        min_interval_ms: int | None = None,
        no_speech_prob_max: float | None = None,
        hallucination_filter: bool = True,
        realtime: bool = True,  # False for fast file feeds: latency is meaningless
        adaptive: bool = True,  # cadence + fenêtre auto-ajustées (temps réel seulement)
        on_partial=None,  # (committed_text: str) — VALIDÉ uniquement, jamais d'attente
        on_finalized=None,  # (SessionSegment)
        on_stats=None,  # (StreamStats)
        on_notice=None,  # (str) operator-facing notices
        on_degraded=None,  # (bool) True = auto-switched to Phrase mode
    ):
        self.engine = engine
        self.store = store
        self.preset = preset
        self.mode = mode
        self.partials = preset.partials
        self.keep_back = preset.keep_back if keep_back is None else keep_back
        self.window_max_ms = int((window_max_s or preset.window_max_s) * 1000)
        self.min_interval_s = (min_interval_ms or MIN_DECODE_INTERVAL_MS) / 1000.0
        self.no_speech_prob_max = NO_SPEECH_PROB_MAX if no_speech_prob_max is None else no_speech_prob_max
        self.hallucination_filter = hallucination_filter
        self.realtime = realtime
        self.adaptive = adaptive
        # Adaptation : la valeur preset/override devient le PLAFOND de fenêtre
        # et le PLANCHER d'intervalle ; le point de fonctionnement suit la machine.
        self._window_ceiling_ms = self.window_max_ms
        self._decode_times_ms: deque[float] = deque(maxlen=ADAPT_DECODE_WINDOW)
        self.on_partial = on_partial or _noop
        self.on_finalized = on_finalized or _noop
        self.on_stats = on_stats or _noop
        self.on_notice = on_notice or _noop
        self.on_degraded = on_degraded or _noop

        # Sortie : FR en mode fr, EN en modes translate et auto (auto = langue
        # source détectée en continu, sous-titres toujours traduits en anglais).
        self.lang = "fr" if mode == "fr" else "en"
        self.src_lang: str | None = None  # langue source détectée (mode auto)

        self._events: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="streamer", daemon=True)
        self._t_start = 0.0

        # Utterance state — decode thread only.
        self._active = False
        self._end_pending = False
        self._chunks: list[np.ndarray] = []
        self._win_samples = 0
        self._win_t0_ms = 0
        self._committed: list[Word] = []
        self._la = LocalAgreement(self.keep_back)
        # Mémoire des derniers mots SORTIS en finalisé. Elle survit à la fin d'un
        # énoncé, et c'est tout l'intérêt : la pré-amorce de 300 ms rejouée au
        # redémarrage du détecteur de parole contient la fin de l'énoncé
        # précédent, donc le doublon arrive APRÈS la remise à zéro de l'énoncé.
        self._emitted_norms: list[str] = []
        self._fed_ms = 0  # session audio time fed so far (from gate sample positions)
        self._fed_ms_at_decode = 0
        self._last_decode_start = 0.0
        self._win_ms_at_decode = 0
        self._lags: deque[float] = deque(maxlen=LAG_MEDIAN_WINDOW)
        self._lag_warned = False
        self._lag_high_since: float | None = None  # sustained-overload timer

    # ------------------------------------------------------------ public API
    def start(self) -> None:
        self._t_start = time.monotonic()
        self._thread.start()

    def restore_partials(self) -> None:
        """Reverse the auto Phrase-mode switch: re-enable partial decoding."""
        if not self.partials and self.preset.partials:
            self.partials = True
            self._lag_high_since = None
            self._lags.clear()
            self.on_degraded(False)
            self.on_notice("Sous-titres partiels réactivés.")
            log.info("Partials restored by operator")

    def on_gate_event(self, ev: GateEvent) -> None:
        """Called from the DSP/feed thread."""
        self._events.put(ev)

    def stop(self, timeout: float = 60.0) -> None:
        """Finalize any in-flight utterance and stop the decode thread."""
        self._events.put(_STOP)
        self._thread.join(timeout)

    # ---------------------------------------------------------- decode thread
    def _run(self) -> None:
        try:
            while True:
                stop = self._drain_events()
                if self._active and self._end_pending:
                    self._decode_final()
                elif self._active and self.partials:
                    since = time.monotonic() - self._last_decode_start
                    fresh_ms = self._fed_ms - self._fed_ms_at_decode
                    if since >= self.effective_interval_s() and fresh_ms >= MIN_NEW_AUDIO_MS:
                        self._decode_partial()
                if stop:
                    if self._active:
                        self._decode_final()
                    return
        except Exception:
            log.exception("Streamer thread crashed")
            raise

    def _drain_events(self) -> bool:
        stop = False
        try:
            ev = self._events.get(timeout=0.03)
        except queue.Empty:
            return False
        while True:
            if ev is _STOP:
                stop = True
            else:
                self._apply_event(ev)
            try:
                ev = self._events.get_nowait()
            except queue.Empty:
                return stop

    def _apply_event(self, ev: GateEvent) -> None:
        self._fed_ms = ev.sample_pos * 1000 // TARGET_SR
        if ev.kind == "speech_start":
            assert ev.audio is not None
            if self._active:
                # Fast feeds / lagging machines: the next utterance queued up
                # before the previous one was finalized — finalize it first, an
                # utterance must never be silently dropped.
                self._decode_final()
            self._active = True
            self._end_pending = False
            self._chunks = [ev.audio.astype(np.float32, copy=False)]
            self._win_samples = ev.audio.size
            self._win_t0_ms = max(0, self._fed_ms - ev.audio.size * 1000 // TARGET_SR)
            self._committed = []
            self._la.reset()
            self._fed_ms_at_decode = self._fed_ms
            self._last_decode_start = time.monotonic()
        elif ev.kind == "frame" and self._active and ev.audio is not None:
            self._chunks.append(ev.audio)
            self._win_samples += ev.audio.size
        elif ev.kind == "speech_end" and self._active:
            self._end_pending = True

    # ---------------------------------------------------------------- decode
    def _window_audio(self) -> np.ndarray:
        if len(self._chunks) > 1:
            audio = np.concatenate(self._chunks)
            self._chunks = [audio]
        elif self._chunks:
            audio = self._chunks[0]
        else:
            audio = np.zeros(0, dtype=np.float32)
        return audio

    def _window_ms(self) -> int:
        return self._win_samples * 1000 // TARGET_SR

    def _decode(self, audio: np.ndarray) -> list:
        self._last_decode_start = time.monotonic()
        self._fed_ms_at_decode = self._fed_ms
        self._win_ms_at_decode = self._window_ms()
        segments = self.engine.transcribe(audio)
        return self._filter_segments(segments)

    def _decode_partial(self) -> None:
        audio = self._window_audio()
        if audio.size < TARGET_SR // 5:
            return
        segments = self._decode(audio)
        self._track_detected_language()
        words = self._extract_words(segments)
        words = self._drop_emitted_overlap(words)
        n = self._la.update([w.norm for w in words])
        n = min(n, len(words))
        n = self._loop_guard(words, n)
        self._la.committed = n
        self._committed = words[:n]
        self._emit_partial(words[:n])
        self._emit_stats()
        self._maybe_trim_window()

    def _decode_final(self) -> None:
        audio = self._window_audio()
        self._end_pending = False
        # La fenêtre finale se termine par le silence qui a justement déclenché la
        # fin d'énoncé. Le donner à whisper, c'est lui demander de le remplir : il
        # redit la phrase précédente, ou invente une formule de fin de vidéo.
        audio = trim_trailing_silence(audio, TARGET_SR, keep_ms=FINAL_TRIM_KEEP_MS)
        if audio.size < TARGET_SR * MIN_FINAL_AUDIO_MS // 1000:
            self._reset_utterance()
            return
        segments = self._decode(audio)
        self._track_detected_language()
        words = self._extract_words(segments)
        words = self._drop_emitted_overlap(words)
        words = self._loop_cut_tail(words)
        self._emit_partial(words)
        for seg in self._finalized_segments(words):
            self.store.add(seg)
            self.on_finalized(seg)
            self._remember_emitted(seg.text.split())
        self._emit_stats()
        self._reset_utterance()

    def _reset_utterance(self) -> None:
        self._active = False
        self._end_pending = False
        self._chunks = []
        self._win_samples = 0
        self._committed = []
        self._la.reset()
        # `_emitted_norms` n'est PAS remis à zéro : c'est lui qui reconnaîtra la
        # fin de cet énoncé quand la pré-amorce du suivant la rejouera.

    # ----------------------------------------------------------------- filters
    def _filter_segments(self, segments: list) -> list:
        out: list = []
        for s in segments:
            if s.no_speech_prob is not None and s.no_speech_prob > self.no_speech_prob_max:
                log.debug("Dropped segment (no_speech_prob=%.2f): %r", s.no_speech_prob, s.text)
                continue
            if self.hallucination_filter and is_hallucination(s.text):
                log.info("Dropped hallucination: %r", s.text)
                continue
            # Whisper resservant le même énoncé en deux segments (horodatages
            # différents, texte identique) : le second est du bruit.
            if out and same_text(out[-1].text, s.text):
                log.info("Dropped duplicate segment: %r", s.text)
                continue
            out.append(s)
        return out

    def _extract_words(self, segments: list) -> list[Word]:
        """Segment text -> words with linearly interpolated absolute timestamps."""
        words: list[Word] = []
        for index, s in enumerate(segments):
            tokens = s.text.split()
            if not tokens:
                continue
            t0 = self._win_t0_ms + s.t0_ms
            span = max(0, s.t1_ms - s.t0_ms)
            for i, tok in enumerate(tokens):
                w0 = t0 + span * i // len(tokens)
                w1 = t0 + span * (i + 1) // len(tokens)
                words.append(Word(w0, w1, tok, norm_token(tok), index))
        return words

    def _remember_emitted(self, tokens: list[str]) -> None:
        """Retient les derniers mots SORTIS, pour reconnaître le recouvrement."""
        self._emitted_norms += words_of(" ".join(tokens))
        del self._emitted_norms[:-OVERLAP_MEMORY_WORDS]

    def _drop_emitted_overlap(self, words: list[Word]) -> list[Word]:
        """Retire du début de l'hypothèse ce qui a DÉJÀ été publié.

        Deux mécanismes rejouent de l'audio déjà décodé : la coupe de fenêtre
        (200 ms de recouvrement) et la pré-amorce du détecteur de parole (300 ms
        avant chaque reprise). Whisper redit donc ces mots ; les laisser passer
        les affiche et les écrit deux fois.

        La recherche est bornée au DÉBUT de la fenêtre : au-delà, un mot identique
        est une vraie répétition de l'orateur, pas un artefact de découpage.
        """
        if not self._emitted_norms or not words:
            return words
        horizon = self._win_t0_ms + OVERLAP_GUARD_MS
        head = 0
        while head < len(words) and words[head].t0_ms < horizon:
            head += 1
        k = overlap_length(self._emitted_norms, [w.norm for w in words[:head]])
        if not k:
            return words
        log.debug("Recouvrement retiré : %s", [w.text for w in words[:k]])
        return words[k:]

    def _loop_guard(self, words: list[Word], n: int) -> int:
        """Refuse un commit dont la fin répète immédiatement ce qui précède."""
        if n <= len(self._committed):
            return n
        k = repeated_tail_length([w.norm for w in words[:n]])
        if k:
            log.info("Boucle : commit refusé, motif de %d mot(s) répété", k)
            return len(self._committed)
        return n

    @staticmethod
    def _loop_cut_words(words: list[Word]) -> list[Word]:
        drop = collapse_repeats([w.norm for w in words])
        return words[: len(words) - drop] if drop else words

    def _loop_cut_tail(self, words: list[Word]) -> list[Word]:
        cut = self._loop_cut_words(words)
        if len(cut) != len(words):
            log.info("Boucle (final) : %d mot(s) répété(s) retiré(s)", len(words) - len(cut))
        return cut

    # ------------------------------------------------------------------ output
    def _emit_partial(self, committed: list[Word]) -> None:
        """Publie le texte VALIDÉ. Les mots en attente de LocalAgreement ne
        quittent JAMAIS le streamer : le public ne doit voir que du définitif,
        pas un mot qui se corrige sous ses yeux."""
        self.on_partial(" ".join(w.text for w in committed))

    def _finalized_segments(self, words: list[Word]) -> list[SessionSegment]:
        """Segments définitifs RECONSTRUITS depuis les mots retenus.

        Les segments bruts de whisper ne doivent pas partir tels quels dans le
        transcript : les gardes anti-recouvrement et anti-boucle travaillent sur
        les MOTS, et publier les segments d'origine à côté remettrait exactement
        ce qu'on vient de retirer.
        """
        out: list[SessionSegment] = []
        for _index, group in groupby(words, key=lambda w: w.seg):
            kept = list(group)
            text = " ".join(w.text for w in kept).strip()
            if not text:
                continue
            out.append(SessionSegment(t0_ms=kept[0].t0_ms, t1_ms=kept[-1].t1_ms, text=text, lang=self.lang))
        return out

    # -------------------------------------------------------------- adaptation
    def effective_interval_s(self) -> float:
        """Intervalle réel entre décodages partiels : suit le décodage médian.

        Machine rapide : le plancher (preset) donne la cadence. Machine lente :
        l'intervalle s'écarte pour garder un rapport décodage/attente sain
        (~45 % de duty max) au lieu de marteler le moteur.
        """
        if not (self.realtime and self.adaptive) or not self._decode_times_ms:
            return self.min_interval_s
        med_s = statistics.median(self._decode_times_ms) / 1000.0
        return max(self.min_interval_s, ADAPT_INTERVAL_FACTOR * med_s)

    def _adapt_window(self, med_lag: float) -> None:
        """La fenêtre max respire : rétrécit sous charge (décodages moins chers),
        regrandit jusqu'au plafond du preset quand la machine est à l'aise."""
        if not (self.realtime and self.adaptive):
            return
        floor_ms = int(ADAPT_WINDOW_MIN_S * 1000)
        if med_lag > ADAPT_SHRINK_LAG and self.window_max_ms > floor_ms:
            self.window_max_ms = max(floor_ms, int(self.window_max_ms * ADAPT_SHRINK_FACTOR))
            log.info("Fenêtre max réduite à %.1f s (lag médian %.2f)", self.window_max_ms / 1000, med_lag)
        elif med_lag < ADAPT_GROW_LAG and self.window_max_ms < self._window_ceiling_ms:
            self.window_max_ms = min(self._window_ceiling_ms, int(self.window_max_ms * ADAPT_GROW_FACTOR))
            log.debug("Fenêtre max élargie à %.1f s (lag médian %.2f)", self.window_max_ms / 1000, med_lag)

    def _emit_stats(self) -> None:
        decode_ms = self.engine.last_decode_ms
        window_s = self._win_ms_at_decode / 1000.0
        rtf = window_s / (decode_ms / 1000.0) if decode_ms > 0 else 0.0
        # Lag = decode time over the audio it decoded (1/RTF). In adaptive
        # cadence the naive decode/fresh ratio sits at ~1.0 regardless of
        # overload (fresh audio grows *during* the decode), so it cannot signal
        # divergence; decode/window does.
        lag = decode_ms / max(1.0, self._win_ms_at_decode)
        self._lags.append(lag)
        self._decode_times_ms.append(decode_ms)
        if self.realtime:
            wall_ms = (time.monotonic() - self._t_start) * 1000.0
            latency = wall_ms - self._fed_ms_at_decode + decode_ms
            # _fed_ms_at_decode ~= capture time of the newest sample in the window
            latency = max(0.0, latency)
        else:
            latency = decode_ms
        self.on_stats(StreamStats(decode_ms, window_s, rtf, lag, latency))
        if len(self._lags) == LAG_MEDIAN_WINDOW:
            med = statistics.median(self._lags)
            self._adapt_window(med)
            if med > LAG_WARN_MEDIAN and not self._lag_warned:
                self._lag_warned = True
                self.on_notice(
                    f"Machine à la traîne (lag médian {med:.2f}) : "
                    f"envisagez un modèle plus petit ou le preset Phrase."
                )
                log.warning("Lag median %.2f > %.2f", med, LAG_WARN_MEDIAN)
            elif med <= LAG_WARN_MEDIAN:
                self._lag_warned = False
            self._check_phrase_switch(med)

    def _check_phrase_switch(self, med: float) -> None:
        """Median lag > 2.0 sustained for 10 s -> auto-switch to Phrase mode
        (no partial decodes), reversible via restore_partials()."""
        now = time.monotonic()
        if med > LAG_PHRASE_SWITCH:
            if self._lag_high_since is None:
                self._lag_high_since = now
            elif self.partials and now - self._lag_high_since >= LAG_PHRASE_SWITCH_S:
                self.partials = False
                log.warning(
                    "Lag median %.2f > %.1f sustained %.0f s -> auto Phrase mode",
                    med, LAG_PHRASE_SWITCH, LAG_PHRASE_SWITCH_S,
                )  # fmt: skip
                self.on_degraded(True)
                self.on_notice(
                    "Machine surchargée : bascule automatique en mode Phrase "
                    "(décodage en fin de phrase uniquement, réversible)."
                )
        else:
            self._lag_high_since = None

    # ------------------------------------------------------------------ window
    def _maybe_trim_window(self) -> None:
        """Bound the window; cut at the last committed sentence end (-200 ms),
        else at the last committed word, else hard-cut."""
        win_ms = self._window_ms()
        if win_ms <= self.window_max_ms:
            return
        flush_upto = 0
        cut_abs_ms: int | None = None
        for i, w in enumerate(self._committed):
            if w.text and w.text[-1] in SENTENCE_END_CHARS:
                flush_upto = i + 1
                cut_abs_ms = w.t1_ms - SENTENCE_CUT_OVERLAP_MS
        if cut_abs_ms is None and self._committed:
            flush_upto = len(self._committed)
            cut_abs_ms = self._committed[-1].t1_ms
        if cut_abs_ms is None:
            cut_abs_ms = self._win_t0_ms + win_ms - (self.window_max_ms - 2000)
        cut_abs_ms = max(self._win_t0_ms, min(cut_abs_ms, self._win_t0_ms + win_ms - 1000))

        flushed = self._committed[:flush_upto]
        if flushed:
            seg = SessionSegment(
                t0_ms=flushed[0].t0_ms,
                t1_ms=flushed[-1].t1_ms,
                text=" ".join(w.text for w in flushed),
                lang=self.lang,
            )
            self.store.add(seg)
            self.on_finalized(seg)
            self._remember_emitted([w.text for w in flushed])

        audio = self._window_audio()
        cut_rel = int((cut_abs_ms - self._win_t0_ms) * TARGET_SR / 1000)
        cut_rel = int(np.clip(cut_rel, 0, audio.size))
        audio = audio[cut_rel:]
        self._chunks = [audio]
        self._win_samples = audio.size
        self._win_t0_ms = cut_abs_ms
        remaining = [w for w in self._committed[flush_upto:] if w.t0_ms >= cut_abs_ms]
        self._committed = remaining
        self._la.reset(committed=len(remaining))
        if flushed:
            # Resynchroniser l'affichage immédiatement : les mots flushés viennent
            # de partir en « finalisé », le partiel courant ne doit plus les
            # contenir (sinon ils apparaissent en double jusqu'au prochain décode).
            self._emit_partial(remaining)
        log.debug(
            "Window trimmed at %d ms (flushed %d words, %d committed kept, window now %.1f s)",
            cut_abs_ms, len(flushed), len(remaining), self._win_samples / TARGET_SR,
        )  # fmt: skip

    # ---------------------------------------------------------------- language
    def _track_detected_language(self) -> None:
        """Mode auto : la langue source est re-détectée par whisper à chaque
        décodage (language=auto) ; on ne fait que suivre les changements pour
        informer l'opérateur — la sortie reste toujours en anglais."""
        if self.mode != "auto":
            return
        lang = self.engine.detected_language()
        if lang and lang != self.src_lang:
            self.src_lang = lang
            self.on_notice(f"Langue détectée : {lang} → sous-titres EN")
