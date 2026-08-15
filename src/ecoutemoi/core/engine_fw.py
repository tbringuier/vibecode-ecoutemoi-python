"""FasterWhisperEngine : moteur faster-whisper (CTranslate2) — CPU.

C'est LE moteur CPU d'Écoute Moi 2.0. Mesuré sur Intel Core Ultra 9 185H,
fenêtre de 9 s, médiane de 6 décodages :

    modèle   whisper.cpp CPU   faster-whisper CPU (int8)
    tiny            338 ms              187 ms
    base            731 ms              336 ms
    small          2893 ms              873 ms

Soit ×3,3 sur `small` — assez pour qu'une machine sans GPU utilisable passe de
« small tout juste tenable » à « small confortable, medium envisageable ».

Ce qu'il ne sait PAS faire, et pourquoi le GPU reste à whisper.cpp : les
binaires officiels de CTranslate2 ne visent que le CPU (x86-64 SSE4.1+ et
AArch64). Ni Vulkan, ni Metal. Sur Apple Silicon il tourne donc sur les cœurs
ARM via Accelerate — vite, mais pas sur le GPU. `core/engines.py` choisit en
conséquence.

Contrat identique à `WhisperEngine` (numpy in, `Segment` out) : le streamer, la
transcription de fichiers et le sous-processus ne voient aucune différence.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from ecoutemoi.constants import (
    TARGET_SR,
    VAD_MIN_SILENCE_MS,
    VAD_MIN_SPEECH_MS,
    VAD_SPEECH_PAD_MS,
    VAD_THRESHOLD,
)
from ecoutemoi.core.engine_base import (
    EngineParams,
    Segment,
    WarmupMixin,
    _LogSink,
    normalize_lexicon,
    pick_n_threads,
)

log = logging.getLogger(__name__)

# Types de calcul CTranslate2 utilisés côté CPU, du plus compact au plus fidèle.
# (`float16` n'existe pas sur CPU.) Demander un type non supporté fait échouer le
# CHARGEMENT, pas le décodage — d'où l'interrogation de la lib plus bas plutôt
# qu'une supposition.
COMPUTE_TYPES = ("int8", "int8_float32", "float32")
COMPUTE_FALLBACK = "int8"

# En dessous, le banc de filtres mel de whisper n'a pas de quoi remplir une
# trame : CTranslate2 lèverait sur un tableau trop court. Le direct borne déjà à
# 200 ms (MIN_FINAL_AUDIO_MS), mais le moteur ne doit pas dépendre de son
# appelant pour ne pas planter.
MIN_AUDIO_SAMPLES = TARGET_SR // 10  # 100 ms


def available() -> bool:
    """faster-whisper est-il importable dans cet interpréteur ?"""
    try:
        import faster_whisper  # noqa: F401
    except Exception as exc:
        log.debug("faster-whisper indisponible : %s", exc)
        return False
    return True


def supported_compute_types() -> list[str]:
    """Types de calcul réellement acceptés par CTranslate2 sur ce CPU.

    Un Xeon sans AVX2 n'a pas les mêmes qu'un portable récent : on interroge la
    lib plutôt que de supposer, et `resolve_compute_type` retombe proprement.
    """
    try:
        import ctranslate2

        available_types = ctranslate2.get_supported_compute_types("cpu")
        return [c for c in COMPUTE_TYPES if c in available_types]
    except Exception as exc:
        log.debug("Types de calcul CTranslate2 inconnus (%s) — repli %s", exc, COMPUTE_FALLBACK)
        return [COMPUTE_FALLBACK]


def resolve_compute_type(wanted: str) -> str:
    """Type demandé s'il est supporté ici, sinon le plus proche disponible."""
    supported = supported_compute_types()
    if wanted in supported:
        return wanted
    for candidate in COMPUTE_TYPES:  # du plus compact au plus fidèle
        if candidate in supported:
            log.warning("Type de calcul %r non supporté par ce CPU — repli sur %r", wanted, candidate)
            return candidate
    return COMPUTE_FALLBACK


def _vad_parameters() -> dict:
    """Les mêmes seuils que ceux passés au VAD interne de whisper.cpp.

    Les deux moteurs embarquent le MÊME Silero v6 ; il serait absurde que le
    découpage de la parole change selon le backend choisi — l'opérateur verrait
    des fins de phrase différentes en passant du GPU au CPU.
    """
    return {
        "threshold": VAD_THRESHOLD,
        "min_speech_duration_ms": VAD_MIN_SPEECH_MS,
        "min_silence_duration_ms": VAD_MIN_SILENCE_MS,
        "speech_pad_ms": VAD_SPEECH_PAD_MS,
    }


class FasterWhisperEngine(WarmupMixin):
    """Un `WhisperModel` CTranslate2 par session ; un seul thread appelle transcribe()."""

    name = "faster-whisper"

    def __init__(self, params: EngineParams):
        self.params = params
        self.sink = _LogSink()
        self.n_threads = pick_n_threads(params.n_threads)
        self.compute_type = resolve_compute_type(params.compute_type)
        self.vad_active = bool(params.vad_filter)
        self.flash_attn_active = False  # notion propre à whisper.cpp
        self.dropped_params: list[str] = []
        self.load_variant = f"cpu-{self.compute_type}"
        self.load_s = 0.0
        self.last_decode_ms = 0.0
        self.warmup_ms: list[float] = []
        self.fallback_reason: str | None = None  # posé par core/engines si repli
        self._detected_language: str | None = None
        self._model = None
        self._load()

    # ------------------------------------------------------------------ load
    def _load(self) -> None:
        from faster_whisper import WhisperModel

        path = self.params.ct2_path
        if path is None:
            raise RuntimeError("faster-whisper : aucun modèle CTranslate2 fourni (ct2_path)")
        t0 = time.perf_counter()
        try:
            self._model = WhisperModel(
                str(path),
                device="cpu",
                compute_type=self.compute_type,
                cpu_threads=self.n_threads,
                num_workers=1,
                # Le modèle est DÉJÀ sur disque (core/models.py le télécharge et
                # le valide) : interdire le réseau ici évite qu'un chargement se
                # transforme en téléchargement silencieux au démarrage d'une
                # session, micro ouvert.
                local_files_only=True,
            )
        except Exception as exc:
            raise RuntimeError(f"Impossible de charger le modèle faster-whisper ({path}) : {exc}") from exc
        self.load_s = time.perf_counter() - t0
        self.sink.write(
            f"faster-whisper: CTranslate2 CPU, compute_type={self.compute_type}, "
            f"threads={self.n_threads}, modele={Path(path).name}\n"
        )
        log.info(
            "Moteur faster-whisper chargé en %.1f s : %s [threads=%d, compute=%s, vad=%s, beam=%d]",
            self.load_s, Path(path).name, self.n_threads, self.compute_type,
            self.vad_active, self.params.beam_size,
        )  # fmt: skip

    # ------------------------------------------------------------- transcribe
    def _decode_kwargs(self) -> dict:
        lexicon = normalize_lexicon(self.params.lexicon)
        kwargs: dict = {
            # « auto » côté application = laisser faster-whisper détecter.
            "language": None if self.params.language == "auto" else self.params.language,
            "task": "translate" if self.params.translate else "transcribe",
            "beam_size": max(1, int(self.params.beam_size)),
            # Température FIXE à 0 : la cascade par défaut (0 → 1.0) relance le
            # décodage jusqu'à 6 fois quand une heuristique juge la sortie
            # douteuse. En direct, ce sursaut de latence est bien pire que le
            # segment médiocre qu'il tente de rattraper.
            "temperature": 0.0,
            "no_speech_threshold": 0.6,
            "condition_on_previous_text": bool(self.params.carry_context),
            "suppress_blank": True,
            "without_timestamps": False,
            "word_timestamps": False,
            "vad_filter": bool(self.params.vad_filter),
        }
        if kwargs["vad_filter"]:
            kwargs["vad_parameters"] = _vad_parameters()
        if lexicon:
            kwargs["initial_prompt"] = lexicon
        return kwargs

    def transcribe(self, audio: np.ndarray) -> list[Segment]:
        """Decode one float32 mono 16 kHz window; returns segments in ms."""
        assert self._model is not None
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if audio.size < MIN_AUDIO_SAMPLES:
            self.last_decode_ms = 0.0
            return []
        t0 = time.perf_counter()
        # `transcribe()` rend un GÉNÉRATEUR : rien n'est décodé tant qu'il n'est
        # pas consommé. Le chronomètre n'aurait aucun sens sans le list().
        raw, info = self._model.transcribe(audio, **self._decode_kwargs())
        segments = list(raw)
        self.last_decode_ms = (time.perf_counter() - t0) * 1000.0
        lang = getattr(info, "language", None)
        if lang:
            self._detected_language = lang
        out: list[Segment] = []
        for s in segments:
            text = (s.text or "").strip()
            if not text:
                continue
            out.append(
                Segment(
                    t0_ms=round(float(s.start) * 1000.0),
                    t1_ms=round(float(s.end) * 1000.0),
                    text=text,
                    no_speech_prob=getattr(s, "no_speech_prob", None),
                )
            )
        return out

    # ---------------------------------------------------------------- helpers
    def detected_language(self) -> str | None:
        return self._detected_language

    def gpu_devices(self) -> list[tuple[int, str]]:
        return []  # CTranslate2 ne voit ni Vulkan ni Metal

    def active_gpu_index(self) -> int | None:
        return None

    def backend_info(self) -> str:
        return f"CPU faster-whisper · {self.compute_type} ({self.n_threads} threads)"

    def gpu_active(self) -> bool:
        return False

    def gpu_diagnostic(self) -> str | None:
        return (
            "faster-whisper (CTranslate2) est le moteur CPU : il ne parle ni "
            "Vulkan ni Metal. Pour utiliser le GPU, choisissez le backend "
            "« GPU » — c'est whisper.cpp qui prend alors la main."
        )

    def diagnostics(self) -> dict:
        return {
            "moteur": "faster-whisper (CTranslate2)",
            "modele": Path(self.params.ct2_path).name if self.params.ct2_path else "—",
            "backend": self.backend_info(),
            "gpu_actif": False,
            "backend_demande": self.params.backend,
            "type_calcul": self.compute_type,
            "types_calcul_supportes": supported_compute_types(),
            "beam_size": max(1, int(self.params.beam_size)),
            "variante_chargement": self.load_variant,
            "chargement_s": round(self.load_s, 2),
            "prechauffage_ms": [round(t) for t in self.warmup_ms],
            "vad": self.vad_active,
            "threads": self.n_threads,
            "params_ignores": self.dropped_params,
            "lexique_caracteres": len(normalize_lexicon(self.params.lexicon)),
            "diagnostic_gpu": self.gpu_diagnostic(),
            "versions": versions(),
        }

    def close(self) -> None:
        self._model = None


def versions() -> str:
    parts = []
    for module, label in (("faster_whisper", "faster-whisper"), ("ctranslate2", "ctranslate2")):
        try:
            mod = __import__(module)
            parts.append(f"{label} {getattr(mod, '__version__', '?')}")
        except Exception:
            parts.append(f"{label} absent")
    return " · ".join(parts)


__all__ = [
    "COMPUTE_TYPES",
    "FasterWhisperEngine",
    "available",
    "resolve_compute_type",
    "supported_compute_types",
    "versions",
]
