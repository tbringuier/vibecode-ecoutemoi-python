"""Socle commun aux deux moteurs de reconnaissance.

Écoute Moi 2.0 embarque DEUX moteurs, et c'est un choix mesuré, pas un
accident historique :

- **faster-whisper** (CTranslate2) sur CPU. Ses binaires officiels ne visent
  que le CPU x86-64/ARM64 — ni Vulkan ni Metal. Sur CPU en revanche il
  écrase whisper.cpp (mesuré : ×3,3 sur `small`, fenêtre de 9 s).
- **whisper.cpp** (pywhispercpp) sur GPU. C'est le seul des deux à savoir
  parler Vulkan (Intel/AMD) et Metal (Apple Silicon), et sur un GPU même
  modeste il reprend la tête (mesuré : ×1,5 face à faster-whisper CPU).

Tout ce qui ne dépend d'AUCUN des deux vit ici : le segment rendu au reste de
l'application, les paramètres de session, la normalisation du lexique, la
politique de threads et le préchauffage. `core/engines.py` choisit le moteur ;
`core/engine.py` et `core/engine_fw.py` l'implémentent, avec la MÊME surface
publique — le streamer, la GUI et le sous-processus ne savent pas lequel des
deux ils manipulent.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ecoutemoi.constants import (
    TARGET_SR,
    WARMUP_AUDIO_S,
    WARMUP_MAX_PASSES,
    WARMUP_STABLE_RATIO,
)
from ecoutemoi.core import cpuinfo

log = logging.getLogger(__name__)


@dataclass
class Segment:
    """Un segment horodaté, en MILLISECONDES relatives au début de la fenêtre.

    whisper.cpp compte en centisecondes et faster-whisper en secondes : les deux
    convertissent ici, à la frontière, pour que rien en aval n'ait à savoir d'où
    vient le segment.
    """

    t0_ms: int
    t1_ms: int
    text: str
    no_speech_prob: float | None = None


@dataclass
class EngineParams:
    """Tout ce qu'il faut pour ouvrir une session de reconnaissance.

    Les deux formats de modèle coexistent dans le même objet : `model_path`
    pointe le `.bin` ggml (whisper.cpp), `ct2_path` le dossier CTranslate2
    (faster-whisper). `engine` tranche ; « auto » laisse `core/engines.py`
    essayer le GPU puis retomber sur le CPU.
    """

    model_path: Path | None = None  # ggml .bin — whisper.cpp
    language: str = "fr"  # "auto" => détection sur le premier énoncé
    translate: bool = False
    n_threads: int | None = None
    backend: str = "auto"  # auto | gpu | cpu
    flash_attn: bool = True
    vad_model_path: Path | None = None
    carry_context: bool = False
    gpu_device: int = 0  # index du périphérique GPU (multi-GPU)
    lexicon: str = ""  # noms propres / acronymes du talk (whisper initial_prompt)
    # --- 2.0 : faster-whisper (CTranslate2) ---
    ct2_path: Path | None = None  # dossier du modèle CTranslate2
    compute_type: str = "int8"  # int8 | int8_float32 | float32
    beam_size: int = 1  # 1 = glouton (direct) ; 5 = transcription de fichiers
    engine: str = "auto"  # auto | whispercpp | faster-whisper
    vad_filter: bool = True  # VAD Silero interne du moteur


# Le prompt initial de whisper est PLAFONNÉ à la moitié de la fenêtre de texte du
# décodeur (224 tokens sur 448) ; au-delà, whisper tronque — par la gauche, donc
# silencieusement et par le début de la liste. On borne donc nous-mêmes, en
# caractères, avec une marge : ~4 caractères par token en français.
LEXICON_MAX_CHARS = 700


def normalize_lexicon(text: str) -> str:
    """Lexique opérateur -> prompt initial whisper exploitable.

    Le prompt est du TEXTE, pas une liste : whisper le lit comme le début d'une
    transcription. Une énumération séparée par des virgules suffit à biaiser le
    décodeur vers ces graphies. On aplatit les retours à la ligne (l'opérateur
    saisit volontiers un mot par ligne) et on tronque proprement sur une
    frontière de mot plutôt que de laisser whisper couper au milieu.
    """
    parts = (part.strip(" \t,;") for part in text.replace("\n", ",").split(","))
    flat = ", ".join(part for part in parts if part)
    if len(flat) <= LEXICON_MAX_CHARS:
        return flat
    cut = flat[:LEXICON_MAX_CHARS]
    head, sep, _ = cut.rpartition(", ")
    truncated = head if sep else cut
    log.warning(
        "Lexique tronqué à %d caractères (%d fournis) : whisper plafonne le prompt initial.",
        len(truncated), len(flat),
    )  # fmt: skip
    return truncated


def warmup_audio() -> np.ndarray:
    """~1 s de PAROLE réelle, pour le préchauffage du moteur.

    Du silence ne conviendrait pas : avec un VAD interne actif, une fenêtre
    muette est écartée AVANT l'encodeur — aucun shader compilé, aucun graphe
    alloué, et le coût du premier vrai décodage reste entier. On réutilise donc
    la voix de référence déjà embarquée pour le benchmark ; à défaut, un signal
    voisé synthétique (harmoniques à 120 Hz, 4 syllabes/s).
    """
    n = int(TARGET_SR * WARMUP_AUDIO_S)
    try:
        import wave

        import ecoutemoi

        path = Path(ecoutemoi.__file__).resolve().parent / "assets" / "calibration" / "calibration_fr.wav"
        with wave.open(str(path), "rb") as w:
            if (w.getframerate(), w.getsampwidth(), w.getnchannels()) == (TARGET_SR, 2, 1):
                w.setpos(min(TARGET_SR, max(0, w.getnframes() - n)))  # saute le silence de tête
                x = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0
                if x.size >= TARGET_SR // 2:
                    return np.ascontiguousarray(x)
    except Exception as exc:
        log.debug("Voix de préchauffage indisponible (%s) — repli synthétique", exc)
    t = np.arange(n, dtype=np.float32) / TARGET_SR
    voiced = sum(np.sin(2.0 * np.pi * 120.0 * k * t) / k for k in (1, 2, 3, 4, 5))
    envelope = 0.5 * (1.0 - np.cos(2.0 * np.pi * 4.0 * t))
    return (0.2 * voiced * envelope).astype(np.float32)


def pick_n_threads(requested: int | None = None) -> int:
    """P-cores physiques (CPU hybride) sinon cœurs physiques ; jamais le SMT.
    Une valeur demandée explicitement est honorée telle quelle."""
    if requested is not None and requested > 0:
        return requested
    return cpuinfo.best_n_threads()


class _LogSink:
    """File-like object capturing engine log lines (backend + language info)."""

    def __init__(self, keep: int = 800):
        self.lines: deque[str] = deque(maxlen=keep)
        self._buf = ""
        self._lock = threading.Lock()

    def write(self, s: str) -> None:
        with self._lock:
            self._buf += s
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                line = line.strip()
                if line:
                    self.lines.append(line)
                    log.debug("moteur: %s", line)

    def flush(self) -> None:
        pass


class WarmupMixin:
    """Décodages à blanc jusqu'à stabilisation, partagés par les deux moteurs.

    Le premier décodage d'un moteur frais paie la compilation des shaders
    Vulkan, l'allocation du graphe et le remplissage des caches : de quelques
    secondes à une minute au tout premier lancement sur une machine donnée.
    Payé PENDANT la session, ce coût produit un tampon d'audio en retard puis
    une rafale de rattrapage avant de revenir au temps réel. Payé avant
    l'ouverture du micro, il ne coûte que de l'attente au démarrage.
    """

    last_decode_ms: float
    warmup_ms: list[float]

    def transcribe(self, audio: np.ndarray) -> list[Segment]:  # pragma: no cover - interface
        raise NotImplementedError

    def warmup(self, on_pass=None, should_stop=None) -> list[float]:
        """`on_pass(i, total)` suit la progression, `should_stop()` interrompt."""
        audio = warmup_audio()
        times: list[float] = []
        for i in range(1, WARMUP_MAX_PASSES + 1):
            if should_stop is not None and should_stop():
                break
            if on_pass is not None:
                on_pass(i, WARMUP_MAX_PASSES)
            try:
                self.transcribe(audio)
            except Exception as exc:  # un échec ici se reproduira en session
                log.warning("Passe de préchauffage %d en échec : %s", i, exc)
                break
            times.append(self.last_decode_ms)
            # Stable dès qu'une passe retombe au niveau de la meilleure observée :
            # le surcoût unique (shaders, caches) est absorbé.
            if len(times) >= 2 and times[-1] <= min(times) * WARMUP_STABLE_RATIO:
                break
        self.warmup_ms = times
        if times:
            log.info(
                "Préchauffage : %d passe(s) — %s ms",
                len(times), ", ".join(f"{t:.0f}" for t in times),
            )  # fmt: skip
        return times


__all__ = [
    "LEXICON_MAX_CHARS",
    "EngineParams",
    "Segment",
    "WarmupMixin",
    "_LogSink",
    "normalize_lexicon",
    "pick_n_threads",
    "warmup_audio",
]
