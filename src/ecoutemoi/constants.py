"""Locked constants.

These values are LOCKED: do not change one without a documented,
contradicting measurement.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

APP_NAME = "EcouteMoi"
APP_DISPLAY_NAME = "Écoute Moi"
OVERLAY_WINDOW_TITLE = "EcouteMoi - Sortie OBS"

# Nom du backend GPU compilé dans le moteur sur cette plateforme.
GPU_BACKEND_NAME = "Metal" if sys.platform == "darwin" else "Vulkan"
BACKENDS = ("auto", "gpu", "cpu")

# --- Moteurs de reconnaissance --------------------------------------------------
# Écoute Moi 2.0 en embarque deux, chacun là où il gagne (voir core/engines.py) :
# whisper.cpp est le seul à parler Vulkan/Metal, faster-whisper est ~3x plus
# rapide sur CPU. Le backend choisi par l'opérateur détermine lequel tourne.
ENGINE_WHISPERCPP = "whispercpp"
ENGINE_FASTER_WHISPER = "faster-whisper"
CPU_ENGINES = (ENGINE_FASTER_WHISPER, ENGINE_WHISPERCPP)
ENGINE_LABELS: dict[str, str] = {
    ENGINE_FASTER_WHISPER: "faster-whisper (CTranslate2) — recommandé",
    ENGINE_WHISPERCPP: "whisper.cpp — repli",
}
# Formats de modèle : un par moteur, incompatibles entre eux.
MODEL_FORMATS = ("ggml", "ct2")
MODEL_FORMAT_LABELS: dict[str, str] = {
    "ggml": f"ggml — whisper.cpp ({GPU_BACKEND_NAME}/CPU)",
    "ct2": "CTranslate2 — faster-whisper (CPU)",
}

# --- Audio capture / DSP ----------------------------------------------------
CAPTURE_BLOCK_MS = 10  # native RNNoise frame duration, ultra-short callback
PREFERRED_CAPTURE_SR = 48_000
TARGET_SR = 16_000  # Whisper input: float32 mono 16 kHz in [-1, 1]
RING_BUFFER_S = 5.0  # recent post-DSP audio kept for the gate pre-roll
HIGHPASS_CUTOFF_HZ = 80.0  # anti-rumble biquad, always cheap
RNNOISE_SR = 48_000
RNNOISE_FRAME = 480  # samples per RNNoise frame (10 ms @ 48 kHz)

# --- Speech gate -------------------------------------------------------------
VAD_FRAME_MS = 20  # webrtcvad frame size @ 16 kHz
VAD_MODE = 2
SPEECH_START_WINDOW = 6  # look at the last 6 VAD frames...
SPEECH_START_MIN_FRAMES = 4  # ...require >= 4 speech frames to start
PRE_ROLL_MS = 300  # audio replayed from the ring buffer at speech start
RNNOISE_SPEECH_PROB = 0.5  # RNNoise per-frame speech probability threshold
RNNOISE_START_LOOKBACK_MS = 200  # max prob over this window required at start

# --- Streaming ----------------------------------------------------------------
MIN_DECODE_INTERVAL_MS = 300  # adaptive cadence floor

# Cadence et fenêtre AUTO-ADAPTATIVES (temps réel uniquement) : l'intervalle
# entre décodages suit le temps de décodage médian (jamais sous le plancher du
# preset), et la fenêtre max rétrécit quand la machine sature / regrandit
# jusqu'au plafond du preset quand elle respire.
ADAPT_INTERVAL_FACTOR = 1.2  # intervalle >= 1.2 x décodage médian (duty <= ~45 %)
ADAPT_DECODE_WINDOW = 5  # médiane glissante des N derniers temps de décodage
ADAPT_WINDOW_MIN_S = 4.0  # plancher de fenêtre quand la machine sature
ADAPT_SHRINK_LAG = 0.8  # lag médian au-delà duquel la fenêtre rétrécit
ADAPT_GROW_LAG = 0.35  # lag médian sous lequel la fenêtre regrandit
ADAPT_SHRINK_FACTOR = 0.85
ADAPT_GROW_FACTOR = 1.10

# Styles d'affichage des sous-titres (fenêtre OBS + prévisualisation).
DISPLAY_STYLES: dict[str, str] = {
    "defilement": "Défilement animé (recommandé)",
    "fondu": "Fondu des nouveaux mots",
    "statique": "Statique (sans animation)",
}
SENTENCE_CUT_OVERLAP_MS = 200  # overlap kept when cutting at a committed sentence end
NO_SPEECH_PROB_MAX = 0.6  # drop segments above this no-speech probability

# --- Répétitions ----------------------------------------------------------------
# L'audio redécodé après une coupe de fenêtre (200 ms) ou au redémarrage du
# détecteur de parole (pré-amorce de 300 ms) contient des mots DÉJÀ sortis en
# « finalisé ». Whisper les redit ; il faut les reconnaître et les retirer. La
# fenêtre de recherche est large (1,2 s) parce que les horodatages de whisper sont
# grossiers : un mot du recouvrement peut être daté bien après le début réel.
OVERLAP_GUARD_MS = 1200

# --- Détection de silence (coupe de queue, découpe de fichier) -------------------
SILENCE_FRAME_MS = 20  # granularité du balayage RMS
SILENCE_REL_RMS = 0.06  # « silence » = 6 % du niveau le plus fort du passage
SILENCE_ABS_RMS = 0.004  # plancher absolu (passage entièrement muet)
FINAL_TRIM_KEEP_MS = 200  # marge gardée après la dernière syllabe
MIN_FINAL_AUDIO_MS = 200  # en dessous, on ne décode pas : il n'y a rien à lire
LAG_WARN_MEDIAN = 1.3  # median lag over 5 decodes -> "machine lagging" banner
LAG_MEDIAN_WINDOW = 5
LAG_PHRASE_SWITCH = 2.0  # sustained lag -> auto switch to Phrase mode
LAG_PHRASE_SWITCH_S = 10.0

SENTENCE_END_CHARS = ".!?…"

# --- Transcription de fichiers (hors direct) -------------------------------------
# Rien à voir avec le temps réel : ici on a tout le fichier, aucune latence à
# tenir, et le seul objectif est la qualité du texte.
MEDIA_BLOCK_S = 8.0  # taille des blocs rendus par le décodeur (mémoire bornée)
FILE_CHUNK_S = 25.0  # durée visée d'une passe de décodage
FILE_CHUNK_SEARCH_S = 6.0  # zone de recherche du creux de silence autour de la cible
FILE_CHUNK_MIN_S = 6.0  # en dessous, on n'ouvre pas une passe de plus
FILE_CHUNK_OVERLAP_S = 0.4  # recouvrement entre passes (le garde anti-doublon fait le reste)

# --- Engine defaults ----------------------------------------------------------
VAD_THRESHOLD = 0.5
VAD_MIN_SPEECH_MS = 200
VAD_MIN_SILENCE_MS = 300
VAD_SPEECH_PAD_MS = 80
VAD_SAMPLES_OVERLAP = 0.1


# --- Presets -------------------------------------------------------------------
@dataclass(frozen=True)
class Preset:
    """Operator-facing latency preset."""

    key: str
    label: str
    silence_ms: int  # continuous silence that ends an utterance
    keep_back: int  # words withheld from commit (LocalAgreement-2)
    window_max_s: float  # decode-window ceiling for this preset
    partials: bool  # False => decode only at speech_end ("Phrase")


PRESETS: dict[str, Preset] = {
    "ultra": Preset("ultra", "Ultra", 350, 1, 7.0, True),
    "equilibre": Preset("equilibre", "Équilibré", 500, 1, 9.0, True),
    "stable": Preset("stable", "Stable", 700, 2, 9.0, True),
    "phrase": Preset("phrase", "Phrase", 700, 0, 9.0, False),
}
DEFAULT_PRESET = "stable"
DEFAULT_MODEL = "small-q5_1"  # fallback when no benchmark result exists

# --- Anti-hallucination blacklist -----------------------------------------------
# Matched on normalized text (lowercase, stripped). Substring match.
HALLUCINATION_BLACKLIST: tuple[str, ...] = (
    "sous-titres réalisés par",
    "sous-titres realises par",
    "sous-titrage société radio-canada",
    "sous-titrage st'",
    "amara.org",
    "merci d'avoir regardé",
    "merci d'avoir regarde",
    "thanks for watching",
    "thank you for watching",
    "abonnez-vous",
    "n'oubliez pas de vous abonner",
    "subscribe to",
)

# --- Calibration texts ----------------------------------------------------------
CALIBRATION_TEXT_FR = (
    "Bonjour à toutes et à tous, et bienvenue dans cette conférence. "
    "Aujourd'hui, nous allons parler de technologie, de réseaux et de stockage distribué. "
    "Trois serveurs, douze disques, quarante-deux téraoctets : les chiffres comptent autant que les idées. "
    "Merci de votre attention, et place à la démonstration."
)
CALIBRATION_TEXT_EN = (
    "Good morning everyone, and welcome to this conference. "
    "Today we will talk about technology, networks, and distributed storage. "
    "Three servers, twelve disks, and forty-two terabytes: numbers matter as much as ideas. "
    "Thank you for your attention, and let's begin the demonstration."
)

# --- Benchmark -------------------------------------------------------------------
BENCH_TIMEOUT_FACTOR = 3.0  # part du timeout proportionnelle au DÉCODAGE (3x l'audio)
# Budget de DÉMARRAGE du sous-processus, hors décodage : extraction onefile
# PyInstaller (~110 Mo), chargement du modèle, compilation des shaders Vulkan au
# premier lancement. Sans lui, des passes légitimes seraient tuées sur les
# machines lentes et comptées « Abandonné (trop lent) ».
BENCH_STARTUP_ALLOWANCE_S = 120.0
BENCH_MIN_RTF = 1.3  # below this the model is "Inadapté"
BENCH_FLUID_RTF = 3.0  # above this (and WER ok) -> "Recommandé"
BENCH_MAX_WER = 0.20

# --- Préchauffage du moteur -------------------------------------------------------
# Le PREMIER décodage d'un moteur frais est bien plus lent que les suivants
# (compilation des shaders Vulkan, allocation du graphe, remplissage des caches).
# Ouvrir le micro avant ce coût unique produit un tampon d'audio en retard, puis
# une rafale de rattrapage : on décode donc du silence AVANT de capturer, et la
# session n'est déclarée « lancée » qu'une fois le moteur chaud.
WARMUP_AUDIO_S = 1.0  # durée du décodage à blanc (silence)
WARMUP_MAX_PASSES = 4  # passes maximum jusqu'à stabilisation
WARMUP_STABLE_RATIO = 1.35  # passe stable si <= 1,35 x la meilleure passe observée

# --- Voyant temps réel (barre d'état) ---------------------------------------------
# lag = temps de décodage / durée de la fenêtre décodée (= 1 / RTF).
#   vert   : RTF >= BENCH_FLUID_RTF -> marge confortable ;
#   orange : RTF >= BENCH_MIN_RTF   -> ça tient, sans marge ;
#   rouge  : RTF <  BENCH_MIN_RTF   -> modèle trop lourd, le direct décroche.
RT_LIGHT_WINDOW = 5  # médiane glissante des N derniers décodages
RT_LIGHT_GREEN_LAG = 1.0 / BENCH_FLUID_RTF  # 0.33
RT_LIGHT_ORANGE_LAG = 1.0 / BENCH_MIN_RTF  # 0.77
# Périodes de clignotement (ms) : plus c'est grave, plus ça clignote vite.
RT_BLINK_MS = {"red": 420, "orange": 900, "green": 1800}
