"""Persisted settings: JSON, atomic writes, tolerant loading."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import platformdirs

from ecoutemoi.constants import (
    APP_NAME,
    CPU_ENGINES,
    DEFAULT_MODEL,
    DEFAULT_PRESET,
    ENGINE_FASTER_WHISPER,
    PRESETS,
)

log = logging.getLogger(__name__)


def atomic_write_text(path: Path, text: str) -> None:
    """Write text to `path` atomically (tmp file + os.replace), UTF-8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


@dataclass
class Settings:
    """All operator-configurable settings; the Display block is GUI-only."""

    # Audio — nettoyage micro OFF par défaut (« Aucun traitement ») : whisper est
    # déjà robuste au bruit et tout filtrage ajoute des artefacts. Mesuré au WER
    # (JFK, modèle base, bruit blanc 5-10 dB), RNNoise dégrade la transcription
    # même en bruit fort ; le passe-haut reste disponible pour les grondements.
    device_index: int | None = None
    gain: float = 1.0
    denoise: bool = False
    highpass: bool = False

    # Recognition
    model: str = DEFAULT_MODEL
    mode: str = "fr"  # fr | translate | auto
    preset: str = DEFAULT_PRESET

    # Lexique de la conférence : noms propres, produits, acronymes du talk.
    # Passé à whisper comme `initial_prompt` + `carry_initial_prompt` (donc
    # rappelé à CHAQUE fenêtre de décodage, sinon seule la première en profite).
    lexicon: str = ""

    # Normes de sous-titrage : débit de lecture constant et largeur de ligne bornée
    pacing_enabled: bool = True
    reading_wpm: int = 180  # 160-180 mots/min = plafond de lecture confortable
    pacing_max_lag_s: float = 2.5  # au-delà, on rattrape pour ne pas décrocher
    max_chars_per_line: int = 42  # 0 => aucune limite (largeur pixel seule)

    # Advanced (None => value comes from the preset / locked constants)
    min_update_interval_ms: int | None = None
    silence_ms: int | None = None
    keep_back: int | None = None
    window_max_s: float | None = None
    no_speech_prob_max: float | None = None
    n_threads: int | None = None
    flash_attn: bool = True
    backend: str = "auto"  # auto | gpu | cpu
    gpu_device: int = 0  # index du périphérique Vulkan/Metal (multi-GPU)
    # Moteur employé quand le décodage tombe sur le CPU. faster-whisper
    # (CTranslate2) y est ~3x plus rapide que whisper.cpp sur `small` ; whisper.cpp
    # reste proposé comme repli si faster-whisper pose problème sur une machine.
    cpu_engine: str = ENGINE_FASTER_WHISPER  # faster-whisper | whispercpp
    # Précision de calcul de faster-whisper. « auto » = déduite de la quantization
    # du modèle choisi (q5_* -> int8, q8_0 -> int8_float32, f16 -> float32).
    cpu_compute_type: str = "auto"  # auto | int8 | int8_float32 | float32
    hallucination_filter: bool = True
    carry_context: bool = False
    # Inférence dans un processus séparé : sort le décodage du processus Qt, donc
    # plus de gigue GIL sur le rendu ni sur le callback audio, et un crash natif
    # du moteur ne tue plus l'application.
    engine_subprocess: bool = True
    prewarm_on_launch: bool = True  # charger/chauffer dès l'ouverture de la fenêtre

    # Second sous-titre simultané (deuxième langue, deuxième fenêtre de sortie)
    dual_enabled: bool = False
    dual_mode: str = "translate"  # fr | translate | auto
    dual_model: str = ""  # "" => même modèle que le sous-titre principal
    dual_overlay_geometry: str | None = None

    # Session
    session_dir: str | None = None  # None => Documents/EcouteMoi/sessions
    autosave: bool = True
    txt_timestamps: bool = False
    export_on_stop: bool = False

    # Transcription de fichiers (hors direct)
    transcribe_formats: str = "txt,srt"  # formats cochés par défaut, voir core/transcript
    transcribe_output_dir: str | None = None  # None => à côté du fichier source
    transcribe_model: str = ""  # "" => le modèle du direct
    transcribe_mode: str = ""  # "" => le mode du direct
    # Chemin d'un ffmpeg explicite. Vide = détection automatique. Ne sert qu'aux
    # formats que libsndfile ne sait pas ouvrir (M4A/AAC, WMA, conteneurs vidéo).
    ffmpeg_path: str = ""

    # Diffusion du texte hors de l'app (voir core/publish.py)
    obs_ws_enabled: bool = False
    obs_ws_host: str = "127.0.0.1"
    obs_ws_port: int = 4455  # port par défaut d'obs-websocket 5.x
    obs_ws_password: str = ""  # écrit EN CLAIR dans settings.json (usage local)
    obs_ws_source: str = "EcouteMoi"  # nom de la source Texte dans OBS
    obs_ws_source_dual: str = "EcouteMoi2"  # source du second sous-titre
    web_enabled: bool = False
    web_port: int = 8777
    web_bind_lan: bool = False  # False => 127.0.0.1 seulement

    # Display (GUI overlay). Il n'existe AUCUNE couleur de « texte en attente » :
    # les mots non validés par LocalAgreement ne sortent jamais du streamer, donc
    # le public ne voit que du texte définitif — jamais un mot qui se corrige.
    display_style: str = "defilement"  # defilement | fondu | statique
    overlay_transparent: bool = False  # fenêtre de sortie translucide (incrustation directe)
    font_family: str = ""  # "" => system default
    font_size: int = 34
    text_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    bg_color: str = "#00FF00"  # chroma green preset
    outline_width: int = 3  # keep >= 2 px
    max_lines: int = 2
    align: str = "center"  # left | center | right
    margin_h: int = 24
    margin_v: int = 12
    always_on_top: bool = True
    overlay_geometry: str | None = None  # "x,y,w,h" persisted
    main_geometry: str | None = None
    preview_checker: bool = False  # preview-only checkerboard behind text


def config_path() -> Path:
    return platformdirs.user_config_path(APP_NAME, appauthor=False) / "settings.json"


def disclaimer_marker_path() -> Path:
    """Marqueur d'acceptation de l'avertissement de premier lancement.

    Fichier SÉPARÉ de settings.json, et c'est délibéré : l'application ne sauvegarde
    jamais les réglages implicitement (seul le bouton « Sauvegarder » écrit
    settings.json). Enregistrer l'acceptation dans settings.json obligerait donc à
    trahir cette règle, ou à réafficher l'avertissement à chaque lancement.
    """
    return platformdirs.user_config_path(APP_NAME, appauthor=False) / "disclaimer_accepted"


def disclaimer_accepted() -> bool:
    return disclaimer_marker_path().exists()


def accept_disclaimer() -> None:
    from ecoutemoi import __version__

    path = disclaimer_marker_path()
    try:
        atomic_write_text(path, f"{__version__}\n")
    except OSError as exc:
        # Config en lecture seule : mieux vaut réafficher l'avertissement à chaque
        # lancement que faire échouer le démarrage.
        log.warning("Acceptation de l'avertissement non enregistrée (%s) : %s", path, exc)


def load_settings(path: Path | None = None) -> Settings:
    """Load settings; unknown keys are ignored, missing keys get defaults."""
    p = path or config_path()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        # Premier lancement : le modèle par défaut est profilé sur la machine
        # (cœurs utiles + RAM) ; le benchmark guidé affinera ensuite.
        from ecoutemoi.core.models import recommended_default_model

        return Settings(model=recommended_default_model())
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Settings file unreadable (%s), using defaults: %s", p, exc)
        return Settings()
    known = {f.name for f in fields(Settings)}
    clean = {k: v for k, v in raw.items() if k in known}
    if "backend" not in clean and raw.get("force_cpu"):  # settings.json d'une version antérieure
        clean["backend"] = "cpu"
    try:
        settings = Settings(**clean)
    except TypeError as exc:
        log.warning("Settings file invalid (%s), using defaults: %s", p, exc)
        return Settings()
    if settings.backend not in ("auto", "gpu", "cpu"):
        log.warning("Backend inconnu %r, retour à 'auto'", settings.backend)
        settings.backend = "auto"
    if settings.cpu_engine not in CPU_ENGINES:
        log.warning("Moteur CPU inconnu %r, retour à %r", settings.cpu_engine, ENGINE_FASTER_WHISPER)
        settings.cpu_engine = ENGINE_FASTER_WHISPER
    if settings.cpu_compute_type not in ("auto", "int8", "int8_float32", "float32"):
        log.warning("Type de calcul CPU inconnu %r, retour à 'auto'", settings.cpu_compute_type)
        settings.cpu_compute_type = "auto"
    try:  # settings.json édité à la main : ne jamais laisser passer un index absurde
        settings.gpu_device = max(0, int(settings.gpu_device))
    except TypeError, ValueError:
        log.warning("gpu_device invalide %r, retour à 0", settings.gpu_device)
        settings.gpu_device = 0
    if settings.display_style not in ("defilement", "fondu", "statique"):
        log.warning("display_style inconnu %r, retour à 'defilement'", settings.display_style)
        settings.display_style = "defilement"
    # Ces trois valeurs indexent directement des widgets/dicts au démarrage de la
    # GUI : une valeur inconnue (settings.json édité à la main, downgrade de
    # version) ferait planter la construction de la fenêtre au lieu de repartir
    # sur le défaut.
    if settings.preset not in PRESETS:
        log.warning("Preset inconnu %r, retour à %r", settings.preset, DEFAULT_PRESET)
        settings.preset = DEFAULT_PRESET
    if settings.mode not in ("fr", "translate", "auto"):
        log.warning("Mode inconnu %r, retour à 'fr'", settings.mode)
        settings.mode = "fr"
    if settings.dual_mode not in ("fr", "translate", "auto"):
        log.warning("Mode du second sous-titre inconnu %r, retour à 'translate'", settings.dual_mode)
        settings.dual_mode = "translate"
    if settings.align not in ("left", "center", "right"):
        log.warning("Alignement inconnu %r, retour à 'center'", settings.align)
        settings.align = "center"
    for name, lo, hi in (("obs_ws_port", 1, 65535), ("web_port", 1, 65535)):
        try:
            value = int(getattr(settings, name))
        except TypeError, ValueError:
            value = -1
        if not lo <= value <= hi:
            log.warning("%s invalide %r, retour au défaut", name, getattr(settings, name))
            value = getattr(Settings(), name)
        setattr(settings, name, value)
    return settings


def clear_settings(path: Path | None = None) -> bool:
    """Efface les réglages enregistrés (retour aux défauts au prochain lancement)."""
    p = path or config_path()
    try:
        p.unlink()
        return True
    except FileNotFoundError:
        return False


def save_settings(settings: Settings, path: Path | None = None) -> None:
    p = path or config_path()
    atomic_write_text(p, json.dumps(asdict(settings), ensure_ascii=False, indent=2) + "\n")
