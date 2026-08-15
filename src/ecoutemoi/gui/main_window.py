"""Main window: controls + live subtitle PREVIEW + status bar + menus,
plus the pipeline controller bridging worker threads to Qt signals.

The preview and the OBS overlay consume the same signals and the same widget
class (SubtitleView) — strictly identical rendering. The same text also feeds the
external outputs (OBS WebSocket / local web page, see core/publish.py).
"""

from __future__ import annotations

import dataclasses
import logging
import statistics
import threading
import time
from collections import deque
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFontComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ecoutemoi import __version__
from ecoutemoi.config import Settings, clear_settings, load_settings, save_settings
from ecoutemoi.constants import (
    APP_DISPLAY_NAME,
    BACKENDS,
    DISPLAY_STYLES,
    GPU_BACKEND_NAME,
    OVERLAY_WINDOW_TITLE,
    PRESETS,
    RT_LIGHT_GREEN_LAG,
    RT_LIGHT_ORANGE_LAG,
    RT_LIGHT_WINDOW,
)
from ecoutemoi.core import models
from ecoutemoi.core.streamer import Streamer, StreamStats
from ecoutemoi.core.transcript import SessionSegment, TranscriptStore, new_session_dir
from ecoutemoi.gui.subtitle_view import SubtitleStyle, SubtitleView
from ecoutemoi.gui.theme import hint as hint_label
from ecoutemoi.gui.theme import mark
from ecoutemoi.gui.widgets import ColorButton, NoticeBanner, StatusLight, VuMeter
from ecoutemoi.logging_setup import log_dir

log = logging.getLogger(__name__)

SAMPLE_COMMITTED = "Bonjour à toutes et à tous, bienvenue dans cette conférence."


MODE_LABELS = {"fr": "FR → FR", "translate": "FR → EN", "auto": "Auto → EN"}


@dataclasses.dataclass(frozen=True)
class ChannelSpec:
    """Un flux de sous-titres : son modèle, son mode, ses fichiers.

    Le second sous-titre est un flux de plus, pas un cas particulier : mêmes
    évènements de gate, même preset, même cadencement — seuls le moteur, le mode
    et le suffixe de fichier changent.
    """

    index: int  # 0 = principal, 1 = second
    model: str
    mode: str
    suffix: str = ""  # "" pour le principal, "_2" pour le second

    @property
    def label(self) -> str:
        return MODE_LABELS.get(self.mode, self.mode)


def channels_for(s: Settings) -> list[ChannelSpec]:
    channels = [ChannelSpec(0, s.model, s.mode)]
    if s.dual_enabled:
        channels.append(ChannelSpec(1, s.dual_model or s.model, s.dual_mode, "_2"))
    return channels


def engine_key(s: Settings, channel: ChannelSpec) -> tuple:
    """Tout ce dont un changement impose de RECHARGER le moteur.

    Sert de clé au cache de préchauffage : si l'opérateur touche le modèle, le
    mode, le backend ou le lexique après l'ouverture de la fenêtre, le moteur
    préchauffé n'est plus le bon et doit être jeté.
    """
    from ecoutemoi.core.engine import normalize_lexicon

    return (
        channel.model,
        channel.mode,
        s.backend,
        s.cpu_engine,
        s.cpu_compute_type,
        int(s.gpu_device),
        s.n_threads,
        bool(s.flash_attn),
        bool(s.carry_context),
        normalize_lexicon(s.lexicon),
        bool(s.engine_subprocess),
    )


class EngineCache:
    """Moteurs chargés ET préchauffés, prêts à ouvrir une session sans attente.

    Rempli par le fil de préchauffage, vidé par le fil du pipeline : d'où le
    verrou. `take()` transfère la propriété — l'appelant devient responsable du
    `close()`.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._engines: dict[tuple, object] = {}

    def put(self, key: tuple, engine) -> None:
        with self._lock:
            previous = self._engines.pop(key, None)
            self._engines[key] = engine
        if previous is not None:
            _close_quietly(previous)

    def take(self, key: tuple):
        with self._lock:
            return self._engines.pop(key, None)

    def keys(self) -> list[tuple]:
        with self._lock:
            return list(self._engines)

    def discard_except(self, wanted: set[tuple]) -> None:
        """Jette les moteurs devenus inutiles (réglages changés depuis)."""
        with self._lock:
            stale = [key for key in self._engines if key not in wanted]
            engines = [self._engines.pop(key) for key in stale]
        for engine in engines:
            _close_quietly(engine)

    def clear(self) -> None:
        with self._lock:
            engines = list(self._engines.values())
            self._engines.clear()
        for engine in engines:
            _close_quietly(engine)


def _close_quietly(engine) -> None:
    try:
        engine.close()
    except Exception:
        log.warning("Fermeture d'un moteur préchauffé en échec", exc_info=True)


class EnginePrewarmer(QObject):
    """Charge et préchauffe les moteurs DÈS l'ouverture de la fenêtre.

    Le coût incompressible d'un démarrage de session, c'est le chargement du
    modèle plus le premier décodage (compilation des shaders Vulkan). Payé au
    clic sur « Démarrer », il fait attendre l'opérateur devant la salle. Payé
    pendant qu'il règle sa police, il est gratuit.

    Ne télécharge JAMAIS : un modèle absent est simplement ignoré (sinon ouvrir
    l'application déclencherait un téléchargement de plusieurs centaines de Mo).
    """

    sig_status = Signal(str)

    def __init__(self, cache: EngineCache, parent=None):
        super().__init__(parent)
        self._cache = cache
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._busy = threading.Lock()

    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def request(self, s: Settings) -> None:
        if not s.prewarm_on_launch or self.busy():
            return
        wanted = {engine_key(s, ch): ch for ch in channels_for(s)}
        self._cache.discard_except(set(wanted))
        missing = {key: ch for key, ch in wanted.items() if key not in set(self._cache.keys())}
        if not missing:
            return
        self._cancel.clear()
        snapshot = dataclasses.replace(s)
        self._thread = threading.Thread(
            target=self._run, args=(snapshot, missing), daemon=True, name="prewarm"
        )
        self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()

    def _status(self, message: str) -> bool:
        """Publie la progression. False si la fenêtre a été détruite entre-temps.

        Émettre sur un QObject dont le pendant C++ est détruit lève RuntimeError
        DANS le fil : sans ce garde-fou, le préchauffage mourait sur cette
        exception et abandonnait derrière lui un moteur (et son sous-processus)
        que personne ne fermait jamais.
        """
        try:
            self.sig_status.emit(message)
            return True
        except RuntimeError:
            log.debug("Fenêtre détruite pendant le préchauffage — abandon.")
            self._cancel.set()
            return False

    def _run(self, s: Settings, missing: dict[tuple, ChannelSpec]) -> None:
        from ecoutemoi.cli import make_engine, plan_engine

        # Le format qui compte est celui du moteur retenu pour ce backend : un
        # `small` présent en ggml ne dispense pas des 464 Mo CTranslate2 quand
        # c'est faster-whisper qui va tourner. Sans cette vérification, ouvrir
        # l'application déclencherait le téléchargement que le préchauffage
        # s'interdit précisément de faire.
        _choice, fmt = plan_engine(s)
        with self._busy:
            for key, channel in missing.items():
                if self._cancel.is_set():
                    return
                spec = models.REGISTRY.get(channel.model)
                if spec is None or not models.is_installed(spec, fmt=fmt):
                    log.info("Préchauffage sauté : %s n'est pas installé en %s.", channel.model, fmt)
                    continue
                if not self._status(f"Préchauffage de {channel.model} ({channel.label})…"):
                    return
                try:
                    engine = make_engine(s, channel.model, channel.mode)
                except Exception as exc:
                    log.warning("Préchauffage de %s impossible : %s", channel.model, exc)
                    self._status("")
                    continue
                try:
                    engine.warmup(should_stop=self._cancel.is_set)
                except Exception:
                    log.warning("Préchauffage de %s en échec", channel.model, exc_info=True)
                if self._cancel.is_set():
                    _close_quietly(engine)
                    return
                self._cache.put(key, engine)
                log.info("Moteur %s préchauffé et gardé au chaud.", channel.model)
            self._status("")


class PipelineController(QObject):
    """Owns the capture->DSP->gate->engine->streamer chain in worker threads.
    All UI updates go through Qt signals.

    Un ou deux canaux : un seul gate alimente tous les streamers, donc la capture
    et le DSP ne sont jamais payés deux fois.
    """

    sig_partial = Signal(int, str)  # (canal, texte VALIDÉ)
    sig_finalized = Signal(int, object)  # (canal, SessionSegment)
    sig_stats = Signal(int, object)  # (canal, StreamStats)
    sig_notice = Signal(str)
    sig_degraded = Signal(bool)
    sig_state = Signal(str)  # "idle" | "loading" | "warmup" | "running"
    sig_warmup = Signal(int, int)  # (passe, total)
    sig_error = Signal(str)

    def __init__(self, cache: EngineCache | None = None, parent=None):
        super().__init__(parent)
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture = None
        self._cache = cache if cache is not None else EngineCache()
        self.streamers: list[Streamer] = []
        self.stores: list[TranscriptStore] = []
        self.channels: list[ChannelSpec] = []
        self.session_dir: Path | None = None
        self.backend = "—"
        self.gpu_devices: list[tuple[int, str]] = []  # énumération Vulkan de la dernière session
        self.engine_diag: dict = {}  # diagnostics() du dernier moteur chargé
        self.running = False

    # ------------------------------------------------------------------ API
    @property
    def streamer(self) -> Streamer | None:
        """Streamer principal (compat : le reste de la GUI ne pilote que lui)."""
        return self.streamers[0] if self.streamers else None

    @property
    def store(self) -> TranscriptStore | None:
        return self.stores[0] if self.stores else None

    def start(self, settings: Settings) -> None:
        if self.running:
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, args=(settings,), daemon=True, name="pipeline")
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()

    @property
    def rms(self) -> float:
        cap = self._capture
        return cap.rms if cap is not None else 0.0

    # ------------------------------------------------------------ worker side
    def _acquire(self, s: Settings, channel: ChannelSpec, warmed: list[bool]):
        """Moteur pour ce canal : celui préchauffé au lancement, sinon un neuf."""
        from ecoutemoi.cli import make_engine

        engine = self._cache.take(engine_key(s, channel))
        if engine is not None:
            log.info("Moteur préchauffé réutilisé pour %s (%s).", channel.model, channel.mode)
            warmed.append(True)
            return engine
        warmed.append(False)
        return make_engine(s, channel.model, channel.mode)

    def _run(self, s: Settings) -> None:
        from functools import partial as bind

        from ecoutemoi.core.audio import AudioCapture
        from ecoutemoi.core.dsp import DspChain
        from ecoutemoi.core.gate import SpeechGate

        engines: list = []
        self.channels = channels_for(s)
        try:
            self.sig_state.emit("loading")
            warmed: list[bool] = []
            for channel in self.channels:
                engines.append(self._acquire(s, channel, warmed))
            primary = engines[0]
            self.backend = primary.backend_info()
            self.gpu_devices = primary.gpu_devices()
            self.engine_diag = primary.diagnostics()
            if s.backend != "cpu" and not primary.gpu_active():
                prefix = "Backend GPU demandé mais indisponible" if s.backend == "gpu" else "GPU indisponible"
                reason = getattr(primary, "fallback_reason", None) or primary.gpu_diagnostic()
                self.sig_notice.emit(f"{prefix} — décodage CPU ({primary.backend_info()}) : {reason}")

            # Préchauffage AVANT d'ouvrir le micro : le surcoût unique du premier
            # décodage (shaders Vulkan, graphe, caches) est payé ici, pas pendant
            # la session — sinon l'audio s'accumule le temps de ce décodage puis
            # sort en rafale de rattrapage avant de revenir au temps réel. Un
            # moteur déjà préchauffé au lancement de l'app saute cette étape.
            cold = [engine for engine, hot in zip(engines, warmed, strict=True) if not hot]
            if cold:
                self.sig_state.emit("warmup")
                t_warm = time.monotonic()
                for engine in cold:
                    engine.warmup(
                        on_pass=lambda i, total: self.sig_warmup.emit(i, total),
                        should_stop=self._stop_evt.is_set,
                    )
                warm_s = time.monotonic() - t_warm
                if warm_s > 5.0:
                    self.sig_notice.emit(
                        f"Moteur préchauffé en {warm_s:.0f} s (compilation des shaders au "
                        "premier lancement) — les prochains démarrages seront plus rapides."
                    )
            self.engine_diag = primary.diagnostics()  # inclut prechauffage_ms
            if self._stop_evt.is_set():
                return

            preset = PRESETS[s.preset]
            root = Path(s.session_dir) if s.session_dir else None
            self.session_dir = new_session_dir(root)
            self.stores = []
            self.streamers = []
            for channel, engine in zip(self.channels, engines, strict=True):
                store = TranscriptStore(self.session_dir, autosave=s.autosave, suffix=channel.suffix)
                self.stores.append(store)
                self.streamers.append(
                    Streamer(
                        engine,
                        store,
                        preset,
                        mode=channel.mode,
                        min_interval_ms=s.min_update_interval_ms,
                        keep_back=s.keep_back,
                        window_max_s=s.window_max_s,
                        no_speech_prob_max=s.no_speech_prob_max,
                        hallucination_filter=s.hallucination_filter,
                        realtime=True,
                        on_partial=bind(self.sig_partial.emit, channel.index),
                        on_finalized=bind(self.sig_finalized.emit, channel.index),
                        on_stats=bind(self.sig_stats.emit, channel.index),
                        on_notice=self.sig_notice.emit,
                        # Le mode Phrase automatique n'est piloté que par le canal
                        # principal : deux bannières pour la même surcharge
                        # n'apprendraient rien de plus à l'opérateur.
                        on_degraded=self.sig_degraded.emit if channel.index == 0 else None,
                    )
                )
            self._capture = AudioCapture(device=s.device_index, gain=s.gain)
            dsp = DspChain(self._capture.sr, denoise=s.denoise, highpass=s.highpass)
            gate = SpeechGate(silence_ms=s.silence_ms or preset.silence_ms, denoise_fusion=s.denoise)
            for streamer in self.streamers:
                streamer.start()
            self._capture.start()
            self.running = True
            self.sig_state.emit("running")

            silent_since: float | None = None
            while not self._stop_evt.is_set():
                block = self._capture.read(timeout=0.5)
                if block is None:
                    continue
                for b16, prob in dsp.process(block):
                    for ev in gate.feed(b16, prob):
                        for streamer in self.streamers:
                            streamer.on_gate_event(ev)
                # Dead-flat signal for 5 s -> probably a mic permission issue
                if self._capture.rms < 1e-6:
                    if silent_since is None:
                        silent_since = time.monotonic()
                    elif time.monotonic() - silent_since > 5.0:
                        silent_since = None
                        self.sig_notice.emit(
                            "Niveau micro nul depuis 5 s — vérifiez la permission micro "
                            "(macOS : Réglages > Confidentialité > Microphone) et le périphérique."
                        )
                else:
                    silent_since = None
        except Exception as exc:
            log.exception("Pipeline failed")
            self.sig_error.emit(str(exc))
        finally:
            self.running = False
            if self._capture is not None:
                self._capture.stop()
                self._capture = None
            for streamer in self.streamers:
                streamer.stop()
            for channel, store in zip(self.channels, self.stores, strict=False):
                store.close()
                if s.export_on_stop and store.segments and self.session_dir:
                    self._export(store, channel, s)
            for engine in engines:
                _close_quietly(engine)
            self.sig_state.emit("idle")

    def _export(self, store: TranscriptStore, channel: ChannelSpec, s: Settings) -> None:
        assert self.session_dir is not None
        base = self.session_dir
        store.export_txt(base / f"transcript_export{channel.suffix}.txt", s.txt_timestamps)
        store.export_srt(base / f"transcript{channel.suffix}.srt")
        store.export_vtt(base / f"transcript{channel.suffix}.vtt")


class MainWindow(QMainWindow):
    def __init__(self, files: list[str] | None = None):
        super().__init__()
        self.settings = load_settings()
        self.setWindowTitle(APP_DISPLAY_NAME)
        # Fichiers reçus en ligne de commande (« Ouvrir avec… ») : la fenêtre de
        # transcription s'ouvrira dessus une fois l'application prête.
        self._pending_files = [Path(f) for f in files or []]
        self.engines = EngineCache()
        self.controller = PipelineController(self.engines, self)
        self.prewarmer = EnginePrewarmer(self.engines, self)
        self.prewarmer.sig_status.connect(self._on_prewarm_status)
        self._t_session_start: float | None = None
        self._lags: dict[int, deque[float]] = {}
        self._degraded = False

        from ecoutemoi.core.pacing import TextPacer
        from ecoutemoi.core.publish import TextPublishers
        from ecoutemoi.gui.overlay import OverlayWindow

        # Un cadenceur par canal : le débit de lecture est borné indépendamment
        # pour chaque langue (une traduction anglaise n'a pas la même longueur).
        self.pacers = [self._new_pacer(TextPacer) for _ in range(2)]
        self.overlay = OverlayWindow(self.settings)
        self.overlay_dual = OverlayWindow(
            self.settings, title_suffix=" 2", geometry=self.settings.dual_overlay_geometry
        )
        self.publishers = TextPublishers(on_status=self._on_publish_status)
        self._build_ui()
        self._build_menus()
        self._connect_controller()
        self._restore_geometry()
        self._refresh_models()
        self._refresh_devices()
        self._apply_style_live()
        self._validate_start()
        self._apply_publishers()

        self._vu_timer = QTimer(self)
        self._vu_timer.setInterval(50)  # 20 Hz vumeter
        self._vu_timer.timeout.connect(lambda: self.vu.set_rms(self.controller.rms))
        self._clock = QTimer(self)
        self._clock.setInterval(1000)
        self._clock.timeout.connect(self._tick_status)
        # Cadencement : 25 Hz suffit largement (le débit plafond est de ~18
        # caractères/s, soit un mot toutes ~350 ms).
        self._pace_timer = QTimer(self)
        self._pace_timer.setInterval(40)
        self._pace_timer.timeout.connect(self._pace_tick)
        self._pace_timer.start()
        # L'avertissement doit passer AVANT le préchauffage, et le préchauffage
        # avant tout le reste : d'où l'enchaînement en un seul différé.
        #
        # Le `self` en deuxième argument n'est pas décoratif : il rattache le
        # différé au cycle de vie de la fenêtre. Sans lui, une fenêtre détruite
        # avant que le différé ne se déclenche laisse Qt appeler `_after_show`
        # sur un arbre de widgets dont les objets C++ n'existent plus
        # (« libshiboken: Internal C++ object (NoticeBanner) already deleted »).
        # Les évènements postés — dont le DeferredDelete de la fenêtre — sont
        # traités AVANT les minuteries : l'ordre joue contre nous.
        QTimer.singleShot(0, self, self._after_show)

    def _after_show(self) -> None:
        from shiboken6 import isValid

        from ecoutemoi.config import accept_disclaimer, disclaimer_accepted

        first_run = not disclaimer_accepted()
        if first_run:
            DisclaimerDialog(self, first_run=True).exec()
            # L'acceptation est enregistrée quoi qu'il arrive ensuite : l'opérateur
            # A lu et validé, même si l'application se ferme dans la foulée.
            accept_disclaimer()
            # `.exec()` fait tourner une boucle d'évènements IMBRIQUÉE. Pendant que
            # l'avertissement est à l'écran, la fenêtre peut disparaître — fermeture
            # de l'application, ou simplement un DeferredDelete déjà en attente que
            # la boucle imbriquée exécute. Tout ce qui suit toucherait alors des
            # objets C++ détruits (« Internal C++ object already deleted »), avec
            # une trace illisible parce que la faute est ici, pas là-bas.
            if not isValid(self):
                return
        self.prewarmer.request(self.settings)
        if self._pending_files:  # « Ouvrir avec… » : on va droit au but
            self._open_transcribe()
        elif first_run:
            # Une bannière, pas une deuxième fenêtre modale : l'opérateur vient
            # d'en fermer une, et il n'y a rien d'urgent à lui faire signer.
            self.banner.show_notice(
                "Première utilisation ? « Prise en main » explique la mise en route en une page.",
                action_label="Ouvrir la prise en main",
            )
            self._banner_action = self._open_quickstart

    def _new_pacer(self, factory):
        return factory(
            wpm=self.settings.reading_wpm,
            max_lag_s=self.settings.pacing_max_lag_s,
            enabled=self.settings.pacing_enabled,
        )

    def _views(self, index: int):
        """Vues alimentées par un canal : (prévisualisation, fenêtre de sortie)."""
        if index == 0:
            return self.preview, self.overlay.view
        return None, self.overlay_dual.view

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        self.banner = NoticeBanner()
        # Le bouton de la bannière change de sens selon le message affiché ; le
        # dernier message qui en propose un désigne l'action.
        self._banner_action = self._restore_partials
        self.banner.action_clicked.connect(lambda: self._banner_action())
        root.addWidget(self.banner)

        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(12)
        grid = QGridLayout()
        grid.setSpacing(12)
        grid.addWidget(self._box_audio(), 0, 0)
        grid.addWidget(self._box_model(), 0, 1)
        grid.addWidget(self._box_settings(), 1, 0)
        grid.addWidget(self._box_appearance(), 1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        root.addLayout(grid)
        root.addWidget(self._box_preview(), 1)
        root.addWidget(self._box_session())
        root.addLayout(self._row_start())
        self.setCentralWidget(central)
        self._build_status_bar()

    def _build_status_bar(self) -> None:
        """Barre d'état : le voyant à gauche, les mesures à droite.

        Chaque valeur porte une infobulle qui dit ce qu'elle mesure et à quoi elle
        sert. « RTF 4,2 » ne veut rien dire pour qui n'a pas écrit le code — et
        c'est pourtant l'indicateur qui décide de tout.
        """
        sb = self.statusBar()
        sb.setSizeGripEnabled(False)
        self.rt_light = StatusLight()
        sb.addWidget(self.rt_light)  # à gauche : le voyant doit sauter aux yeux
        # Progression du préchauffage dans un libellé À NOUS, pas via
        # showMessage() : ce dernier masque tous les widgets non permanents — donc
        # le voyant temps réel, qui est précisément ce qu'il ne faut pas cacher.
        self.st_prewarm = QLabel("")
        mark(self.st_prewarm, "hint")
        sb.addWidget(self.st_prewarm, 1)
        self.st_backend = QLabel("Backend : —")
        self.st_rtf = QLabel("RTF : —")
        self.st_rtf.setToolTip(
            "Facteur temps réel : durée d'audio décodée par seconde de calcul. "
            "1 = juste à la limite, 3 = confortable. En dessous de 1,3 le direct "
            "prend du retard qu'il ne rattrapera pas."
        )
        self.st_lat = QLabel("Retard : —")
        self.st_lat.setToolTip(
            "Délai entre ce qui est prononcé et ce qui s'affiche, mesuré de bout "
            "en bout (détection de parole, décodage, cadencement)."
        )
        self.st_win = QLabel("Fenêtre : —")
        self.st_win.setToolTip(
            "Durée d'audio redécodée à chaque passe. Elle rétrécit d'elle-même "
            "quand la machine sature, et regrandit quand elle respire."
        )
        self.st_dur = QLabel("Durée : 00:00")
        self.st_dur.setToolTip("Temps écoulé depuis le démarrage de la session.")
        self.st_words = QLabel("Mots : 0")
        self.st_words.setToolTip("Mots validés depuis le début de la session.")
        self.st_pace = QLabel("")
        self.st_pace.setToolTip(
            "Retard introduit VOLONTAIREMENT par le débit de lecture constant "
            "(Réglages avancés → Sous-titrage). Il se résorbe dès que l'orateur "
            "ralentit ; le plafond évite qu'il grandisse sans fin."
        )
        self.st_out = QLabel("")
        self.st_out.setToolTip("Sorties externes actives (Réglages avancés → Diffusion)")
        for lbl in (self.st_out, self.st_backend, self.st_rtf, self.st_lat,
                    self.st_win, self.st_pace, self.st_dur, self.st_words):  # fmt: skip
            sb.addPermanentWidget(lbl)

    def _box_audio(self) -> QGroupBox:
        box = QGroupBox("Entrée audio")
        lay = QGridLayout(box)
        lay.setColumnStretch(1, 1)
        self.device = QComboBox()
        refresh = QPushButton("↻")
        refresh.setObjectName("icon")
        refresh.setFixedWidth(30)
        refresh.setToolTip("Re-scanner les périphériques d'entrée")
        refresh.clicked.connect(self._refresh_devices)
        self.vu = VuMeter()
        self.gain = QDoubleSpinBox(minimum=0.25, maximum=4.0, singleStep=0.05,
                                   value=self.settings.gain)  # fmt: skip
        self.gain.setMaximumWidth(110)
        self.gain.setToolTip(
            "Multiplie le signal d'entrée avant tout traitement. À n'utiliser que "
            "si le niveau reste dans le bas de l'échelle malgré le réglage système."
        )
        self.audio_clean = QComboBox()
        self.audio_clean.addItem("Aucun traitement (recommandé)", "none")
        self.audio_clean.addItem("Passe-haut 80 Hz (anti-grondement)", "highpass")
        self.audio_clean.addItem("Passe-haut + RNNoise (bruit fort constant)", "full")
        self.audio_clean.setToolTip(
            "Nettoyage du micro, désactivé par défaut : whisper est déjà robuste au "
            "bruit et tout filtrage ajoute des artefacts. RNNoise peut dégrader la "
            "transcription (mesuré au WER) — réservez-le aux souffles constants "
            "type ventilation."
        )
        self.audio_clean.setCurrentIndex(self.audio_clean.findData(self._audio_clean_key(self.settings)))
        lay.addWidget(QLabel("Micro"), 0, 0)
        lay.addWidget(self.device, 0, 1)
        lay.addWidget(refresh, 0, 2)
        lay.addWidget(self.vu, 1, 0, 1, 3)
        lay.addWidget(QLabel("Gain"), 2, 0)
        lay.addWidget(self.gain, 2, 1)
        lay.addWidget(QLabel("Nettoyage"), 3, 0)
        lay.addWidget(self.audio_clean, 3, 1, 1, 2)
        return box

    @staticmethod
    def _audio_clean_key(s: Settings) -> str:
        if s.denoise:
            return "full"  # RNNoise implique le passe-haut (fusion des filtres)
        return "highpass" if s.highpass else "none"

    def _box_model(self) -> QGroupBox:
        box = QGroupBox("Modèle et langue")
        lay = QGridLayout(box)
        self.model = QComboBox()
        self.model.currentIndexChanged.connect(self._validate_start)
        manage = QPushButton("Gérer les modèles…")
        manage.clicked.connect(self._open_model_manager)
        self.mode_fr = QRadioButton("FR → FR")
        self.mode_tr = QRadioButton("FR → EN")
        self.mode_auto = QRadioButton("Auto → EN")
        self.mode_auto.setToolTip(
            "Langue parlée détectée en continu, sous-titres toujours traduits en anglais."
        )
        {"fr": self.mode_fr, "translate": self.mode_tr, "auto": self.mode_auto}[
            self.settings.mode
        ].setChecked(True)
        for rb in (self.mode_fr, self.mode_tr, self.mode_auto):
            rb.toggled.connect(self._validate_start)
        self.backend_combo = QComboBox()
        self.backend_combo.addItem("Auto — GPU si disponible, sinon CPU", "auto")
        self.backend_combo.addItem(f"GPU ({GPU_BACKEND_NAME}) — whisper.cpp", "gpu")
        self.backend_combo.addItem("CPU — faster-whisper", "cpu")
        self.backend_combo.setToolTip(
            "Deux moteurs, chacun là où il gagne :\n"
            f"• GPU ({GPU_BACKEND_NAME}) : whisper.cpp — le seul à savoir parler "
            "Vulkan et Metal.\n"
            "• CPU : faster-whisper — environ trois fois plus rapide que "
            "whisper.cpp sur CPU, mais aveugle au GPU.\n\n"
            "Chaque moteur a son propre format de modèle : changer de backend "
            "peut demander un téléchargement (voir « Gérer les modèles »)."
        )
        if self.settings.backend in BACKENDS:
            self.backend_combo.setCurrentIndex(BACKENDS.index(self.settings.backend))
        self.backend_combo.currentIndexChanged.connect(self._backend_changed)
        self.gpu_combo = QComboBox()
        self.gpu_combo.setToolTip(
            "Périphérique GPU à utiliser (machines multi-GPU) ; un index invalide "
            "retombe automatiquement sur le GPU 0."
        )
        self._populate_gpu_combo([], want=self.settings.gpu_device)

        # Second sous-titre : une case + son propre mode et son propre modèle.
        # Deux fenêtres de sortie, deux transcripts, deux sources OBS.
        self.dual_check = QCheckBox("Second sous-titre :")
        self.dual_check.setChecked(self.settings.dual_enabled)
        self.dual_check.setToolTip(
            "Deuxième flux simultané dans SA propre fenêtre de sortie (« "
            f"{OVERLAY_WINDOW_TITLE} 2 »), avec son transcript et sa source OBS.\n\n"
            "Deux moteurs tournent alors en parallèle : comptez le double de RAM et "
            "de charge de calcul. Vérifiez le voyant temps réel après activation."
        )
        self.dual_mode = QComboBox()
        for key, label in MODE_LABELS.items():
            self.dual_mode.addItem(label, key)
        self.dual_mode.setCurrentIndex(max(0, self.dual_mode.findData(self.settings.dual_mode)))
        self.dual_model = QComboBox()
        self.dual_model.setToolTip(
            "Modèle du second sous-titre. « Même modèle » réutilise celui du "
            "principal — deux instances distinctes, pas de partage de mémoire."
        )
        self.dual_check.toggled.connect(self._dual_toggled)
        self.dual_mode.currentIndexChanged.connect(self._validate_start)
        self.dual_model.currentIndexChanged.connect(self._validate_start)

        self.model_warn = QLabel("")
        self.model_warn.setWordWrap(True)
        mark(self.model_warn, "warn")
        modes = QHBoxLayout()
        modes.setSpacing(14)
        for rb in (self.mode_fr, self.mode_tr, self.mode_auto):
            modes.addWidget(rb)
        modes.addStretch(1)
        dual = QHBoxLayout()
        dual.setSpacing(8)
        dual.addWidget(self.dual_mode, 1)
        dual.addWidget(self.dual_model, 1)
        lay.setColumnStretch(1, 1)
        lay.addWidget(QLabel("Modèle"), 0, 0)
        lay.addWidget(self.model, 0, 1)
        lay.addWidget(manage, 0, 2)
        lay.addWidget(QLabel("Langue"), 1, 0)
        lay.addLayout(modes, 1, 1, 1, 2)
        lay.addWidget(self.dual_check, 2, 0)
        lay.addLayout(dual, 2, 1, 1, 2)
        lay.addWidget(QLabel("Backend"), 3, 0)
        lay.addWidget(self.backend_combo, 3, 1, 1, 2)
        lay.addWidget(QLabel("GPU"), 4, 0)
        lay.addWidget(self.gpu_combo, 4, 1, 1, 2)
        lay.addWidget(self.model_warn, 5, 0, 1, 3)
        # Pas d'appel à _dual_toggled ici : btn_start n'existe pas encore. L'état
        # complet est établi par __init__ (_validate_start + _apply_publishers).
        self.dual_mode.setEnabled(self.settings.dual_enabled)
        self.dual_model.setEnabled(self.settings.dual_enabled)
        return box

    def _dual_toggled(self, on: bool) -> None:
        self.dual_mode.setEnabled(on)
        self.dual_model.setEnabled(on)
        self.settings = dataclasses.replace(self.settings, dual_enabled=on)
        self.overlay_dual.setVisible(on and self.overlay.isVisible())
        if not on:
            self.pacers[1].clear()
            self.overlay_dual.view.clear()
        self._validate_start()
        self._apply_publishers()

    def _populate_gpu_combo(self, devices: list[tuple[int, str]], want: int | None = None) -> None:
        """(Re)peuple le sélecteur de GPU ; conserve la sélection courante."""
        if want is None:
            want = self.gpu_combo.currentData() if self.gpu_combo.count() else self.settings.gpu_device
            if want is None:
                want = self.settings.gpu_device
        self.gpu_combo.blockSignals(True)
        self.gpu_combo.clear()
        if devices:
            for i, name in devices:
                self.gpu_combo.addItem(f"{i} — {name}", i)
        else:
            self.gpu_combo.addItem("automatique", 0)
        if self.gpu_combo.findData(want) < 0:  # index hérité d'une autre machine
            self.gpu_combo.addItem(f"{want} — (index manuel)", int(want))
        self.gpu_combo.setCurrentIndex(self.gpu_combo.findData(want))
        self.gpu_combo.blockSignals(False)
        self._backend_changed()

    def _backend_changed(self) -> None:
        self.gpu_combo.setEnabled(self.backend_combo.currentData() != "cpu")

    def _box_settings(self) -> QGroupBox:
        """Point d'entrée de TOUS les réglages du direct — pas seulement la latence."""
        box = QGroupBox("Réglages")
        lay = QGridLayout(box)
        self.preset = QComboBox()
        for key, p in PRESETS.items():
            self.preset.addItem(p.label, key)
        self.preset.setCurrentIndex(list(PRESETS).index(self.settings.preset))
        self.preset.setToolTip(
            "Compromis latence / stabilité. Ultra réagit le plus vite, Stable "
            "valide plus prudemment, Phrase ne décode qu'en fin d'énoncé."
        )
        self.btn_lexicon = QPushButton("Lexique de la conférence…")
        self.btn_lexicon.clicked.connect(self._open_lexicon)
        adv = QPushButton("Réglages avancés…")
        adv.clicked.connect(self._open_settings)
        lay.setColumnStretch(1, 1)
        lay.addWidget(QLabel("Latence"), 0, 0)
        lay.addWidget(self.preset, 0, 1)
        lay.addWidget(self.btn_lexicon, 1, 0, 1, 2)
        lay.addWidget(adv, 2, 0, 1, 2)
        lay.addWidget(
            hint_label(
                "Le <b>lexique</b> est le réglage qui rapporte le plus : les noms "
                "propres et les acronymes du talk, écrits comme ils doivent "
                "apparaître. Tout le reste a des valeurs par défaut mesurées."
            ),
            3, 0, 1, 2,
        )  # fmt: skip
        lay.setRowStretch(4, 1)  # l'espace libre tombe en bas, pas au milieu
        self._refresh_lexicon_button()
        return box

    def _refresh_lexicon_button(self) -> None:
        words = [w for w in self.settings.lexicon.replace("\n", ",").split(",") if w.strip()]
        self.btn_lexicon.setText(
            f"Lexique de la conférence… ({len(words)} entrées)" if words else "Lexique de la conférence…"
        )
        self.btn_lexicon.setToolTip(
            "Noms propres, produits et acronymes du talk. Whisper les reçoit comme "
            "amorce de transcription à CHAQUE fenêtre décodée : c'est le levier le "
            "plus efficace sur les mots que la quantization dégrade en premier."
        )

    def _open_lexicon(self) -> None:
        dlg = LexiconDialog(self.settings.lexicon, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.settings = dataclasses.replace(self.settings, lexicon=dlg.text())
            self._refresh_lexicon_button()
            # Le lexique fait partie de la clé moteur : le préchauffé n'est plus bon.
            self.prewarmer.request(self.settings)

    def _box_appearance(self) -> QGroupBox:
        from ecoutemoi.gui.settings_dialog import BG_PRESETS

        box = QGroupBox("Apparence")
        lay = QGridLayout(box)
        self.font = QFontComboBox()
        if self.settings.font_family:
            self.font.setCurrentText(self.settings.font_family)
        self.font_size = QSpinBox(minimum=10, maximum=200, value=self.settings.font_size)
        self.c_text = ColorButton(self.settings.text_color)
        self.c_outline = ColorButton(self.settings.outline_color)
        self.c_bg = ColorButton(self.settings.bg_color)
        self.bg_preset = QComboBox()
        self.bg_preset.addItem("personnalisé", "")
        for label, color in BG_PRESETS:
            self.bg_preset.addItem(label, color)
        self.bg_preset.addItem("Transparent — incrustation directe (sans OBS)", "__transparent__")
        self.bg_preset.setToolTip(
            "Vert/Magenta : à incruster dans OBS (chroma). Transparent : les "
            "sous-titres flottent directement au-dessus des autres fenêtres — "
            "idéal sans OBS ; la capture OBS, elle, ne préserve pas l'alpha."
        )
        if self.settings.overlay_transparent:
            self.bg_preset.setCurrentIndex(self.bg_preset.findData("__transparent__"))
        elif self.settings.bg_color.upper() == "#00FF00":
            self.bg_preset.setCurrentIndex(1)
        self.outline_w = QSpinBox(minimum=2, maximum=12, value=self.settings.outline_width)
        self.lines = QSpinBox(minimum=1, maximum=3, value=self.settings.max_lines)
        self.align = QComboBox()
        self.align.addItem("Gauche", "left")
        self.align.addItem("Centre", "center")
        self.align.addItem("Droite", "right")
        self.align.setCurrentIndex(["left", "center", "right"].index(self.settings.align))
        # Les petits réglages numériques restent petits : un « 3 » dans un champ
        # large se lit comme une valeur importante, ce qu'il n'est pas.
        for small in (self.font_size, self.outline_w, self.lines):
            small.setMaximumWidth(80)
        self.font_size.setSuffix(" px")
        self.font_size.setToolTip("Hauteur du texte dans la fenêtre de sortie, en pixels.")
        self.outline_w.setSuffix(" px")
        self.outline_w.setToolTip(
            "Épaisseur du contour noir. Deux pixels au minimum : c'est lui qui "
            "masque le liseré vert laissé par l'incrustation chromatique."
        )
        self.lines.setToolTip("Nombre de lignes de sous-titre affichées simultanément.")
        colors = QHBoxLayout()
        colors.setSpacing(8)
        colors.addWidget(QLabel("Texte"))
        colors.addWidget(self.c_text)
        colors.addWidget(QLabel("Contour"))
        colors.addWidget(self.c_outline)
        colors.addWidget(QLabel("Fond"))
        colors.addWidget(self.c_bg)
        colors.addStretch(1)
        lay.setColumnStretch(1, 1)
        lay.addWidget(QLabel("Police"), 0, 0)
        lay.addWidget(self.font, 0, 1)
        lay.addWidget(self.font_size, 0, 2, 1, 2)
        lay.addWidget(QLabel("Sous-titres"), 1, 0)
        lay.addLayout(colors, 1, 1, 1, 3)
        lay.addWidget(QLabel("Préréglage de fond"), 2, 0)
        lay.addWidget(self.bg_preset, 2, 1, 1, 3)
        lay.addWidget(QLabel("Contour"), 3, 0)
        lay.addWidget(self.outline_w, 3, 1)
        lay.addWidget(QLabel("Lignes"), 3, 2, Qt.AlignmentFlag.AlignRight)
        lay.addWidget(self.lines, 3, 3)
        self.display_style = QComboBox()
        for key, label in DISPLAY_STYLES.items():
            self.display_style.addItem(label, key)
        idx = self.display_style.findData(self.settings.display_style)
        self.display_style.setCurrentIndex(max(0, idx))
        self.display_style.setToolTip(
            "Défilement : les lignes glissent vers le haut. Fondu : les nouveaux "
            "mots émergent du fond. Statique : aucun mouvement."
        )
        self.margin_h = QSpinBox(minimum=0, maximum=300, value=self.settings.margin_h)
        self.margin_v = QSpinBox(minimum=0, maximum=300, value=self.settings.margin_v)
        for margin, tip in (
            (self.margin_h, "Marge gauche et droite, en pixels."),
            (self.margin_v, "Marge basse : distance entre le texte et le bord bas."),
        ):
            margin.setMaximumWidth(80)
            margin.setSuffix(" px")
            margin.setToolTip(tip)
        self.on_top = QCheckBox("Fenêtre de sortie toujours au premier plan")
        self.on_top.setChecked(self.settings.always_on_top)
        margins = QHBoxLayout()
        margins.setSpacing(8)
        margins.addWidget(self.margin_h)
        margins.addWidget(QLabel("vertical"))
        margins.addWidget(self.margin_v)
        margins.addStretch(1)
        lay.addWidget(QLabel("Alignement"), 4, 0)
        lay.addWidget(self.align, 4, 1, 1, 3)
        lay.addWidget(QLabel("Animation"), 5, 0)
        lay.addWidget(self.display_style, 5, 1, 1, 3)
        lay.addWidget(QLabel("Marges"), 6, 0)
        lay.addLayout(margins, 6, 1, 1, 3)
        lay.addWidget(self.on_top, 7, 0, 1, 4)
        for w in (self.font_size, self.outline_w, self.lines, self.margin_h, self.margin_v):
            w.valueChanged.connect(self._apply_style_live)
        self.font.currentFontChanged.connect(self._apply_style_live)
        self.align.currentIndexChanged.connect(self._apply_style_live)
        self.display_style.currentIndexChanged.connect(self._apply_style_live)
        self.bg_preset.currentIndexChanged.connect(self._bg_preset_changed)
        self.on_top.toggled.connect(self._on_top_toggled)
        for cb in (self.c_text, self.c_outline, self.c_bg):
            cb.color_changed.connect(self._apply_style_live)
        return box

    def _on_top_toggled(self, on: bool) -> None:
        self.settings = dataclasses.replace(self.settings, always_on_top=on)
        for overlay in (self.overlay, self.overlay_dual):
            overlay.set_always_on_top(on)

    def _box_preview(self) -> QGroupBox:
        box = QGroupBox("Prévisualisation des sous-titres (rendu identique à la fenêtre OBS)")
        lay = QVBoxLayout(box)
        self.preview = SubtitleView(SubtitleStyle.from_settings(self.settings))
        self.preview.setMinimumHeight(130)
        self.chk_checker = QCheckBox("Damier (visualiser la zone chroma)")
        self.chk_checker.setChecked(self.settings.preview_checker)
        self.chk_checker.toggled.connect(self._toggle_checker)
        self.chk_sample = QCheckBox("Texte d'exemple")
        self.chk_sample.toggled.connect(self._toggle_sample)
        row = QHBoxLayout()
        row.addWidget(self.chk_checker)
        row.addWidget(self.chk_sample)
        row.addStretch(1)
        lay.addWidget(self.preview, 1)
        lay.addLayout(row)
        return box

    def _box_session(self) -> QGroupBox:
        """Où va le texte de la session, et comment le sortir.

        Un seul bouton d'export : le menu <i>Fichier → Exporter la session</i>
        offre les huit formats, aligner six boutons ici ne ferait que du bruit
        au-dessus du bouton qui compte vraiment.
        """
        box = QGroupBox("Session")
        lay = QHBoxLayout(box)
        self.session_label = QLabel(self._session_root_text())
        self.session_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.session_label.setToolTip(
            "Le transcript est écrit au fil de l'eau dans ce dossier : une coupure "
            "de courant ne coûte pas la conférence."
        )
        btn_open = QPushButton("Ouvrir le dossier")
        btn_open.clicked.connect(self._open_session_dir)
        self.btn_export = QPushButton("Exporter…")
        self.btn_export.setToolTip(
            "Texte, Markdown, SRT, WebVTT, JSON, CSV, TSV, LRC — ou n'importe "
            "quelle autre extension, qui recevra du texte brut."
        )
        self.btn_export.clicked.connect(self._export_as)
        btn_clear = QPushButton("Effacer l'affichage")
        btn_clear.setToolTip(
            "Vide les sous-titres affichés (et les sorties externes). Le transcript "
            "déjà écrit n'est pas touché."
        )
        btn_clear.clicked.connect(self._clear_views)
        lay.addWidget(self.session_label, 1)
        lay.addWidget(btn_open)
        lay.addWidget(self.btn_export)
        lay.addWidget(btn_clear)
        return box

    def _row_start(self) -> QHBoxLayout:
        """Pied de fenêtre : les réglages à gauche, le direct à droite.

        Le bouton qui met l'application à l'antenne est le plus gros de la fenêtre
        et le seul en couleur pleine : à quinze secondes du début d'un talk, on ne
        doit pas avoir à le chercher.
        """
        row = QHBoxLayout()
        row.setSpacing(8)
        btn_save = QPushButton("Sauvegarder les réglages")
        btn_save.setToolTip(
            "Les réglages ne sont JAMAIS enregistrés automatiquement : ce bouton "
            "écrit l'état actuel (y compris la position des fenêtres) dans "
            "settings.json.\n\nRaccourci : Ctrl+S"
        )
        btn_save.clicked.connect(self._save_settings_clicked)
        btn_reset = QPushButton("Réinitialiser")
        btn_reset.setObjectName("danger")
        btn_reset.setToolTip(
            "Revient à TOUS les réglages par défaut et efface le fichier "
            "settings.json enregistré. Confirmation demandée."
        )
        btn_reset.clicked.connect(self._clear_saved_settings)
        self.btn_overlay = QPushButton("Afficher la fenêtre de sortie OBS")
        self.btn_overlay.setCheckable(True)
        self.btn_overlay.setToolTip(
            "La fenêtre à fond chroma que capture OBS. Elle s'ouvre de toute façon "
            "au démarrage du sous-titrage.\n\nRaccourci : Ctrl+M"
        )
        self.btn_overlay.toggled.connect(self._toggle_overlay)
        self.btn_start = QPushButton("Démarrer le sous-titrage")
        self.btn_start.setObjectName("primary")
        self.btn_start.setCheckable(True)
        self.btn_start.setMinimumHeight(46)
        self.btn_start.setToolTip("Ouvre le micro et lance le direct.\n\nRaccourci : Ctrl+Entrée")
        self.btn_start.toggled.connect(self._toggle_start)
        row.addWidget(btn_save)
        row.addWidget(btn_reset)
        row.addStretch(1)
        row.addWidget(self.btn_overlay)
        row.addWidget(self.btn_start, 1)
        return row

    # ------------------------------------------------------------------ menus
    def _menu_action(self, menu, label: str, slot, *, shortcut: str = "", tip: str = "") -> QAction:
        act = QAction(label, self)
        if shortcut:
            act.setShortcut(shortcut)
        if tip:
            act.setStatusTip(tip)
            act.setToolTip(tip)
        act.triggered.connect(slot)
        menu.addAction(act)
        return act

    def _build_menus(self) -> None:
        from ecoutemoi.core.transcript import TRANSCRIPT_FORMATS

        m_file = self.menuBar().addMenu("Fichier")
        self._menu_action(
            m_file, "Transcrire des fichiers audio…", self._open_transcribe,
            shortcut="Ctrl+O",
            tip="Transcrire des enregistrements en fichiers texte, hors direct",
        )  # fmt: skip
        m_file.addSeparator()
        # Référence gardée côté Python : un sous-menu créé par addMenu() n'est pas
        # toujours considéré comme possédé par le C++, et le ramasse-miettes le
        # détruisait derrière nous.
        self.menu_export = m_export = m_file.addMenu("Exporter la session")
        for key, fmt in TRANSCRIPT_FORMATS.items():
            self._menu_action(
                m_export, f"{fmt.label} ({fmt.extension})", lambda _=False, k=key: self._export(k)
            )
        m_export.addSeparator()
        self._menu_action(m_export, "Enregistrer sous…", self._export_as, shortcut="Ctrl+Shift+S")
        self._menu_action(
            m_file, "Ouvrir le dossier de session", self._open_session_dir, shortcut="Ctrl+Shift+O"
        )
        m_file.addSeparator()
        self._menu_action(m_file, "Quitter", self.close, shortcut="Ctrl+Q")

        m_session = self.menuBar().addMenu("Direct")
        self._menu_action(
            m_session, "Démarrer / arrêter le sous-titrage", self.btn_start.toggle, shortcut="Ctrl+Return"
        )
        self._menu_action(
            m_session, "Afficher / masquer la fenêtre de sortie", self.btn_overlay.toggle, shortcut="Ctrl+M"
        )
        m_session.addSeparator()
        self._menu_action(m_session, "Lexique de la conférence…", self._open_lexicon, shortcut="Ctrl+L")
        self._menu_action(m_session, "Réglages avancés…", self._open_settings, shortcut="Ctrl+,")
        m_session.addSeparator()
        self._menu_action(m_session, "Effacer les sous-titres affichés", self._clear_views)

        m_tools = self.menuBar().addMenu("Outils")
        self._menu_action(m_tools, "Benchmark — Tester ma machine…", self._open_bench, shortcut="Ctrl+B")
        self._menu_action(m_tools, "Gérer les modèles…", self._open_model_manager, shortcut="Ctrl+G")
        self._menu_action(m_tools, "Diagnostic moteur (GPU/Vulkan)…", self._open_diag)
        self._menu_action(m_tools, "Ouvrir la page web des sous-titres", self._open_web_page)
        m_tools.addSeparator()
        models_dir = models.models_dir
        self._menu_action(m_tools, "Ouvrir le dossier des modèles", lambda: self._open_dir(models_dir()))
        self._menu_action(m_tools, "Ouvrir le dossier des logs", lambda: self._open_dir(log_dir()))
        m_tools.addSeparator()
        self._menu_action(m_tools, "Sauvegarder les réglages", self._save_settings_clicked, shortcut="Ctrl+S")
        self._menu_action(m_tools, "Effacer les réglages enregistrés", self._clear_saved_settings)

        m_help = self.menuBar().addMenu("Aide")
        self._menu_action(m_help, "Prise en main…", self._open_quickstart, shortcut="F1")
        self._menu_action(m_help, "Guide OBS (par système)…", self._open_obs_guide)
        self._menu_action(m_help, "Quantizations — aide-mémoire…", self._open_quant_help)
        m_help.addSeparator()
        self._menu_action(
            m_help, "Avertissement et licence…", lambda: DisclaimerDialog(self, first_run=False).exec()
        )
        self._menu_action(m_help, "À propos…", self._about)

    # ---------------------------------------------------------- controller IO
    def _connect_controller(self) -> None:
        c = self.controller
        c.sig_partial.connect(self._on_partial)
        c.sig_finalized.connect(self._on_finalized)
        c.sig_stats.connect(self._on_stats)
        c.sig_notice.connect(lambda msg: self.banner.show_notice(msg))
        c.sig_degraded.connect(self._on_degraded)
        c.sig_state.connect(self._on_state)
        c.sig_warmup.connect(self._on_warmup)
        c.sig_error.connect(self._on_error)

    # Le texte des canaux ne va PAS directement aux vues : il passe par le
    # cadenceur, qui le libère à débit de lecture constant (voir core/pacing.py).
    def _on_partial(self, index: int, committed: str) -> None:
        if not self.chk_sample.isChecked():
            self.pacers[index].set_live(committed)

    def _on_finalized(self, index: int, seg: SessionSegment) -> None:
        if not self.chk_sample.isChecked():
            self.pacers[index].push_final(seg.text)
        if index == 0:
            store = self.controller.store
            self.st_words.setText(f"Mots : {store.word_count() if store else 0}")

    def _pace_tick(self) -> None:
        """25 Hz : pousse aux vues et aux sorties ce que le débit autorise."""
        if self.chk_sample.isChecked():
            return
        changed = False
        for index, pacer in enumerate(self.pacers):
            if not pacer.tick():
                continue
            changed = True
            text = pacer.text()
            preview, overlay_view = self._views(index)
            if preview is not None:
                preview.set_live(text)
            overlay_view.set_live(text)
        if changed:
            self._publish()
        if self.controller.running:
            lag = self.pacers[0].lag_s()
            self.st_pace.setText(f"Cadence : +{lag:.1f} s" if lag > 0.05 else "Cadence : à jour")

    def _on_stats(self, index: int, s: StreamStats) -> None:
        if index == 0:
            self.st_rtf.setText(f"RTF : {s.rtf:.1f}")
            self.st_lat.setText(f"Retard : {s.latency_ms / 1000:.1f} s")
            self.st_win.setText(f"Fenêtre : {s.window_s:.1f} s")
        self._lags.setdefault(index, deque(maxlen=RT_LIGHT_WINDOW)).append(s.lag)
        self._update_rt_light()

    def _update_rt_light(self) -> None:
        """Voyant temps réel à partir du lag médian (décodage / audio décodé).

        Avec deux canaux, c'est le PIRE des deux qui décide : si l'un décroche, le
        public voit un sous-titre en retard — le voyant doit le dire.
        """
        if self._degraded:  # bascule auto en mode Phrase : la machine a déjà lâché
            self.rt_light.set_level(
                "red", "Bascule automatique en mode Phrase : modèle trop lourd pour le direct."
            )
            return
        medians = [statistics.median(lags) for lags in self._lags.values() if lags]
        if not medians:
            self.rt_light.set_level("off")
            return
        median = max(medians)
        rtf = 1.0 / median if median > 0 else 0.0
        if median <= RT_LIGHT_GREEN_LAG:
            level = "green"
        elif median <= RT_LIGHT_ORANGE_LAG:
            level = "orange"
        else:
            level = "red"
        advice = {
            "green": "Marge confortable.",
            "orange": "Ça tient, sans marge : une pointe de charge fera décrocher le direct.",
            "red": "Modèle trop lourd pour cette machine : prenez un modèle plus petit, "
                   "le preset Phrase, désactivez le second sous-titre, ou activez le GPU.",
        }[level]  # fmt: skip
        scope = " (pire des deux canaux)" if len(medians) > 1 else ""
        self.rt_light.set_level(level, f"RTF médian {rtf:.1f}{scope}. {advice}")

    def _on_degraded(self, degraded: bool) -> None:
        self._degraded = degraded
        self._update_rt_light()
        if degraded:
            self._banner_action = self._restore_partials
            self.banner.show_notice(
                "Machine surchargée : bascule automatique en mode Phrase.",
                action_label="Réactiver les partiels",
                level="warn",
            )

    def _restore_partials(self) -> None:
        for streamer in self.controller.streamers:
            streamer.restore_partials()
        self._degraded = False
        self._lags.clear()
        self._update_rt_light()
        self.banner.hide()

    def _on_warmup(self, index: int, total: int) -> None:
        self.btn_start.setText(f"Préchauffage du moteur… ({index}/{total})")

    def _on_state(self, state: str) -> None:
        if state == "loading":
            self.btn_start.setText("Chargement du modèle…")
            self.btn_start.setEnabled(False)
            self._degraded = False
            self._lags.clear()
            self.rt_light.set_level("off", "Chargement du modèle…")
        elif state == "warmup":
            self.btn_start.setText("Préchauffage du moteur…")
            self.btn_start.setEnabled(False)
            self.rt_light.set_level(
                "off",
                "Préchauffage : le premier décodage (compilation des shaders, caches) "
                "est absorbé avant l'ouverture du micro.",
            )
        elif state == "running":
            self.btn_start.setEnabled(True)
            self.btn_start.setText("Arrêter le sous-titrage")
            self.st_backend.setText(f"Backend : {self.controller.backend}")
            self.st_backend.setToolTip(self._backend_tooltip())
            self._populate_gpu_combo(self.controller.gpu_devices)
            self._t_session_start = time.monotonic()
            self._vu_timer.start()
            self._clock.start()
        else:  # idle
            self._vu_timer.stop()
            self._clock.stop()
            self.vu.set_rms(0.0)
            self.btn_start.setEnabled(True)
            with_sig = self.btn_start.blockSignals(True)
            self.btn_start.setChecked(False)
            self.btn_start.blockSignals(with_sig)
            self.btn_start.setText("Démarrer le sous-titrage")
            self._update_session_label()
            self.rt_light.set_level("off")

    # ------------------------------------------------------------- diffusion
    def _publish(self) -> None:
        """Envoie aux sorties externes exactement les lignes des fenêtres OBS."""
        secondary = self.overlay_dual.view.visible_text() if self.settings.dual_enabled else ""
        self.publishers.set_text(self.overlay.view.visible_text(), secondary)

    def _apply_publishers(self) -> None:
        self.publishers.apply(self.settings)
        self._refresh_outputs_label()
        if self.publishers.web_error:
            self.banner.show_notice(f"Page web : {self.publishers.web_error}", level="warn")

    def _refresh_outputs_label(self) -> None:
        active = self.publishers.active()
        self.st_out.setText(f"Sorties : {' + '.join(active)}" if active else "")

    def _on_publish_status(self, status: str) -> None:
        """Appelé depuis le fil OBS : les widgets ne se touchent QUE via un signal."""
        self.controller.sig_notice.emit(status)

    def _on_error(self, msg: str) -> None:
        QMessageBox.critical(self, "Erreur", msg)

    # ----------------------------------------------------------------- actions
    def _toggle_start(self, on: bool) -> None:
        if on:
            # Volontairement AUCUNE sauvegarde implicite : les réglages ne sont
            # écrits que par le bouton « Sauvegarder les réglages ».
            self._collect_settings()
            self.chk_sample.setChecked(False)
            self._clear_views()
            if not self.overlay.isVisible():
                self.btn_overlay.setChecked(True)
            self.controller.start(self.settings)
        else:
            self.controller.stop()

    def _toggle_overlay(self, on: bool) -> None:
        self.overlay.setVisible(on)
        self.overlay_dual.setVisible(on and self.settings.dual_enabled)
        self.btn_overlay.setText(
            "Masquer la fenêtre de sortie OBS" if on else "Afficher la fenêtre de sortie OBS"
        )

    def _toggle_checker(self, on: bool) -> None:
        self.preview.set_checker(on)  # preview only; the overlay keeps the real bg

    def _toggle_sample(self, on: bool) -> None:
        if on:
            self.preview.set_texts(SAMPLE_COMMITTED)
        else:
            self.preview.clear()

    def _bg_preset_changed(self) -> None:
        data = self.bg_preset.currentData()
        if data and data != "__transparent__":
            self.c_bg.set_color(data)
        self._apply_style_live()

    def _apply_style_live(self) -> None:
        self._collect_appearance()
        style = SubtitleStyle.from_settings(self.settings)
        self.preview.set_style(style)
        for overlay in (self.overlay, self.overlay_dual):
            overlay.view.set_style(style)
            overlay.set_transparent(self.settings.overlay_transparent)
        # La page web suit la même apparence ; sans redémarrage de serveur, elle
        # ne bouge qu'au prochain chargement de l'onglet.
        self.publishers.apply(self.settings)

    def _collect_appearance(self) -> None:
        self.settings = dataclasses.replace(
            self.settings,
            font_family=self.font.currentText(),
            font_size=self.font_size.value(),
            text_color=self.c_text.color(),
            outline_color=self.c_outline.color(),
            bg_color=self.c_bg.color(),
            outline_width=self.outline_w.value(),
            max_lines=self.lines.value(),
            align=self.align.currentData(),
            display_style=self.display_style.currentData(),
            overlay_transparent=self.bg_preset.currentData() == "__transparent__",
            margin_h=self.margin_h.value(),
            margin_v=self.margin_v.value(),
            always_on_top=self.on_top.isChecked(),
            preview_checker=self.chk_checker.isChecked(),
        )

    def _collect_settings(self) -> None:
        self._collect_appearance()
        mode = "fr" if self.mode_fr.isChecked() else ("translate" if self.mode_tr.isChecked() else "auto")
        clean = self.audio_clean.currentData()
        self.settings = dataclasses.replace(
            self.settings,
            device_index=self.device.currentData(),
            gain=self.gain.value(),
            denoise=clean == "full",
            highpass=clean in ("highpass", "full"),
            model=self.model.currentText(),
            mode=mode,
            preset=self.preset.currentData(),
            backend=self.backend_combo.currentData(),
            gpu_device=int(self.gpu_combo.currentData() or 0),
            dual_enabled=self.dual_check.isChecked(),
            dual_mode=self.dual_mode.currentData() or "translate",
            dual_model=self.dual_model.currentData() or "",
        )

    def _validate_start(self) -> None:
        """Verrou traduction + avertissement RAM, en direct, sur les DEUX canaux.

        Avec le second sous-titre, la RAM se cumule : deux moteurs, deux modèles
        chargés. C'est le total qui est comparé à la mémoire disponible, sinon
        l'avertissement passerait à côté du cas qui compte.
        """
        import psutil

        mode = "fr" if self.mode_fr.isChecked() else ("translate" if self.mode_tr.isChecked() else "auto")
        wanted: list[tuple[str, str, str]] = [(self.model.currentText(), mode, "principal")]
        if self.dual_check.isChecked():
            second = self.dual_model.currentData() or self.model.currentText()
            wanted.append((second, self.dual_mode.currentData() or "translate", "second sous-titre"))

        problems: list[str] = []
        total_ram = 0.0
        ok = True
        for key, channel_mode, who in wanted:
            spec = models.REGISTRY.get(key)
            if spec is None:
                problems.append(f"⚠ Modèle inconnu ({who}) : {key or '—'}.")
                ok = False
                continue
            total_ram += spec.ram_gb
            if channel_mode in ("translate", "auto") and not spec.translate:
                problems.append(
                    f"⚠ {key} n'est pas entraîné à la traduction ({who}) : il ressortirait "
                    f"la langue source. Choisissez un autre modèle pour "
                    f"{MODE_LABELS[channel_mode]}, ou le mode FR → FR."
                )
                ok = False
        avail = psutil.virtual_memory().available / 1e9
        if ok and total_ram > avail:
            label = "RAM cumulée estimée" if len(wanted) > 1 else "RAM estimée"
            problems.append(f"⚠ {label} ~{total_ram:.1f} Go, disponible {avail:.1f} Go.")
        self.model_warn.setText("\n".join(problems))
        self.btn_start.setEnabled(ok)

    def _refresh_devices(self) -> None:
        from ecoutemoi.core.audio import list_input_devices

        self.device.clear()
        self.device.addItem("Périphérique par défaut", None)
        try:
            for d in list_input_devices():
                self.device.addItem(f"{d.name} ({d.hostapi})", d.index)
        except Exception as exc:
            log.warning("Device scan failed: %s", exc)
        if self.settings.device_index is not None:
            idx = self.device.findData(self.settings.device_index)
            if idx >= 0:
                self.device.setCurrentIndex(idx)

    def _refresh_models(self) -> None:
        current = self.settings.model
        self.model.blockSignals(True)
        self.model.clear()
        installed = models.installed_models()
        for key in installed or list(models.REGISTRY):
            spec = models.REGISTRY[key]
            badge = "trad. EN : oui" if spec.translate else "trad. EN : NON"
            quant_short = models.QUANT_NOTES.get(spec.quant, ("", ""))[0]
            # Deux formats : dire lequel est là évite de découvrir au démarrage
            # qu'un changement de backend implique un téléchargement.
            present = models.installed_formats(spec) or ["aucun"]
            self.model.addItem(key)
            self.model.setItemData(
                self.model.count() - 1,
                f"{spec.role}\n{spec.size_mb} Mo · quantization {spec.quant} "
                f"({quant_short}) · RAM ~{spec.ram_gb:.1f} Go · {badge}\n"
                f"Installé : {' + '.join(present)} "
                f"(ggml = GPU whisper.cpp, ct2 = CPU faster-whisper)",
                Qt.ItemDataRole.ToolTipRole,
            )
        if current in (installed or list(models.REGISTRY)):
            self.model.setCurrentText(current)
        self.model.blockSignals(False)

        # Second sous-titre : « Même modèle » d'abord, puis les mêmes candidats.
        self.dual_model.blockSignals(True)
        wanted_dual = self.settings.dual_model
        self.dual_model.clear()
        self.dual_model.addItem("Même modèle que le principal", "")
        for key in installed or list(models.REGISTRY):
            self.dual_model.addItem(key, key)
        index = self.dual_model.findData(wanted_dual)
        self.dual_model.setCurrentIndex(index if index >= 0 else 0)
        self.dual_model.blockSignals(False)
        self._validate_start()

    # ------------------------------------------------------------ dialogs etc.
    def _open_settings(self) -> None:
        from ecoutemoi.gui.settings_dialog import SettingsDialog

        self._collect_settings()
        dlg = SettingsDialog(self.settings, self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.result_settings:
            # En mémoire seulement (« Sauvegarder les réglages » pour persister).
            self.settings = dlg.result_settings
            self._sync_widgets_from_settings()
            self._apply_style_live()
            self._apply_publishers()
            self._apply_pacing()
            # Modèle, mode, backend ou lexique ont pu changer : le moteur gardé au
            # chaud n'est peut-être plus le bon.
            self.prewarmer.request(self.settings)

    def _apply_pacing(self) -> None:
        for pacer in self.pacers:
            pacer.configure(
                wpm=self.settings.reading_wpm,
                max_lag_s=self.settings.pacing_max_lag_s,
                enabled=self.settings.pacing_enabled,
            )
        if not self.settings.pacing_enabled:
            self.st_pace.setText("")

    def _sync_widgets_from_settings(self) -> None:
        s = self.settings
        self.gain.setValue(s.gain)
        self.audio_clean.setCurrentIndex(self.audio_clean.findData(self._audio_clean_key(s)))
        self.preset.setCurrentIndex(list(PRESETS).index(s.preset))
        if s.backend in BACKENDS:
            self.backend_combo.setCurrentIndex(BACKENDS.index(s.backend))
        self._populate_gpu_combo(self.controller.gpu_devices, want=s.gpu_device)
        {"fr": self.mode_fr, "translate": self.mode_tr, "auto": self.mode_auto}[s.mode].setChecked(True)
        if s.font_family:
            self.font.setCurrentText(s.font_family)
        self.font_size.setValue(s.font_size)
        self.c_text.set_color(s.text_color)
        self.c_outline.set_color(s.outline_color)
        self.c_bg.set_color(s.bg_color)
        self.outline_w.setValue(s.outline_width)
        self.lines.setValue(s.max_lines)
        self.align.setCurrentIndex(["left", "center", "right"].index(s.align))
        self.display_style.setCurrentIndex(max(0, self.display_style.findData(s.display_style)))
        if s.overlay_transparent:
            self.bg_preset.setCurrentIndex(self.bg_preset.findData("__transparent__"))
        elif self.bg_preset.currentData() == "__transparent__":
            self.bg_preset.setCurrentIndex(0)  # revenu à un fond peint
        self.margin_h.setValue(s.margin_h)
        self.margin_v.setValue(s.margin_v)
        self.on_top.setChecked(s.always_on_top)
        for overlay in (self.overlay, self.overlay_dual):
            overlay.set_transparent(s.overlay_transparent)
            overlay.set_always_on_top(s.always_on_top)
        self.dual_check.setChecked(s.dual_enabled)
        self.dual_mode.setCurrentIndex(max(0, self.dual_mode.findData(s.dual_mode)))
        self._refresh_models()
        self._refresh_lexicon_button()
        self.session_label.setText(self._session_root_text())

    def _open_model_manager(self) -> None:
        from ecoutemoi.gui.model_manager_dialog import ModelManagerDialog

        dlg = ModelManagerDialog(self)
        dlg.exec()
        self._refresh_models()

    def _backend_tooltip(self) -> str:
        """Détails moteur au survol du label Backend (variante, GPU, VAD, threads…)."""
        d = self.controller.engine_diag
        if not d:
            return ""
        rows = []
        for key in ("moteur", "modele", "backend", "variante_chargement", "chargement_s",
                    "type_calcul", "flash_attn", "vad", "threads", "gpu_devices",
                    "diagnostic_gpu"):  # fmt: skip
            v = d.get(key)
            if v in (None, [], ""):
                continue
            if isinstance(v, list):
                v = " ; ".join(str(x) for x in v)
            rows.append(f"{key} : {v}")
        return "\n".join(rows)

    def _open_diag(self) -> None:
        dlg = DiagDialog(self.controller.engine_diag, self)
        dlg.exec()

    def _open_bench(self) -> None:
        from ecoutemoi.gui.bench_dialog import BenchDialog

        self._collect_settings()
        dlg = BenchDialog(self.settings, self)
        dlg.exec()
        if dlg.applied:
            model_key, preset_key = dlg.applied
            self.settings = dataclasses.replace(self.settings, model=model_key, preset=preset_key)
            self._refresh_models()
            self.preset.setCurrentIndex(list(PRESETS).index(preset_key))
            self.banner.show_notice(
                f"Recommandation appliquée : modèle {model_key}, preset {preset_key}. "
                f"« Sauvegarder les réglages » pour la conserver.",
                level="ok",
            )

    def _save_settings_clicked(self) -> None:
        self._collect_settings()
        g = self.geometry()
        self.settings = dataclasses.replace(
            self.settings,
            main_geometry=f"{g.x()},{g.y()},{g.width()},{g.height()}",
            overlay_geometry=self.overlay.geometry_string(),
            dual_overlay_geometry=self.overlay_dual.geometry_string(),
        )
        save_settings(self.settings)
        self.banner.show_notice(
            "Réglages sauvegardés — ils seront rechargés au prochain lancement.", level="ok"
        )

    def _clear_saved_settings(self) -> None:
        if (
            QMessageBox.question(
                self,
                "Réinitialiser les réglages",
                "Revenir à TOUS les réglages par défaut et effacer le fichier "
                "settings.json enregistré ?\n\nLa session en cours n'est pas interrompue.",
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        erased = clear_settings()
        self.settings = Settings()
        self._sync_widgets_from_settings()
        self._apply_style_live()
        self._apply_publishers()
        self._apply_pacing()
        self.prewarmer.request(self.settings)
        msg = (
            "Réglages réinitialisés et settings.json effacé ✔"
            if erased
            else "Réglages réinitialisés ✔ (aucun fichier n'était enregistré)"
        )
        self.banner.show_notice(msg, level="ok")

    def _exportable(self):
        """(store, dossier) si la session a quelque chose à exporter, sinon None."""
        store = self.controller.store
        sdir = self.controller.session_dir
        if store is None or not store.segments or sdir is None:
            QMessageBox.information(
                self,
                "Export",
                "Aucun segment à exporter pour l'instant.\n\nCette commande exporte la "
                "session de sous-titrage en cours. Pour transcrire un enregistrement, "
                "utilisez « Fichier → Transcrire des fichiers audio… ».",
            )
            return None
        return store, sdir

    def _export(self, key: str) -> None:
        from ecoutemoi.core.transcript import TRANSCRIPT_FORMATS

        target = self._exportable()
        if target is None:
            return
        store, sdir = target
        fmt = TRANSCRIPT_FORMATS[key]
        # « transcript.txt » appartient à l'autosave : l'export TXT prend un autre
        # nom, sinon on remplacerait l'un par l'autre sans le dire.
        stem = "transcript_export" if key == "txt" else "transcript"
        path = store.export(sdir / f"{stem}{fmt.extension}", key=key,
                            timestamps=self.settings.txt_timestamps)  # fmt: skip
        self.banner.show_notice(f"{fmt.label} écrit : {path}", level="ok")

    def _export_as(self) -> None:
        """Export vers n'importe quel chemin, dans n'importe quelle extension."""
        from ecoutemoi.core.transcript import format_filter
        from ecoutemoi.gui.desktop import pick_save_path

        target = self._exportable()
        if target is None:
            return
        store, sdir = target
        chosen = pick_save_path(self, "Enregistrer le transcript", format_filter(), str(sdir / "transcript"))
        if chosen is None:
            return
        try:
            path = store.export(chosen, timestamps=self.settings.txt_timestamps)
        except OSError as exc:
            QMessageBox.warning(self, "Export", f"Écriture impossible :\n{chosen}\n\n{exc}")
            return
        self.banner.show_notice(f"Transcript écrit : {path}", level="ok")

    def _open_transcribe(self) -> None:
        """Transcription de fichiers : une fenêtre à part, hors du direct."""
        from ecoutemoi.gui.transcribe_dialog import TranscribeDialog

        self._collect_settings()
        dlg = TranscribeDialog(self.settings, self, initial=self._pending_files)
        self._pending_files = []
        dlg.exec()
        if dlg.result_settings is not None:
            self.settings = dlg.result_settings  # en mémoire (« Sauvegarder les réglages » pour persister)

    def _clear_views(self) -> None:
        for pacer in self.pacers:
            pacer.clear()
        self.preview.clear()
        self.overlay.view.clear()
        self.overlay_dual.view.clear()
        self.st_pace.setText("")
        self._publish()

    def _open_web_page(self) -> None:
        url = self.publishers.web_url()
        if url is None:
            QMessageBox.information(
                self,
                "Page web des sous-titres",
                "La page web est désactivée.\n\nActivez-la dans « Réglages avancés… » "
                "→ onglet Diffusion, puis validez.",
            )
            return
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        if not QDesktopServices.openUrl(QUrl(url)):
            from PySide6.QtWidgets import QApplication

            QApplication.clipboard().setText(url)
            self.banner.show_notice(f"Navigateur injoignable — URL copiée : {url}", level="warn")

    def _session_root_text(self) -> str:
        root = self.settings.session_dir or str(new_session_dir().parent)
        return f"Dossier : {root}"

    def _update_session_label(self) -> None:
        if self.controller.session_dir is not None:
            self.session_label.setText(f"Session : {self.controller.session_dir}")

    def _open_session_dir(self) -> None:
        target = self.controller.session_dir or Path(self.settings.session_dir or new_session_dir().parent)
        self._open_dir(target)

    def _open_dir(self, path: Path) -> None:
        """Ouvre un dossier ; en cas d'échec, affiche le chemin plutôt que rien."""
        from ecoutemoi.gui.desktop import open_path

        error = open_path(path)
        if error is None:
            return
        log.warning("Ouverture de %s impossible : %s", path, error)
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Ouverture du dossier")
        box.setText(f"Impossible d'ouvrir le dossier :\n{path}")
        box.setInformativeText(f"{error}\n\nLe chemin est copiable ci-dessous.")
        box.setDetailedText(str(path))
        copy = box.addButton("Copier le chemin", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Close)
        box.exec()
        if box.clickedButton() is copy:
            from PySide6.QtWidgets import QApplication

            QApplication.clipboard().setText(str(path))
            self.banner.show_notice(f"Chemin copié : {path}")

    def _tick_status(self) -> None:
        if self._t_session_start is not None:
            s = int(time.monotonic() - self._t_session_start)
            self.st_dur.setText(f"Durée : {s // 60:02d}:{s % 60:02d}")

    # ------------------------------------------------------------------ help
    def _open_obs_guide(self) -> None:
        dlg = ObsGuideDialog(self)
        dlg.exec()

    def _open_quant_help(self) -> None:
        from ecoutemoi.gui.model_manager_dialog import QuantizationHelpDialog

        QuantizationHelpDialog(self).exec()

    def _open_quickstart(self) -> None:
        HelpDialog("Prise en main", QUICKSTART_HTML, self).exec()

    def _about(self) -> None:
        QMessageBox.about(
            self,
            "À propos",
            f"<b>{APP_DISPLAY_NAME}</b> v{__version__}<br><br>"
            "Sous-titrage temps réel et transcription de fichiers, 100 % en local.<br>"
            "Deux moteurs : <b>faster-whisper</b> (CTranslate2) sur CPU, "
            "<b>whisper.cpp</b> sur GPU Vulkan · Metal.<br><br>"
            "Licence <b>GNU GPL v3 ou ultérieure</b> · logiciel gratuit, "
            "fourni sans aucune garantie.<br>"
            "Interface Qt via PySide6 (LGPL) · lecture audio via libsndfile (LGPL).<br>"
            "Modèles Whisper © OpenAI · conversions ggml-org (ggml) et "
            "Systran / mobiuslabs (CTranslate2).",
        )

    # ------------------------------------------------------------------ close
    def _restore_geometry(self) -> None:
        if self.settings.main_geometry:
            try:
                x, y, w, h = (int(v) for v in self.settings.main_geometry.split(","))
                self.setGeometry(x, y, w, h)
                return
            except ValueError:
                pass
        self.resize(980, 720)

    def _on_prewarm_status(self, status: str) -> None:
        """Progression du préchauffage : dans la barre d'état, sans bannière —
        l'opérateur n'a rien à faire, il n'a pas besoin d'être interrompu."""
        self.st_prewarm.setText(status)

    def closeEvent(self, event) -> None:
        # Pas de sauvegarde implicite à la fermeture : seuls le bouton et le menu
        # « Sauvegarder les réglages » écrivent settings.json.
        self._pace_timer.stop()
        self.prewarmer.cancel()
        self.controller.stop()
        self.publishers.stop()
        # Les moteurs préchauffés détiennent des sous-processus : sans ce nettoyage
        # explicite, ils survivraient à la fenêtre.
        self.engines.clear()
        self.overlay.close()
        self.overlay_dual.close()
        super().closeEvent(event)


DISCLAIMER_HTML = f"""
<h2>À lire avant d'utiliser {APP_DISPLAY_NAME}</h2>

<p><b>Ce logiciel est intégralement « vibecodé ».</b> Il a été écrit en dialoguant
avec un modèle de langage, du premier commit au dernier. Le code est commenté,
testé et relu, mais il n'a pas la maturité d'un produit industriel : ni audit
externe, ni support, ni engagement de maintenance.</p>

<p><b>Fourni EN L'ÉTAT, sans aucune garantie</b> — ni de bon fonctionnement, ni
d'exactitude des transcriptions, ni d'adéquation à un usage particulier. Les
binaires ne sont pas signés. Vous l'utilisez à vos risques ; l'auteur ne peut être
tenu responsable d'aucun dommage, d'aucune perte de données, ni d'un sous-titre
faux en pleine conférence.</p>

<p>Une transcription automatique se trompe. En contexte d'accessibilité, elle ne
remplace <b>pas</b> un vélotypiste ni un interprète : traitez-la comme une aide,
pas comme une source fiable.</p>

<p><b>Ce logiciel est et doit rester gratuit.</b> Si quelqu'un vous l'a fait payer,
vous avez été trompé. Tout tourne en local : aucune donnée audio ni texte ne quitte
votre machine, sauf si vous activez explicitement une sortie réseau (page web
ouverte au réseau local, ou OBS distant).</p>

<p><b>Licence : GNU GPL, version 3 ou ultérieure.</b> Vous pouvez l'utiliser,
l'étudier, le modifier et le redistribuer ; toute redistribution, modifiée ou non,
doit rester libre et sous la même licence. C'est la traduction juridique de la
phrase précédente. Le texte complet est dans le fichier <code>LICENSE</code>.</p>

<p>Les modèles Whisper (ggml) sont © OpenAI / ggml-org et suivent leurs propres
licences. L'interface utilise Qt via PySide6 (LGPL, bibliothèques dynamiques) ;
la lecture des fichiers audio passe par libsndfile (LGPL).</p>
"""


class DisclaimerDialog(QDialog):
    """Avertissement de premier lancement. Fermable UNIQUEMENT en acceptant.

    Ni croix de fenêtre ni Échap : c'est le sens même d'une décharge — il faut
    l'avoir sous les yeux avant de s'en servir. Reconsultable ensuite via le
    menu Aide.
    """

    def __init__(self, parent=None, *, first_run: bool = True):
        super().__init__(parent)
        self.setWindowTitle(f"Avertissement — {APP_DISPLAY_NAME}")
        self.resize(680, 560)
        self.setModal(True)
        browser = QTextBrowser()
        browser.setHtml(DISCLAIMER_HTML)
        browser.setOpenExternalLinks(True)
        if first_run:
            self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)
            label = "J'ai lu et compris"
            buttons = QDialogButtonBox()
            buttons.addButton(label, QDialogButtonBox.ButtonRole.AcceptRole)
            buttons.accepted.connect(self.accept)
        else:
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(browser, 1)
        lay.addWidget(buttons)
        self._first_run = first_run

    def keyPressEvent(self, event) -> None:
        # Échap = reject : au premier lancement il n'y a rien à rejeter.
        if self._first_run and event.key() == Qt.Key.Key_Escape:
            event.ignore()
            return
        super().keyPressEvent(event)

    def reject(self) -> None:
        if self._first_run:
            return  # seul « J'ai lu et compris » ferme cette fenêtre
        super().reject()


class LexiconDialog(QDialog):
    """Lexique de la conférence : noms propres, produits, acronymes du talk.

    Whisper reçoit cette liste comme amorce de transcription (`initial_prompt`,
    rappelée à chaque fenêtre décodée). C'est le levier le plus efficace sur
    exactement ce que la quantization dégrade en premier : les mots rares.
    """

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        from ecoutemoi.core.engine import LEXICON_MAX_CHARS

        self.setWindowTitle("Lexique de la conférence")
        self.resize(560, 460)
        self._limit = LEXICON_MAX_CHARS
        hint = QLabel(
            "Un terme par ligne (ou séparés par des virgules) : <b>noms propres, "
            "produits, acronymes, noms des intervenants</b>. Whisper est amorcé "
            "avec cette liste à chaque décodage, il écrira donc « Kubernetes » "
            "plutôt qu'une transcription phonétique.<br><br>"
            "Écrivez les termes <b>exactement</b> comme ils doivent apparaître "
            "(casse et accents compris). Inutile d'y mettre du vocabulaire "
            "courant : seuls les mots que le modèle ne connaît pas ou orthographie "
            "mal ont un intérêt."
        )
        hint.setWordWrap(True)
        self.editor = QPlainTextEdit(text)
        self.editor.setPlaceholderText("Kubernetes\nCeph\nOpenStack\nPrometheus\nGrafana")
        self.counter = QLabel("")
        self.editor.textChanged.connect(self._refresh_counter)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(hint)
        lay.addWidget(self.editor, 1)
        lay.addWidget(self.counter)
        lay.addWidget(buttons)
        self._refresh_counter()

    def text(self) -> str:
        return self.editor.toPlainText()

    def _refresh_counter(self) -> None:
        from ecoutemoi.core.engine import normalize_lexicon

        flat = normalize_lexicon(self.text())
        raw_len = len(", ".join(p.strip() for p in self.text().replace("\n", ",").split(",") if p.strip()))
        entries = len([p for p in flat.split(",") if p.strip()])
        if raw_len > self._limit:
            self.counter.setText(
                f"⚠ {entries} entrées retenues — whisper plafonne son amorce, "
                f"la liste est tronquée à {self._limit} caractères ({raw_len} fournis)."
            )
            mark(self.counter, "warn")
            self.counter.style().polish(self.counter)
        else:
            self.counter.setText(f"{entries} entrées · {raw_len}/{self._limit} caractères")
            mark(self.counter, "hint")
            self.counter.style().polish(self.counter)


class DiagDialog(QDialog):
    """Diagnostic moteur : environnement + dernier moteur chargé + queue du log.

    Tout est copiable en un clic — c'est le contenu attendu d'un rapport de bug.
    """

    def __init__(self, engine_diag: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Diagnostic moteur (GPU/Vulkan)")
        self.resize(760, 560)
        self._text = self._build(engine_diag)
        browser = QTextBrowser()
        from PySide6.QtGui import QFontDatabase

        browser.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        browser.setPlainText(self._text)
        btn_copy = QPushButton("Copier le rapport")
        btn_copy.clicked.connect(self._copy)
        btn_close = QPushButton("Fermer")
        btn_close.clicked.connect(self.accept)
        row = QHBoxLayout()
        row.addWidget(btn_copy)
        row.addStretch(1)
        row.addWidget(btn_close)
        lay = QVBoxLayout(self)
        lay.addWidget(browser, 1)
        lay.addLayout(row)

    def _copy(self) -> None:
        from PySide6.QtWidgets import QApplication

        QApplication.clipboard().setText(self._text)

    @staticmethod
    def _build(engine_diag: dict) -> str:
        import platform
        import sys

        from ecoutemoi.config import config_path
        from ecoutemoi.core import engine_fw, gpuprobe, media
        from ecoutemoi.core.engine import gpu_backend_libs

        frozen = "binaire PyInstaller" if getattr(sys, "frozen", False) else "environnement Python"
        libs = gpu_backend_libs()
        fw = (
            f"faster-whisper (CPU) — {engine_fw.versions()} · "
            f"calcul : {', '.join(engine_fw.supported_compute_types())}"
            if engine_fw.available()
            else "faster-whisper (CPU) : ABSENT de cet environnement"
        )
        probe = gpuprobe.load_cache()
        lines = [
            f"EcouteMoi {__version__} — diagnostic",
            f"OS         : {platform.platform()} ({platform.machine()})",
            f"Python     : {platform.python_version()} · {frozen}",
            f"Config     : {config_path()}",
            f"Log        : {log_dir() / 'ecoutemoi.log'}",
            f"Modèles    : {models.models_dir()}",
            "",
            "Moteurs :",
            "  whisper.cpp (GPU/CPU) — libs backend GPU : "
            + (", ".join(libs) if libs else "AUCUNE (moteur CPU pur — wheel PyPI ?)"),
            f"  {fw}",
            f"  Sondage GPU : {probe.summary if probe else 'pas encore effectué'}",
            "",
            "Modèles installés :",
            *(
                f"  {key:<22} {'+'.join(models.installed_formats(spec))}"
                for key, spec in models.REGISTRY.items()
                if models.installed_formats(spec)
            ),
            "",
            "Décodeurs de fichiers :",
            *(f"  {row}" for row in media.decoder_report()),
            "",
        ]
        if engine_diag:
            lines.append("Dernier moteur chargé :")
            for k, v in engine_diag.items():
                if isinstance(v, list):
                    v = " ; ".join(str(x) for x in v) if v else "—"
                lines.append(f"  {k:<22}: {v}")
        else:
            lines.append(
                "Aucune session lancée dans cette instance — démarrez une session "
                "pour remplir la partie moteur (ou lancez « EcouteMoi --diag » en console)."
            )
        try:
            tail = (log_dir() / "ecoutemoi.log").read_text(encoding="utf-8", errors="replace")
            lines += ["", "Queue du log :", *("  " + ln for ln in tail.splitlines()[-25:])]
        except OSError:
            pass
        return "\n".join(lines)


QUICKSTART_HTML = f"""
<h2>Prise en main</h2>

<p>{APP_DISPLAY_NAME} fait deux choses, et elles ne se ressemblent pas.</p>

<h3>Sous-titrer en direct</h3>
<ol>
<li><b>Un modèle.</b> <i>Outils → Gérer les modèles…</i> pour en télécharger un, puis
<i>Outils → Benchmark — Tester ma machine…</i> : l'application mesure vitesse et
qualité sur VOTRE machine et applique sa recommandation en un clic. C'est cinq
minutes une fois pour toutes, et ça évite de découvrir en salle que le modèle
choisi ne tient pas le direct.</li>
<li><b>Un micro.</b> Rubrique <i>Entrée audio</i>. Parlez : le niveau doit vivre
autour des deux tiers de l'échelle.</li>
<li><b>Démarrer.</b> La fenêtre de sortie à fond vert s'ouvre — c'est elle
qu'OBS capture (<i>Aide → Guide OBS</i>). Surveillez le voyant en bas à gauche :
vert, la machine a de la marge ; rouge, le direct décroche.</li>
</ol>
<p>Avant la conférence, remplissez le <b>lexique</b> (<i>Direct → Lexique de la
conférence…</i>) avec les noms propres, produits et acronymes du talk : c'est le
réglage qui rapporte le plus, très loin devant tous les autres.</p>

<h3>Transcrire des enregistrements</h3>
<p><i>Fichier → Transcrire des fichiers audio…</i> (Ctrl+O). Déposez vos fichiers,
choisissez les formats de sortie, lancez. Rien ne presse ici : prenez le modèle le
plus lourd que la machine accepte, la qualité s'en ressentira. Le texte sort en
TXT, Markdown, SRT, WebVTT, JSON, CSV, TSV ou LRC — et dans n'importe quelle autre
extension, qui recevra du texte brut.</p>

<h3>Ce qu'il faut savoir</h3>
<ul>
<li><b>Rien ne quitte la machine.</b> Aucune transcription n'est envoyée nulle
part, sauf si vous activez explicitement une sortie réseau.</li>
<li><b>Les réglages ne sont jamais enregistrés tout seuls.</b> Le bouton
<i>Sauvegarder les réglages</i> (Ctrl+S) est le seul à écrire sur le disque.</li>
<li><b>Une transcription automatique se trompe.</b> En contexte d'accessibilité,
elle ne remplace ni un vélotypiste ni un interprète.</li>
</ul>
"""


class HelpDialog(QDialog):
    """Page d'aide : du HTML, refermable, copiable, imprimable."""

    def __init__(self, title: str, html: str, parent=None, *, size: tuple[int, int] = (720, 620)):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(*size)
        browser = QTextBrowser()
        browser.setHtml(html)
        browser.setOpenExternalLinks(True)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(browser, 1)
        lay.addWidget(buttons)


class ObsGuideDialog(QDialog):
    """OBS capture guide per OS, with recommended chroma-key parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Guide OBS — capturer la fenêtre de sous-titres")
        self.resize(680, 520)
        tabs = QTabWidget(self)
        for name, html in (
            ("Windows", _OBS_WINDOWS),
            ("Linux", _OBS_LINUX),
            ("macOS", _OBS_MACOS),
        ):
            browser = QTextBrowser()
            browser.setHtml(html)
            browser.setOpenExternalLinks(True)
            tabs.addTab(browser, name)
        lay = QVBoxLayout(self)
        lay.addWidget(tabs)


_CHROMA_COMMON = """
<h3>Incrustation chromatique (filtre OBS)</h3>
<ol>
<li>Clic droit sur la source → <b>Filtres</b> → « + » → <b>Incrustation chromatique</b>.</li>
<li>Couleur : <b>Vert</b> (ou Magenta si votre contenu contient du vert).</li>
<li>Réglages recommandés : <b>similarité ≈ 400</b> · <b>lissage ≈ 80</b> ·
<b>réduction du débordement ≈ 100</b>.</li>
<li>Gardez un contour de texte ≥ 2 px (réglé dans Écoute Moi) : il masque le liseré.</li>
</ol>
"""

_OBS_WINDOWS = f"""
<h2>OBS sous Windows</h2>
<ol>
<li>Sources → « + » → <b>Capture de fenêtre</b>.</li>
<li>Fenêtre : <b>[EcouteMoi.exe] : EcouteMoi - Sortie OBS</b>.</li>
<li>Méthode de capture : <b>Windows 10 (1903 et ultérieur)</b>.</li>
</ol>
{_CHROMA_COMMON}
"""

_OBS_LINUX = f"""
<h2>OBS sous Linux</h2>
<ul>
<li><b>Wayland</b> : Sources → <b>Capture d'écran/fenêtre (PipeWire)</b> — le portail du
système vous laisse choisir la fenêtre « EcouteMoi - Sortie OBS ».</li>
<li><b>X11</b> : Sources → <b>Capture de fenêtre (Xcomposite)</b>.</li>
<li>Si le déplacement de la fenêtre de sortie pose problème sous Wayland, lancez :
<code>QT_QPA_PLATFORM=xcb ./EcouteMoi-*.AppImage</code>.</li>
<li>AppImage : <code>chmod +x</code> d'abord ; sans FUSE :
<code>./EcouteMoi-*.AppImage --appimage-extract-and-run</code>.</li>
</ul>
{_CHROMA_COMMON}
"""

_OBS_MACOS = f"""
<h2>OBS sous macOS</h2>
<ul>
<li>Autorisez OBS : Réglages Système → Confidentialité et sécurité →
<b>Enregistrement de l'écran</b>.</li>
<li>Autorisez Écoute Moi : Confidentialité → <b>Microphone</b> (sinon signal nul).</li>
<li>Sources → <b>Capture de fenêtre macOS</b> → fenêtre « EcouteMoi - Sortie OBS ».</li>
<li>Premier lancement de l'app non signée : clic droit → Ouvrir, ou
<code>xattr -cr EcouteMoi.app</code>.</li>
</ul>
{_CHROMA_COMMON}
"""


def run_gui(files: list[str] | None = None) -> int:
    """GUI entry point (called from app.main when no CLI flag is given)."""
    import gc

    from PySide6.QtWidgets import QApplication

    from ecoutemoi.gui.theme import apply_theme

    app = QApplication.instance() or QApplication([])
    app.setApplicationName(APP_DISPLAY_NAME)
    app.setApplicationDisplayName(APP_DISPLAY_NAME)
    apply_theme(app)
    win = MainWindow(files)
    win.show()
    code = app.exec()
    # PySide 6.11 : des wrappers de widgets vivants à Py_Finalize font
    # segfaulter le nettoyage atexit de PySide (destroyQCoreApplication) —
    # sortie propre : détruire la fenêtre et collecter les cycles
    # (vues <-> animations) tant que Qt est encore entier.
    del win
    gc.collect()
    app.processEvents()
    return code
