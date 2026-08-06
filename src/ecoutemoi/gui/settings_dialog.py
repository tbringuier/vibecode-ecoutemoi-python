"""Settings dialog: recognition/session tuning (returned in memory; the main
window's « Sauvegarder les réglages » button is the only writer).

Les réglages d'APPARENCE (police, couleurs, style, marges) vivent uniquement
dans la rubrique « Apparence » de la fenêtre principale — pas de doublon ici.

Spin boxes use their minimum as a sentinel with specialValueText("preset/auto")
to represent None (= value comes from the preset / locked constants).
"""

from __future__ import annotations

import dataclasses
import logging
import threading

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ecoutemoi.config import Settings
from ecoutemoi.constants import BACKENDS, GPU_BACKEND_NAME, PRESETS
from ecoutemoi.core import models
from ecoutemoi.gui.theme import mono_font

log = logging.getLogger(__name__)

BG_PRESETS = [
    ("Vert (chroma)", "#00FF00"),
    ("Magenta (chroma)", "#FF00FF"),
    ("Bleu", "#0000FF"),
    ("Noir", "#000000"),
]

# Sondes OBS en vol. Le fil de travail émet sur le worker, PAS sur le dialogue :
# si l'opérateur ferme la fenêtre avant la réponse, Qt a déjà coupé la connexion
# vers le dialogue détruit et l'émission devient inoffensive — alors qu'émettre
# sur le dialogue lui-même segfaulterait. Le set garde le worker en vie.
_PROBES: set[_ObsProbe] = set()


class _ObsProbe(QObject):
    """Test de connexion OBS hors du fil GUI : (message, sources | None)."""

    done = Signal(str, object)

    def run(self, host: str, port: int, password: str) -> None:
        _PROBES.add(self)

        def work() -> None:
            from ecoutemoi.core.publish import obs_probe

            try:
                version, sources = obs_probe(host, port, password)
                self.done.emit(f"Connecté — {version}", sources)
            except Exception as exc:
                log.info("Sonde OBS %s:%s en échec : %s", host, port, exc)
                self.done.emit(f"Échec : {exc}", None)
            finally:
                _PROBES.discard(self)

        threading.Thread(target=work, daemon=True, name="obs-probe").start()


class SettingsDialog(QDialog):
    """Returns an updated Settings via .result_settings after exec()."""

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Réglages")
        self.resize(560, 520)
        self._in = settings
        self.result_settings: Settings | None = None

        tabs = QTabWidget(self)
        tabs.addTab(self._tab_audio(settings), "Audio")
        tabs.addTab(self._tab_reco(settings), "Reconnaissance")
        tabs.addTab(self._tab_subtitling(settings), "Sous-titrage")
        tabs.addTab(self._tab_session(settings), "Session")
        tabs.addTab(self._tab_files(settings), "Fichiers")
        tabs.addTab(self._tab_publish(settings), "Diffusion")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(tabs, 1)
        lay.addWidget(buttons)

    # ------------------------------------------------------------------ tabs
    def _tab_audio(self, s: Settings) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self.gain = QDoubleSpinBox(minimum=0.25, maximum=4.0, singleStep=0.05, value=s.gain)
        self.audio_clean = QComboBox()
        self.audio_clean.addItem("Aucun traitement (recommandé)", "none")
        self.audio_clean.addItem("Passe-haut 80 Hz (anti-grondement)", "highpass")
        self.audio_clean.addItem("Passe-haut + RNNoise (bruit fort constant)", "full")
        key = "full" if s.denoise else ("highpass" if s.highpass else "none")
        self.audio_clean.setCurrentIndex(self.audio_clean.findData(key))
        form.addRow("Gain d'entrée", self.gain)
        form.addRow("Nettoyage micro", self.audio_clean)
        return w

    def _tab_reco(self, s: Settings) -> QWidget:
        w = QWidget()
        outer = QVBoxLayout(w)
        base = QGroupBox("Base")
        form = QFormLayout(base)
        self.model = QComboBox()
        installed = models.installed_models()
        self.model.addItems(installed or list(models.REGISTRY))
        if s.model in [self.model.itemText(i) for i in range(self.model.count())]:
            self.model.setCurrentText(s.model)
        self.mode = QComboBox()
        self.mode.addItem("FR → FR", "fr")
        self.mode.addItem("FR → EN", "translate")
        self.mode.addItem("Auto → EN (détection continue, sortie anglaise)", "auto")
        self.mode.setCurrentIndex(max(0, ["fr", "translate", "auto"].index(s.mode)))
        self.preset = QComboBox()
        for key, p in PRESETS.items():
            self.preset.addItem(p.label, key)
        self.preset.setCurrentIndex(max(0, list(PRESETS).index(s.preset)))
        form.addRow("Modèle", self.model)
        form.addRow("Mode", self.mode)
        form.addRow("Preset de latence", self.preset)

        adv = QGroupBox("Avancé (0 / « preset » = valeur du preset)")
        aform = QFormLayout(adv)
        self.min_interval = QSpinBox(minimum=0, maximum=2000, value=s.min_update_interval_ms or 0)
        self.min_interval.setSpecialValueText("auto-adaptatif (plancher 300 ms)")
        self.silence = QSpinBox(minimum=0, maximum=3000, singleStep=50, value=s.silence_ms or 0)
        self.silence.setSpecialValueText("preset")
        self.keep_back = QSpinBox(minimum=-1, maximum=4, value=-1 if s.keep_back is None else s.keep_back)
        self.keep_back.setSpecialValueText("preset")
        self.window_max = QDoubleSpinBox(minimum=0.0, maximum=30.0, singleStep=0.5,
                                         value=s.window_max_s or 0.0)  # fmt: skip
        self.window_max.setSpecialValueText("preset")
        nsp = s.no_speech_prob_max if s.no_speech_prob_max is not None else 0.0
        self.no_speech = QDoubleSpinBox(minimum=0.0, maximum=1.0, singleStep=0.05, value=nsp)
        self.no_speech.setSpecialValueText("défaut (0,6)")
        self.threads = QSpinBox(minimum=0, maximum=64, value=s.n_threads or 0)
        self.threads.setSpecialValueText("auto (P-cores, sinon cœurs physiques)")
        self.flash = QCheckBox("flash attention (repli automatique si échec)")
        self.flash.setChecked(s.flash_attn)
        self.backend = QComboBox()
        self.backend.addItem("Auto — GPU si disponible, repli CPU", "auto")
        self.backend.addItem(f"GPU ({GPU_BACKEND_NAME})", "gpu")
        self.backend.addItem("CPU uniquement", "cpu")
        self.backend.setCurrentIndex(max(0, BACKENDS.index(s.backend) if s.backend in BACKENDS else 0))
        self.halluc = QCheckBox("Filtre anti-hallucinations")
        self.halluc.setChecked(s.hallucination_filter)
        self.carry = QCheckBox("Report du contexte entre énoncés (déconseillé)")
        self.carry.setChecked(s.carry_context)
        self.subproc = QCheckBox("Décoder dans un sous-processus dédié (recommandé)")
        self.subproc.setChecked(s.engine_subprocess)
        self.subproc.setToolTip(
            "Sort l'inférence du processus graphique. Deux bénéfices : plus de gigue "
            "GIL sur le rendu et sur le callback audio (donc plus de blocs perdus), "
            "et un crash natif du moteur (pilote Vulkan, modèle corrompu) ne tue "
            "plus l'application — il est relancé automatiquement.\n\n"
            "À décocher seulement pour diagnostiquer un problème de moteur."
        )
        self.prewarm = QCheckBox("Précharger le moteur dès l'ouverture de l'application")
        self.prewarm.setChecked(s.prewarm_on_launch)
        self.prewarm.setToolTip(
            "Charge et préchauffe le modèle en tâche de fond dès l'ouverture de la "
            "fenêtre : « Démarrer » devient immédiat au lieu d'attendre le "
            "chargement devant la salle. Coût : la mémoire du modèle est occupée "
            "même sans session. Ne télécharge jamais un modèle absent."
        )
        aform.addRow("Backend moteur", self.backend)
        aform.addRow("Intervalle min entre décodes (ms)", self.min_interval)
        aform.addRow("Fin d'énoncé : silence (ms)", self.silence)
        aform.addRow("Mots retenus (keep_back)", self.keep_back)
        aform.addRow("Fenêtre max (s)", self.window_max)
        aform.addRow("Seuil no_speech", self.no_speech)
        aform.addRow("Threads", self.threads)
        aform.addRow(self.flash)
        aform.addRow(self.halluc)
        aform.addRow(self.carry)
        aform.addRow(self.subproc)
        aform.addRow(self.prewarm)
        outer.addWidget(base)
        outer.addWidget(adv)
        outer.addStretch(1)
        return w

    def _tab_subtitling(self, s: Settings) -> QWidget:
        """Normes de lisibilité : débit de lecture et largeur de ligne."""
        w = QWidget()
        outer = QVBoxLayout(w)

        pace = QGroupBox("Débit de lecture constant")
        form = QFormLayout(pace)
        self.pacing = QCheckBox("Lisser le flot à un débit constant (recommandé)")
        self.pacing.setChecked(s.pacing_enabled)
        self.pacing.setToolTip(
            "Whisper valide les mots par rafales : sans lissage, deux lignes "
            "surgissent d'un bloc puis rien pendant deux secondes. Le public n'a "
            "pas le temps de lire, alors que le texte est juste."
        )
        self.reading_wpm = QSpinBox(minimum=60, maximum=1200, singleStep=10, value=s.reading_wpm)
        self.reading_wpm.setSuffix(" mots/min")
        self.reading_wpm.setToolTip(
            "Vitesse de lecture maximale. Les normes de sous-titrage (BBC, EBU-TT) "
            "situent le confort entre 160 et 180 mots/min pour un public qui suit "
            "AUSSI un orateur et des diapositives."
        )
        self.pacing_lag = QDoubleSpinBox(minimum=0.5, maximum=10.0, singleStep=0.5,
                                         value=s.pacing_max_lag_s)  # fmt: skip
        self.pacing_lag.setSuffix(" s")
        self.pacing_lag.setToolTip(
            "Plafond de retard toléré. Au-delà, le débit accélère pour revenir sous "
            "le plafond : sans ce garde-fou, un orateur rapide creuserait un "
            "décalage sans fin et les sous-titres finiraient par commenter la "
            "diapositive précédente."
        )
        self.cpl = QSpinBox(minimum=0, maximum=120, value=s.max_chars_per_line)
        self.cpl.setSpecialValueText("aucune limite (largeur de fenêtre seule)")
        self.cpl.setToolTip(
            "Largeur maximale d'une ligne, en caractères. Norme : 37 à 42 — au-delà, "
            "l'œil perd la ligne au retour chariot, même si la fenêtre est large."
        )
        form.addRow(self.pacing)
        form.addRow("Vitesse de lecture", self.reading_wpm)
        form.addRow("Retard maximal", self.pacing_lag)
        form.addRow("Caractères par ligne", self.cpl)

        note = QLabel(
            "Le lissage ajoute par construction un léger retard, affiché en bas de "
            "la fenêtre principale (« Cadence »). C'est un compromis assumé : un "
            "sous-titre lisible avec une seconde de retard vaut mieux qu'un "
            "sous-titre instantané que personne ne peut suivre.<br><br>"
            "Rappel : les mots non encore validés ne sont <b>jamais</b> affichés — "
            "le public ne voit que du texte définitif, pas un mot qui se corrige."
        )
        note.setWordWrap(True)
        outer.addWidget(pace)
        outer.addWidget(note)
        outer.addStretch(1)
        return w

    def _tab_session(self, s: Settings) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self.session_dir = QLineEdit(s.session_dir or "")
        self.session_dir.setPlaceholderText("Documents/EcouteMoi/sessions (défaut)")
        browse = QPushButton("Parcourir…")

        def pick():
            from ecoutemoi.gui.desktop import pick_directory

            d = pick_directory(self, "Dossier des sessions", self.session_dir.text().strip())
            if d:
                self.session_dir.setText(d)

        browse.clicked.connect(pick)
        row = QHBoxLayout()
        row.addWidget(self.session_dir, 1)
        row.addWidget(browse)
        self.autosave = QCheckBox("Autosave continu (transcript.txt + segments.jsonl)")
        self.autosave.setChecked(s.autosave)
        self.txt_ts = QCheckBox("Horodatage [HH:MM:SS] dans l'export TXT")
        self.txt_ts.setChecked(s.txt_timestamps)
        self.export_stop = QCheckBox("Exporter TXT/SRT/VTT automatiquement à l'arrêt")
        self.export_stop.setChecked(s.export_on_stop)
        form.addRow("Dossier", row)
        form.addRow(self.autosave)
        form.addRow(self.txt_ts)
        form.addRow(self.export_stop)
        return w

    def _tab_files(self, s: Settings) -> QWidget:
        """Transcription de fichiers : décodeurs et dossier de sortie par défaut."""
        from ecoutemoi.core import media
        from ecoutemoi.gui.theme import hint

        w = QWidget()
        outer = QVBoxLayout(w)

        decoders = QGroupBox("Décodeurs disponibles")
        dlay = QVBoxLayout(decoders)
        media.set_ffmpeg_path(s.ffmpeg_path)
        self.decoder_report = hint("\n".join(media.decoder_report()))
        self.decoder_report.setFont(mono_font())
        dlay.addWidget(self.decoder_report)
        dlay.addWidget(
            hint(
                "WAV, FLAC, MP3, OGG, Opus, AIFF et CAF sont lus par la bibliothèque "
                "livrée avec l'application — rien à installer. Les M4A/AAC, WMA et les "
                "pistes audio de fichiers vidéo demandent <b>ffmpeg</b> : installez-le, "
                "ou déposez simplement l'exécutable à côté de celui d'Écoute Moi."
            )
        )

        ff = QGroupBox("ffmpeg")
        fform = QFormLayout(ff)
        self.ffmpeg_path = QLineEdit(s.ffmpeg_path)
        self.ffmpeg_path.setPlaceholderText("détection automatique")
        self.ffmpeg_path.setToolTip(
            "Chemin explicite de l'exécutable ffmpeg. À ne renseigner que si la "
            "détection automatique ne le trouve pas — par exemple une version "
            "installée dans un dossier à vous."
        )
        browse_ff = QPushButton("Parcourir…")

        def pick_ffmpeg():
            from ecoutemoi.gui.desktop import pick_files

            wanted = "ffmpeg (ffmpeg ffmpeg.exe);;Tous les fichiers (*)"
            chosen = pick_files(self, "Choisir l'exécutable ffmpeg", wanted)
            if chosen:
                self.ffmpeg_path.setText(str(chosen[0]))
                media.set_ffmpeg_path(str(chosen[0]))
                self.decoder_report.setText("\n".join(media.decoder_report()))

        browse_ff.clicked.connect(pick_ffmpeg)
        ffrow = QHBoxLayout()
        ffrow.addWidget(self.ffmpeg_path, 1)
        ffrow.addWidget(browse_ff)
        fform.addRow("Exécutable", ffrow)

        out = QGroupBox("Sortie par défaut")
        oform = QFormLayout(out)
        self.transcribe_dir = QLineEdit(s.transcribe_output_dir or "")
        self.transcribe_dir.setPlaceholderText("à côté du fichier d'origine")
        browse_out = QPushButton("Parcourir…")

        def pick_out():
            from ecoutemoi.gui.desktop import pick_directory

            chosen = pick_directory(self, "Dossier des transcriptions", self.transcribe_dir.text().strip())
            if chosen:
                self.transcribe_dir.setText(chosen)

        browse_out.clicked.connect(pick_out)
        orow = QHBoxLayout()
        orow.addWidget(self.transcribe_dir, 1)
        orow.addWidget(browse_out)
        oform.addRow("Dossier", orow)
        oform.addRow(
            hint(
                "Le choix fait dans la fenêtre « Transcrire des fichiers audio… » "
                "reste prioritaire ; celui-ci n'est que la valeur de départ."
            )
        )

        outer.addWidget(decoders)
        outer.addWidget(ff)
        outer.addWidget(out)
        outer.addStretch(1)
        return w

    def _tab_publish(self, s: Settings) -> QWidget:
        """Sorties texte hors de l'app : OBS WebSocket et/ou page web locale."""
        w = QWidget()
        outer = QVBoxLayout(w)

        obs = QGroupBox("OBS Studio — texte poussé par WebSocket")
        obs_form = QFormLayout(obs)
        self.obs_enabled = QCheckBox("Envoyer les sous-titres à OBS")
        self.obs_enabled.setChecked(s.obs_ws_enabled)
        self.obs_host = QLineEdit(s.obs_ws_host)
        self.obs_port = QSpinBox(minimum=1, maximum=65535, value=s.obs_ws_port)
        self.obs_password = QLineEdit(s.obs_ws_password)
        self.obs_password.setEchoMode(QLineEdit.EchoMode.Password)
        self.obs_password.setPlaceholderText("vide si l'authentification est désactivée")
        self.obs_password.setToolTip(
            "Enregistré EN CLAIR dans settings.json (usage local assumé). "
            "OBS : Outils → Paramètres du serveur WebSocket → Afficher les informations."
        )
        self.obs_source = QComboBox()
        self.obs_source.setEditable(True)
        self.obs_source.addItem(s.obs_ws_source)
        self.obs_source.setCurrentText(s.obs_ws_source)
        self.obs_source.setToolTip(
            "Nom EXACT d'une source « Texte (GDI+) » / « Texte (FreeType 2) » de la "
            "scène OBS. « Tester » remplit la liste avec les sources trouvées."
        )
        self.obs_source_dual = QComboBox()
        self.obs_source_dual.setEditable(True)
        self.obs_source_dual.addItem(s.obs_ws_source_dual)
        self.obs_source_dual.setCurrentText(s.obs_ws_source_dual)
        self.obs_source_dual.setEnabled(bool(s.dual_enabled))
        self.obs_source_dual.setToolTip(
            "Source Texte du SECOND sous-titre. Utilisée seulement si le second "
            "sous-titre est activé dans la fenêtre principale."
        )
        self.obs_test = QPushButton("Tester la connexion")
        self.obs_test.clicked.connect(self._test_obs)
        self.obs_status = QLabel("")
        self.obs_status.setWordWrap(True)
        hint = QLabel(
            "Dans OBS : <b>Outils → Paramètres du serveur WebSocket</b> → activer le "
            "serveur (port 4455), puis ajouter une source <b>Texte (GDI+)</b>. Cette "
            "voie remplace la capture de fenêtre + incrustation chromatique : OBS "
            "compose le texte lui-même, donc net et redimensionnable."
        )
        hint.setWordWrap(True)
        obs_form.addRow(self.obs_enabled)
        obs_form.addRow("Hôte", self.obs_host)
        obs_form.addRow("Port", self.obs_port)
        obs_form.addRow("Mot de passe", self.obs_password)
        obs_form.addRow("Source texte", self.obs_source)
        obs_form.addRow("Source texte n°2", self.obs_source_dual)
        obs_form.addRow(self.obs_test)
        obs_form.addRow(self.obs_status)
        obs_form.addRow(hint)

        web = QGroupBox("Page web locale")
        web_form = QFormLayout(web)
        self.web_enabled = QCheckBox("Servir une page de sous-titres")
        self.web_enabled.setChecked(s.web_enabled)
        self.web_port = QSpinBox(minimum=1, maximum=65535, value=s.web_port)
        # Case formulée en « localhost uniquement » et COCHÉE par défaut : l'état
        # sûr doit être celui qu'on obtient sans rien faire, et il doit se lire
        # comme une garantie, pas comme une restriction à lever.
        self.web_localhost = QCheckBox("Localhost uniquement (recommandé)")
        self.web_localhost.setChecked(not s.web_bind_lan)
        self.web_localhost.setToolTip(
            "Coché : le serveur n'écoute que sur 127.0.0.1, rien ne sort de la "
            "machine.\n\nDécoché : il écoute sur toutes les interfaces et n'importe "
            "qui sur le réseau peut lire les sous-titres — il n'y a AUCUNE "
            "authentification. À réserver à un réseau de confiance (poste de régie "
            "sur un second écran, par exemple)."
        )
        self.web_urls = QLabel("")
        self.web_urls.setWordWrap(True)
        self.web_urls.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        web_hint = QLabel(
            "Ouvrable dans un navigateur (second écran, retour orateur) ou comme "
            "<b>Source navigateur</b> OBS. Ajoutez <code>?bg=transparent</code> à l'URL "
            "pour un fond transparent au lieu du fond chroma."
        )
        web_hint.setWordWrap(True)
        web_form.addRow(self.web_enabled)
        web_form.addRow("Port", self.web_port)
        web_form.addRow(self.web_localhost)
        web_form.addRow("Adresses", self.web_urls)
        web_form.addRow(web_hint)
        self.web_port.valueChanged.connect(self._refresh_web_url)
        self.web_localhost.toggled.connect(self._refresh_web_url)
        self._refresh_web_url()

        outer.addWidget(obs)
        outer.addWidget(web)
        outer.addStretch(1)
        return w

    def _refresh_web_url(self) -> None:
        """Localhost seul : une URL. Ouvert au LAN : toutes les IP de la machine.

        L'opérateur doit pouvoir LIRE l'adresse à taper sur le poste distant ;
        aller la chercher dans les réglages système au moment du direct n'est pas
        une option."""
        from ecoutemoi.core.publish import local_ip_addresses

        port = self.web_port.value()
        lines = [f"http://127.0.0.1:{port}/"]
        if not self.web_localhost.isChecked():
            addresses = local_ip_addresses()
            if addresses:
                lines += [f"http://{ip}:{port}/" for ip in addresses]
            else:
                lines.append("(aucune adresse réseau détectée sur cette machine)")
        self.web_urls.setText("<br>".join(lines))

    def _test_obs(self) -> None:
        self.obs_test.setEnabled(False)
        self.obs_status.setText("Connexion à OBS…")
        probe = _ObsProbe()
        probe.done.connect(self._on_obs_probe)
        probe.run(self.obs_host.text().strip() or "127.0.0.1", self.obs_port.value(),
                  self.obs_password.text())  # fmt: skip

    def _on_obs_probe(self, message: str, sources: object) -> None:
        self.obs_test.setEnabled(True)
        if sources is None:
            self.obs_status.setText(message)
            return
        names = list(sources)
        if not names:
            self.obs_status.setText(
                f"{message} — aucune source texte dans la scène : ajoutez une source "
                "« Texte (GDI+) » dans OBS, puis retestez."
            )
            return
        wanted = self.obs_source.currentText().strip()
        self.obs_source.clear()
        self.obs_source.addItems(names)
        self.obs_source.setCurrentText(wanted if wanted in names else names[0])
        self.obs_status.setText(f"{message} — {len(names)} source(s) texte trouvée(s).")

    # ---------------------------------------------------------------- accept
    def _accept(self) -> None:
        clean = self.audio_clean.currentData()
        s = dataclasses.replace(
            self._in,
            gain=self.gain.value(),
            denoise=clean == "full",
            highpass=clean in ("highpass", "full"),
            model=self.model.currentText(),
            mode=self.mode.currentData(),
            preset=self.preset.currentData(),
            min_update_interval_ms=self.min_interval.value() or None,
            silence_ms=self.silence.value() or None,
            keep_back=None if self.keep_back.value() < 0 else self.keep_back.value(),
            window_max_s=self.window_max.value() or None,
            no_speech_prob_max=self.no_speech.value() or None,
            n_threads=self.threads.value() or None,
            flash_attn=self.flash.isChecked(),
            backend=self.backend.currentData(),
            hallucination_filter=self.halluc.isChecked(),
            carry_context=self.carry.isChecked(),
            engine_subprocess=self.subproc.isChecked(),
            prewarm_on_launch=self.prewarm.isChecked(),
            pacing_enabled=self.pacing.isChecked(),
            reading_wpm=self.reading_wpm.value(),
            pacing_max_lag_s=self.pacing_lag.value(),
            max_chars_per_line=self.cpl.value(),
            session_dir=self.session_dir.text().strip() or None,
            autosave=self.autosave.isChecked(),
            txt_timestamps=self.txt_ts.isChecked(),
            export_on_stop=self.export_stop.isChecked(),
            obs_ws_enabled=self.obs_enabled.isChecked(),
            obs_ws_host=self.obs_host.text().strip() or "127.0.0.1",
            obs_ws_port=self.obs_port.value(),
            obs_ws_password=self.obs_password.text(),
            obs_ws_source=self.obs_source.currentText().strip() or "EcouteMoi",
            obs_ws_source_dual=self.obs_source_dual.currentText().strip() or "EcouteMoi2",
            web_enabled=self.web_enabled.isChecked(),
            web_port=self.web_port.value(),
            web_bind_lan=not self.web_localhost.isChecked(),
            ffmpeg_path=self.ffmpeg_path.text().strip(),
            transcribe_output_dir=self.transcribe_dir.text().strip() or None,
        )
        self.result_settings = s
        self.accept()
