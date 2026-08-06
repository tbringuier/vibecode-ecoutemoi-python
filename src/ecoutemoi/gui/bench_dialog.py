"""Guided benchmark wizard: record FR + EN calibration texts, run the
isolated subprocess benchmark, show the table, apply the recommendation.

The 16 kHz recordings are kept in the app data dir so the operator can re-run
the benchmark without reading the texts again.
"""

from __future__ import annotations

import logging
import threading
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ecoutemoi.config import Settings
from ecoutemoi.constants import CALIBRATION_TEXT_EN, CALIBRATION_TEXT_FR, TARGET_SR
from ecoutemoi.core import bench, models
from ecoutemoi.gui.widgets import VuMeter

log = logging.getLogger(__name__)

AUTO_STOP_AFTER_S = 10.0  # auto-stop allowed only beyond 10 s...
AUTO_STOP_SILENCE_S = 1.2  # ...after 1.2 s of continuous silence
SILENCE_RMS = 0.004


def calibration_dir() -> Path:
    from ecoutemoi.cli import calibration_dir as _cd

    return _cd()


def _write_wav16(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TARGET_SR)
        w.writeframes(pcm.tobytes())


class RecordWorker(QObject):
    """Records through the real DSP chain (same signal the engine will see)."""

    sig_level = Signal(float)
    sig_time = Signal(float)
    sig_done = Signal(object)  # np.ndarray float32 @ 16 kHz
    sig_error = Signal(str)

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="bench-record")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        try:
            from ecoutemoi.core.audio import AudioCapture
            from ecoutemoi.core.dsp import DspChain

            cap = AudioCapture(device=self._settings.device_index, gain=self._settings.gain)
            dsp = DspChain(cap.sr, denoise=self._settings.denoise, highpass=self._settings.highpass)
            blocks: list[np.ndarray] = []
            silent_s = 0.0
            cap.start()
            try:
                while not self._stop.is_set():
                    block = cap.read(timeout=0.5)
                    if block is None:
                        continue
                    self.sig_level.emit(cap.rms)
                    for b16, _prob in dsp.process(block):
                        blocks.append(b16)
                        rms = float(np.sqrt(np.mean(b16 * b16)))
                        silent_s = silent_s + 0.01 if rms < SILENCE_RMS else 0.0
                    dur = len(blocks) * 0.01
                    self.sig_time.emit(dur)
                    if dur > AUTO_STOP_AFTER_S and silent_s >= AUTO_STOP_SILENCE_S:
                        break
            finally:
                cap.stop()
            audio = np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)
            self.sig_done.emit(audio)
        except Exception as exc:
            log.exception("Recording failed")
            self.sig_error.emit(str(exc))


class _RecordPage(QWidget):
    def __init__(self, lang: str, text: str, settings: Settings, parent=None):
        super().__init__(parent)
        self.lang = lang
        self.settings = settings
        self.audio: np.ndarray | None = None
        self._worker: RecordWorker | None = None
        self._countdown = 0

        title = QLabel(f"Lisez ce texte en {'français' if lang == 'fr' else 'anglais'} :")
        title.setStyleSheet("font-weight: bold;")
        self.text = QPlainTextEdit(text)
        self.text.setReadOnly(True)
        self.text.setStyleSheet("font-size: 16px;")
        self.vu = VuMeter()
        self.status = QLabel("Prêt. Cliquez sur « Enregistrer » (décompte de 3 s).")
        self.btn_rec = QPushButton("● Enregistrer")
        self.btn_stop = QPushButton("■ Arrêter")
        self.btn_stop.setEnabled(False)
        self.btn_play = QPushButton("▶ Réécouter")
        self.btn_play.setEnabled(False)
        self.btn_rec.clicked.connect(self._start_countdown)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_play.clicked.connect(self._play)

        row = QHBoxLayout()
        row.addWidget(self.btn_rec)
        row.addWidget(self.btn_stop)
        row.addWidget(self.btn_play)
        row.addStretch(1)
        lay = QVBoxLayout(self)
        lay.addWidget(title)
        lay.addWidget(self.text, 1)
        lay.addWidget(self.vu)
        lay.addWidget(self.status)
        lay.addLayout(row)

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)
        self._load_existing()

    def _load_existing(self) -> None:
        path = calibration_dir() / f"calibration_{self.lang}.wav"
        if path.is_file():
            try:
                with wave.open(str(path), "rb") as w:
                    raw = w.readframes(w.getnframes())
                self.audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                self.btn_play.setEnabled(True)
                self.status.setText(
                    f"Enregistrement existant ({len(self.audio) / TARGET_SR:.1f} s) — "
                    "réutilisable tel quel, ou réenregistrez."
                )
            except OSError, wave.Error:
                pass

    def _start_countdown(self) -> None:
        self._countdown = 3
        self.btn_rec.setEnabled(False)
        self.status.setText("Début dans 3…")
        self._timer.start()

    def _tick(self) -> None:
        self._countdown -= 1
        if self._countdown > 0:
            self.status.setText(f"Début dans {self._countdown}…")
            return
        self._timer.stop()
        self._record()

    def _record(self) -> None:
        self.status.setText("Enregistrement en cours… (arrêt auto après un silence)")
        self.btn_stop.setEnabled(True)
        self.btn_play.setEnabled(False)
        self._worker = RecordWorker(self.settings, self)
        self._worker.sig_level.connect(self.vu.set_rms)
        self._worker.sig_time.connect(lambda t: self.status.setText(f"Enregistrement : {t:.1f} s"))
        self._worker.sig_done.connect(self._done)
        self._worker.sig_error.connect(self._error)
        self._worker.start()

    def _stop(self) -> None:
        if self._worker:
            self._worker.stop()

    def _done(self, audio: np.ndarray) -> None:
        self.btn_rec.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.vu.set_rms(0.0)
        if audio.size < TARGET_SR * 3:
            self.status.setText("Enregistrement trop court (< 3 s) — recommencez.")
            return
        self.audio = audio
        _write_wav16(calibration_dir() / f"calibration_{self.lang}.wav", audio)
        self.btn_play.setEnabled(True)
        self.status.setText(
            f"Enregistré : {audio.size / TARGET_SR:.1f} s ✔ (conservé pour les prochains bench)"
        )

    def _error(self, msg: str) -> None:
        self.btn_rec.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.status.setText(f"Erreur d'enregistrement : {msg}")

    def _play(self) -> None:
        if self.audio is None:
            return
        try:
            import sounddevice as sd

            sd.play(self.audio, TARGET_SR)
        except Exception as exc:
            self.status.setText(f"Lecture impossible : {exc}")


class _BenchRunner(QObject):
    sig_progress = Signal(str)
    sig_done = Signal(object)  # list[BenchResult]
    sig_error = Signal(str)

    def run(
        self,
        keys: list[str],
        wav_fr: Path | None,
        wav_en: Path | None,
        backend: str,
        gpu_device: int = 0,
    ) -> None:
        def work():
            try:
                results = bench.run_benchmark(
                    keys, wav_fr, wav_en, backend=backend, gpu_device=gpu_device,
                    progress=self.sig_progress.emit,
                )  # fmt: skip
                self.sig_done.emit(results)
            except Exception as exc:
                log.exception("Benchmark failed")
                self.sig_error.emit(str(exc))

        threading.Thread(target=work, daemon=True, name="bench-run").start()


class BenchDialog(QDialog):
    """« Tester ma machine » — the guided operator benchmark flow.

    .applied is set to (model_key, preset_key) when the recommendation is applied.
    """

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Benchmark — Tester ma machine")
        self.resize(820, 560)
        self.settings = settings
        self.applied: tuple[str, str] | None = None
        self._results: list[bench.BenchResult] | None = None

        self.stack = QStackedWidget(self)
        self.page_fr = _RecordPage("fr", CALIBRATION_TEXT_FR, settings)
        self.page_en = _RecordPage("en", CALIBRATION_TEXT_EN, settings)
        self.page_run = self._build_run_page()
        self.stack.addWidget(self.page_fr)
        self.stack.addWidget(self.page_en)
        self.stack.addWidget(self.page_run)

        self.btn_prev = QPushButton("← Précédent")
        self.btn_next = QPushButton("Suivant →")
        self.btn_prev.clicked.connect(lambda: self._nav(-1))
        self.btn_next.clicked.connect(lambda: self._nav(+1))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)

        nav = QHBoxLayout()
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.btn_next)
        nav.addStretch(1)
        nav.addWidget(buttons)
        lay = QVBoxLayout(self)
        lay.addWidget(self.stack, 1)
        lay.addLayout(nav)
        # Ouvrir directement sur la page « Lancer » : le bench marche en un clic
        # grâce aux voix de référence embarquées ; l'enregistrement de SA voix
        # (pages précédentes) reste possible pour un WER personnalisé.
        self.stack.setCurrentIndex(2)
        self._nav(0)
        self._load_previous_results()

    # ------------------------------------------------------------- run page
    def _build_run_page(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.models_box = QVBoxLayout()
        self._model_checks: dict[str, QCheckBox] = {}
        installed = set(models.installed_models())
        for key, spec in models.REGISTRY.items():
            cb = QCheckBox(f"{key} ({spec.size_mb} Mo{'' if spec.translate else ', pas de traduction'})")
            cb.setChecked(key in installed)
            cb.setEnabled(key in installed)
            self._model_checks[key] = cb
            self.models_box.addWidget(cb)
        self.progress = QLabel("Modèles installés cochés — « Lancer le benchmark ».")
        self.btn_run = QPushButton("Lancer le benchmark")
        self.btn_run.clicked.connect(self._run)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Modèle", "Backend", "RTF", "WER FR", "WER EN", "Trad. (indicatif)", "Verdict"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.reco_label = QLabel("")
        self.reco_label.setStyleSheet("font-weight: bold;")
        self.btn_apply = QPushButton("Appliquer la recommandation")
        self.btn_apply.setEnabled(False)
        self.btn_apply.clicked.connect(self._apply)

        lay.addLayout(self.models_box)
        lay.addWidget(self.btn_run)
        lay.addWidget(self.progress)
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        row.addWidget(self.reco_label, 1)
        row.addWidget(self.btn_apply)
        lay.addLayout(row)
        return w

    def _nav(self, delta: int) -> None:
        idx = max(0, min(2, self.stack.currentIndex() + delta))
        self.stack.setCurrentIndex(idx)
        self.btn_prev.setEnabled(idx > 0)
        self.btn_next.setEnabled(idx < 2)

    def _run(self) -> None:
        keys = [k for k, cb in self._model_checks.items() if cb.isChecked() and cb.isEnabled()]
        if not keys:
            QMessageBox.information(self, "Benchmark", "Aucun modèle installé sélectionné.")
            return
        wav_fr, wav_en, is_reference = bench.resolve_calibration_wavs(
            calibration_dir() / "calibration_fr.wav",
            calibration_dir() / "calibration_en.wav",
        )
        if wav_fr is None and wav_en is None:
            QMessageBox.warning(
                self, "Benchmark",
                "Aucune voix disponible (voix de référence absentes du bundle et aucun "
                "enregistrement) — réinstallez l'application ou enregistrez les textes.",
            )  # fmt: skip
            return
        note = (
            " — voix de référence intégrée (enregistrez votre voix aux pages "
            "précédentes pour un WER personnalisé)"
            if is_reference
            else " — vos enregistrements"
        )
        self.btn_run.setEnabled(False)
        self.progress.setText(f"Benchmark en cours (passes isolées, timeout large){note}…")
        self._runner = _BenchRunner(self)
        self._runner.sig_progress.connect(self.progress.setText)
        self._runner.sig_done.connect(self._done)
        self._runner.sig_error.connect(self._failed)
        self._runner.run(
            keys, wav_fr, wav_en,
            backend=self.settings.backend, gpu_device=self.settings.gpu_device,
        )  # fmt: skip

    def _failed(self, msg: str) -> None:
        self.btn_run.setEnabled(True)
        self.progress.setText(f"Échec du benchmark : {msg}")

    def _done(self, results: list[bench.BenchResult]) -> None:
        self.btn_run.setEnabled(True)
        self.progress.setText("Benchmark terminé.")
        self._results = results
        bench.save_results(results, stamp=datetime.now().isoformat(timespec="seconds"))
        self._fill_table(results)

    def _fill_table(self, results: list[bench.BenchResult]) -> None:
        pct = lambda v: "—" if v is None else f"{v:.1%}"  # noqa: E731
        self.table.setRowCount(len(results))
        for row, r in enumerate(results):
            cells = [
                r.model_key,
                r.backend,
                "—" if r.rtf is None else f"{r.rtf:.1f}",
                pct(r.wer_fr),
                pct(r.wer_en),
                pct(r.translate_wer),
                bench.verdict(r),
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if r.error:  # erreur complète au survol (le libellé peut être tronqué)
                    item.setToolTip(r.error)
                self.table.setItem(row, col, item)
        self.table.resizeColumnsToContents()
        reco = bench.recommend(results)
        if reco:
            self.reco_label.setText(f"Recommandation : {reco[0]} + preset {reco[1]}")
            self.btn_apply.setEnabled(True)
        else:
            self.reco_label.setText("Aucun modèle utilisable sur cette machine (RTF < 1,3 partout).")
            self.btn_apply.setEnabled(False)

    def _load_previous_results(self) -> None:
        data = bench.load_results()
        if not data:
            return
        results = [bench.BenchResult(**r) for r in data.get("results", [])]
        if results:
            self._results = results
            self._fill_table(results)
            self.progress.setText(f"Derniers résultats ({data.get('stamp', '?')}) — relancez si besoin.")

    def _apply(self) -> None:
        if not self._results:
            return
        reco = bench.recommend(self._results)
        if reco:
            self.applied = reco
            self.accept()
