"""« Transcrire des fichiers » : le mode hors direct, avec sa propre fenêtre.

Une conférence enregistrée, un entretien, la piste audio d'une visioconférence :
il n'y a ni micro, ni latence, ni public — donc aucune raison d'emprunter
l'interface du direct. Cette fenêtre ne demande que ce qui compte ici : quels
fichiers, quel modèle, quels formats de sortie, où les écrire.

Le travail vit dans un fil séparé et rend la main entre chaque passe de décodage :
la fenêtre reste vivante, la progression est réelle (pas une animation), et
« Interrompre » interrompt vraiment. Un fichier en échec n'arrête pas le lot — il
est marqué en rouge, avec sa raison au survol, et les suivants continuent.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ecoutemoi.config import Settings
from ecoutemoi.core import media, models
from ecoutemoi.core.transcript import (
    TRANSCRIPT_FORMATS,
    output_path,
    unique_path,
    write_transcript,
)
from ecoutemoi.gui.theme import hint, mark, palette
from ecoutemoi.gui.widgets import NoticeBanner

log = logging.getLogger(__name__)

MODE_LABELS = {"fr": "Français → français", "translate": "Français → anglais",
               "auto": "Langue détectée → anglais"}  # fmt: skip

COL_FILE, COL_INFO, COL_STATE = range(3)


class _Runner(QObject):
    """Le lot, dans un fil. Tout ce qui touche l'interface passe par un signal."""

    sig_loading = Signal(str)  # étape avant la première transcription
    sig_progress = Signal(int, str, float)  # (rang, étape, avancement 0..1 ou -1)
    sig_file_done = Signal(int, object)  # (rang, filejob.Transcript)
    sig_all_done = Signal(int, int)  # (réussis, échoués)
    sig_failed = Signal(str)  # échec global (modèle introuvable, etc.)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def cancel(self) -> None:
        self._stop.set()

    def _emit(self, signal, *args) -> bool:
        """Émet un signal ; False si la fenêtre a été détruite entre-temps.

        Émettre depuis le fil de travail sur un QObject dont le pendant C++ est
        déjà détruit lève RuntimeError DANS le fil. Sans ce garde-fou, fermer la
        fenêtre pendant un lot laissait une trace d'exception et, surtout, sautait
        la fermeture propre du moteur.
        """
        try:
            signal.emit(*args)
            return True
        except RuntimeError:
            log.debug("Fenêtre de transcription fermée — abandon du lot.")
            self._stop.set()
            return False

    def start(
        self,
        settings: Settings,
        paths: list[Path],
        *,
        model_key: str,
        mode: str,
        formats: list[str],
        out_dir: Path | None,
        timestamps: bool,
    ) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(settings, paths, model_key, mode, formats, out_dir, timestamps),
            daemon=True,
            name="transcribe-files",
        )
        self._thread.start()

    def _run(self, settings, paths, model_key, mode, formats, out_dir, timestamps) -> None:
        from ecoutemoi.cli import make_engine
        from ecoutemoi.core import filejob

        engine = None
        ok = failed = 0
        try:
            self._emit(self.sig_loading, f"Chargement du modèle {model_key}…")
            engine = make_engine(settings, model_key, mode)
            self._emit(self.sig_loading, "")
            written: set[Path] = set()

            def on_progress(p: filejob.Progress) -> None:
                ratio = p.ratio
                self._emit(self.sig_progress, p.index, p.stage, -1.0 if ratio is None else ratio)

            for result in filejob.transcribe_many(
                engine,
                paths,
                lang="fr" if mode == "fr" else "en",
                hallucination_filter=settings.hallucination_filter,
                no_speech_prob_max=settings.no_speech_prob_max,
                on_progress=on_progress,
                should_stop=self._stop.is_set,
            ):
                index = paths.index(result.path)
                if result.ok:
                    self._emit(self.sig_progress, index, "écriture", 1.0)
                    for key in formats:
                        fmt = TRANSCRIPT_FORMATS[key]
                        target = unique_path(output_path(result.path, fmt, out_dir), written)
                        try:
                            write_transcript(target, result.segments, key=key,
                                             timestamps=timestamps, title=result.path.name)  # fmt: skip
                        except OSError as exc:
                            result.error = f"écriture impossible ({target.name}) : {exc}"
                            break
                        written.add(target)
                        result.outputs.append(target)
                ok, failed = (ok + 1, failed) if result.ok else (ok, failed + 1)
                self._emit(self.sig_file_done, index, result)
                if self._stop.is_set():
                    break
            self._emit(self.sig_all_done, ok, failed)
        except Exception as exc:
            log.exception("Lot de transcription en échec")
            self._emit(self.sig_failed, str(exc) or type(exc).__name__)
        finally:
            if engine is not None:
                try:
                    engine.close()
                except Exception:
                    log.warning("Fermeture du moteur en échec", exc_info=True)


class TranscribeDialog(QDialog):
    """Fenêtre de transcription de fichiers.

    Les réglages choisis ici ressortent dans `result_settings` : comme partout
    dans l'application, rien n'est écrit sur le disque sans un « Sauvegarder les
    réglages » explicite.
    """

    def __init__(self, settings: Settings, parent=None, *, initial: list[Path] | None = None):
        super().__init__(parent)
        self.setWindowTitle("Transcrire des fichiers audio")
        self.resize(920, 680)
        self.setAcceptDrops(True)
        self.settings = settings
        self.result_settings: Settings | None = None
        self.paths: list[Path] = []
        self.results: dict[int, object] = {}
        self._last_out_dir: Path | None = None
        self.runner = _Runner(self)
        self.runner.sig_loading.connect(self._on_loading)
        self.runner.sig_progress.connect(self._on_progress)
        self.runner.sig_file_done.connect(self._on_file_done)
        self.runner.sig_all_done.connect(self._on_all_done)
        self.runner.sig_failed.connect(self._on_failed)

        media.set_ffmpeg_path(settings.ffmpeg_path)
        self._build_ui()
        self._refresh_actions()
        for path in initial or []:
            self._add_paths([Path(path)])

    # --------------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(12)
        self.banner = NoticeBanner()
        root.addWidget(self.banner)
        root.addWidget(self._box_files(), 1)
        row = QHBoxLayout()
        row.setSpacing(12)
        row.addWidget(self._box_reco(), 1)
        row.addWidget(self._box_output(), 1)
        root.addLayout(row)
        root.addWidget(self._box_run())

    def _box_files(self) -> QGroupBox:
        box = QGroupBox("Fichiers à transcrire")
        lay = QVBoxLayout(box)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Fichier", "Format d'origine", "État"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(COL_FILE, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(COL_INFO, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(COL_STATE, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(COL_STATE, 220)
        self.table.itemSelectionChanged.connect(self._refresh_actions)

        self.btn_add = QPushButton("Ajouter des fichiers…")
        self.btn_add.clicked.connect(self._pick_files)
        self.btn_remove = QPushButton("Retirer")
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_clear = QPushButton("Vider la liste")
        self.btn_clear.clicked.connect(self._clear)
        buttons = QHBoxLayout()
        buttons.addWidget(self.btn_add)
        buttons.addWidget(self.btn_remove)
        buttons.addWidget(self.btn_clear)
        buttons.addStretch(1)
        self.count_label = mark(QLabel(""), "hint")
        buttons.addWidget(self.count_label)

        lay.addLayout(buttons)
        lay.addWidget(self.table, 1)
        lay.addWidget(hint(self._decoders_hint()))
        return box

    def _decoders_hint(self) -> str:
        """Dit franchement ce qui est lisible sur CETTE machine, et pourquoi."""
        base = (
            "Glissez-déposez des fichiers ici, ou utilisez « Ajouter des fichiers… ». "
            "WAV, FLAC, MP3, OGG, Opus, AIFF et CAF sont lus directement."
        )
        if media.ffmpeg_executable() or media.pyav_available():
            return base + " M4A/AAC, WMA et les pistes audio de fichiers vidéo le sont aussi."
        return (
            base + " Pour les M4A/AAC, WMA et les fichiers vidéo, installez ffmpeg "
            "(ou indiquez son chemin dans Réglages avancés → Fichiers)."
        )

    def _box_reco(self) -> QGroupBox:
        box = QGroupBox("Reconnaissance")
        lay = QGridLayout(box)
        self.model = QComboBox()
        installed = models.installed_models()
        for key in installed or list(models.REGISTRY):
            spec = models.REGISTRY[key]
            self.model.addItem(key, key)
            self.model.setItemData(
                self.model.count() - 1,
                f"{spec.role}\n{spec.size_mb} Mo · RAM ~{spec.ram_gb:.1f} Go",
                Qt.ItemDataRole.ToolTipRole,
            )
        wanted = self.settings.transcribe_model or self.settings.model
        if (found := self.model.findData(wanted)) >= 0:
            self.model.setCurrentIndex(found)
        self.model.setToolTip(
            "Hors direct, rien ne presse : prenez le modèle le plus lourd que la "
            "machine accepte. Une transcription deux fois plus lente qu'un direct "
            "reste largement plus rapide que le temps réel."
        )
        self.model.currentIndexChanged.connect(self._validate)

        self.mode = QComboBox()
        for key, label in MODE_LABELS.items():
            self.mode.addItem(label, key)
        mode = self.settings.transcribe_mode or self.settings.mode
        self.mode.setCurrentIndex(max(0, self.mode.findData(mode)))
        self.mode.currentIndexChanged.connect(self._validate)

        btn_models = QPushButton("Gérer les modèles…")
        btn_models.clicked.connect(self._open_models)

        self.warn = mark(QLabel(""), "warn")
        self.warn.setWordWrap(True)

        lay.addWidget(QLabel("Modèle"), 0, 0)
        lay.addWidget(self.model, 0, 1)
        lay.addWidget(QLabel("Langue"), 1, 0)
        lay.addWidget(self.mode, 1, 1)
        lay.addWidget(btn_models, 2, 0, 1, 2)
        lay.addWidget(self.warn, 3, 0, 1, 2)
        lay.setColumnStretch(1, 1)
        return box

    def _box_output(self) -> QGroupBox:
        box = QGroupBox("Sortie")
        lay = QVBoxLayout(box)
        wanted = {k.strip() for k in self.settings.transcribe_formats.split(",") if k.strip()}
        self.format_checks: dict[str, QCheckBox] = {}
        grid = QGridLayout()
        for i, (key, fmt) in enumerate(TRANSCRIPT_FORMATS.items()):
            check = QCheckBox(fmt.label.split(" (")[0])
            check.setChecked(key in wanted)
            check.setToolTip(f"{fmt.label} — {' '.join(fmt.extensions)}")
            check.toggled.connect(self._validate)
            self.format_checks[key] = check
            grid.addWidget(check, i // 2, i % 2)
        lay.addLayout(grid)

        self.out_dir = QLineEdit(self.settings.transcribe_output_dir or "")
        self.out_dir.setPlaceholderText("À côté du fichier d'origine")
        browse = QPushButton("Parcourir…")
        browse.clicked.connect(self._pick_out_dir)
        row = QHBoxLayout()
        row.addWidget(self.out_dir, 1)
        row.addWidget(browse)
        lay.addWidget(QLabel("Dossier de sortie"))
        lay.addLayout(row)

        self.timestamps = QCheckBox("Horodater aussi le texte brut")
        self.timestamps.setChecked(self.settings.txt_timestamps)
        self.timestamps.setToolTip(
            "Les formats de sous-titres portent toujours leurs horodatages. "
            "Cette case ne concerne que le texte brut, où ils sont parfois de trop."
        )
        lay.addWidget(self.timestamps)
        lay.addStretch(1)
        return box

    def _box_run(self) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        self.preview = mark(QLabel(""), "hint")
        self.preview.setWordWrap(False)
        self.preview.setMinimumHeight(20)
        self.preview.setToolTip("Dernier passage transcrit — pour vérifier d'un œil que ça tient la route.")
        self.total = QProgressBar()
        self.total.setRange(0, 1000)
        self.total.setValue(0)
        self.total.setFormat("")
        self.status = QLabel("Ajoutez des fichiers pour commencer.")

        self.btn_run = QPushButton("Transcrire")
        self.btn_run.setObjectName("primary")
        self.btn_run.setMinimumHeight(38)
        self.btn_run.clicked.connect(self._start)
        self.btn_cancel = QPushButton("Interrompre")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_open = QPushButton("Ouvrir le dossier")
        self.btn_open.setEnabled(False)
        self.btn_open.clicked.connect(self._open_out_dir)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close.rejected.connect(self.reject)

        row = QHBoxLayout()
        row.addWidget(self.btn_run, 1)
        row.addWidget(self.btn_cancel)
        row.addWidget(self.btn_open)
        row.addWidget(close)
        lay.addWidget(self.preview)
        lay.addWidget(self.total)
        lay.addWidget(self.status)
        lay.addLayout(row)
        return box

    # ------------------------------------------------------------ liste de fichiers
    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self._add_paths(paths)
            event.acceptProposedAction()

    def _pick_files(self) -> None:
        from ecoutemoi.gui.desktop import pick_files

        start = str(self.paths[-1].parent) if self.paths else ""
        chosen = pick_files(self, "Choisir des fichiers audio", media.file_dialog_filter(), start)
        if chosen:
            self._add_paths(chosen)

    def _add_paths(self, paths: list[Path]) -> None:
        """Ajoute des fichiers (un dossier déposé apporte ce qu'il contient)."""
        expanded: list[Path] = []
        for path in paths:
            if path.is_dir():
                expanded += sorted(p for p in path.iterdir() if p.is_file() and media.looks_like_media(p))
            elif path.is_file():
                expanded.append(path)
        added = 0
        for path in expanded:
            resolved = path.resolve()
            if resolved in self.paths:
                continue
            self.paths.append(resolved)
            self._append_row(resolved)
            added += 1
        if expanded and not added:
            self.banner.show_notice("Ces fichiers sont déjà dans la liste.", level="info")
        self._refresh_actions()

    def _append_row(self, path: Path) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        name = QTableWidgetItem(path.name)
        name.setToolTip(str(path))
        self.table.setItem(row, COL_FILE, name)
        self.table.setItem(row, COL_INFO, QTableWidgetItem("…"))
        bar = QProgressBar()
        bar.setRange(0, 1000)
        bar.setValue(0)
        bar.setFormat("en attente")
        self.table.setCellWidget(row, COL_STATE, bar)
        # Le format d'origine est lu tout de suite : mieux vaut apprendre MAINTENANT
        # qu'un fichier est illisible qu'au bout de dix minutes de lot.
        try:
            info = media.probe(path)
            self.table.item(row, COL_INFO).setText(info.label)
        except media.MediaError as exc:
            item = self.table.item(row, COL_INFO)
            item.setText("format non reconnu")
            item.setForeground(palette().q("warn"))
            item.setToolTip(str(exc))
            bar.setFormat("illisible")

    def _remove_selected(self) -> None:
        for row in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(row)
            del self.paths[row]
        self.results.clear()
        self._refresh_actions()

    def _clear(self) -> None:
        self.table.setRowCount(0)
        self.paths.clear()
        self.results.clear()
        self.total.setValue(0)
        self.preview.setText("")
        self._refresh_actions()

    # ---------------------------------------------------------------- validation
    def _formats(self) -> list[str]:
        return [key for key, check in self.format_checks.items() if check.isChecked()]

    def _validate(self) -> None:
        problems: list[str] = []
        key = self.model.currentData() or ""
        mode = self.mode.currentData() or "fr"
        spec = models.REGISTRY.get(key)
        if spec is None:
            problems.append("Aucun modèle installé — ouvrez « Gérer les modèles… ».")
        elif mode in ("translate", "auto") and not spec.translate:
            problems.append(
                f"{key} n'est pas entraîné à la traduction : il ressortirait la "
                f"langue source. Choisissez un autre modèle, ou la langue "
                f"« {MODE_LABELS['fr']} »."
            )
        if not self._formats():
            problems.append("Choisissez au moins un format de sortie.")
        self.warn.setText("\n".join(problems))
        self.btn_run.setEnabled(bool(self.paths) and not problems and not self.runner.busy())

    def _refresh_actions(self) -> None:
        count = len(self.paths)
        self.count_label.setText(
            "aucun fichier" if not count else f"{count} fichier{'s' if count > 1 else ''}"
        )
        selected = bool(self.table.selectedIndexes())
        busy = self.runner.busy()
        self.btn_remove.setEnabled(selected and not busy)
        self.btn_clear.setEnabled(bool(count) and not busy)
        self.btn_add.setEnabled(not busy)
        if not busy and not count:
            self.status.setText("Ajoutez des fichiers pour commencer.")
        self._validate()

    # ------------------------------------------------------------------- travail
    def _start(self) -> None:
        if self.runner.busy() or not self.paths:
            return
        out_dir = Path(self.out_dir.text().strip()) if self.out_dir.text().strip() else None
        if out_dir is not None:
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                QMessageBox.warning(
                    self, "Dossier de sortie", f"Impossible d'utiliser ce dossier :\n{out_dir}\n\n{exc}"
                )
                return
        self._last_out_dir = out_dir or self.paths[0].parent
        self.results.clear()
        self.settings = dataclasses.replace(
            self.settings,
            transcribe_model=self.model.currentData() or "",
            transcribe_mode=self.mode.currentData() or "",
            transcribe_formats=",".join(self._formats()),
            transcribe_output_dir=str(out_dir) if out_dir else None,
            txt_timestamps=self.timestamps.isChecked(),
        )
        self.result_settings = self.settings
        for row in range(self.table.rowCount()):
            bar: QProgressBar = self.table.cellWidget(row, COL_STATE)
            bar.setRange(0, 1000)
            bar.setValue(0)
            bar.setFormat("en attente")
        self.banner.hide()
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_open.setEnabled(False)
        self._refresh_actions()
        self.status.setText("Préparation…")
        self.runner.start(
            self.settings,
            list(self.paths),
            model_key=self.model.currentData(),
            mode=self.mode.currentData(),
            formats=self._formats(),
            out_dir=out_dir,
            timestamps=self.timestamps.isChecked(),
        )

    def _cancel(self) -> None:
        self.runner.cancel()
        self.btn_cancel.setEnabled(False)
        self.status.setText("Interruption demandée — fin de la passe en cours…")

    def _on_loading(self, message: str) -> None:
        if message:
            self.status.setText(message)

    def _on_progress(self, index: int, stage: str, ratio: float) -> None:
        if not 0 <= index < self.table.rowCount():
            return
        bar: QProgressBar = self.table.cellWidget(index, COL_STATE)
        if ratio < 0:  # durée inconnue : barre indéterminée plutôt que fausse
            bar.setRange(0, 0)
            bar.setFormat(stage)
        else:
            bar.setRange(0, 1000)
            bar.setValue(int(ratio * 1000))
            bar.setFormat(f"{stage} — %p %")
        done = index + max(0.0, ratio)
        self.total.setValue(int(1000 * done / max(1, len(self.paths))))
        self.status.setText(f"{index + 1}/{len(self.paths)} · {self.paths[index].name} · {stage}")

    def _on_file_done(self, index: int, result) -> None:
        self.results[index] = result
        bar: QProgressBar = self.table.cellWidget(index, COL_STATE)
        bar.setRange(0, 1000)
        item = self.table.item(index, COL_FILE)
        if result.ok:
            bar.setValue(1000)
            speed = f" · x{result.speed:.1f}" if result.speed else ""
            bar.setFormat(f"{len(result.outputs)} fichier(s) écrit(s){speed}")
            bar.setToolTip("\n".join(str(p) for p in result.outputs))
            item.setToolTip("\n".join([str(result.path), "", *(str(p) for p in result.outputs)]))
            if result.segments:
                self.preview.setText(self._elide(result.segments[-1].text))
        else:
            bar.setValue(0)
            bar.setFormat("échec" if result.error != "interrompu" else "interrompu")
            bar.setToolTip(result.error or "")
            item.setForeground(palette().q("error" if result.error != "interrompu" else "text_muted"))
            item.setToolTip(f"{result.path}\n\n{result.error}")

    def _elide(self, text: str, limit: int = 110) -> str:
        flat = " ".join(text.split())
        return flat if len(flat) <= limit else "… " + flat[-limit:]

    def _on_all_done(self, ok: int, failed: int) -> None:
        self.btn_cancel.setEnabled(False)
        self.btn_open.setEnabled(self._last_out_dir is not None)
        self.total.setValue(1000 if not failed else self.total.value())
        words = sum(r.word_count for r in self.results.values() if r.ok)
        if failed and ok:
            self.banner.show_notice(
                f"{ok} fichier(s) transcrit(s), {failed} en échec — la raison est au "
                f"survol de la ligne en rouge.",
                level="warn",
            )
        elif failed:
            self.banner.show_notice(
                f"{failed} fichier(s) en échec — la raison est au survol de la ligne.", level="error"
            )
        elif ok:
            self.banner.show_notice(f"{ok} fichier(s) transcrit(s), {words} mots au total ✔", level="ok")
        self.status.setText("Terminé." if not failed else "Terminé, avec des échecs.")
        self._refresh_actions()

    def _on_failed(self, message: str) -> None:
        self.btn_cancel.setEnabled(False)
        self.status.setText("Échec.")
        self.banner.show_notice(f"Transcription impossible : {message}", level="error")
        self._refresh_actions()

    # -------------------------------------------------------------------- divers
    def _pick_out_dir(self) -> None:
        from ecoutemoi.gui.desktop import pick_directory

        chosen = pick_directory(self, "Dossier de sortie", self.out_dir.text().strip())
        if chosen:
            self.out_dir.setText(chosen)

    def _open_out_dir(self) -> None:
        from ecoutemoi.gui.desktop import open_path

        target = self._last_out_dir
        if target is None:
            return
        error = open_path(target)
        if error is not None:
            QMessageBox.warning(self, "Ouverture du dossier", f"{target}\n\n{error}")

    def _open_models(self) -> None:
        from ecoutemoi.gui.model_manager_dialog import ModelManagerDialog

        ModelManagerDialog(self).exec()
        current = self.model.currentData()
        self.model.clear()
        for key in models.installed_models() or list(models.REGISTRY):
            self.model.addItem(key, key)
        if (found := self.model.findData(current)) >= 0:
            self.model.setCurrentIndex(found)
        self._validate()

    def reject(self) -> None:
        if self.runner.busy():
            answer = QMessageBox.question(
                self,
                "Transcription en cours",
                "Une transcription est en cours. L'interrompre et fermer la fenêtre ?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            self.runner.cancel()
        super().reject()
