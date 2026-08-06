"""Model manager: registry table, parallel downloads (pool 2) with real
per-file progress + MB/s, RAM warning.

Progression : PIÈGE tqdm — avec `disable=True` (pas de sortie console),
`tqdm.update()` sort immédiatement SANS incrémenter `self.n` : lire `self.n`
donnait « 0 % 0.0 Mo/s » en permanence. On compte donc les octets NOUS-MÊMES
dans l'override, avant l'early-return de tqdm. (Le sondage disque n'est pas
une alternative : huggingface_hub 1.x télécharge dans %TEMP%\\<uuid>.tmp,
inattribuable à un modèle précis.)
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
)

from ecoutemoi.core import models

log = logging.getLogger(__name__)

COL_KEY, COL_QUANT, COL_SIZE, COL_TRAD, COL_RAM, COL_STATE, COL_ACTION = range(7)


def _make_qt_tqdm(cb):
    """tqdm subclass forwarding cumulated done_bytes to `cb`.

    `disable=True` (aucune sortie console) implique que tqdm n'incrémente
    JAMAIS self.n — le compteur est tenu ici, dans l'override, qui s'exécute
    quoi qu'il arrive.
    """
    from tqdm.auto import tqdm as _tqdm

    class QtTqdm(_tqdm):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("disable", True)  # no console output, Qt only
            self._done_bytes = int(kwargs.get("initial", 0) or 0)
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            self._done_bytes += int(n or 0)
            cb(self._done_bytes)
            return super().update(n)

    return QtTqdm


class DownloadManager(QObject):
    """Runs downloads in a 2-worker pool; signals are Qt-thread-safe."""

    sig_progress = Signal(str, int, float)  # key, percent, MB/s
    sig_done = Signal(str, bool, str)  # key, ok, message
    sig_all_done = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pool: ThreadPoolExecutor | None = None
        self._pending = 0

    def busy(self) -> bool:
        return self._pending > 0

    def download(self, keys: list[str]) -> None:
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=models.DOWNLOAD_POOL)
        for key in keys:
            self._pending += 1
            self._pool.submit(self._one, key)

    def _one(self, key: str) -> None:
        spec = models.REGISTRY[key]
        expected = spec.size_mb * 1024 * 1024
        seen = {"t": time.monotonic(), "prev": 0}

        def report(nbytes: int) -> None:
            now = time.monotonic()
            dt = now - seen["t"]
            if dt < 0.4:  # débit lissé, UI rafraîchie ~2.5x/s
                return
            speed = max(0.0, (nbytes - seen["prev"]) / dt / 1e6)
            seen["t"], seen["prev"] = now, nbytes
            pct = min(99, nbytes * 100 // expected) if expected else 0
            self.sig_progress.emit(key, int(pct), speed)

        try:
            models.download_model(spec, tqdm_class=_make_qt_tqdm(report))
            models.ensure_vad_model()  # mandatory companion model
            self.sig_done.emit(key, True, "")
        except Exception as exc:
            log.exception("Download failed: %s", key)
            self.sig_done.emit(key, False, str(exc))
        finally:
            self._pending -= 1
            if self._pending == 0:
                self.sig_all_done.emit()


class ModelManagerDialog(QDialog):
    """« Gérer les modèles… » — download + RAM estimates."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Gérer les modèles")
        self.resize(900, 560)
        self.manager = DownloadManager(self)
        self.manager.sig_progress.connect(self._on_progress)
        self.manager.sig_done.connect(self._on_done)
        self._downloading: set[str] = set()

        self.table = QTableWidget(len(models.REGISTRY), 7, self)
        self.table.setHorizontalHeaderLabels(
            ["Modèle", "Quantization", "Taille", "Trad. EN", "RAM est.", "État", ""]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)

        avail_gb = psutil.virtual_memory().available / 1e9
        self._rows: dict[str, int] = {}
        for row, (key, spec) in enumerate(models.REGISTRY.items()):
            self._rows[key] = row
            item_key = QTableWidgetItem(key)
            item_key.setToolTip(spec.role)
            self.table.setItem(row, COL_KEY, item_key)
            quant = QTableWidgetItem(spec.quant)
            short, detail = models.QUANT_NOTES.get(spec.quant, ("", ""))
            quant.setToolTip(f"{short}\n\n{detail}" if detail else "")
            self.table.setItem(row, COL_QUANT, quant)
            self.table.setItem(row, COL_SIZE, QTableWidgetItem(f"{spec.size_mb} Mo"))
            trad = QTableWidgetItem("oui" if spec.translate else "non")
            if not spec.translate:
                trad.setToolTip(
                    "Modèle distillé sans la tâche de traduction : en FR→EN ou "
                    "Auto→EN il ressortirait la langue source. Réservé au FR→FR."
                )
            self.table.setItem(row, COL_TRAD, trad)
            ram = QTableWidgetItem(f"~{spec.ram_gb:.1f} Go")
            if spec.ram_gb > avail_gb:
                ram.setText(f"~{spec.ram_gb:.1f} Go ⚠ RAM dispo {avail_gb:.1f} Go")
            self.table.setItem(row, COL_RAM, ram)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setTextVisible(True)
            self.table.setCellWidget(row, COL_STATE, bar)
            btn = QPushButton()
            btn.clicked.connect(lambda _=False, k=key: self._download(k))
            self.table.setCellWidget(row, COL_ACTION, btn)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setStretchLastSection(True)

        self._hint = QLabel("")
        self._hint.setWordWrap(True)
        self._hint.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        btn_quant = QPushButton("Quantizations — aide-mémoire…")
        btn_quant.clicked.connect(lambda: QuantizationHelpDialog(self).exec())
        btn_import = QPushButton("Importer un fichier…")
        btn_import.setToolTip(
            "Installe un .bin obtenu ailleurs (clé USB, miroir interne). Le nom du "
            "fichier doit être celui du registre — il identifie le modèle — et le "
            "contenu est validé (magie ggml + taille) avant installation."
        )
        btn_import.clicked.connect(self._import_file)
        btn_dir = QPushButton("Ouvrir le dossier")
        btn_dir.clicked.connect(self._open_models_dir)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)

        row = QHBoxLayout()
        row.addWidget(btn_quant)
        row.addWidget(btn_import)
        row.addWidget(btn_dir)
        row.addStretch(1)
        row.addWidget(buttons)
        lay = QVBoxLayout(self)
        lay.addWidget(self.table, 1)
        lay.addWidget(self._hint)
        lay.addLayout(row)
        self._refresh_hint()
        self._refresh_states()

    def _refresh_hint(self) -> None:
        self._hint.setText(f"Dossier des modèles : {models.models_dir()}")

    def _open_models_dir(self) -> None:
        from ecoutemoi.gui.desktop import open_path

        error = open_path(models.models_dir())
        if error is not None:
            QMessageBox.warning(self, "Ouverture du dossier", f"{models.models_dir()}\n\n{error}")

    def _import_file(self) -> None:
        """Import manuel d'un .bin (clé USB, miroir interne)."""
        import sys

        from PySide6.QtWidgets import QFileDialog

        options = QFileDialog.Option(0)
        if sys.platform.startswith("linux"):  # portail xdg absent en conteneur/bundle
            options |= QFileDialog.Option.DontUseNativeDialog
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Importer des modèles ggml", "", "Modèles ggml (*.bin);;Tous les fichiers (*)",
            options=options,
        )  # fmt: skip
        if not paths:
            return
        imported: list[str] = []
        failures: list[str] = []
        for raw in paths:
            try:
                key, _ = models.import_model_file(Path(raw))
                imported.append(key)
            except (OSError, ValueError) as exc:
                failures.append(f"{Path(raw).name} : {exc}")
        if imported:
            QMessageBox.information(
                self, "Import terminé", "Modèles installés :\n" + "\n".join(sorted(imported))
            )
        if failures:
            QMessageBox.warning(self, "Import partiel", "\n\n".join(failures))
        self._refresh_states()

    # ------------------------------------------------------------------ state
    def _refresh_states(self) -> None:
        installed = set(models.installed_models())
        for key, row in self._rows.items():
            if key in self._downloading:
                continue  # ne pas écraser une barre de téléchargement en cours
            bar: QProgressBar = self.table.cellWidget(row, COL_STATE)
            btn: QPushButton = self.table.cellWidget(row, COL_ACTION)
            bar.setRange(0, 100)
            if key in installed:
                bar.setValue(100)
                bar.setFormat("installé")
                btn.setText("Re-télécharger")
            else:
                bar.setValue(0)
                bar.setFormat("—")
                btn.setText("Télécharger")

    def _download(self, key: str) -> None:
        self._downloading.add(key)
        row = self._rows[key]
        bar: QProgressBar = self.table.cellWidget(row, COL_STATE)
        btn: QPushButton = self.table.cellWidget(row, COL_ACTION)
        btn.setEnabled(False)
        btn.setText("Téléchargement…")
        bar.setRange(0, 0)  # indéterminé tant qu'aucun octet n'est observé
        self.manager.download([key])

    def _on_progress(self, key: str, pct: int, mbps: float) -> None:
        bar: QProgressBar = self.table.cellWidget(self._rows[key], COL_STATE)
        if bar.maximum() == 0:  # premiers octets : passer en barre déterminée
            bar.setRange(0, 100)
        bar.setValue(pct)
        bar.setFormat(f"{pct} % — {mbps:.1f} Mo/s")

    def _on_done(self, key: str, ok: bool, msg: str) -> None:
        self._downloading.discard(key)
        if not ok:
            QMessageBox.warning(self, "Téléchargement échoué", f"{key} : {msg}")
        self._refresh_states()


def quantization_cheatsheet_html() -> str:
    """Aide-mémoire quantizations, CONSTRUIT depuis le registre.

    Généré plutôt qu'écrit à la main : ajouter un modèle ou une quantization met
    l'aide à jour tout seul, elle ne peut pas mentir sur ce qui est réellement
    proposé.
    """
    quant_rows = "".join(
        f"<tr><td><b>{q}</b></td><td>{models.QUANT_NOTES[q][0]}</td><td>{models.QUANT_NOTES[q][1]}</td></tr>"
        for q in models.QUANT_ORDER
        if q in models.QUANT_NOTES
    )

    families: dict[str, list] = {}
    for spec in models.REGISTRY.values():
        families.setdefault(spec.family, []).append(spec)
    model_rows = ""
    for family, specs in families.items():
        variants = " · ".join(f"<code>{s.quant}</code> {s.size_mb} Mo" for s in specs)
        trad = "oui" if specs[0].translate else "<b>NON</b>"
        model_rows += (
            f"<tr><td><b>{family}</b></td><td>{variants}</td>"
            f"<td>~{min(s.ram_gb for s in specs):.1f}–{max(s.ram_gb for s in specs):.1f} Go</td>"
            f"<td>{trad}</td></tr>"
        )

    return f"""
<h2>Quantizations — aide-mémoire</h2>

<p><b>Quantizer</b> = stocker les poids du modèle sur moins de bits. Le fichier
rétrécit, il y a moins d'octets à lire à chaque décodage — donc c'est plus
rapide — et la qualité baisse un peu. La perte ne se répartit pas
uniformément : elle frappe d'abord les <b>noms propres, les acronymes et les
mots rares</b>, exactement ce qui compte en conférence technique. D'où le
<b>Lexique de la conférence</b> (Réglages), qui compense sur ces mots-là.</p>

<h3>Les variantes proposées</h3>
<table border="1" cellpadding="6" cellspacing="0" width="100%">
<tr><th>Format</th><th>Taille</th><th>Ce que ça vaut</th></tr>
{quant_rows}
</table>

<h3>Comment choisir</h3>
<ul>
<li><b>Sur CPU</b>, le facteur limitant est la mémoire à parcourir : prenez la
quantization la plus compacte à qualité acceptable, soit <code>q5_1</code>
(ou <code>q5_0</code> sur medium/large). Monter d'une <i>famille</i>
(base → small) rapporte bien plus que monter d'une quantization
(small-q5_1 → small-q8_0).</li>
<li><b>Sur GPU</b>, la mémoire est rapide et souvent abondante :
<code>q8_0</code> est quasi gratuit en temps et gagne sur les mots rares.
<code>f16</code> n'a d'intérêt que pour mesurer la perte des autres.</li>
<li><b>Ne devinez pas</b> : <i>Outils → Benchmark — Tester ma machine…</i>
mesure RTF (vitesse) <i>et</i> WER (qualité) de chaque variante installée sur
VOTRE machine, et refuse celles qui ne tiennent pas le direct.</li>
</ul>

<h3>Ce que la quantization ne change PAS</h3>
<p>La latence perçue vient surtout de la <b>fenêtre de décodage</b>, du preset et
du réglage de fin d'énoncé — pas du format des poids. Une quantization plus
légère donne de la <b>marge</b> (RTF), elle ne raccourcit pas le délai à elle
seule.</p>

<h3>Familles disponibles</h3>
<table border="1" cellpadding="6" cellspacing="0" width="100%">
<tr><th>Famille</th><th>Variantes</th><th>RAM estimée</th><th>Traduction</th></tr>
{model_rows}
</table>

<h3>Pièges</h3>
<ul>
<li><b>large-v3-turbo ne traduit pas.</b> Son décodeur a été distillé sans la
tâche de traduction : en FR→EN ou Auto→EN il ressortirait du français.
Excellent en FR→FR, inutilisable pour les autres modes — l'application le
refuse d'ailleurs explicitement.</li>
<li><b>large-v1/v2/v3 en f16 (2,9 Go) ne sont pas proposés</b> : aucune machine
ne les tient en temps réel sur des fenêtres de 9 s. Les versions quantifiées
de large-v2 et large-v3, elles, sont là.</li>
<li><b>Les modèles <code>.en</code> ne sont pas proposés</b> : anglais
uniquement et sans traduction, aucun des trois modes ne peut les employer.</li>
<li><b>La RAM affichée est une estimation</b> (poids + activations). Une valeur
au-dessus de la mémoire disponible est signalée avant le démarrage.</li>
</ul>
"""


class QuantizationHelpDialog(QDialog):
    """Aide-mémoire des quantizations, copiable et imprimable."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Quantizations — aide-mémoire")
        self.resize(760, 620)
        browser = QTextBrowser()
        browser.setHtml(quantization_cheatsheet_html())
        browser.setOpenExternalLinks(True)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(browser, 1)
        lay.addWidget(buttons)
