"""Petits widgets réutilisables : vumètre, sélecteur de couleur, bannière, voyant.

Tous prennent leurs couleurs dans `gui/theme.palette()` — aucune valeur en dur
ici, sinon un changement de thème se ferait à moitié.
"""

from __future__ import annotations

import math
from typing import ClassVar

from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath
from PySide6.QtWidgets import (
    QColorDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from ecoutemoi.constants import RT_BLINK_MS
from ecoutemoi.gui.theme import palette

# Vumètre : plage affichée et repères. −60 dB est le plancher, 0 dB la saturation ;
# les repères tombent là où l'opérateur décide quelque chose (−18 dB : niveau de
# travail, −6 dB : marge avant écrêtage).
VU_FLOOR_DB = -60.0
VU_MARKS_DB = (-40.0, -18.0, -6.0)
VU_WARN = 0.75  # au-delà, ambre
VU_CLIP = 0.92  # au-delà, rouge
PEAK_DECAY = 0.012  # décroissance du témoin de crête, par rafraîchissement


class VuMeter(QWidget):
    """Niveau d'entrée, rafraîchi ~20 Hz par l'appelant, avec témoin de crête.

    Le témoin de crête n'est pas un ornement : sur un micro-cravate, l'écrêtage
    passe inaperçu à l'oreille dans la salle et dégrade la transcription. Une
    marque qui reste deux secondes dans le rouge, elle, se voit.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._level = 0.0  # 0..1
        self._peak = 0.0
        self.setFixedHeight(16)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setToolTip(
            "Niveau du micro. Visez les deux tiers de l'échelle sur la voix : "
            "trop bas, le détecteur de parole hésite ; dans le rouge, l'écrêtage "
            "abîme la transcription."
        )

    @staticmethod
    def _to_level(rms: float) -> float:
        db = VU_FLOOR_DB if rms <= 1e-6 else max(VU_FLOOR_DB, 20.0 * math.log10(rms))
        return (db - VU_FLOOR_DB) / -VU_FLOOR_DB

    def set_rms(self, rms: float) -> None:
        level = self._to_level(rms)
        peak = max(level, self._peak - PEAK_DECAY)
        if abs(level - self._level) > 0.005 or abs(peak - self._peak) > 0.005:
            self._level, self._peak = level, peak
            self.update()

    def _color(self, level: float) -> QColor:
        p = palette()
        if level > VU_CLIP:
            return p.q("error")
        return p.q("warn") if level > VU_WARN else p.q("ok")

    def paintEvent(self, event) -> None:
        p = palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = rect.height() / 2

        track = QPainterPath()
        track.addRoundedRect(rect, radius, radius)
        painter.fillPath(track, p.q("meter_bg"))

        if self._level > 0.001:
            filled = QRectF(rect)
            filled.setWidth(max(rect.height(), rect.width() * self._level))
            painter.setClipPath(track)
            painter.fillRect(filled, self._color(self._level))
            painter.setClipping(False)

        painter.setPen(p.q("border"))
        for db in VU_MARKS_DB:
            x = rect.left() + rect.width() * (db - VU_FLOOR_DB) / -VU_FLOOR_DB
            painter.drawLine(int(x), int(rect.top() + 3), int(x), int(rect.bottom() - 3))

        if self._peak > 0.02:  # témoin de crête
            x = rect.left() + rect.width() * self._peak
            painter.setPen(self._color(self._peak))
            painter.drawLine(int(x), int(rect.top() + 1), int(x), int(rect.bottom() - 1))

        painter.setPen(p.q("border"))
        painter.drawPath(track)
        painter.end()


class ColorButton(QPushButton):
    """Bouton montrant une couleur ; ouvre le sélecteur de couleur du système."""

    color_changed = Signal(str)

    def __init__(self, color: str, parent=None):
        super().__init__(parent)
        self._color = QColor(color)
        self.setFixedSize(46, 26)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clicked.connect(self._pick)
        self._refresh()

    def color(self) -> str:
        return self._color.name().upper()

    def set_color(self, color: str) -> None:
        c = QColor(color)
        if c.isValid():
            self._color = c
            self._refresh()

    def _refresh(self) -> None:
        self.setToolTip(f"{self.color()} — cliquer pour changer")
        self.setStyleSheet(
            f"QPushButton {{ background-color: {self._color.name()}; "
            f"border: 1px solid {palette().border}; border-radius: 6px; }}"
            f"QPushButton:hover {{ border-color: {palette().accent}; }}"
        )

    def _pick(self) -> None:
        c = QColorDialog.getColor(self._color, self, "Choisir une couleur")
        if c.isValid():
            self._color = c
            self._refresh()
            self.color_changed.emit(self.color())


class NoticeBanner(QFrame):
    """Message dans la fenêtre, refermable, avec une action facultative.

    Trois niveaux, et ils ne se ressemblent pas : une confirmation ne doit pas
    avoir l'air d'un avertissement. C'est ce qui permet d'ignorer les messages
    verts d'un coup d'œil et de s'arrêter sur les ambres.
    """

    action_clicked = Signal()

    GLYPHS: ClassVar[dict[str, str]] = {"info": "i", "ok": "✔", "warn": "!", "error": "✕"}

    def __init__(self, parent=None):
        super().__init__(parent)
        # QLabel dérive de QFrame : une règle « QFrame » non qualifiée encadrerait
        # aussi les libellés à l'intérieur. D'où le nom d'objet.
        self.setObjectName("notice")
        self._glyph = QLabel("")
        self._glyph.setFixedWidth(16)
        self._glyph.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label = QLabel("")
        self._label.setWordWrap(True)
        self._label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._action = QPushButton("")
        self._action.setVisible(False)
        self._action.clicked.connect(self.action_clicked.emit)
        close = QPushButton("✕")
        close.setObjectName("icon")
        close.setFixedWidth(26)
        close.setToolTip("Fermer ce message")
        close.clicked.connect(self.hide)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 6, 6, 6)
        lay.setSpacing(8)
        lay.addWidget(self._glyph)
        lay.addWidget(self._label, 1)
        lay.addWidget(self._action)
        lay.addWidget(close)
        self._apply_level("info")
        self.hide()

    def _apply_level(self, level: str) -> None:
        p = palette()
        color = {"info": p.accent, "ok": p.ok, "warn": p.warn, "error": p.error}.get(level, p.accent)
        tint = QColor(color)
        tint.setAlpha(28)
        self.setStyleSheet(
            f"QFrame#notice {{ background: rgba({tint.red()},{tint.green()},{tint.blue()},{tint.alpha()});"
            f" border: 1px solid {color}; border-left: 3px solid {color}; border-radius: 8px; }}"
            f"QFrame#notice QLabel {{ background: transparent; border: none; color: {p.text}; }}"
        )
        self._glyph.setText(self.GLYPHS.get(level, "i"))
        self._glyph.setStyleSheet(f"color: {color}; font-weight: 700;")

    def show_notice(self, text: str, action_label: str | None = None, level: str = "info") -> None:
        self._apply_level(level)
        self._label.setText(text)
        self._action.setVisible(action_label is not None)
        if action_label:
            self._action.setText(action_label)
        self.show()


class _Dot(QWidget):
    """Pastille ronde, allumée ou en veilleuse (clignotement)."""

    def __init__(self, diameter: int = 11, parent=None):
        super().__init__(parent)
        self._color = QColor(palette().text_faint)
        self._lit = True
        self.setFixedSize(diameter, diameter)

    def set_color(self, color: QColor) -> None:
        self._color = color
        self.update()

    def set_lit(self, lit: bool) -> None:
        if lit != self._lit:
            self._lit = lit
            self.update()

    def toggle(self) -> None:
        self.set_lit(not self._lit)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        color = self._color if self._lit else self._color.darker(280)
        p.setBrush(color)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(self.rect().adjusted(1, 1, -1, -1))
        p.end()


class StatusLight(QWidget):
    """Voyant « le direct tient-il ? » — vert / ambre / rouge, clignotant.

    Le niveau vient du lag médian mesuré (temps de décodage / durée décodée) :
    vert = marge confortable, ambre = ça passe sans marge, rouge = le modèle est
    trop lourd pour le temps réel sur cette machine. Plus c'est grave, plus le
    clignotement est rapide (voir RT_BLINK_MS).
    """

    LABELS: ClassVar[dict[str, str]] = {
        "off": "Temps réel : —",
        "green": "Temps réel : OK",
        "orange": "Temps réel : limite",
        "red": "Temps réel : insuffisant",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._level = "off"
        self._dot = _Dot()
        self._label = QLabel(self.LABELS["off"])
        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 0, 8, 0)
        lay.setSpacing(6)
        lay.addWidget(self._dot)
        lay.addWidget(self._label)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._blink)
        self.set_level("off")

    @staticmethod
    def _color(level: str) -> str:
        p = palette()
        return {"off": p.text_faint, "green": p.ok, "orange": p.warn, "red": p.error}.get(level, p.text_faint)

    def level(self) -> str:
        return self._level

    def set_level(self, level: str, detail: str = "") -> None:
        """`level` ∈ off | green | orange | red ; `detail` alimente l'infobulle."""
        color = self._color(level)
        text = self.LABELS.get(level, self.LABELS["off"])
        if level != self._level:
            self._level = level
            self._dot.set_color(QColor(color))
            self._dot.set_lit(True)
            period = RT_BLINK_MS.get(level)
            if period:
                self._timer.start(period)
            else:
                self._timer.stop()
        self._label.setText(text)
        self._label.setStyleSheet(f"color: {color}; font-weight: 600;")
        tip = f"{text}\n{detail}" if detail else text
        self.setToolTip(tip)
        self._label.setToolTip(tip)

    def _blink(self) -> None:
        self._dot.toggle()


__all__ = ["ColorButton", "NoticeBanner", "StatusLight", "VuMeter"]
