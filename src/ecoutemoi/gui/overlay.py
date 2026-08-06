"""Chroma-key output window captured by OBS.

Frameless, fixed title (OBS matches on it), drag anywhere via
windowHandle().startSystemMove() (required on Wayland), QSizeGrip,
context menu (always on top / hide). Geometry persisted in Settings.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu, QSizeGrip, QVBoxLayout, QWidget

from ecoutemoi.config import Settings
from ecoutemoi.constants import OVERLAY_WINDOW_TITLE
from ecoutemoi.gui.subtitle_view import SubtitleStyle, SubtitleView

DEFAULT_SIZE = (1280, 220)


class OverlayWindow(QWidget):
    """Fenêtre de sortie capturée par OBS.

    `title_suffix` distingue la fenêtre du SECOND sous-titre : OBS reconnaît ses
    sources au titre exact, deux fenêtres homonymes seraient indiscernables dans
    le sélecteur de capture. `geometry` permet de persister les deux positions
    séparément.
    """

    def __init__(
        self,
        settings: Settings,
        parent=None,
        *,
        title_suffix: str = "",
        geometry: str | None = None,
    ):
        super().__init__(parent)
        self.title_suffix = title_suffix
        self.setWindowTitle(OVERLAY_WINDOW_TITLE + title_suffix)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
        self.view = SubtitleView(SubtitleStyle.from_settings(settings), overlay=True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.view)
        self._grip = QSizeGrip(self)
        self._grip.resize(16, 16)
        self._apply_geometry(settings, geometry)
        self.set_transparent(settings.overlay_transparent)
        self.set_always_on_top(settings.always_on_top)

    def set_transparent(self, on: bool) -> None:
        """Fenêtre translucide : les sous-titres flottent directement sur les
        applis en dessous (incrustation SANS OBS). Pour la capture OBS, rester
        en fond chroma : la capture de fenêtre ne préserve pas l'alpha."""
        if bool(self.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)) == bool(on):
            return
        visible = self.isVisible()
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, on)
        if visible:
            self.hide()
            self.show()  # recrée la surface avec/sans canal alpha
        self.view.update()

    # --------------------------------------------------------------- geometry
    def _apply_geometry(self, settings: Settings, geometry: str | None = None) -> None:
        wanted = geometry if geometry is not None else settings.overlay_geometry
        if wanted:
            try:
                x, y, w, h = (int(v) for v in wanted.split(","))
                self.setGeometry(x, y, max(200, w), max(80, h))
                return
            except ValueError:
                pass
        self.resize(*DEFAULT_SIZE)

    def geometry_string(self) -> str:
        g = self.geometry()
        return f"{g.x()},{g.y()},{g.width()},{g.height()}"

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._grip.move(self.width() - self._grip.width(), self.height() - self._grip.height())

    # ------------------------------------------------------------ interaction
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self.windowHandle():
            self.windowHandle().startSystemMove()  # Wayland-safe drag
            event.accept()

    def contextMenuEvent(self, event) -> None:
        menu = QMenu(self)
        top = QAction("Toujours au premier plan", menu, checkable=True)
        top.setChecked(bool(self.windowFlags() & Qt.WindowType.WindowStaysOnTopHint))
        top.toggled.connect(self.set_always_on_top)
        hide = QAction("Masquer la fenêtre de sortie", menu)
        hide.triggered.connect(self.hide)
        menu.addAction(top)
        menu.addSeparator()
        menu.addAction(hide)
        menu.exec(event.globalPos() if hasattr(event, "globalPos") else QPoint(0, 0))

    def set_always_on_top(self, on: bool) -> None:
        visible = self.isVisible()
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, on)
        if visible:
            self.show()  # re-apply flags
