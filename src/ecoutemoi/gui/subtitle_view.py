"""Shared subtitle rendering widget: the SAME class instance type renders
the OBS overlay and the main-window preview — strictly identical output.

Affichage VÉRIDIQUE : seul le texte validé est rendu (le texte en attente de
LocalAgreement n'apparaît jamais). Pour éviter l'écran vide entre deux énoncés,
la vue tient un historique roulant des mots finalisés + l'énoncé en cours.

Styles (réglage « Apparence ») :
- defilement : les lignes glissent vers le haut quand une nouvelle ligne naît ;
- fondu      : les nouveaux mots émergent du fond (interpolation OPAQUE vers la
               couleur de fond — jamais d'alpha, le chroma OBS reste propre) ;
- statique   : rendu instantané sans animation.

Rendering: opaque background color; word-wrap keeping the last `max_lines`
lines; outline via QPainterPath.addText -> strokePath(QPen(outline, 2*w,
RoundJoin)) then fillPath; antialiasing ON (masks chroma fringing).
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QEasingCurve, Qt, QVariantAnimation
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from ecoutemoi.config import Settings

_CHECKER_SIZE = 12
MAX_HISTORY_WORDS = 80  # mots finalisés conservés pour le défilement continu
SCROLL_MS = 240
FADE_MS = 220


def _lerp_color(a: QColor, b: QColor, t: float) -> QColor:
    """Interpolation RGB opaque (chroma-safe : aucune transparence)."""
    t = max(0.0, min(1.0, t))
    return QColor(
        round(a.red() + (b.red() - a.red()) * t),
        round(a.green() + (b.green() - a.green()) * t),
        round(a.blue() + (b.blue() - a.blue()) * t),
    )


@dataclass
class SubtitleStyle:
    font_family: str = ""
    font_size: int = 34
    text_color: str = "#FFFFFF"
    outline_color: str = "#000000"
    bg_color: str = "#00FF00"
    outline_width: int = 3
    max_lines: int = 2
    align: str = "center"
    margin_h: int = 24
    margin_v: int = 12
    display_style: str = "defilement"  # defilement | fondu | statique
    transparent_bg: bool = False  # incrustation directe : aucun fond peint
    # Norme de sous-titrage : 37-42 caractères par ligne. Au-delà, l'œil perd la
    # ligne au retour chariot — la coupe est faite AVANT la limite pixel, sinon
    # une grande fenêtre produirait des lignes illisiblement longues.
    max_chars_per_line: int = 42  # 0 => largeur pixel seule

    @classmethod
    def from_settings(cls, s: Settings) -> SubtitleStyle:
        return cls(
            font_family=s.font_family,
            font_size=s.font_size,
            text_color=s.text_color,
            outline_color=s.outline_color,
            bg_color=s.bg_color,
            outline_width=max(2, s.outline_width),  # contour >= 2 px minimum
            max_lines=max(1, min(3, s.max_lines)),
            align=s.align,
            margin_h=s.margin_h,
            margin_v=s.margin_v,
            display_style=s.display_style,
            transparent_bg=s.overlay_transparent,
            max_chars_per_line=max(0, s.max_chars_per_line),
        )


class SubtitleView(QWidget):
    """Renders validated subtitle text (rolling history + live utterance)."""

    def __init__(self, style: SubtitleStyle | None = None, parent=None, *, overlay: bool = False):
        super().__init__(parent)
        self._style = style or SubtitleStyle()
        self._is_overlay = overlay  # la prévisualisation montre la transparence en damier
        self._final_words: list[str] = []
        self._live = ""  # énoncé en cours (déjà validé par LocalAgreement)
        self._checker = False  # preview-only: show checkerboard instead of bg
        self.setMinimumHeight(60)

        # Animations : un seul objet réutilisé par type, relancé à chaque event.
        self._scroll_px = 0.0
        self._scroll_anim = QVariantAnimation(self, duration=SCROLL_MS)
        self._scroll_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._scroll_anim.valueChanged.connect(self._on_scroll_tick)
        self._fade = 1.0
        self._fade_from = 0  # index du premier mot en cours de fondu
        self._static_count = 0  # mots déjà affichés pleinement (jamais re-fondus)
        self._fade_anim = QVariantAnimation(self, duration=FADE_MS)
        self._fade_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._fade_anim.valueChanged.connect(self._on_fade_tick)
        self._prev_line_count = 0

    # ------------------------------------------------------------------ API
    def set_live(self, committed: str) -> None:
        """Texte validé de l'énoncé en cours (vide = énoncé terminé/pas commencé)."""
        if committed == self._live:
            return
        grew = len(committed) > len(self._live)
        self._live = committed
        self._text_changed(grew)

    def push_final(self, text: str) -> None:
        """Mots finalisés : rejoignent l'historique roulant du défilement."""
        words = text.split()
        if not words:
            return
        combined = self._final_words + words
        dropped = max(0, len(combined) - MAX_HISTORY_WORDS)
        self._final_words = combined[dropped:]
        if dropped:  # les index de fondu suivent le roulement de l'historique
            self._static_count = max(0, self._static_count - dropped)
            self._fade_from = max(0, self._fade_from - dropped)
        self._text_changed(grew=True)

    def set_texts(self, committed: str) -> None:
        """Remplacement complet (texte d'exemple, tests)."""
        self._final_words = []
        self._live = committed
        self._text_changed(grew=bool(committed))

    def clear(self) -> None:
        self._final_words = []
        self._live = ""
        self._reset_anims()
        self.update()

    def set_style(self, style: SubtitleStyle) -> None:
        self._style = style
        self._prev_line_count = self._layout_line_count()  # pas d'anim sur restyle
        self.update()

    def set_checker(self, on: bool) -> None:
        self._checker = on
        self.update()

    def current_text(self) -> str:
        """Texte affiché (historique + énoncé en cours) — debug et tests."""
        return " ".join(self._display_words())

    def visible_text(self) -> str:
        """Les lignes RÉELLEMENT visibles (au plus `max_lines`), séparées par \\n.

        C'est ce texte qui part vers les sorties externes (OBS WebSocket, page
        web) : elles affichent exactement ce que montre la fenêtre de sortie, pas
        l'historique déjà sorti de l'écran.
        """
        words = self._display_words()
        if not words:
            return ""
        fm = QFontMetricsF(self._font())
        avail_w = max(10.0, self.width() - 2.0 * self._style.margin_h)
        lines = self._wrap_all(words, avail_w, fm, self._style.max_chars_per_line)[
            -max(1, self._style.max_lines) :
        ]
        return "\n".join(" ".join(word for word, _idx in line) for line in lines)

    # ------------------------------------------------------------- animation
    def _display_words(self) -> list[str]:
        return self._final_words + self._live.split()

    def _text_changed(self, grew: bool) -> None:
        style = self._style.display_style
        if grew and style == "fondu":
            self._start_fade()
        elif grew and style == "defilement":
            count = self._layout_line_count()
            if self._prev_line_count and count > self._prev_line_count:
                self._start_scroll()
            self._prev_line_count = count
        if not self._display_words():
            self._reset_anims()
        self.update()

    def _start_fade(self) -> None:
        """Fondu des mots au-delà de _static_count ; les mots déjà pleinement
        affichés ne re-fondent jamais."""
        total = len(self._display_words())
        if self._static_count >= total:
            self._static_count = max(0, total - 1)
        self._fade_from = self._static_count
        self._fade_anim.stop()
        self._fade_anim.setStartValue(0.0)
        self._fade_anim.setEndValue(1.0)
        self._fade = 0.0
        self._fade_anim.start()

    def _on_fade_tick(self, v) -> None:
        self._fade = float(v)
        if self._fade >= 1.0:
            self._static_count = len(self._display_words())
        self.update()

    def _start_scroll(self) -> None:
        fm = QFontMetricsF(self._font())
        self._scroll_anim.stop()
        self._scroll_anim.setStartValue(float(fm.height()))
        self._scroll_anim.setEndValue(0.0)
        self._scroll_px = float(fm.height())
        self._scroll_anim.start()

    def _on_scroll_tick(self, v) -> None:
        self._scroll_px = float(v)
        self.update()

    def _reset_anims(self) -> None:
        self._scroll_anim.stop()
        self._fade_anim.stop()
        self._scroll_px = 0.0
        self._fade = 1.0
        self._static_count = 0
        self._prev_line_count = 0

    def _layout_line_count(self) -> int:
        words = self._display_words()
        if not words:
            return 0
        fm = QFontMetricsF(self._font())
        avail_w = max(10.0, self.width() - 2.0 * self._style.margin_h)
        return len(self._wrap_all(words, avail_w, fm, self._style.max_chars_per_line))

    # ------------------------------------------------------------- rendering
    def _font(self) -> QFont:
        font = QFont(self._style.font_family) if self._style.font_family else QFont()
        font.setPixelSize(max(8, self._style.font_size))
        font.setWeight(QFont.Weight.DemiBold)
        return font

    @staticmethod
    def _wrap_all(
        words: list[str], width: float, fm: QFontMetricsF, max_chars: int = 0
    ) -> list[list[tuple[str, int]]]:
        """Word-wrap complet en lignes de (mot, index_global).

        Deux bornes en parallèle : la largeur en PIXELS (la fenêtre) et la largeur
        en CARACTÈRES (la norme de lisibilité). La première qui saute coupe.
        """
        space = fm.horizontalAdvance(" ")
        lines: list[list[tuple[str, int]]] = [[]]
        x = 0.0
        chars = 0
        for idx, word in enumerate(words):
            w = fm.horizontalAdvance(word)
            over_width = bool(lines[-1]) and x + space + w > width
            over_chars = bool(max_chars) and bool(lines[-1]) and chars + 1 + len(word) > max_chars
            if over_width or over_chars:
                lines.append([])
                x = 0.0
                chars = 0
            if lines[-1]:
                x += space
                chars += 1
            lines[-1].append((word, idx))
            x += w
            chars += len(word)
        return [ln for ln in lines if ln]

    def _fading_color(self, base: str, st: SubtitleStyle) -> QColor:
        """Couleur d'un élément en cours de fondu.

        Fond opaque : interpolation RGB vers la couleur de fond (chroma-safe).
        Fond transparent : alpha réel — la fenêtre est translucide, le
        compositeur fait le fondu proprement sur n'importe quel contenu.
        """
        if st.transparent_bg:
            c = QColor(base)
            c.setAlphaF(max(0.0, min(1.0, self._fade)))
            return c
        return _lerp_color(QColor(st.bg_color), QColor(base), self._fade)

    def _word_color(self, idx: int, st: SubtitleStyle) -> QColor:
        if st.display_style == "fondu" and self._fade < 1.0 and idx >= self._fade_from:
            return self._fading_color(st.text_color, st)
        return QColor(st.text_color)

    def _outline_color(self, line: list[tuple[str, int]], st: SubtitleStyle) -> QColor:
        # Une ligne entièrement en fondu fond aussi son contour (émersion propre).
        if (
            st.display_style == "fondu"
            and self._fade < 1.0
            and line
            and all(idx >= self._fade_from for _w, idx in line)
        ):
            return self._fading_color(st.outline_color, st)
        return QColor(st.outline_color)

    def paintEvent(self, event) -> None:
        st = self._style
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        show_checker = self._checker or (st.transparent_bg and not self._is_overlay)
        if show_checker:  # prévisualisation : damier = « ceci sera transparent/chroma »
            painter.fillRect(self.rect(), QColor("#808080"))
            dark = QColor("#606060")
            for y in range(0, self.height(), _CHECKER_SIZE):
                for x in range(0, self.width(), _CHECKER_SIZE):
                    if (x // _CHECKER_SIZE + y // _CHECKER_SIZE) % 2:
                        painter.fillRect(x, y, _CHECKER_SIZE, _CHECKER_SIZE, dark)
        elif st.transparent_bg:
            pass  # fenêtre translucide : ne RIEN peindre derrière le texte
        else:
            painter.fillRect(self.rect(), QColor(st.bg_color))

        words = self._display_words()
        if not words:
            painter.end()
            return

        font = self._font()
        fm = QFontMetricsF(font)
        avail_w = max(10.0, self.width() - 2.0 * st.margin_h)
        all_lines = self._wrap_all(words, avail_w, fm, st.max_chars_per_line)
        # Pendant le défilement, garder une ligne de plus : celle qui sort de
        # l'écran glisse vers le haut au lieu de disparaître brutalement.
        scrolling = st.display_style == "defilement" and self._scroll_px > 0.5
        keep = st.max_lines + (1 if scrolling else 0)
        lines = all_lines[-keep:]

        line_h = fm.height()
        total_h = line_h * min(len(lines), st.max_lines)
        y = self.height() - st.margin_v - total_h + fm.ascent()  # bottom-anchored
        if scrolling:
            y += self._scroll_px  # le bloc part une ligne plus bas et remonte
            if len(lines) > st.max_lines:
                y -= line_h  # la ligne sortante commence au-dessus de la zone

        outline_w = 2.0 * st.outline_width
        space = fm.horizontalAdvance(" ")
        for line in lines:
            line_w = sum(fm.horizontalAdvance(w) for w, _ in line) + space * (len(line) - 1)
            if st.align == "left":
                x = float(st.margin_h)
            elif st.align == "right":
                x = self.width() - st.margin_h - line_w
            else:
                x = (self.width() - line_w) / 2.0

            outline_pen = QPen(self._outline_color(line, st), outline_w)
            outline_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            outline_pen.setCapStyle(Qt.PenCapStyle.RoundCap)

            # Pass 1: one path for the whole line -> uniform outline.
            line_path = QPainterPath()
            word_paths: list[tuple[QPainterPath, int]] = []
            wx = x
            for word, idx in line:
                p = QPainterPath()
                p.addText(wx, y, font, word)
                line_path.addPath(p)
                word_paths.append((p, idx))
                wx += fm.horizontalAdvance(word) + space
            painter.strokePath(line_path, outline_pen)
            # Pass 2: fill each word in its (opaque) color.
            for p, idx in word_paths:
                painter.fillPath(p, self._word_color(idx, st))
            y += line_h
        painter.end()
