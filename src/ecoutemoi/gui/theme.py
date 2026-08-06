"""L'apparence de l'application, tenue en un seul endroit.

Trois partis pris, assumés :

**Un thème sombre, le même partout.** L'application vit à côté d'OBS, dans une
régie, souvent sur un écran mal calibré au fond d'une salle. Un fond graphite
laisse la place aux seules couleurs qui portent une information : le vert du fond
chroma, et le voyant temps réel. Le même rendu sur les trois systèmes vaut mieux
qu'un rendu « natif » ici et là : une capture d'écran d'aide correspond alors à ce
que l'opérateur a sous les yeux.

**La palette Qt d'abord, la feuille de style ensuite.** Tout ce que Qt dessine
lui-même — coches, flèches, poignées — suit la `QPalette` ; l'habiller à la
feuille de style demanderait des images et casserait les cases à cocher sur la
moitié des systèmes. La feuille de style ne reprend donc la main que sur ce qui
est un rectangle et du texte : cartes, boutons, champs, onglets, tableaux.

**Les couleurs ont un sens.** Une seule teinte d'accent pour « ici, vous pouvez
agir » (sélection, focus, bouton principal), et trois couleurs d'état — vert,
ambre, rouge — réservées à ce que la machine a à dire. Rien de décoratif : si
quelque chose est coloré dans cette interface, c'est que ça veut dire quelque
chose.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPalette

# Familles d'interface, par ordre de préférence. Qt prend la première installée ;
# la dernière est présente partout où tourne Qt.
UI_FONTS = (
    "Inter",
    "Segoe UI Variable Text",
    "Segoe UI",
    "SF Pro Text",
    "Noto Sans",
    "DejaVu Sans",
    "Sans Serif",
)
MONO_FONTS = ("JetBrains Mono", "Cascadia Mono", "SF Mono", "DejaVu Sans Mono", "monospace")


@dataclass(frozen=True)
class Palette:
    """Les seules couleurs de l'application. Aucune autre valeur en dur ailleurs."""

    window: str  # fond des fenêtres
    surface: str  # fond des cartes
    field: str  # fond des champs de saisie
    raised: str  # survol, en-têtes de tableau
    border: str  # trait des cartes et des champs
    border_light: str  # séparateurs discrets
    text: str
    text_muted: str  # libellés secondaires, aides
    text_faint: str  # texte désactivé
    accent: str  # sélection, focus, action principale
    accent_hover: str
    accent_dim: str  # remplissage sous du texte clair (barres de progression)
    accent_text: str  # texte posé sur l'accent
    live: str  # « à l'antenne » : session en cours
    live_hover: str
    ok: str
    warn: str
    error: str
    meter_bg: str  # fond du vumètre

    def q(self, name: str) -> QColor:
        return QColor(getattr(self, name))


# « Régie » : graphite froid, accent bleu-cyan calme, rouge d'antenne pour l'arrêt.
DARK = Palette(
    window="#12151b",
    surface="#191d25",
    field="#10131a",
    raised="#222834",
    border="#2c3341",
    border_light="#232a35",
    text="#e6e9ef",
    text_muted="#98a2b3",
    text_faint="#5f6875",
    accent="#3d8fb0",
    accent_hover="#4aa3c6",
    accent_dim="#235a70",
    accent_text="#f2fbff",
    live="#a5332f",
    live_hover="#bb3a35",
    ok="#42bd77",
    warn="#dfa03a",
    error="#dd5b50",
    meter_bg="#0c0f14",
)


def palette() -> Palette:
    """La palette active. Un point de passage unique, pour le jour où il en
    faudra une seconde (thème clair, fort contraste)."""
    return DARK


def qt_palette(p: Palette) -> QPalette:
    """La palette Qt : c'est elle qui habille ce que Qt dessine sans nous."""
    pal = QPalette()
    roles = {
        QPalette.ColorRole.Window: p.window,
        QPalette.ColorRole.WindowText: p.text,
        QPalette.ColorRole.Base: p.field,
        QPalette.ColorRole.AlternateBase: p.raised,
        QPalette.ColorRole.Text: p.text,
        QPalette.ColorRole.Button: p.surface,
        QPalette.ColorRole.ButtonText: p.text,
        QPalette.ColorRole.BrightText: p.accent_text,
        QPalette.ColorRole.Highlight: p.accent,
        QPalette.ColorRole.HighlightedText: p.accent_text,
        QPalette.ColorRole.ToolTipBase: p.raised,
        QPalette.ColorRole.ToolTipText: p.text,
        QPalette.ColorRole.PlaceholderText: p.text_faint,
        QPalette.ColorRole.Link: p.accent_hover,
        QPalette.ColorRole.LinkVisited: p.accent,
    }
    for role, color in roles.items():
        pal.setColor(role, QColor(color))
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        pal.setColor(QPalette.ColorGroup.Disabled, role, QColor(p.text_faint))
    return pal


def stylesheet(p: Palette) -> str:
    """Feuille de style : cartes, boutons, champs, onglets, tableaux.

    Les noms d'objet servent de variantes : `primary` pour l'action principale,
    `live` pour l'état « en cours », `danger` pour ce qui détruit.
    """
    return f"""
QMainWindow, QDialog {{ background: {p.window}; }}
QWidget {{ color: {p.text}; }}

QGroupBox {{
    background: {p.surface};
    border: 1px solid {p.border};
    border-radius: 10px;
    margin-top: 14px;
    padding: 14px 12px 12px 12px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    color: {p.text_muted};
}}

QLabel[role="hint"] {{ color: {p.text_muted}; }}
QLabel[role="warn"] {{ color: {p.warn}; }}
QLabel[role="error"] {{ color: {p.error}; }}
QLabel[role="heading"] {{ font-weight: 600; font-size: 13px; }}
QLabel:disabled {{ color: {p.text_faint}; }}

QPushButton {{
    background: {p.raised};
    color: {p.text};
    border: 1px solid {p.border};
    border-radius: 8px;
    padding: 6px 14px;
    min-height: 20px;
}}
QPushButton:hover {{ background: {p.border}; border-color: {p.accent}; }}
QPushButton:pressed {{ background: {p.field}; }}
QPushButton:disabled {{ background: {p.surface}; color: {p.text_faint}; border-color: {p.border_light}; }}
QPushButton:focus {{ border-color: {p.accent}; }}

QPushButton#primary {{
    background: {p.accent};
    color: {p.accent_text};
    border: 1px solid {p.accent};
    font-weight: 600;
}}
QPushButton#primary:hover {{ background: {p.accent_hover}; border-color: {p.accent_hover}; }}
QPushButton#primary:disabled {{
    background: {p.surface}; color: {p.text_faint}; border-color: {p.border_light};
}}
QPushButton#primary:checked {{ background: {p.live}; border-color: {p.live}; }}
QPushButton#primary:checked:hover {{ background: {p.live_hover}; border-color: {p.live_hover}; }}
QPushButton#danger:hover {{ border-color: {p.error}; color: {p.error}; }}
/* Bouton réduit à un glyphe : la marge des boutons normaux le rognerait. */
QPushButton#icon {{ padding: 4px 2px; min-width: 24px; font-size: 13px; }}

QLineEdit, QPlainTextEdit, QTextEdit, QTextBrowser, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget {{
    background: {p.field};
    border: 1px solid {p.border};
    border-radius: 8px;
    padding: 4px 6px;
    selection-background-color: {p.accent};
    selection-color: {p.accent_text};
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QSpinBox:focus,
QDoubleSpinBox:focus, QComboBox:focus, QListWidget:focus {{ border-color: {p.accent}; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    background: {p.surface}; color: {p.text_faint};
}}
QComboBox::drop-down {{ border: none; width: 20px; }}
/* Les flèches des compteurs sont dessinées par Qt : leur laisser la place que la
   marge intérieure des champs leur prendrait. */
QSpinBox, QDoubleSpinBox {{ padding-right: 2px; }}
QComboBox QAbstractItemView {{
    background: {p.surface};
    border: 1px solid {p.border};
    selection-background-color: {p.accent};
    selection-color: {p.accent_text};
    padding: 4px;
}}

QMenuBar {{ background: {p.window}; border-bottom: 1px solid {p.border_light}; }}
QMenuBar::item {{ padding: 6px 10px; background: transparent; border-radius: 6px; }}
QMenuBar::item:selected {{ background: {p.raised}; }}
QMenu {{ background: {p.surface}; border: 1px solid {p.border}; padding: 6px; }}
QMenu::item {{ padding: 6px 24px 6px 12px; border-radius: 6px; }}
QMenu::item:selected {{ background: {p.accent}; color: {p.accent_text}; }}
QMenu::separator {{ height: 1px; background: {p.border_light}; margin: 5px 8px; }}

QTabWidget::pane {{ border: 1px solid {p.border}; border-radius: 10px; top: -1px; }}
QTabBar::tab {{
    background: transparent;
    color: {p.text_muted};
    padding: 7px 14px;
    margin-right: 2px;
    border: 1px solid transparent;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}}
QTabBar::tab:hover {{ color: {p.text}; }}
QTabBar::tab:selected {{
    background: {p.surface};
    color: {p.text};
    border-color: {p.border};
    border-bottom-color: {p.surface};
}}

QTableWidget, QTableView {{
    background: {p.field};
    border: 1px solid {p.border};
    border-radius: 10px;
    gridline-color: {p.border_light};
}}
QHeaderView::section {{
    background: {p.raised};
    color: {p.text_muted};
    border: none;
    border-bottom: 1px solid {p.border};
    padding: 6px 8px;
    font-weight: 600;
}}
QTableWidget::item {{ padding: 4px 6px; }}
QTableWidget::item:selected {{ background: {p.accent}; color: {p.accent_text}; }}

QProgressBar {{
    background: {p.field};
    border: 1px solid {p.border};
    border-radius: 7px;
    min-height: 16px;
    text-align: center;
    color: {p.text};
}}
/* Remplissage assombri : le texte de la barre passe par-dessus, il doit rester
   lisible aussi bien sur le fond vide que sur la partie remplie. */
QProgressBar::chunk {{ background: {p.accent_dim}; border-radius: 6px; }}

QScrollBar:vertical {{ background: transparent; width: 12px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 12px; margin: 2px; }}
QScrollBar::handle {{ background: {p.border}; border-radius: 5px; min-height: 28px; min-width: 28px; }}
QScrollBar::handle:hover {{ background: {p.text_faint}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QStatusBar {{ background: {p.window}; border-top: 1px solid {p.border_light}; }}
QStatusBar QLabel {{ color: {p.text_muted}; padding: 0 6px; }}
QStatusBar::item {{ border: none; }}

QToolTip {{
    background: {p.raised};
    color: {p.text};
    border: 1px solid {p.border};
    border-radius: 6px;
    padding: 6px 8px;
}}

QSplitter::handle {{ background: {p.border_light}; }}
QFrame[frameShape="4"], QFrame[frameShape="5"] {{ color: {p.border}; }}
"""


def icon_path() -> Path:
    import ecoutemoi

    return Path(ecoutemoi.__file__).resolve().parent / "assets" / "icon.png"


def app_icon() -> QIcon:
    path = icon_path()
    return QIcon(str(path)) if path.is_file() else QIcon()


def ui_font(size: int = 10, *, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    font = QFont()
    font.setFamilies(list(UI_FONTS))
    font.setPointSize(size)
    font.setWeight(weight)
    return font


def mono_font(size: int = 9) -> QFont:
    font = QFont()
    font.setFamilies(list(MONO_FONTS))
    font.setPointSize(size)
    return font


# Les traducteurs Qt doivent survivre à l'appel : Qt ne garde qu'un pointeur.
_TRANSLATORS: list = []


def install_french(app) -> bool:
    """Traduit les boutons et dialogues fournis par Qt.

    Sans ça, l'application est en français mais Qt répond « Close », « Yes », et
    le sélecteur de fichiers s'ouvre en anglais : le genre de détail qui fait
    tout de suite bricolé. Les traductions voyagent avec PySide6 ; si le fichier
    manque (bundle allégé), on continue simplement sans.
    """
    from PySide6.QtCore import QLibraryInfo, QLocale, QTranslator

    translations = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
    for name in ("qtbase_fr", "qt_fr"):
        translator = QTranslator(app)
        if translator.load(name, translations):
            app.installTranslator(translator)
            _TRANSLATORS.append(translator)
    if not _TRANSLATORS:
        return False
    QLocale.setDefault(QLocale(QLocale.Language.French))
    return True


def apply_theme(app) -> Palette:
    """Habille l'application entière. À appeler avant de créer la fenêtre."""
    p = palette()
    app.setStyle("Fusion")  # base commune aux trois systèmes, sans surprise
    app.setPalette(qt_palette(p))
    app.setFont(ui_font())
    app.setStyleSheet(stylesheet(p))
    install_french(app)
    icon = app_icon()
    if not icon.isNull():
        app.setWindowIcon(icon)
    return p


def mark(widget, role: str):
    """Étiquette un widget pour la feuille de style (`hint`, `warn`, `heading`…)."""
    widget.setProperty("role", role)
    return widget


def hint(text: str, *, role: str = "hint"):
    """Libellé d'aide : gris, retour à la ligne automatique, sélectionnable.

    Ces phrases sont la documentation de l'application. Elles doivent pouvoir être
    copiées (un chemin, un port, une adresse) sans être prises pour du texte
    principal — d'où le gris.
    """
    from PySide6.QtWidgets import QLabel

    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return mark(label, role)


__all__ = [
    "DARK",
    "Palette",
    "app_icon",
    "apply_theme",
    "hint",
    "icon_path",
    "mark",
    "mono_font",
    "palette",
    "qt_palette",
    "stylesheet",
    "ui_font",
]
