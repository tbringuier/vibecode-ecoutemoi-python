"""GUI smoke tests (offscreen platform): windows build, widgets exist, the
shared SubtitleView renders committed/pending text with outline (invariant:
preview and overlay use the SAME widget class and style).
"""

import os

import pytest

pytestmark = pytest.mark.integration

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# exc_type=ImportError: sans libs système Qt (libEGL…), l'import échoue avec un
# ImportError qui n'est pas un ModuleNotFoundError — il doit sauter, pas casser
# la collecte pytest.
pyside = pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
    # PySide 6.11 : tout widget C++ encore vivant à Py_Finalize expose
    # PySide::destroyQCoreApplication à un segfault d'ordre de destruction
    # (exit 139 en CI, backtrace QObject::~QObject via visitAllPyObjects).
    import gc

    gc.collect()
    app.processEvents()


@pytest.fixture
def reg(qapp):
    """Enregistre chaque widget créé par un test pour le DÉTRUIRE explicitement
    (deleteLater + processEvents) pendant que Qt est encore entier — sans quoi
    des arbres de fenêtres entiers survivent jusqu'à l'atexit de PySide et la
    destruction en vrac segfaute aléatoirement (heisencrash CI, exit 139)."""
    created = []

    def _register(widget):
        created.append(widget)
        return widget

    yield _register
    for w in created:
        w.close()
        w.deleteLater()
    qapp.processEvents()  # exécuter les deleteLater maintenant, pas à l'atexit


@pytest.fixture(autouse=True)
def isolate_disclaimer_marker(tmp_path, monkeypatch):
    """Redirige le marqueur d'acceptation vers tmp_path, pour TOUS les tests.

    `disclaimer_marker_path` ne dérive pas de `config_path` — le marqueur est
    délibérément séparé de settings.json — donc patcher `config_path` ne le
    couvrait pas. Les tests qui construisaient une MainWindow lisaient, et
    ÉCRIVAIENT, dans le vrai dossier de config de la machine.

    Le symptôme n'était pas la pollution mais la non-reproductibilité : sur un
    poste ayant déjà lancé l'application le marqueur existe, le chemin « premier
    lancement » n'est jamais pris et tout passe ; sur un runner neuf il est pris,
    ouvre une boucle d'évènements imbriquée pendant la destruction de la fenêtre,
    et la CI tombait sur un défaut invisible en local. Ce chemin est désormais
    pris PARTOUT.
    """
    import ecoutemoi.config as config

    monkeypatch.setattr(config, "disclaimer_marker_path", lambda: tmp_path / "disclaimer_accepted")


def test_subtitle_view_renders_text(qapp, reg, tmp_path):
    from PySide6.QtGui import QColor, QImage

    from ecoutemoi.gui.subtitle_view import SubtitleStyle, SubtitleView

    view = reg(SubtitleView(SubtitleStyle(bg_color="#00FF00", font_size=40, display_style="statique")))
    view.resize(800, 200)
    view.set_texts("Bonjour à tous")
    img: QImage = view.grab().toImage()
    assert img.width() == 800

    colors = set()
    # pas de 2 : un contour de ~3 px peut passer entre les mailles d'une grille de 4
    for x in range(0, 800, 2):
        for y in range(0, 200, 2):
            colors.add(img.pixelColor(x, y).name().upper())
    assert "#00FF00" in colors  # chroma background
    assert "#FFFFFF" in colors  # committed text fill
    assert any(c != "#00FF00" and c != "#FFFFFF" for c in colors)  # outline

    # empty text -> pure background
    view.clear()
    img2 = view.grab().toImage()
    corner = img2.pixelColor(5, 5)
    assert corner == QColor("#00FF00")


def test_subtitle_view_takes_only_validated_text(qapp, reg):
    """La vue n'a AUCUN canal pour du texte non validé : `set_texts` prend un seul
    argument. Sans texte validé, elle ne peint que le fond."""
    import inspect

    from PySide6.QtGui import QImage

    from ecoutemoi.gui.subtitle_view import SubtitleStyle, SubtitleView

    params = list(inspect.signature(SubtitleView.set_texts).parameters)
    assert params == ["self", "committed"], "un paramètre « pending » est réapparu"

    view = reg(SubtitleView(SubtitleStyle(bg_color="#00FF00", font_size=40, display_style="statique")))
    view.resize(800, 200)
    view.set_texts("")
    assert view.current_text() == ""
    img: QImage = view.grab().toImage()
    colors = {img.pixelColor(x, y).name().upper() for x in range(0, 800, 8) for y in range(0, 200, 8)}
    assert colors == {"#00FF00"}  # fond pur : rien d'autre n'a été peint


def test_subtitle_view_rolling_history(qapp, reg):
    from ecoutemoi.gui.subtitle_view import SubtitleStyle, SubtitleView

    view = reg(SubtitleView(SubtitleStyle(display_style="statique")))
    view.resize(800, 200)
    view.set_live("bonjour à tous")
    view.push_final("bonjour à tous.")
    view.set_live("")  # fin d'énoncé : l'historique garde l'écran plein
    assert view.current_text() == "bonjour à tous."
    view.set_live("la suite arrive")
    assert view.current_text() == "bonjour à tous. la suite arrive"


def test_subtitle_view_transparent_background(qapp, reg):
    """Mode transparent : l'overlay ne peint AUCUN fond ; la préviz montre un damier."""
    from PySide6.QtGui import QImage

    from ecoutemoi.gui.subtitle_view import SubtitleStyle, SubtitleView

    style = SubtitleStyle(bg_color="#00FF00", transparent_bg=True, display_style="statique")
    overlay_view = reg(SubtitleView(style, overlay=True))
    overlay_view.resize(400, 100)
    img: QImage = overlay_view.grab().toImage()
    colors = {img.pixelColor(x, y).name().upper() for x in range(0, 400, 8) for y in range(0, 100, 8)}
    assert "#00FF00" not in colors  # le chroma n'est plus peint

    preview = reg(SubtitleView(style))  # préviz : damier de visualisation
    preview.resize(400, 100)
    img2 = preview.grab().toImage()
    colors2 = {img2.pixelColor(x, y).name().upper() for x in range(0, 400, 8) for y in range(0, 100, 8)}
    assert {"#808080", "#606060"} <= colors2


def test_overlay_uses_same_widget_class_and_style(qapp, reg):
    from ecoutemoi.config import Settings
    from ecoutemoi.gui.overlay import OverlayWindow
    from ecoutemoi.gui.subtitle_view import SubtitleStyle, SubtitleView

    settings = Settings(bg_color="#FF00FF", font_size=28)
    overlay = reg(OverlayWindow(settings))
    assert type(overlay.view) is SubtitleView  # same widget class as the preview
    assert overlay.windowTitle() == "EcouteMoi - Sortie OBS"
    style = SubtitleStyle.from_settings(settings)
    assert overlay.view._style == style
    overlay.close()


def test_main_window_builds(qapp, reg, tmp_path, monkeypatch):
    import ecoutemoi.config as config

    monkeypatch.setattr(config, "config_path", lambda: tmp_path / "settings.json")
    from ecoutemoi.gui.main_window import MainWindow

    win = reg(MainWindow())
    reg(win.overlay)  # top-level sans parent Qt : détruit explicitement
    assert win.btn_start is not None
    assert win.preview is not None and win.overlay.view is not None
    assert type(win.preview) is type(win.overlay.view)  # identical rendering path
    # appearance change propagates to BOTH views
    win.font_size.setValue(50)
    assert win.preview._style.font_size == 50
    assert win.overlay.view._style.font_size == 50
    # sample text toggle fills the preview
    win.chk_sample.setChecked(True)
    assert win.preview.current_text() != ""
    # préréglage « Transparent » -> overlay translucide, préviz prévenue
    win.bg_preset.setCurrentIndex(win.bg_preset.findData("__transparent__"))
    assert win.settings.overlay_transparent is True
    assert win.preview._style.transparent_bg is True
    win.close()


def test_settings_dialog_roundtrip(qapp, reg):
    from ecoutemoi.config import Settings
    from ecoutemoi.gui.settings_dialog import SettingsDialog

    dlg = reg(SettingsDialog(Settings(), None))
    dlg.silence.setValue(650)
    dlg.backend.setCurrentIndex(2)  # "cpu"
    dlg.audio_clean.setCurrentIndex(dlg.audio_clean.findData("full"))
    dlg._accept()
    assert dlg.result_settings is not None
    assert dlg.result_settings.silence_ms == 650
    assert dlg.result_settings.keep_back is None  # sentinel -> preset
    assert dlg.result_settings.backend == "cpu"
    assert dlg.result_settings.denoise is True  # « Passe-haut + RNNoise »
    assert dlg.result_settings.highpass is True


def test_settings_dialog_defaults_to_no_audio_cleanup(qapp, reg):
    from ecoutemoi.config import Settings
    from ecoutemoi.gui.settings_dialog import SettingsDialog

    dlg = reg(SettingsDialog(Settings(), None))
    assert dlg.audio_clean.currentData() == "none"
    dlg._accept()
    assert dlg.result_settings.denoise is False
    assert dlg.result_settings.highpass is False


def test_settings_dialog_publish_roundtrip(qapp, reg):
    """Onglet Diffusion : OBS WebSocket + page web font l'aller-retour."""
    from ecoutemoi.config import Settings
    from ecoutemoi.gui.settings_dialog import SettingsDialog

    dlg = reg(SettingsDialog(Settings(), None))
    # La case est formulée en « localhost uniquement » et doit être COCHÉE par
    # défaut : l'état sûr est celui qu'on obtient sans rien toucher.
    assert dlg.web_localhost.isChecked()
    dlg.obs_enabled.setChecked(True)
    dlg.obs_host.setText("192.168.1.20")
    dlg.obs_port.setValue(4460)
    dlg.obs_password.setText("secret")
    dlg.obs_source.setCurrentText("Sous-titres")
    dlg.obs_source_dual.setCurrentText("Sous-titres EN")
    dlg.web_enabled.setChecked(True)
    dlg.web_port.setValue(9123)
    dlg.web_localhost.setChecked(False)  # ouvert au réseau local
    dlg._accept()
    s = dlg.result_settings
    assert (s.obs_ws_enabled, s.obs_ws_host, s.obs_ws_port) == (True, "192.168.1.20", 4460)
    assert (s.obs_ws_password, s.obs_ws_source) == ("secret", "Sous-titres")
    assert s.obs_ws_source_dual == "Sous-titres EN"
    assert (s.web_enabled, s.web_port, s.web_bind_lan) == (True, 9123, True)


def test_settings_dialog_lists_lan_addresses_when_opened(qapp, reg):
    """Décocher « localhost uniquement » doit AFFICHER les IP à taper ailleurs."""
    from ecoutemoi.config import Settings
    from ecoutemoi.core.publish import local_ip_addresses
    from ecoutemoi.gui.settings_dialog import SettingsDialog

    dlg = reg(SettingsDialog(Settings(web_port=8777), None))
    assert dlg.web_urls.text() == "http://127.0.0.1:8777/"
    dlg.web_localhost.setChecked(False)
    shown = dlg.web_urls.text()
    assert "http://127.0.0.1:8777/" in shown
    for ip in local_ip_addresses():
        assert f"http://{ip}:8777/" in shown


def test_settings_dialog_subtitling_roundtrip(qapp, reg):
    from ecoutemoi.config import Settings
    from ecoutemoi.gui.settings_dialog import SettingsDialog

    dlg = reg(SettingsDialog(Settings(), None))
    assert dlg.pacing.isChecked()  # normes actives par défaut
    assert dlg.reading_wpm.value() == 180
    assert dlg.cpl.value() == 42
    dlg.reading_wpm.setValue(160)
    dlg.pacing_lag.setValue(4.0)
    dlg.cpl.setValue(37)
    dlg.pacing.setChecked(False)
    dlg.subproc.setChecked(False)
    dlg.prewarm.setChecked(False)
    dlg._accept()
    s = dlg.result_settings
    assert (s.reading_wpm, s.pacing_max_lag_s, s.max_chars_per_line) == (160, 4.0, 37)
    assert s.pacing_enabled is False
    assert s.engine_subprocess is False and s.prewarm_on_launch is False


def test_status_light_blinks_by_severity(qapp, reg):
    """Voyant temps réel : couleur ET période de clignotement suivent la gravité."""
    from ecoutemoi.constants import RT_BLINK_MS
    from ecoutemoi.gui.widgets import StatusLight

    light = reg(StatusLight())
    assert light.level() == "off"
    assert not light._timer.isActive()  # au repos : pas de clignotement
    for level in ("green", "orange", "red"):
        light.set_level(level, f"détail {level}")
        assert light.level() == level
        assert light._timer.isActive()
        assert light._timer.interval() == RT_BLINK_MS[level]
        assert f"détail {level}" in light.toolTip()
    assert RT_BLINK_MS["red"] < RT_BLINK_MS["orange"] < RT_BLINK_MS["green"]


def test_subtitle_view_visible_text_keeps_only_shown_lines(qapp, reg):
    """Le texte diffusé (OBS/web) = les lignes VISIBLES, pas tout l'historique."""
    from ecoutemoi.gui.subtitle_view import SubtitleStyle, SubtitleView

    view = reg(SubtitleView(SubtitleStyle(font_size=40, max_lines=2, display_style="statique")))
    view.resize(420, 160)
    assert view.visible_text() == ""
    view.set_live(" ".join(f"mot{i:02d}" for i in range(40)))
    visible = view.visible_text()
    assert visible.count("\n") == 1  # exactement max_lines lignes
    assert "mot39" in visible  # la fin du texte est bien celle qu'on voit
    assert "mot00" not in visible  # le début est sorti de l'écran


def test_main_window_rt_light_follows_stats(qapp, reg, tmp_path, monkeypatch):
    import ecoutemoi.config as config

    monkeypatch.setattr(config, "config_path", lambda: tmp_path / "settings.json")
    from ecoutemoi.core.streamer import StreamStats
    from ecoutemoi.gui.main_window import MainWindow

    win = reg(MainWindow())
    reg(win.overlay)
    reg(win.overlay_dual)
    assert win.rt_light.level() == "off"

    def feed(lag: float, channel: int = 0) -> None:
        for _ in range(5):  # la médiane porte sur RT_LIGHT_WINDOW décodages
            win._on_stats(channel, StreamStats(decode_ms=100.0, window_s=5.0, rtf=1 / lag,
                                               lag=lag, latency_ms=500.0))  # fmt: skip

    feed(0.2)  # RTF 5 : large marge
    assert win.rt_light.level() == "green"
    feed(0.6)  # RTF ~1,7 : ça tient sans marge
    assert win.rt_light.level() == "orange"
    feed(1.4)  # RTF < 1 : le direct décroche
    assert win.rt_light.level() == "red"
    win._on_degraded(True)  # bascule auto en mode Phrase -> rouge quoi qu'il arrive
    assert win.rt_light.level() == "red"
    win.close()


def test_rt_light_follows_the_worst_channel(qapp, reg, tmp_path, monkeypatch):
    """Double sous-titre : si UN canal décroche, le public le voit — le voyant aussi."""
    import ecoutemoi.config as config

    monkeypatch.setattr(config, "config_path", lambda: tmp_path / "settings.json")
    from ecoutemoi.core.streamer import StreamStats
    from ecoutemoi.gui.main_window import MainWindow

    win = reg(MainWindow())
    reg(win.overlay)
    reg(win.overlay_dual)
    for _ in range(5):
        win._on_stats(0, StreamStats(100.0, 5.0, 5.0, 0.2, 500.0))  # canal sain
        win._on_stats(1, StreamStats(100.0, 5.0, 0.7, 1.4, 500.0))  # canal noyé
    assert win.rt_light.level() == "red"
    assert "pire des deux" in win.rt_light.toolTip()
    win.close()


def test_main_window_title_is_app_name(qapp, reg, tmp_path, monkeypatch):
    import ecoutemoi.config as config

    monkeypatch.setattr(config, "config_path", lambda: tmp_path / "settings.json")
    from ecoutemoi.gui.main_window import MainWindow

    win = reg(MainWindow())
    reg(win.overlay)
    assert win.windowTitle() == "Écoute Moi"
    # Rubriques sans numérotation, et « prévisualisation » en clair
    titles = [box.title() for box in win.findChildren(pyside.QGroupBox)]
    assert "Entrée audio" in titles and "Session" in titles
    assert not any(title[:1] in "①②③④⑤⑥" for title in titles)
    assert any(title.startswith("Prévisualisation") for title in titles)
    win.close()


def test_disclaimer_dialog_cannot_be_dismissed_on_first_run(qapp, reg):
    """Une décharge doit être ACCEPTÉE : ni croix, ni Échap, ni reject()."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeyEvent

    from ecoutemoi.gui.main_window import DisclaimerDialog

    accepted = int(DisclaimerDialog.DialogCode.Accepted)
    rejected = int(DisclaimerDialog.DialogCode.Rejected)
    escape = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier(0))

    # `result()` vaut 0 = Rejected avant toute interaction : c'est le signal
    # `finished` qui dit si la fenêtre s'est réellement fermée.
    dlg = reg(DisclaimerDialog(None, first_run=True))
    closed: list[int] = []
    dlg.finished.connect(closed.append)
    dlg.reject()
    assert closed == [], "reject() a fermé l'avertissement de premier lancement"
    dlg.keyPressEvent(escape)
    assert closed == [], "Échap a fermé l'avertissement de premier lancement"
    dlg.accept()  # seule sortie
    assert closed == [accepted]
    # La croix de fenêtre est retirée, pas seulement ignorée
    assert not dlg.windowFlags() & Qt.WindowType.WindowCloseButtonHint

    # Reconsulté depuis le menu Aide : là il se ferme normalement.
    again = reg(DisclaimerDialog(None, first_run=False))
    closed_again: list[int] = []
    again.finished.connect(closed_again.append)
    again.reject()
    assert closed_again == [rejected]


def test_disclaimer_states_the_essentials():
    from ecoutemoi.gui.main_window import DISCLAIMER_HTML

    text = DISCLAIMER_HTML.lower()
    assert "vibecodé" in text
    assert "sans aucune garantie" in text
    assert "gratuit" in text


def test_disclaimer_shown_once_then_remembered(qapp, reg, tmp_path, monkeypatch):
    import ecoutemoi.config as config

    monkeypatch.setattr(config, "config_path", lambda: tmp_path / "settings.json")
    monkeypatch.setattr(config, "disclaimer_marker_path", lambda: tmp_path / "ack")
    from ecoutemoi.gui.main_window import MainWindow

    shown = []
    win = reg(MainWindow())
    reg(win.overlay)
    reg(win.overlay_dual)
    monkeypatch.setattr(
        "ecoutemoi.gui.main_window.DisclaimerDialog",
        lambda *a, **k: type("Fake", (), {"exec": lambda self: shown.append(1)})(),
    )
    win._after_show()
    assert shown == [1] and (tmp_path / "ack").exists()
    win._after_show()  # deuxième lancement : plus d'avertissement
    assert shown == [1]
    win.close()


def test_gpu_combo_says_just_automatique(qapp, reg, tmp_path, monkeypatch):
    import ecoutemoi.config as config

    monkeypatch.setattr(config, "config_path", lambda: tmp_path / "settings.json")
    from ecoutemoi.gui.main_window import MainWindow

    win = reg(MainWindow())
    reg(win.overlay)
    assert win.gpu_combo.itemText(0) == "automatique"
    assert win.mode_tr.text() == "FR → EN"
    win.close()


def test_backend_combo_choices(qapp, reg, tmp_path, monkeypatch):
    import ecoutemoi.config as config

    monkeypatch.setattr(config, "config_path", lambda: tmp_path / "settings.json")
    from ecoutemoi.gui.main_window import MainWindow

    win = reg(MainWindow())
    reg(win.overlay)
    values = [win.backend_combo.itemData(i) for i in range(win.backend_combo.count())]
    assert values == ["auto", "gpu", "cpu"]
    win.backend_combo.setCurrentIndex(1)
    win._collect_settings()
    assert win.settings.backend == "gpu"
    win.close()


def test_theme_is_a_single_source_of_colours(qapp):
    """Le thème s'applique à l'application entière, sans valeurs en dur ailleurs."""
    from ecoutemoi.gui.theme import DARK, apply_theme, palette, stylesheet

    assert palette() is DARK
    sheet = stylesheet(DARK)
    assert DARK.accent in sheet and DARK.surface in sheet
    applied = apply_theme(qapp)
    assert applied is DARK
    assert qapp.styleSheet() == sheet
    # La palette Qt porte les mêmes couleurs : c'est elle qui habille les coches
    # et les flèches, que la feuille de style ne touche pas.
    from PySide6.QtGui import QPalette

    assert qapp.palette().color(QPalette.ColorRole.Window).name() == DARK.window
    assert qapp.palette().color(QPalette.ColorRole.Highlight).name() == DARK.accent


def test_transcribe_dialog_builds_and_validates(qapp, reg, tmp_path):
    """Fenêtre de transcription : liste, formats, garde-fous."""
    import numpy as np

    sf = pytest.importorskip("soundfile")
    from ecoutemoi.config import Settings
    from ecoutemoi.gui.transcribe_dialog import TranscribeDialog

    voice = (np.random.default_rng(4).standard_normal(16000) * 0.2).astype("float32")
    audio = tmp_path / "entretien.wav"
    sf.write(str(audio), voice, 16000)

    dlg = reg(TranscribeDialog(Settings(transcribe_formats="txt,srt"), None))
    assert dlg.paths == []
    assert not dlg.btn_run.isEnabled()  # rien à transcrire
    assert dlg._formats() == ["txt", "srt"]

    dlg._add_paths([audio])
    assert dlg.paths == [audio.resolve()]
    assert dlg.table.rowCount() == 1
    assert "16 kHz" in dlg.table.item(0, 1).text()  # format d'origine sondé
    assert dlg.btn_run.isEnabled()

    dlg._add_paths([audio])  # deux fois le même fichier : ignoré
    assert dlg.paths == [audio.resolve()]

    for check in dlg.format_checks.values():  # aucun format : on ne peut pas partir
        check.setChecked(False)
    assert not dlg.btn_run.isEnabled()
    assert "format de sortie" in dlg.warn.text()

    dlg.format_checks["md"].setChecked(True)
    assert dlg._formats() == ["md"] and dlg.btn_run.isEnabled()

    dlg._clear()
    assert dlg.paths == [] and dlg.table.rowCount() == 0


def test_transcribe_dialog_refuses_translation_with_a_model_that_cannot(qapp, reg):
    from ecoutemoi.config import Settings
    from ecoutemoi.core import models
    from ecoutemoi.gui.transcribe_dialog import TranscribeDialog

    turbo = next(k for k, spec in models.REGISTRY.items() if not spec.translate)
    dlg = reg(TranscribeDialog(Settings(transcribe_mode="translate"), None))
    # La liste ne contient que les modèles installés : on injecte celui qui nous
    # intéresse pour éprouver la validation, pas l'état de la machine.
    if dlg.model.findData(turbo) < 0:
        dlg.model.addItem(turbo, turbo)
    dlg.model.setCurrentIndex(dlg.model.findData(turbo))
    assert "traduction" in dlg.warn.text()
    dlg.mode.setCurrentIndex(dlg.mode.findData("fr"))
    assert "traduction" not in dlg.warn.text()


def test_file_menu_offers_file_transcription(qapp, reg, tmp_path, monkeypatch):
    import ecoutemoi.config as config

    monkeypatch.setattr(config, "config_path", lambda: tmp_path / "settings.json")
    from ecoutemoi.gui.main_window import MainWindow

    win = reg(MainWindow())
    reg(win.overlay)
    reg(win.overlay_dual)
    menus = {m.title(): m for m in win.menuBar().findChildren(pyside.QMenu)}
    assert {"Fichier", "Direct", "Outils", "Aide"} <= set(menus)
    labels = [a.text() for a in menus["Fichier"].actions()]
    assert any("Transcrire des fichiers audio" in label for label in labels)
    # Les huit formats sont proposés à l'export de session
    export = menus["Exporter la session"]
    assert len([a for a in export.actions() if not a.isSeparator()]) >= 8
    assert any(".md" in a.text() for a in export.actions())
    win.close()


def test_main_window_accepts_files_from_the_command_line(qapp, reg, tmp_path, monkeypatch):
    """« Ouvrir avec… » : les fichiers attendent la fenêtre de transcription."""
    import ecoutemoi.config as config

    monkeypatch.setattr(config, "config_path", lambda: tmp_path / "settings.json")
    from ecoutemoi.gui.main_window import MainWindow

    win = reg(MainWindow(["/tmp/a.mp3", "/tmp/b.m4a"]))
    reg(win.overlay)
    reg(win.overlay_dual)
    assert [p.name for p in win._pending_files] == ["a.mp3", "b.m4a"]
    win.close()
