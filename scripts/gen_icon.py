"""Generate the app icon (assets/icon.png 256px + icon.ico) with QPainter.

Run once on a dev machine: uv run python scripts/gen_icon.py
The macOS .icns is derived from icon.png in CI (scripts/make_macos_app.sh).
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QApplication

ASSETS = Path(__file__).resolve().parents[1] / "src" / "ecoutemoi" / "assets"


def draw(size: int) -> QImage:
    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    s = size / 256.0

    # Rounded dark tile
    tile = QPainterPath()
    tile.addRoundedRect(8 * s, 8 * s, 240 * s, 240 * s, 40 * s, 40 * s)
    p.fillPath(tile, QColor("#1d2430"))

    # Chroma-green subtitle bar
    bar = QPainterPath()
    bar.addRoundedRect(28 * s, 164 * s, 200 * s, 56 * s, 12 * s, 12 * s)
    p.fillPath(bar, QColor("#00FF00"))

    # Subtitle text on the bar (white with black outline, like the app)
    font = QFont("Segoe UI", -1)
    font.setPixelSize(int(38 * s))
    font.setWeight(QFont.Weight.Bold)
    text_path = QPainterPath()
    text_path.addText(58 * s, 204 * s, font, "Écoute")
    pen = QPen(QColor("#000000"), 6 * s)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.strokePath(text_path, pen)
    p.fillPath(text_path, QColor("#FFFFFF"))

    # Microphone glyph
    mic = QPainterPath()
    mic.addRoundedRect(108 * s, 44 * s, 40 * s, 72 * s, 20 * s, 20 * s)
    p.fillPath(mic, QColor("#e8edf5"))
    pen = QPen(QColor("#e8edf5"), 10 * s)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    p.setPen(pen)
    p.drawArc(QRect(int(94 * s), int(70 * s), int(68 * s), int(64 * s)), 200 * 16, 140 * 16)
    p.drawLine(int(128 * s), int(134 * s), int(128 * s), int(150 * s))
    p.end()
    return img


def main() -> int:
    QApplication.instance() or QApplication(sys.argv)
    ASSETS.mkdir(parents=True, exist_ok=True)
    draw(256).save(str(ASSETS / "icon.png"))
    # Multi-size .ico for the Windows executable
    sizes = [16, 24, 32, 48, 64, 128, 256]
    images = [draw(n) for n in sizes]
    images[-1].save(str(ASSETS / "icon.ico"))  # Qt writes single-image ICO...
    try:  # ...upgrade to a proper multi-size ICO when Pillow is available
        from PIL import Image

        pngs = []
        for n, im in zip(sizes, images, strict=True):
            tmp = ASSETS / f"_icon_{n}.png"
            im.save(str(tmp))
            with Image.open(tmp) as pil_im:
                pngs.append(pil_im.copy())  # detach from the file handle (Windows)
        pngs[-1].save(ASSETS / "icon.ico", sizes=[(n, n) for n in sizes])
        for n in sizes:
            (ASSETS / f"_icon_{n}.png").unlink(missing_ok=True)
    except ImportError:
        pass
    print(f"Icons written to {ASSETS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
