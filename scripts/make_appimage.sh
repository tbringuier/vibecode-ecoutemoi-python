#!/usr/bin/env bash
# Build EcouteMoi-linux-x86_64.AppImage from the PyInstaller onedir.
# Prereq: uv run pyinstaller packaging/ecoutemoi.spec  (dist/EcouteMoi exists)
# Runs inside the ubuntu:22.04 CI container: appimagetool is executed with
# --appimage-extract-and-run (no FUSE required).
set -euo pipefail
cd "$(dirname "$0")/.."

DIST=dist/EcouteMoi
[ -d "$DIST" ] || { echo "dist/EcouteMoi introuvable — lancer pyinstaller d'abord" >&2; exit 1; }

APPDIR=build/AppDir
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/icons/hicolor/256x256/apps" "$APPDIR/usr/share/applications"

cp -r "$DIST"/. "$APPDIR/usr/bin/"
cp src/ecoutemoi/assets/ecoutemoi.desktop "$APPDIR/"
cp src/ecoutemoi/assets/ecoutemoi.desktop "$APPDIR/usr/share/applications/"
cp src/ecoutemoi/assets/icon.png "$APPDIR/ecoutemoi.png"
cp src/ecoutemoi/assets/icon.png "$APPDIR/usr/share/icons/hicolor/256x256/apps/ecoutemoi.png"

cat > "$APPDIR/AppRun" <<'EOF'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/bin/EcouteMoi" "$@"
EOF
chmod +x "$APPDIR/AppRun"

TOOL=build/appimagetool
if [ ! -f "$TOOL" ]; then
  curl -fsSL -o "$TOOL" \
    https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage
  chmod +x "$TOOL"
fi

ARCH=x86_64 "$TOOL" --appimage-extract-and-run "$APPDIR" EcouteMoi-linux-x86_64.AppImage
ls -lh EcouteMoi-linux-x86_64.AppImage
