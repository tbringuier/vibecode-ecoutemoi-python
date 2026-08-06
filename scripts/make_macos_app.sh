#!/usr/bin/env bash
# Build EcouteMoi.app (arm64, Metal) then EcouteMoi-macos-arm64.zip.
# Does: icon.png -> .icns, pyinstaller (BUNDLE with NSMicrophoneUsageDescription),
# smoke run, ditto zip.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p build
ICONSET=build/ecoutemoi.iconset
rm -rf "$ICONSET"
mkdir -p "$ICONSET"
for n in 16 32 64 128 256 512; do
  sips -z "$n" "$n" src/ecoutemoi/assets/icon.png --out "$ICONSET/icon_${n}x${n}.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o build/ecoutemoi.icns

# --no-sync : un `uv run` synchronisant restaurerait la wheel CPU de PyPI
# par-dessus la wheel moteur Metal installée juste avant.
uv run --no-sync pyinstaller packaging/ecoutemoi.spec --noconfirm

# Smoke: the inner binary must run and exit 0 (works without any Python installed)
./dist/EcouteMoi.app/Contents/MacOS/EcouteMoi --version

ditto -c -k --keepParent dist/EcouteMoi.app EcouteMoi-macos-arm64.zip
ls -lh EcouteMoi-macos-arm64.zip
