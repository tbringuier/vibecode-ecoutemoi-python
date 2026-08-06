#!/usr/bin/env bash
# Build the pywhispercpp engine wheel with Metal for macOS arm64.
# Metal est activé par défaut par ggml sur Apple Silicon : rien à forcer côté
# backend, seul le RPATH doit être corrigé.
#
# Ce script existe SÉPARÉMENT du YAML de CI pour que le cache de la wheel puisse
# être clé sur son contenu (hashFiles) : les runners macOS coûtent 10x, une
# recompilation inutile est la dépense la plus chère du pipeline.
set -euo pipefail
cd "$(dirname "$0")/.."

WHEEL_DIR="${WHEEL_DIR:-wheelhouse}"
# Pin to the locked version so the artifact matches uv.lock.
VERSION="$(uv run python -c 'import importlib.metadata as m; print(m.version("pywhispercpp"))')"

# RPATH @loader_path : sans delocate, le RUNPATH du build pointerait vers un
# dossier temporaire détruit — les libs ggml/whisper vivent à côté de l'extension.
export CMAKE_ARGS="-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_INSTALL_RPATH=@loader_path"

echo "Building pywhispercpp==${VERSION} from source with: ${CMAKE_ARGS}"
# --no-cache-dir : la clé de cache de pip ignore CMAKE_ARGS et resservirait une
# wheel compilée avec d'anciens flags.
uv run --with pip python -m pip wheel "pywhispercpp==${VERSION}" \
  --no-binary pywhispercpp --no-deps --no-cache-dir -w "${WHEEL_DIR}"

ls -l "${WHEEL_DIR}"
echo "Install into the project venv with:"
echo "  uv pip install ${WHEEL_DIR}/pywhispercpp-*.whl --force-reinstall"
echo "Then ALWAYS run with --no-sync (uv run would restore the PyPI CPU wheel):"
echo "  uv run --no-sync ecoutemoi"
