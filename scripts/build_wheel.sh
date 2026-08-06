#!/usr/bin/env bash
# Build the pywhispercpp engine wheel with Vulkan for Linux x86_64.
# Used by the packaging CI job (ubuntu:22.04 container = glibc 2.35 floor); runnable
# locally for GPU testing.
#
# Requirements: build-essential cmake git libvulkan-dev + glslc (LunarG apt repo,
# package vulkan-sdk).
# Verification: logs mention Vulkan; CPU fallback loads; the wheel runs on a
# machine other than the builder (no -march=native).
set -euo pipefail
cd "$(dirname "$0")/.."

WHEEL_DIR="${WHEEL_DIR:-wheelhouse}"
# Pin to the locked version so the artifact matches uv.lock.
VERSION="$(uv run python -c 'import importlib.metadata as m; print(m.version("pywhispercpp"))')"

# GGML_NATIVE=OFF: portable artifact, not runner-tuned. AVX2+FMA+F16C+BMI2 =
# le variant « avx2 » canonique de ggml (Intel Haswell 2013+ / AMD Zen 2017+).
# Pas d'AVX-512 : absent de nombreux CPU grand public, l'artefact ne démarrerait pas.
# RPATH $ORIGIN : sans auditwheel, le RUNPATH du build pointerait vers un dossier
# temporaire détruit — les libs ggml/whisper vivent à côté de l'extension.
export CMAKE_ARGS="-DGGML_VULKAN=1 -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_INSTALL_RPATH=\$ORIGIN"

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
