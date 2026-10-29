#!/usr/bin/env bash
# Creates the Python 3.13 environment used by the persistent WD14 tagger worker.
#
# onnxruntime-gpu needs CUDA 13 + cuDNN 9 and has no cp314 wheels yet, so this
# builds a Python 3.13 env (via conda) instead of using the system interpreter.
# The tagger finds the CUDA libraries inside the env on its own; no CUDA toolkit
# installation or LD_LIBRARY_PATH is required.
#
# Usage: scripts/setup-gpu-tagger.sh [env-dir]
#   env-dir defaults to ~/venvs/allusion-gpu (override with DEEPDANBOORU_CUDA_PYTHON
#   in the app, or by pointing at the resulting bin/python).
set -euo pipefail

ENV_DIR="${1:-${HOME}/venvs/allusion-gpu}"
PY_VERSION="3.13"

say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }

if [ -x "${ENV_DIR}/bin/python" ] && [ -f "${ENV_DIR}/.allusion-gpu-ready" ]; then
  say "Environment already set up at ${ENV_DIR}"
  exit 0
fi

find_bootstrap() {
  if command -v conda >/dev/null 2>&1; then
    echo "conda"
    return
  fi
  if [ -x /tmp/allusion-micromamba/bin/micromamba ]; then
    echo "/tmp/allusion-micromamba/bin/micromamba"
    return
  fi
  ARCH="$(uname -m)"
  case "${ARCH}" in
    x86_64) MM_ARCH="linux-64" ;;
    aarch64|arm64) MM_ARCH="osx-arm64" ;;
    *) echo "Unsupported architecture: ${ARCH}" >&2; exit 1 ;;
  esac
  say "Downloading micromamba (no conda found)"
  mkdir -p /tmp/allusion-micromamba
  curl -Ls "https://micro.mamba.pm/api/micromamba/${MM_ARCH}/latest" \
    | tar -xvj -C /tmp/allusion-micromamba bin/micromamba >/dev/null
  echo "/tmp/allusion-micromamba/bin/micromamba"
}

BOOTSTRAP="$(find_bootstrap)"

say "Creating Python ${PY_VERSION} environment at ${ENV_DIR}"
if [ "${BOOTSTRAP}" = "conda" ]; then
  conda create -y --override-channels -c conda-forge -p "${ENV_DIR}" "python=${PY_VERSION}" >/dev/null
else
  "${BOOTSTRAP}" create -y -p "${ENV_DIR}" -c conda-forge "python=${PY_VERSION}" >/dev/null
fi

say 'Installing onnxruntime-gpu (CUDA 13) with its runtime libraries'
"${ENV_DIR}/bin/python" -m pip install --upgrade pip >/dev/null
"${ENV_DIR}/bin/python" -m pip install \
  onnxruntime-gpu pandas pillow \
  "nvidia-cublas==13.*" "nvidia-cudnn-cu13==9.*" \
  "nvidia-cuda-runtime==13.*" "nvidia-cuda-nvrtc==13.*" \
  "nvidia-cufft==12.*" "nvidia-curand==10.*"

say 'Checking imports and execution providers'
"${ENV_DIR}/bin/python" - <<'PY'
import numpy, onnxruntime as ort, pandas
from PIL import Image
providers = ort.get_available_providers()
print('available providers:', providers)
if 'CUDAExecutionProvider' not in providers:
    raise SystemExit('CUDAExecutionProvider is missing from this onnxruntime build')
PY

touch "${ENV_DIR}/.allusion-gpu-ready"

say "Done. The tagger will use ${ENV_DIR}/bin/python automatically."
say 'Set DEEPDANBOORU_CUDA_PYTHON=/path/to/python to use a different environment.'
