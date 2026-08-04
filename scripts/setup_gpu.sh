#!/usr/bin/env bash
# Optional GPU acceleration for StoMar (RTX 50-series / CUDA 12.8+ machines).
#
# Detects an NVIDIA GPU and swaps the CPU-only torch for a CUDA build.
# Run once per GPU machine:  bash scripts/setup_gpu.sh
# Safe to skip on CPU-only machines — StoMar works fine without it.

set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "No NVIDIA GPU detected — CPU-only torch stays (nothing to do)."
    exit 0
fi

echo "NVIDIA GPU detected:"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

VENV_PY=".venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
    echo "Error: .venv not found. Run: uv sync  (or pip install -e .)"
    exit 1
fi

echo "Swapping torch -> CUDA 12.8 build (cu128, supports RTX 50-series)..."
"$VENV_PY" -m pip install --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu128 \
    "torch>=2.0,<3"

echo
echo "Verifying..."
"$VENV_PY" -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"

if "$VENV_PY" -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "GPU acceleration enabled. The auto-pipeline will use CUDA automatically."
else
    echo "WARNING: CUDA still unavailable — check driver version (needs >= 570 for RTX 50-series)."
fi
