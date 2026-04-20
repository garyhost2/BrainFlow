#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Panther — one-time environment setup
# Creates a conda env `brainflow` with PyTorch + all deps.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ENV_NAME="${ENV_NAME:-brainflow}"
PY_VERSION="${PY_VERSION:-3.10}"

# Load conda (assumes Anaconda installed in $HOME/anaconda3 per Panther guide)
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Anaconda not found at $HOME/anaconda3 — install per Panther guide first."
    exit 1
fi

# Load CUDA module (Panther provides cuda10.1; for newer PyTorch we use the bundled CUDA runtime via pip wheels)
module load cuda10.1/toolkit || true

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Env '${ENV_NAME}' already exists — activating."
else
    echo "Creating conda env '${ENV_NAME}' (Python ${PY_VERSION}) …"
    conda create -y -n "${ENV_NAME}" python="${PY_VERSION}"
fi

conda activate "${ENV_NAME}"

# Install PyTorch with CUDA 12.1 wheels (works on A100/V100/T4 nodes)
pip install --upgrade pip
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

# Project deps
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pip install -r "${REPO_DIR}/requirements.txt"

echo
echo "✓ Environment '${ENV_NAME}' ready. To use:"
echo "    conda activate ${ENV_NAME}"
