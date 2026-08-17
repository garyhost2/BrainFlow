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

# Try to load any available CUDA module (PyTorch wheels carry their own runtime,
# so this is best-effort).
for m in cuda12.1/toolkit/12.1.1 cuda11.8/toolkit/11.8.0 cuda12.4/toolkit/12.4.1 cuda12.6/toolkit/12.6.2; do
    if module load "$m" 2>/dev/null; then
        echo "Loaded module: $m"; break
    fi
done

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

# The decoder half of the environment. requirements.txt alone leaves out
# transformers, kornia and pytorch-lightning, so an env built from this script
# could train but died at the first --decode-eval. setup_step1b.sh owns the
# pinned set (and is idempotent: it skips the MindEyeV2 clone and the 5 GB
# unclip6 checkpoint if they are already present).
bash "${REPO_DIR}/scripts/setup_step1b.sh"

python - <<'PY'
import importlib, sys
need = ["torch", "torchvision", "diffusers", "transformers", "open_clip",
        "kornia", "pytorch_lightning", "omegaconf", "h5py", "webdataset",
        "scipy", "pytest"]
missing = []
for m in need:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
if missing:
    print("INCOMPLETE ENV, missing:", ", ".join(missing)); sys.exit(1)
print("env import check: all", len(need), "modules present")
PY

echo
echo "✓ Environment '${ENV_NAME}' ready. To use:"
echo "    conda activate ${ENV_NAME}"
