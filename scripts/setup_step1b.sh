#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Step-1b one-time setup: vendored sgm (SDXL-unCLIP decoder) + checkpoint.
# Run on the A100 node from the repo root:  bash scripts/setup_step1b.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

THIRD_PARTY="third_party"
MINDEYE_DIR="${THIRD_PARTY}/MindEyeV2"
CKPT="${THIRD_PARTY}/unclip6_epoch0_step110000.ckpt"

mkdir -p "${THIRD_PARTY}"

# 1) Vendored sgm + unclip6.yaml come from the MindEyeV2 repo (src/generative_models).
if [ ! -d "${MINDEYE_DIR}" ]; then
  echo "▶ Cloning MedARC-AI/MindEyeV2 (for vendored sgm + unclip6.yaml)…"
  git clone --depth 1 https://github.com/MedARC-AI/MindEyeV2 "${MINDEYE_DIR}"
else
  echo "✓ ${MINDEYE_DIR} already present"
fi

if [ ! -d "${MINDEYE_DIR}/src/generative_models/sgm" ]; then
  echo "✗ Expected ${MINDEYE_DIR}/src/generative_models/sgm — repo layout changed?" >&2
  exit 1
fi

# 2) Dependencies the vendored sgm + bigG embedder need (pinned to MindEye2's set).
echo "▶ Installing Step-1b deps (sgm / bigG / decoder)…"
pip install --upgrade pip
pip install \
  "open_clip_torch==2.24.0" "kornia==0.7.1" "omegaconf==2.3.0" \
  "pytorch-lightning==2.0.1" "transformers>=4.37" "diffusers>=0.23" \
  "torchmetrics>=1.3" einops ftfy regex "huggingface_hub>=0.20" safetensors
# xformers is optional (sgm uses softmax-xformers attn); install if your CUDA matches:
#   pip install xformers==0.0.22.post7

# 3) Download the SDXL-unCLIP decoder checkpoint (~5 GB) from the NSD dataset repo.
if [ ! -f "${CKPT}" ]; then
  echo "▶ Downloading unclip6 decoder checkpoint…"
  python scripts/download_unclip6.py --out "${THIRD_PARTY}"
else
  echo "✓ ${CKPT} already downloaded"
fi

echo ""
echo "✅ Step-1b setup complete."
echo "   sgm src : ${MINDEYE_DIR}/src"
echo "   ckpt    : ${CKPT}"
echo ""
echo "Next:"
echo "  python -m scripts.train_step1b --data-dir ./mindeyev2_cache --subjects 1 \\"
echo "      --mindeye-src ${MINDEYE_DIR}/src --ckpt-path ${CKPT} \\"
echo "      --epochs 150 --batch-size 48 --out outputs/step1b"
