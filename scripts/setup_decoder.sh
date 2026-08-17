#!/usr/bin/env bash
set -euo pipefail

THIRD_PARTY="third_party"
MINDEYE_DIR="${THIRD_PARTY}/MindEyeV2"
CKPT="${THIRD_PARTY}/unclip6_epoch0_step110000.ckpt"

mkdir -p "${THIRD_PARTY}"

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

echo "▶ Installing Step-1b deps (sgm / bigG / decoder)…"
pip install --upgrade pip
pip install \
  "numpy<2" "setuptools<81" scipy \
  "open_clip_torch==2.24.0" "kornia==0.7.1" "omegaconf==2.3.0" \
  "pytorch-lightning==2.0.1" "transformers==4.40.2" "diffusers==0.27.2" \
  "torchmetrics>=1.3" einops ftfy regex "huggingface_hub==0.23.4" safetensors

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
