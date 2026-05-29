#!/bin/bash -l
# ─────────────────────────────────────────────────────────────────────────────
# BrainFlow V6 — 64-token encoder + LPIPS perceptual loss (USE_V6=1).
# Single V100 32GB. Effective batch = 32 * 1 * 3 = 96.
#
# V6 preset (config_overrides.py):
#   n_tokens    = 64    (vs 16 baseline — richer brain representation)
#   lambda_percep = 0.15  LPIPS perceptual loss
#   percep_loss = "lpips"
#   mixup_alpha = 0.1
#
# Requires corrected CLIP+latent caches — run rebuild_cache_v2.sbatch first
# if you haven't trained since PR #6 (CLIP normalisation fix).
#
#   Submit:  sbatch slurm/train_v6_v100_32gb.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -J brainflow_v6_v100_32
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -p gpu-all
#SBATCH --gres=gpu:v100nv_32GB:1
#SBATCH -c 8
#SBATCH --mem=160G
#SBATCH -t 3-00:00:00

set -euo pipefail

module load slurm/slurm/21.08.8 2>/dev/null || module load slurm 2>/dev/null || true
for m in cuda12.1/toolkit/12.1.1 cuda11.8/toolkit/11.8.0 cuda12.4/toolkit/12.4.1 cuda12.6/toolkit/12.6.2; do
    if module load "$m" 2>/dev/null; then
        echo "Loaded module: $m"; break
    fi
done

ENV_NAME="${ENV_NAME:-brainflow}"
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate "${ENV_NAME}"

REPO_DIR="${REPO_DIR:-$HOME/BrainFlow}"
cd "${REPO_DIR}"
mkdir -p logs outputs

if [ -f "${REPO_DIR}/.env" ]; then
    set -a; source "${REPO_DIR}/.env"; set +a
fi

NGPUS=$(nvidia-smi -L | wc -l)
echo "Node: $(hostname) | GPUs visible: ${NGPUS}"
nvidia-smi || true

export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export DDP_TIMEOUT=1800
export DISABLE_COMPILE=1        # V100 compute cap 7.0 — no torch.compile

# ── V6 experiment settings ────────────────────────────────────────────────────
export USE_V6=1                 # activates the V6 preset in config_overrides.py
export EXPERIMENT_NAME=v6_v100_32

# Throughput settings for single V100 32GB with 64-token encoder
# 64 tokens × 4 enc blocks × 1024 dim → larger encoder footprint
# batch=32, accum=3 → effective batch=96 (same as baseline)
export BATCH_SIZE_PER_GPU=32
export GRAD_ACCUM=3

export SUBJECTS=1,2,3,4,5,6,7,8

torchrun \
    --standalone \
    --nproc_per_node="${NGPUS}" \
    -m scripts.train

echo "Done at $(date)"
