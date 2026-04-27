#!/bin/bash -l
# ─────────────────────────────────────────────────────────────────────────────
# Stage-1 pretrain: BrainEncoder + CLIPPrior only.
# Runs on a single V100 32GB (much faster than full training; no VAE/UNet/DiT).
#   Submit:    sbatch slurm/pretrain_prior.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -J brainflow_stage1_prior
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -p gpu-all
#SBATCH --gres=gpu:v100nv_32GB:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 1-00:00:00

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

# Same encoder/prior dims as XL DiT so weights load cleanly into the full model
export METHOD=dit
export SUBJECTS=1,2,3,4,5,6,7,8
export EXPERIMENT_NAME=stage1_prior

export ENC_HIDDEN=2048
export ENC_BLOCKS=6
export PRIOR_DIM=1024
export PRIOR_BLOCKS=8

# Loss weights
export LAMBDA_ALIGN=1.0
export LAMBDA_PRIOR=1.0

# No VAE/UNet → can use a much larger batch
export BATCH_SIZE_PER_GPU=128
export GRAD_ACCUM=1
export NUM_EPOCHS=80         # prior typically converges fast

# Slightly higher LR than full training (smaller model, no grad-clip pressure from VAE loss)
# export LR=2e-4

torchrun \
    --standalone \
    --nproc_per_node="${NGPUS}" \
    -m scripts.pretrain_prior

echo "Done at $(date)"
