#!/bin/bash -l
# ─────────────────────────────────────────────────────────────────────────────
# BrainFlow MC-XL with HRF method on V100 32GB (2 GPUs).
# HRF gave the best *visual* results (mb_hrf_viz.png) despite lower PC.
# Scale up: bigger fMRI encoder + bigger UNet + more tokens.
#   Submit:    sbatch slurm/train_mc_xl_hrf_v100_32gb.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -J brainflow_mc_xl_hrf_32
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -p gpu-all
#SBATCH --gres=gpu:v100nv_32GB:2
#SBATCH -c 8
#SBATCH --mem=120G
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

export METHOD=hrf
export SUBJECTS=1,2,3,4,5,6,7,8
export EXPERIMENT_NAME=mc_xl_hrf_v100_32
export PERCEP_LOSS=lpips
export LAMBDA_PERCEP=0.10

# XL fMRI encoder
export ENC_HIDDEN=4096
export ENC_BLOCKS=8

# XL UNet (decoder) + more CLIP tokens for richer conditioning
export N_TOKENS=64
export UNET_BASE_CH=320

# Effective batch ≈ 96 (24 * 2 GPUs * 2 accum)
export BATCH_SIZE_PER_GPU=24
export GRAD_ACCUM=2
export DISABLE_COMPILE=1

torchrun \
    --standalone \
    --nproc_per_node="${NGPUS}" \
    -m scripts.train

echo "Done at $(date)"
