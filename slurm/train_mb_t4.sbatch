#!/bin/bash -l
# ─────────────────────────────────────────────────────────────────────────────
# BrainFlow MB on 2× T4 — Brain-manifold flow with HRF-aware velocity.
#   Submit:    sbatch slurm/train_mb_t4.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -J brainflow_mb_t4
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -p gpu-all
#SBATCH --gres=gpu:T4_16G              # T4 nodes appear to be single-GPU on this cluster
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 3-00:00:00

set -euo pipefail

# ── Modules ──
module load slurm/slurm/21.08.8 2>/dev/null || module load slurm 2>/dev/null || true
for m in cuda12.1/toolkit/12.1.1 cuda11.8/toolkit/11.8.0 cuda12.4/toolkit/12.4.1 cuda12.6/toolkit/12.6.2; do
    if module load "$m" 2>/dev/null; then
        echo "Loaded module: $m"; break
    fi
done

# ── Conda env ──
ENV_NAME="${ENV_NAME:-brainflow}"
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate "${ENV_NAME}"

# ── Repo ──
REPO_DIR="${REPO_DIR:-$HOME/BrainFlow}"
cd "${REPO_DIR}"
mkdir -p logs outputs

# ── Secrets ──
if [ -f "${REPO_DIR}/.env" ]; then
    set -a; source "${REPO_DIR}/.env"; set +a
fi

# ── Launch (single GPU) ──
NGPUS=$(nvidia-smi -L | wc -l)
echo "Node: $(hostname) | GPUs visible: ${NGPUS}"
nvidia-smi || true

export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export DDP_TIMEOUT=1800

# ── MB experiment knobs (T4 16GB, single-GPU) ──
export METHOD=hrf
export SUBJECTS=1,2,3,4,5,6,7,8
export EXPERIMENT_NAME=mb_hrf_full
export PERCEP_LOSS=lpips
export LAMBDA_PERCEP=0.10
# 1× T4 16GB. Per-GPU batch = 8; grad_accum = 12
# Effective batch = 8 * 1 GPU * 12 = 96 (matches V100 mb_hrf_full default)
export BATCH_SIZE_PER_GPU=8
export GRAD_ACCUM=12

torchrun \
    --standalone \
    --nproc_per_node="${NGPUS}" \
    -m scripts.train

echo "Done at $(date)"
