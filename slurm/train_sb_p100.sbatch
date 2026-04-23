#!/bin/bash -l
# ─────────────────────────────────────────────────────────────────────────────
# BrainFlow SB on P100 — Schrödinger-bridge flow, 16GB GPU variant.
#   Submit:    sbatch slurm/train_sb_p100.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -J brainflow_sb_p100
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -p gpu-all
#SBATCH --gres=gpu:P100_16GB:1
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

# ── Distributed launch ──
NGPUS=$(nvidia-smi -L | wc -l)
echo "Node: $(hostname) | GPUs visible: ${NGPUS}"
nvidia-smi || true

export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export DDP_TIMEOUT=1800

# ── SB experiment knobs (P100-sized) ──
export METHOD=sb
export SUBJECTS=1,2,3,4,5,6,7,8
export EXPERIMENT_NAME=sb_bridge_full_p100
export PERCEP_LOSS=lpips
export LAMBDA_PERCEP=0.10
# P100 is 16GB (half of V100 32GB). Halve per-GPU batch, double grad_accum
# to keep the effective batch identical to the V100 run.
export BATCH_SIZE_PER_GPU=8
export GRAD_ACCUM=8

torchrun \
    --standalone \
    --nproc_per_node="${NGPUS}" \
    -m scripts.train

echo "Done at $(date)"
