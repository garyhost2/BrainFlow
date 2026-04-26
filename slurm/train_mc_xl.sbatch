#!/bin/bash -l
# ─────────────────────────────────────────────────────────────────────────────
# BrainFlow MC-XL — Scaled DiT + bigger encoder + bigger CLIP prior.
# Targets: bigger lever toward MindEye-v2 quality.
# 2× V100 16GB; smaller batch since model is larger.
#   Submit:    sbatch slurm/train_mc_xl.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -J brainflow_mc_xl
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -p gpu-all
#SBATCH --gres=gpu:v100_16GB:2
#SBATCH -c 8
#SBATCH --mem=90G
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

# ── MC-XL experiment knobs ──
export METHOD=dit
export SUBJECTS=1,2,3,4,5,6,7,8
export EXPERIMENT_NAME=mc_xl
export PERCEP_LOSS=lpips
export LAMBDA_PERCEP=0.10

# Bigger encoder (≈ 4× wider, 2× deeper)
export ENC_HIDDEN=4096
export ENC_BLOCKS=8

# Bigger CLIP prior (d 512→1024, blocks 3→8) + stronger supervision
export PRIOR_DIM=1024
export PRIOR_BLOCKS=8
export LAMBDA_PRIOR=0.4
export PRIOR_ODE_STEPS=20

# Compensate memory: smaller per-GPU batch, more grad_accum.
# Effective batch = 16 * 2 GPUs * 3 = 96
export BATCH_SIZE_PER_GPU=16
export GRAD_ACCUM=3
# export NUM_EPOCHS=2                     # uncomment for smoke test

torchrun \
    --standalone \
    --nproc_per_node="${NGPUS}" \
    -m scripts.train

echo "Done at $(date)"
