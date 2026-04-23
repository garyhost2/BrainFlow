#!/bin/bash -l
# ─────────────────────────────────────────────────────────────────────────────
# BrainFlow MB on 2× V100 32GB — Brain-manifold flow with HRF-aware velocity.
#   Submit:    sbatch slurm/train_mb_v100.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -J brainflow_mb_v100
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -p gpu-all
#SBATCH --gres=gpu:v100nv_32GB:2       # 2× V100 32GB SXM (NVLink)
#SBATCH -c 16                          # 8 CPUs per GPU
#SBATCH --mem=180G
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

# ── Distributed launch (single-node, 2× V100) ──
NGPUS=$(nvidia-smi -L | wc -l)
echo "Node: $(hostname) | GPUs visible: ${NGPUS}"
nvidia-smi || true

export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export DDP_TIMEOUT=1800

# ── MB experiment knobs (V100 32GB, 2-GPU) ──
export METHOD=hrf
export SUBJECTS=1,2,3,4,5,6,7,8
export EXPERIMENT_NAME=mb_hrf_full
export PERCEP_LOSS=lpips
export LAMBDA_PERCEP=0.10
# 2× V100 32GB. Use the new repo defaults (batch=96, grad_accum=1) per GPU.
# Effective batch = 96 * 2 = 192. Override below if memory tight.
# export BATCH_SIZE_PER_GPU=96
# export GRAD_ACCUM=1

torchrun \
    --standalone \
    --nproc_per_node="${NGPUS}" \
    -m scripts.train

echo "Done at $(date)"
