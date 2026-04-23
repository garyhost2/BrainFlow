#!/bin/bash -l
# ─────────────────────────────────────────────────────────────────────────────
# BrainFlow MB — Brain-manifold flow with HRF-aware velocity
# Single subject (subj 1), 150 epochs. Goal: beat MindEye2 SSIM (0.431).
#   Submit:    sbatch slurm/train_mb.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -J brainflow_mb
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -p gpu-all
#SBATCH --gres=gpu:v100nv_32GB:2
#SBATCH -c 16
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

# ── Distributed launch ──
NGPUS=$(nvidia-smi -L | wc -l)
echo "Node: $(hostname) | GPUs visible: ${NGPUS}"
nvidia-smi || true

export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export DDP_TIMEOUT=1800

# ── MB experiment knobs (overrides via brainflow/config_overrides.py) ──
export METHOD=hrf
export SUBJECTS=1                       # single subject for focused MB run
export EXPERIMENT_NAME=mb_hrf_subj1
export PERCEP_LOSS=lpips                # SSIM-friendly perceptual supervision
export LAMBDA_PERCEP=0.10
# Optional: shorter run for smoke test → uncomment next line
# export NUM_EPOCHS=2

torchrun \
    --standalone \
    --nproc_per_node="${NGPUS}" \
    -m scripts.train

echo "Done at $(date)"
