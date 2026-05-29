#!/bin/bash -l
# ─────────────────────────────────────────────────────────────────────────────
# BrainFlow V8 — MindEye-v2 quality target: CLIP prior head + pixel L1 loss.
# Two V100 32GB GPUs. Effective batch = 16 * 2 * 3 = 96.
#
# V8 preset (config_overrides.py):
#   n_tokens     = 32         (2× baseline — richer representation)
#   use_clip_prior = True     (CLIPPriorHead: brain → predicted CLIP CLS)
#   lambda_prior = 0.3        (strong prior supervision)
#   lambda_pixel = 0.3        (pixel-space L1 for t > 0.85 only)
#   lambda_align = 0.2        (balanced InfoNCE)
#   method       = "baseline" (Gaussian source — CLIP prior handles alignment)
#
# Target: PC ≥ 0.32, SSIM ≥ 0.42, CLIP_Sim ≥ 0.94 at 60 epochs (subj-1).
#
# Requires corrected CLIP+latent caches — run rebuild_cache_v2.sbatch first.
#
#   Submit:  sbatch slurm/train_v8_v100_32gb.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -J brainflow_v8_v100_32
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -p gpu-all
#SBATCH --gres=gpu:v100nv_32GB:2
#SBATCH -c 16
#SBATCH --mem=180G
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

# ── V8 experiment settings ────────────────────────────────────────────────────
export USE_V8=1                 # activates the V8 preset in config_overrides.py
export EXPERIMENT_NAME=v8_v100_32

# Encoder: modest size to fit 2× V100 32GB with 32 tokens + CLIP prior overhead
export ENC_HIDDEN=1024
export ENC_BLOCKS=4

# CLIP prior head sizing (CLIPPriorHead via use_clip_prior=True)
export PRIOR_DIM=512
export PRIOR_BLOCKS=3
export PRIOR_ODE_STEPS=10

# Effective batch = 16 * 2 GPUs * 3 = 96
export BATCH_SIZE_PER_GPU=16
export GRAD_ACCUM=3

export SUBJECTS=1,2,3,4,5,6,7,8

torchrun \
    --standalone \
    --nproc_per_node="${NGPUS}" \
    -m scripts.train

echo "Done at $(date)"
