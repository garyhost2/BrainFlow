#!/bin/bash -l
# ─────────────────────────────────────────────────────────────────────────────
# BrainFlow MC-XL V9 on V100 32GB (2 GPUs) — SOTA Push
#
# V9 applies all audit-identified improvements to the baseline MC experiment
# in a single training run. The baseline MC run plateaued at EP45 with
# PC=0.090, SSIM=0.334, CLIP=0.740 — far below SOTA (~0.35, ~0.42, ~0.94).
#
# Key changes vs baseline:
#   • CLIPPriorHead enabled (use_clip_prior=True) — semantic alignment
#   • InfoNCE raised to λ=0.8, CFM lowered to λ=0.5 — fixes 9:1 gradient imbalance
#   • n_tokens=64 — 4× richer brain representation
#   • enc_blocks=6 — deeper encoder
#   • LPIPS perceptual loss (λ=0.15) — pixel-quality gradient signal
#   • cfg_scale=6.0 — stronger semantic guidance at inference
#   • eval_solver=midpoint, eval_ode_steps=25 — accurate benchmarking
#
# Expected training milestones:
#   Early (EP15–30): PC ~0.10–0.17, SSIM ~0.34–0.40, CLIP ~0.79–0.87
#   Full convergence (EP60+): PC ~0.18–0.22, SSIM ~0.38–0.42, CLIP ~0.85–0.90
#
#   Submit:    sbatch slurm/train_mc_xl_v9_v100_32gb.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -J brainflow_mc_xl_v9
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

# V9 experiment — UNet backbone + CLIPPriorHead (not full DiT)
# All improvements are applied together via USE_V9=1; see header for details.
export METHOD=baseline
export SUBJECTS=1,2,3,4,5,6,7,8
export EXPERIMENT_NAME=v9
export USE_V9=1
export PERCEP_LOSS=lpips
export LAMBDA_PERCEP=0.15

# Encoder: keep compatible with 32GB VRAM given n_tokens=64 + LPIPS overhead
export ENC_HIDDEN=1024
export ENC_BLOCKS=6

# Effective batch = 48 * 2 GPUs * 2 = 192
# Reduced slightly vs baseline to account for 64 tokens + LPIPS memory overhead
export BATCH_SIZE_PER_GPU=48
export GRAD_ACCUM=2
export DISABLE_COMPILE=1

torchrun \
    --standalone \
    --nproc_per_node="${NGPUS}" \
    -m scripts.train

echo "Done at $(date)"
