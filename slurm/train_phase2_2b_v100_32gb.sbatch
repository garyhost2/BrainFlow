#!/bin/bash -l
# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 Stage 2B — Joint fine-tune: BrainEncoder + FlowCLIPDiT + FlowUNet.
# Two V100 32GB GPUs; effective batch = 16 * 2 * 4 = 128.
#   Submit:  sbatch slurm/train_phase2_2b_v100_32gb.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -J phase2_2b_v100_32
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -p gpu-all
#SBATCH --gres=gpu:v100nv_32GB:2
#SBATCH -c 16
#SBATCH --mem=180G
#SBATCH -t 1-02:00:00

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

# ── Phase 2 Stage 2B config ──────────────────────────────────────────────────
# BRAINFLOW_CONFIG selects the phase2_stage2b.yaml config
export BRAINFLOW_CONFIG="${REPO_DIR}/configs/phase2_stage2b.yaml"

# Warm-start all weights from the best Stage 2A checkpoint
export INIT_FROM="${REPO_DIR}/outputs/phase2_stage2a/best_clip_cos.pt"

torchrun \
    --standalone \
    --nproc_per_node="${NGPUS}" \
    -m scripts.train_phase2

echo "Done at $(date)"
