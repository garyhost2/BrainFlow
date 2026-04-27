#!/bin/bash -l
# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 Stage 2A — FlowCLIPDiT training (BrainEncoder frozen).
# Single V100 32GB; effective batch = 32 * 1 * 2 = 64.
#   Submit:  sbatch slurm/train_phase2_2a_v100_32gb.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -J phase2_2a_v100_32
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -p gpu-all
#SBATCH --gres=gpu:v100nv_32GB:1
#SBATCH -c 8
#SBATCH --mem=160G                    # tensors(56G)+latents(12G)+patches(38G)+workers+overhead
#SBATCH -t 0-12:00:00

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

# ── Phase 2 Stage 2A config ──────────────────────────────────────────────────
# BRAINFLOW_CONFIG selects the phase2_stage2a.yaml config (overrides root config.yaml)
export BRAINFLOW_CONFIG="${REPO_DIR}/configs/phase2_stage2a.yaml"

# Warm-start BrainEncoder from the best v5 multi-subject checkpoint
export INIT_FROM="${REPO_DIR}/outputs/mc_xl_v100_32/best_pc_v5.pt"

torchrun \
    --standalone \
    --nproc_per_node="${NGPUS}" \
    -m scripts.train_phase2

echo "Done at $(date)"
