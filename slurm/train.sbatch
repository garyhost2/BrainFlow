#!/bin/bash -l
# ─────────────────────────────────────────────────────────────────────────────
# BrainFlow v5 — multi-subject training on Panther
# Submit:    sbatch slurm/train.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -J brainflow_v5
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err
#SBATCH -p gpu-all
#SBATCH --gres=gpu:v100_16GB:4        # 4 V100s (max 8 per Panther policy)
#SBATCH -c 16                          # 4 CPUs per GPU
#SBATCH --mem=120G
#SBATCH -t 7-00:00:00                  # 7 days

set -euo pipefail

# ── Modules ──
module load slurm
module load cuda10.1/toolkit || true

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

# ── Secrets (rank-0 reads .env via python-dotenv) ──
if [ -f "${REPO_DIR}/.env" ]; then
    set -a; source "${REPO_DIR}/.env"; set +a
fi

# ── Distributed launch (single-node multi-GPU) ──
NGPUS=$(nvidia-smi -L | wc -l)
echo "Node: $(hostname) | GPUs visible: ${NGPUS}"
nvidia-smi || true

export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

torchrun \
    --standalone \
    --nproc_per_node="${NGPUS}" \
    -m scripts.train

echo "Done at $(date)"
