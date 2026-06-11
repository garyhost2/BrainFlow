#!/bin/bash -l
# ─────────────────────────────────────────────────────────────────────────────
# Step 1b — fMRI -> bigG token grid -> SDXL-unCLIP (MindEye2 decoder).
# The MindEye2-competitive path.  Needs a big GPU (A100 80GB recommended) and the
# one-time setup (bash scripts/setup_step1b.sh) already run on the LOGIN node.
# Use a DEDICATED conda env `me2` (MindEye2's pinned deps) — see docs/STEP1B.md.
#   Submit:  sbatch slurm/train_step1b_a100.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -J step1b
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -p gpu-all
#SBATCH --gres=gpu:a100_80GB:1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 2-00:00:00

set -euo pipefail

module load slurm 2>/dev/null || true
for m in cuda12.1/toolkit/12.1.1 cuda11.8/toolkit/11.8.0; do
    module load "$m" 2>/dev/null && { echo "Loaded module: $m"; break; }
done

# Activate the dedicated Step-1b env. venv by default (login-node conda solver
# OOMs on Panther); falls back to a conda env named $ENV_NAME if no venv exists.
VENV="${VENV:-$HOME/envs/me2}"
if [ -f "${VENV}/bin/activate" ]; then
    source "${VENV}/bin/activate"
    echo "Activated venv: ${VENV}"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"; conda activate "${ENV_NAME:-me2}"
fi

REPO_DIR="${REPO_DIR:-$HOME/BrainFlow}"
cd "${REPO_DIR}"
mkdir -p logs outputs
[ -f "${REPO_DIR}/.env" ] && { set -a; source "${REPO_DIR}/.env"; set +a; }

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
echo "Node: $(hostname)"; nvidia-smi || true

MINDEYE_SRC="${MINDEYE_SRC:-third_party/MindEyeV2/src}"
UNCLIP_CKPT="${UNCLIP_CKPT:-third_party/unclip6_epoch0_step110000.ckpt}"

# First launch also builds the bigG target cache (~8 GB/subject) via the embedder.
python -m scripts.train_step1b \
    --data-dir ./mindeyev2_cache \
    --subjects 1 \
    --mindeye-src "${MINDEYE_SRC}" \
    --ckpt-path "${UNCLIP_CKPT}" \
    --epochs 150 --batch-size 48 --lr 3e-4 \
    --eval-freq 10 --decode-eval --decode-n 16 \
    --out outputs/step1b

echo "Done at $(date)"
