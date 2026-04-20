#!/bin/bash -l
# Inference on a trained checkpoint — single GPU.
#SBATCH -J brainflow_v5_infer
#SBATCH -o logs/infer_%j.out
#SBATCH -e logs/infer_%j.err
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 2:00:00

set -euo pipefail
module load slurm
module load cuda10.1/toolkit || true

ENV_NAME="${ENV_NAME:-brainflow}"
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

REPO_DIR="${REPO_DIR:-$HOME/BrainFlow}"
cd "${REPO_DIR}"
[ -f .env ] && { set -a; source .env; set +a; }

CKPT="${CKPT:-outputs/best_pc_v5.pt}"
python -m scripts.inference --ckpt "${CKPT}"
