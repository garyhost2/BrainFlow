#!/bin/bash -l

#SBATCH -J ledger_thm
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p gpu-A100
#SBATCH --account=a100_intern
#SBATCH --qos=a100_intern
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 06:00:00

set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/BrainFlow}"; cd "${REPO_DIR}"
mkdir -p logs runs data
[ -f "${REPO_DIR}/.env" ] && { set -a; source "${REPO_DIR}/.env"; set +a; }
source slurm/_env.sh

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv || true

python scripts/cluster_job.py --part theorem \
    --out runs/thm \
    --thm_dims 2,4,8,16,32,64 \
    --thm_batch 2048 \
    --thm_reps 6 \
    --thm_steps 2000 \
    --wandb 1 --wandb_project coupling-ledger

echo "theorem done. results in runs/thm/results"
