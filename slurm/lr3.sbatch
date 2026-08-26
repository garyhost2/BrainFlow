#!/bin/bash
#SBATCH -J lr3
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p gpu-A100
#SBATCH --account=a100_intern
#SBATCH --qos=a100_intern
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 10:00:00

set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/BrainFlow}"; cd "${REPO_DIR}"
mkdir -p logs runs
[ -f "${REPO_DIR}/.env" ] && { set -a; source "${REPO_DIR}/.env"; set +a; }
source slurm/_env.sh

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv || true

python notebooks/lr3.py

echo "lr3 done. results in runs/lr3/results"
grep -h '^reg=' logs/lr3_${SLURM_JOB_ID}.out || true
