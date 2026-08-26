#!/bin/bash
#SBATCH -J probe
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p gpu-A100
#SBATCH --account=a100_intern
#SBATCH --qos=a100_intern
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00

set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/BrainFlow}"; cd "${REPO_DIR}"
mkdir -p logs runs
[ -f "${REPO_DIR}/.env" ] && { set -a; source "${REPO_DIR}/.env"; set +a; }
source slurm/_env.sh

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv || true

python -c "import open_clip" 2>/dev/null || pip install -q open_clip_torch

LEDGER_STAGES=real python notebooks/ledger6.py

echo "probe done. results in runs/ledger6/results/clip.csv"
grep -h '^reg=\|^probe' logs/probe_${SLURM_JOB_ID}.out || true
