#!/bin/bash
#SBATCH -J sr_figs
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p gpu-A100
#SBATCH --account=a100_intern
#SBATCH --qos=a100_intern
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00

set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/BrainFlow}"; cd "${REPO_DIR}"
mkdir -p logs runs data
[ -f "${REPO_DIR}/.env" ] && { set -a; source "${REPO_DIR}/.env"; set +a; }
source slurm/_env.sh

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv || true

# one seed only, at the resolution used for the figure. seeding is deterministic, so
# these are the same three models the reported table was computed from.
python scripts/cluster_job.py --part sr \
    --out runs/sr_figs \
    --sr_data stl10 --sr_root data \
    --sr_res 48 --sr_factor 4 \
    --sr_seeds 0 \
    --sr_steps 20000 --sr_batch 64 --sr_base 64 --sr_ode_k 32 \
    --wandb 0

echo "sr figure panel in runs/sr_figs/figures"
ls -la runs/sr_figs/figures || true
