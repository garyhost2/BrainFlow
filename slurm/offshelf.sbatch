#!/bin/bash
#SBATCH -J offshelf
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p gpu-A100
#SBATCH --account=a100_intern
#SBATCH --qos=a100_intern
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00

set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/BrainFlow}"; cd "${REPO_DIR}"
mkdir -p logs runs data
[ -f "${REPO_DIR}/.env" ] && { set -a; source "${REPO_DIR}/.env"; set +a; }
source slurm/_env.sh

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv || true

for pkg in diffusers transformers accelerate safetensors open_clip_torch; do
    python -c "import ${pkg/open_clip_torch/open_clip}" 2>/dev/null && continue
    echo "installing $pkg"
    pip install -q "$pkg" || echo "WARN: $pkg install failed"
done

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

python scripts/cluster_job.py --part offshelf \
    --out runs/offshelf \
    --sr_root data \
    --os_n 64 --os_factors 2,4,8,16 \
    --os_strengths 0.1,0.2,0.3,0.4,0.5,0.6,0.8 \
    --os_steps 30 --os_batch 16 --os_seeds 0,1,2 \
    --wandb 1 --wandb_project coupling-ledger

echo "offshelf done. results in runs/offshelf/results"
grep -h '^reg=' logs/offshelf_${SLURM_JOB_ID}.out || true
