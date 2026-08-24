#!/bin/bash -l

#SBATCH -J ledger_smoke
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -p gpu-A100
#SBATCH --account=a100_intern
#SBATCH --qos=a100_intern
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 02:00:00

set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/BrainFlow}"; cd "${REPO_DIR}"
mkdir -p logs runs data
[ -f "${REPO_DIR}/.env" ] && { set -a; source "${REPO_DIR}/.env"; set +a; }
source slurm/_env.sh

echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv || true

if [ ! -d data/stl10_binary ]; then
    echo "stl10 absent, fetching (~2.6 GB, allow 15-20 min)"
    python - <<'PYFETCH'
import torchvision as tv
tv.datasets.STL10("data", split="train", download=True)
tv.datasets.STL10("data", split="test", download=True)
print("fetched stl10", flush=True)
PYFETCH
else
    echo "stl10 present, skipping fetch"
fi

python scripts/cluster_job.py --part theorem --out runs/smoke_thm \
    --thm_dims 2,8 --thm_batch 256 --thm_reps 1 --thm_steps 400 --wandb 0

python scripts/cluster_job.py --part sr --out runs/smoke_sr \
    --sr_data stl10 --sr_root data --sr_res 32 --sr_factor 4 \
    --sr_steps 200 --sr_batch 32 --sr_base 16 --sr_seeds 0 --sr_ode_k 8 --wandb 0

echo "smoke done"
grep -h '^reg=' logs/ledger_smoke_${SLURM_JOB_ID}.out || true
