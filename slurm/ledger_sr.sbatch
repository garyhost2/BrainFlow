#!/bin/bash -l

#SBATCH -J ledger_sr
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -p gpu-A100
#SBATCH --account=a100_intern
#SBATCH --qos=a100_intern
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 12:00:00
#SBATCH --array=0-2

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

RES_LIST=(32 48 96)
RES=${RES_LIST[$SLURM_ARRAY_TASK_ID]}
echo "resolution ${RES}"

python scripts/cluster_job.py --part sr \
    --out runs/sr_res${RES} \
    --sr_data stl10 --sr_root data \
    --sr_res ${RES} --sr_factor 4 \
    --sr_steps 20000 --sr_batch 64 --sr_base 64 \
    --sr_seeds 0,1,2 --sr_ode_k 32 \
    --wandb 1 --wandb_project coupling-ledger

echo "sr res=${RES} done. results in runs/sr_res${RES}/results"
