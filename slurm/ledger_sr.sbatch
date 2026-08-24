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
module load slurm 2>/dev/null || true
for m in cuda12.1/toolkit/12.1.1 cuda11.8/toolkit/11.8.0; do
    module load "$m" 2>/dev/null && { echo "Loaded module: $m"; break; }
done

VENV="${VENV:-$HOME/envs/me2}"
if [ -f "${VENV}/bin/activate" ]; then
    source "${VENV}/bin/activate"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"; conda activate "${ENV_NAME:-me2}"
fi

REPO_DIR="${REPO_DIR:-$HOME/BrainFlow}"; cd "${REPO_DIR}"
mkdir -p logs runs data
[ -f "${REPO_DIR}/.env" ] && { set -a; source "${REPO_DIR}/.env"; set +a; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS=1
echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv || true

if [ ! -d data/stl10_binary ] && [ ! -d data/cifar-10-batches-py ]; then
    echo "datasets absent, fetching"
    python - <<'PYFETCH'
import torchvision as tv
jobs = [("stl10-train", lambda: tv.datasets.STL10("data", split="train", download=True)),
        ("stl10-test",  lambda: tv.datasets.STL10("data", split="test", download=True)),
        ("cifar-train", lambda: tv.datasets.CIFAR10("data", train=True, download=True)),
        ("cifar-test",  lambda: tv.datasets.CIFAR10("data", train=False, download=True))]
for name, fn in jobs:
    try:
        fn()
        print("fetched", name, flush=True)
    except Exception as e:
        print("FAILED", name, repr(e)[:160], flush=True)
PYFETCH
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
