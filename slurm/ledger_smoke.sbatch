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
#SBATCH -t 00:40:00

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

python scripts/cluster_job.py --part theorem --out runs/smoke_thm \
    --thm_dims 2,8 --thm_batch 256 --thm_reps 1 --thm_steps 400 --wandb 0

python scripts/cluster_job.py --part sr --out runs/smoke_sr \
    --sr_data cifar10 --sr_root data --sr_res 32 --sr_factor 4 \
    --sr_steps 200 --sr_batch 32 --sr_base 16 --sr_seeds 0 --sr_ode_k 8 --wandb 0

echo "smoke done"
grep -h '^reg=' logs/ledger_smoke_${SLURM_JOB_ID}.out || true
