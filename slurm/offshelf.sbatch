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

if ! python - <<'PYV'
import sys
try:
    import diffusers, transformers, accelerate
    ok = (diffusers.__version__.startswith("0.27")
          and transformers.__version__.startswith("4.38")
          and accelerate.__version__.startswith("0.27"))
except Exception:
    ok = False
sys.exit(0 if ok else 1)
PYV
then
    echo "pinning the diffusers stack to versions built for torch 2.1"
    pip install -q "diffusers==0.27.2" "transformers==4.38.2" "accelerate==0.27.2" \
        "huggingface_hub==0.23.5" "safetensors>=0.4.2" || echo "WARN: pin install failed"
fi
python -c "import open_clip" 2>/dev/null || pip install -q open_clip_torch
python -c "import wandb" 2>/dev/null || pip install -q wandb || true
python -c "import diffusers, transformers, accelerate; print('diffusers', diffusers.__version__, 'transformers', transformers.__version__, 'accelerate', accelerate.__version__)"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

python scripts/cluster_job.py --part offshelf \
    --out runs/offshelf \
    --sr_root data \
    --os_n 64 --os_factors 2,4,8,16 \
    --os_strengths 0.1,0.2,0.3,0.4,0.5,0.6,0.8 \
    --os_steps 30 --os_batch 16 --os_seeds 0,1,2 \
    --wandb 1 --wandb_project coupling-ledger

echo "offshelf done. results in runs/offshelf/results"
grep -h '^reg=\|^factor=\|model_' logs/offshelf_${SLURM_JOB_ID}.out || true
