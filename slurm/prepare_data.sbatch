#!/bin/bash -l
# One-shot CPU job to download + cache MindEyeV2 (run before training).
#SBATCH -J brainflow_prep
#SBATCH -o logs/prep_%j.out
#SBATCH -e logs/prep_%j.err
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1                  # need 1 GPU for CLIP+VAE precompute
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 6:00:00

set -euo pipefail
module load slurm/slurm/21.08.8 2>/dev/null || module load slurm 2>/dev/null || true
for m in cuda12.1/toolkit/12.1.1 cuda11.8/toolkit/11.8.0 cuda12.4/toolkit/12.4.1 cuda12.6/toolkit/12.6.2; do
    if module load "$m" 2>/dev/null; then
        echo "Loaded module: $m"; break
    fi
done

ENV_NAME="${ENV_NAME:-brainflow}"
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

REPO_DIR="${REPO_DIR:-$HOME/BrainFlow}"
cd "${REPO_DIR}"
[ -f .env ] && { set -a; source .env; set +a; }
mkdir -p logs

# Imports the data builder which downloads + precomputes CLIP & VAE caches
python -c "
from brainflow.config import load_config
from brainflow.data import build_or_load_tensors, compute_or_load_clip, compute_or_load_latents
cfg = load_config()
print('Subjects:', cfg.subjects)
t = build_or_load_tensors(cfg)
c = compute_or_load_clip(t, cfg)
l = compute_or_load_latents(t, cfg)
print('All caches built.')
"
