#!/bin/bash -l

#SBATCH -J brainflow_rebuild_cache
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -p gpu-all
#SBATCH --gres=gpu:1                  # 1 GPU for CLIP ViT-L/14 encoding
#SBATCH -c 8
#SBATCH --mem=140G                    # tensors(56G) + models + overhead
#SBATCH -t 04:00:00

set -euo pipefail

module load slurm/slurm/21.08.8 2>/dev/null || module load slurm 2>/dev/null || true
for m in cuda12.1/toolkit/12.1.1 cuda11.8/toolkit/11.8.0 cuda12.4/toolkit/12.4.1 cuda12.6/toolkit/12.6.2; do
    if module load "$m" 2>/dev/null; then
        echo "Loaded module: $m"; break
    fi
done

ENV_NAME="${ENV_NAME:-brainflow}"
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate "${ENV_NAME}"

REPO_DIR="${REPO_DIR:-$HOME/BrainFlow}"
cd "${REPO_DIR}"
[ -f .env ] && { set -a; source .env; set +a; }
mkdir -p logs

echo "Rebuilding CLIP CLS + VAE latent caches with v2 format..."
nvidia-smi || true

python -c "
from brainflow.config import load_config
from brainflow.data import build_or_load_tensors, compute_or_load_clip, compute_or_load_latents

cfg = load_config()
cfg.force_rebuild = True
print('Subjects:', cfg.subjects)
print('Data dir:', cfg.data_dir)

print('Loading existing tensors (not rebuilding)...')
cfg.force_rebuild = False
t = build_or_load_tensors(cfg)

cfg.force_rebuild = True
print('Rebuilding CLIP CLS cache (with corrected ImageNet norm)...')
c = compute_or_load_clip(t, cfg)

print('Rebuilding VAE latent cache (adds _meta key)...')
l = compute_or_load_latents(t, cfg)

print('Done. Both caches rebuilt with _meta and correct CLIP normalisation.')
"

echo "Cache rebuild complete at $(date)"
