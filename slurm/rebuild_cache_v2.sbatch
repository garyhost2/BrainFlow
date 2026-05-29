#!/bin/bash -l
# ─────────────────────────────────────────────────────────────────────────────
# Force-rebuild CLIP CLS and VAE latent caches to v2 format.
#
# WHY THIS IS NEEDED (PR #6 — brainflow-v6-full-refactor):
#   1. CLIP CLS bug fix: compute_or_load_clip() now applies ImageNet
#      normalization (mean/std) before ViT-L/14 encoding. Old caches have
#      incorrect CLIP embeddings — CLIP_Sim was inflated by ~0.02–0.05.
#   2. Cache validation: both clip and latent caches now require a _meta key.
#      Old caches without _meta raise RuntimeError on load.
#
# WHAT IS REBUILT:
#   all_subjects_clip_v2.pt      — CLIP CLS embeddings (corrected norm)
#   all_subjects_latents_v2.pt   — VAE latents (unchanged math, adds _meta)
#
# NOTE: all_subjects_tensors.pt and all_subjects_clip_patches_v2.pt are NOT
# rebuilt here (tensors are large/slow; patches are unaffected by this PR).
# If patch cache also lacks _meta, add compute_or_load_clip_patches(t, cfg).
#
#   Submit:  sbatch slurm/rebuild_cache_v2.sbatch
# ─────────────────────────────────────────────────────────────────────────────
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
cfg.force_rebuild = True        # skip cache-hit checks, overwrite existing files
print('Subjects:', cfg.subjects)
print('Data dir:', cfg.data_dir)

# Tensors must already exist (run prepare_data.sbatch first if not)
print('Loading existing tensors (not rebuilding)...')
cfg.force_rebuild = False       # keep tensors as-is
t = build_or_load_tensors(cfg)

# Rebuild CLIP CLS cache — applies corrected ImageNet normalisation
cfg.force_rebuild = True
print('Rebuilding CLIP CLS cache (with corrected ImageNet norm)...')
c = compute_or_load_clip(t, cfg)

# Rebuild latent cache — same VAE math, adds _meta key
print('Rebuilding VAE latent cache (adds _meta key)...')
l = compute_or_load_latents(t, cfg)

print('Done. Both caches rebuilt with _meta and correct CLIP normalisation.')
"

echo "Cache rebuild complete at $(date)"
