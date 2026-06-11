#!/bin/bash -l
# ─────────────────────────────────────────────────────────────────────────────
# Step-1b decoder smoke test on an A100 — decode GROUND-TRUTH bigG tokens.
# Validates the SDXL-unCLIP decoder path before committing to training.
# Run AFTER scripts/setup_step1b.sh has finished (env + sgm + ckpt present).
#   Submit:  sbatch slurm/smoke_step1b_a100.sbatch
#   Output:  outputs/step1b/smoke/smoke_grid.png  (top=GT, bottom=decoded-from-GT)
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -J step1b_smoke
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 02:00:00

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
echo "Node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv || true

python -m scripts.smoke_decoder_step1b --subject 1 --n 8 \
    --mindeye-src third_party/MindEyeV2/src \
    --ckpt-path third_party/unclip6_epoch0_step110000.ckpt \
    --out outputs/step1b/smoke

echo "Done. Inspect outputs/step1b/smoke/smoke_grid.png"
