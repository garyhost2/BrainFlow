#!/bin/bash -l

#SBATCH -J diag_tokens
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -p gpu-A100
#SBATCH --account=a100_intern
#SBATCH --qos=a100_intern
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 00:20:00

# CPU-only token diagnostic (radius spread + direction concentration on real bigG
# tokens). Runs through SLURM because the login node has no usable RAM/CPU budget.
# Usage:  SUBJECT=1 sbatch slurm/diagnose_tokens.sbatch

set -euo pipefail

module load slurm 2>/dev/null || true
for m in cuda12.1/toolkit/12.1.1 cuda11.8/toolkit/11.8.0; do
    module load "$m" 2>/dev/null && { echo "Loaded module: $m"; break; }
done

VENV="${VENV:-$HOME/envs/rxfm}"
if [ -f "${VENV}/bin/activate" ]; then
    source "${VENV}/bin/activate"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"; conda activate "${ENV_NAME:-rxfm}"
fi

REPO_DIR="${REPO_DIR:-$HOME/BrainFlow}"; cd "${REPO_DIR}"
echo "Node: $(hostname)"

TARGET_DIR="${TARGET_DIR:-./mindeyev2_cache}"
SUBJECT="${SUBJECT:-1}"
SPLIT="${SPLIT:-test}"
N_IMAGES="${N_IMAGES:-400}"

# unbuffered so the verdict shows up even if the job is requeued
python -u -m scripts.diagnose_tokens \
    --target-dir "${TARGET_DIR}" \
    --subject "${SUBJECT}" \
    --split "${SPLIT}" \
    --n-images "${N_IMAGES}"

echo "Done at $(date)."
