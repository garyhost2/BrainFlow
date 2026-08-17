#!/bin/bash -l

#SBATCH -J build_latent
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -p gpu-A100
#SBATCH --account=a100_intern
#SBATCH --qos=a100_intern
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 06:00:00

# Precompute the blurry-latent low-level target, once, per subject.
# ~2 GB per subject on disk; roughly 28k VAE encodes at 768^2 per subject, so
# budget ~10-15 min/subject. Run this BEFORE any --ll-target latent training job
# (the training job refuses to build the cache itself).
#
# Usage:  SUBJECTS="1 2 3 4 5 6 7 8" sbatch slurm/build_latent_targets.sbatch
# Split it if the wall limit is tight:  SUBJECTS="1 2 3 4" sbatch ...

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
[ -f "${REPO_DIR}/.env" ] && { set -a; source "${REPO_DIR}/.env"; set +a; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
echo "Node: $(hostname)"

MINDEYE_SRC="${MINDEYE_SRC:-third_party/MindEyeV2/src}"
UNCLIP_CKPT="${UNCLIP_CKPT:-third_party/unclip6_epoch0_step110000.ckpt}"
DATA_DIR="${DATA_DIR:-./mindeyev2_cache}"
TARGET_DIR="${TARGET_DIR:-${DATA_DIR}}"
SUBJECTS="${SUBJECTS:-1}"
LL_SIZE="${LL_SIZE:-64}"

python -u -m scripts.build_latent_targets \
    --data-dir "${DATA_DIR}" \
    --target-dir "${TARGET_DIR}" \
    --mindeye-src "${MINDEYE_SRC}" \
    --ckpt-path "${UNCLIP_CKPT}" \
    --subjects ${SUBJECTS} \
    --ll-size "${LL_SIZE}" \
    ${BUILD_EXTRA:-}

echo "Done at $(date)."
