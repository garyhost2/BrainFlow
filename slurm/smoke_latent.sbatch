#!/bin/bash -l

#SBATCH -J smoke_latent
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -p gpu-A100
#SBATCH --account=a100_intern
#SBATCH --qos=a100_intern
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 01:00:00

# GATE the VAE-latent low-level pathway before committing a 150-epoch run to it.
# Decodes GT tokens with three img2img inits (GT-blurry RGB / the same blur as a
# latent / the MEAN latent) across strengths. Read the two gate lines at the end:
#   * latent must match rgb   -> the new code path is correct
#   * latent minus mean       -> how much PixCorr is really attributable to
#                                predicting the layout, rather than to any smooth init
# Short job -- backfills ahead of the 3-day queue.
# Usage:  SUBJECT=1 N=16 sbatch slurm/smoke_latent.sbatch

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
SUBJECT="${SUBJECT:-1}"
N="${N:-16}"
LL_SIZE="${LL_SIZE:-64}"
STRENGTHS="${STRENGTHS:-1.0 0.7 0.6 0.5}"

python -u -m scripts.smoke_latent \
    --data-dir "${DATA_DIR}" \
    --mindeye-src "${MINDEYE_SRC}" \
    --ckpt-path "${UNCLIP_CKPT}" \
    --subject "${SUBJECT}" \
    --n "${N}" \
    --ll-size "${LL_SIZE}" \
    --strengths ${STRENGTHS} \
    --out "outputs/step1c_smoke_latent"

echo "Done at $(date)."
