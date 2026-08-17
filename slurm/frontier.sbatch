#!/bin/bash -l
#SBATCH -J bf_frontier
#SBATCH -o logs/frontier_%j.out
#SBATCH -e logs/frontier_%j.err
#SBATCH -p gpu-A100
#SBATCH --account=a100_intern
#SBATCH --qos=a100_intern
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 12:00:00

set -euo pipefail

module load slurm 2>/dev/null || true
for m in cuda12.1/toolkit/12.1.1 cuda11.8/toolkit/11.8.0; do
    module load "$m" 2>/dev/null && { echo "module: $m"; break; }
done

ENV_NAME="${ENV_NAME:-rxfm}"
[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ] && source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

REPO_DIR="${REPO_DIR:-$HOME/BrainFlow}"
cd "${REPO_DIR}"
mkdir -p logs outputs
[ -f .env ] && { set -a; source .env; set +a; }

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONHASHSEED=0

CKPT="${CKPT:-outputs/gate1_A3_tier1/best.pt}"
KS="${KS:-0 1 2 4 8 16}"
STRENGTHS="${STRENGTHS:-1.0 0.8 0.7 0.6 0.5}"
N_IMAGES="${N_IMAGES:-200}"
SUBJECTS="${SUBJECTS:-1}"
OUT="${OUT:-outputs/frontier}"
LATENT_ONLY="${LATENT_ONLY:-0}"

if [ ! -f "${CKPT}" ]; then
    echo "missing checkpoint ${CKPT}; set CKPT=<path>"
    exit 1
fi

EXTRA=""
[ "${LATENT_ONLY}" = "1" ] && EXTRA="--latent-only"

srun python -m scripts.sweep_frontier \
    --ckpt "${CKPT}" \
    --ks ${KS} \
    --strengths ${STRENGTHS} \
    --max-images "${N_IMAGES}" \
    --eval-subjects ${SUBJECTS} \
    --out "${OUT}" \
    ${EXTRA} \
    ${SWEEP_EXTRA:-}

echo "-- frontier done $(date) --"
