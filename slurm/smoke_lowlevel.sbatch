#!/bin/bash -l

#SBATCH -J smoke_lowlevel
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
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

VENV="${VENV:-$HOME/envs/rxfm}"
if [ -f "${VENV}/bin/activate" ]; then
    source "${VENV}/bin/activate"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"; conda activate "${ENV_NAME:-rxfm}"
fi

REPO_DIR="${REPO_DIR:-$HOME/BrainFlow}"; cd "${REPO_DIR}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
echo "Node: $(hostname)"

MINDEYE_SRC="${MINDEYE_SRC:-third_party/MindEyeV2/src}"
UNCLIP_CKPT="${UNCLIP_CKPT:-third_party/unclip6_epoch0_step110000.ckpt}"
DATA_DIR="${DATA_DIR:-./mindeyev2_cache}"

python -u -m scripts.smoke_lowlevel \
    --data-dir "${DATA_DIR}" \
    --subject "${SUBJECT:-1}" \
    --n "${N:-16}" \
    --mindeye-src "${MINDEYE_SRC}" \
    --ckpt-path "${UNCLIP_CKPT}"

echo "Done at $(date)."
