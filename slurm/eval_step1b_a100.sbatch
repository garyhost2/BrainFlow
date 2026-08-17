#!/bin/bash -l

#SBATCH -J step1b_eval
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -p gpu-A100
#SBATCH --account=a100_intern
#SBATCH --qos=a100_intern
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 06:00:00

set -euo pipefail

module load slurm 2>/dev/null || true
for m in cuda12.1/toolkit/12.1.1 cuda11.8/toolkit/11.8.0; do
    module load "$m" 2>/dev/null && { echo "Loaded module: $m"; break; }
done

VENV="${VENV:-$HOME/envs/brainflow}"
if [ -f "${VENV}/bin/activate" ]; then
    source "${VENV}/bin/activate"
    echo "Activated venv: ${VENV}"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"; conda activate "${ENV_NAME:-brainflow}"
fi

REPO_DIR="${REPO_DIR:-$HOME/BrainFlow}"
cd "${REPO_DIR}"
mkdir -p logs outputs
[ -f "${REPO_DIR}/.env" ] && { set -a; source "${REPO_DIR}/.env"; set +a; }

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
echo "Node: $(hostname)"; nvidia-smi || true

MINDEYE_SRC="${MINDEYE_SRC:-third_party/MindEyeV2/src}"
UNCLIP_CKPT="${UNCLIP_CKPT:-third_party/unclip6_epoch0_step110000.ckpt}"

CKPT="${CKPT:-outputs/step1b/last.pt}"
OUT="${OUT:-eval_last}"
DATA_DIR="${DATA_DIR:-./mindeyev2_cache}"
TARGET_DIR="${TARGET_DIR:-${DATA_DIR}}"

python -m scripts.eval_step1b \
    --ckpt "${CKPT}" \
    --data-dir "${DATA_DIR}" \
    --target-dir "${TARGET_DIR}" \
    --mindeye-src "${MINDEYE_SRC}" \
    --ckpt-path "${UNCLIP_CKPT}" \
    --cond-source prior --cfg-scale 3.0 --steps 50 \
    --out "outputs/step1b/${OUT}"

echo "Done at $(date). Metrics -> outputs/step1b/${OUT}/metrics.json"
