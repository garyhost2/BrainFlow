#!/bin/bash -l

#SBATCH -J step1b_sweep
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -p gpu-A100
#SBATCH --account=a100_intern
#SBATCH --qos=a100_intern
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 06:00:00

# Sampling-calibration sweep on a trained checkpoint: finds the PixCorr-optimal
# and CLIP-optimal (cond_source, cfg_scale) without retraining.
# Usage:  CKPT=outputs/step1c_sphere/last.pt sbatch slurm/sweep_step1b.sbatch

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
[ -f "${REPO_DIR}/.env" ] && { set -a; source "${REPO_DIR}/.env"; set +a; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
echo "Node: $(hostname)"

MINDEYE_SRC="${MINDEYE_SRC:-third_party/MindEyeV2/src}"
UNCLIP_CKPT="${UNCLIP_CKPT:-third_party/unclip6_epoch0_step110000.ckpt}"
CKPT="${CKPT:-outputs/step1c_sphere/last.pt}"
DATA_DIR="${DATA_DIR:-./mindeyev2_cache}"
TARGET_DIR="${TARGET_DIR:-${DATA_DIR}}"
N_IMAGES="${N_IMAGES:-200}"
# Widen the grid without editing code, e.g.:
#   COND_SOURCES="regression blend prior" CFG_SCALES="1.0 1.5 2.0 3.0" N_IMAGES=982 ...
COND_SOURCES="${COND_SOURCES:-regression prior}"
CFG_SCALES="${CFG_SCALES:-1.0 3.0}"
# blend_w was hardcoded at 0.5 until 2026-08 and never swept; it is the direct
# PixCorr <-> CLIP dial (1.0 = pure prior, 0.0 = pure regression).
BLEND_WS="${BLEND_WS:-0.5}"
# Budget: ~1.2 h per config at N_IMAGES=982 against the 6 h wall limit -> <=4 configs.
# OUT_DIR defaults next to the checkpoint; set it to keep two sweeps of the same
# run from overwriting each other.
OUT_DIR="${OUT_DIR:-$(dirname "${CKPT}")/sweep}"

python -u -m scripts.sweep_step1b \
    --ckpt "${CKPT}" \
    --data-dir "${DATA_DIR}" \
    --target-dir "${TARGET_DIR}" \
    --mindeye-src "${MINDEYE_SRC}" \
    --ckpt-path "${UNCLIP_CKPT}" \
    --n-images "${N_IMAGES}" \
    --cond-sources ${COND_SOURCES} \
    --cfg-scales ${CFG_SCALES} \
    --blend-ws ${BLEND_WS} \
    --out "${OUT_DIR}"

echo "Done at $(date)."
