#!/bin/bash -l
# ─────────────────────────────────────────────────────────────────────────────
# Step 1b — fMRI -> bigG token grid -> SDXL-unCLIP (MindEye2 decoder).
# The MindEye2-competitive path.  Needs a big GPU (A100 80GB recommended) and the
# one-time setup (bash scripts/setup_step1b.sh) already run on the LOGIN node.
# Use a DEDICATED conda env `me2` (MindEye2's pinned deps) — see docs/STEP1B.md.
#   Submit:  sbatch slurm/train_step1b_a100.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -J step1b
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -p gpu-A100
#SBATCH --account=a100_intern
#SBATCH --qos=a100_intern
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 2-00:00:00
# For an H200 instead (bigger/faster): change the two lines above to
#   -p gpu-H200   and   --gres=gpu:H200_141GB:1   (no other changes needed).

set -euo pipefail

module load slurm 2>/dev/null || true
for m in cuda12.1/toolkit/12.1.1 cuda11.8/toolkit/11.8.0; do
    module load "$m" 2>/dev/null && { echo "Loaded module: $m"; break; }
done

# Activate the dedicated Step-1b env. venv by default (login-node conda solver
# OOMs on Panther); falls back to a conda env named $ENV_NAME if no venv exists.
VENV="${VENV:-$HOME/envs/me2}"
if [ -f "${VENV}/bin/activate" ]; then
    source "${VENV}/bin/activate"
    echo "Activated venv: ${VENV}"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"; conda activate "${ENV_NAME:-me2}"
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

# Subjects to pool (shared backbone + per-subject input proj, MindEye2-style).
#   Default: the 4 full-data subjects MindEye2 reports on (1,2,5,7) — directly
#   comparable. Override for all 8:  SUBJECTS="1 2 3 4 5 6 7 8" sbatch ...
SUBJECTS="${SUBJECTS:-1 2 5 7}"
CACHE_DIR="${CACHE_DIR:-./mindeyev2_cache}"

# ── Disk preflight ──────────────────────────────────────────────────────────
# Each subject's bigG token cache is ~26 GB fp16 (train ~24 GB + test ~2.5 GB):
#   ~27.7k train trials × 256 tok × 1664 dim × 2 B ≈ 23.6 GB. Refuse to start a
# multi-subject build that would exceed free space rather than die mid-encode.
mkdir -p "${CACHE_DIR}"
N_SUBJ=$(echo ${SUBJECTS} | wc -w)
NEED_GB=$(( N_SUBJ * 26 ))
# subtract already-built per-subject caches from the requirement
HAVE_GB=0
for s in ${SUBJECTS}; do
    f="${CACHE_DIR}/step1b_bigg_s${s}.pt"
    [ -f "$f" ] && HAVE_GB=$(( HAVE_GB + $(du -m "$f" 2>/dev/null | cut -f1) / 1024 ))
done
NEED_GB=$(( NEED_GB > HAVE_GB ? NEED_GB - HAVE_GB : 0 ))
FREE_GB=$(df -BG "${CACHE_DIR}" 2>/dev/null | tail -1 | awk '{print $4}' | tr -dc '0-9')
FREE_GB=${FREE_GB:-0}
echo "── disk preflight ──────────────────────────────────────────────"
echo "subjects=[${SUBJECTS}]  to-build≈${NEED_GB}G (already cached≈${HAVE_GB}G)  free=${FREE_GB}G on ${CACHE_DIR}"
df -h "${CACHE_DIR}" || true
command -v quota >/dev/null 2>&1 && quota -s 2>/dev/null || true
if [ "${FREE_GB}" -lt "$(( NEED_GB + 10 ))" ]; then
    echo "ERROR: need ~${NEED_GB}G (+10G headroom) for the bigG target build but only ${FREE_GB}G free." >&2
    echo "       Free space, point CACHE_DIR at a larger volume, or train fewer subjects." >&2
    exit 1
fi
echo "────────────────────────────────────────────────────────────────"

# First launch also builds the bigG target cache (~26 GB/subject) via the embedder.
python -m scripts.train_step1b \
    --data-dir "${CACHE_DIR}" \
    --subjects ${SUBJECTS} \
    --mindeye-src "${MINDEYE_SRC}" \
    --ckpt-path "${UNCLIP_CKPT}" \
    --epochs 150 --batch-size 48 --lr 3e-4 \
    --eval-freq 10 --decode-eval --decode-n 16 \
    --out outputs/step1b

echo "Done at $(date)"
