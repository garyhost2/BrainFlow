#!/bin/bash -l

#SBATCH -J step1b_full
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -p gpu-A100
#SBATCH --account=a100_intern
#SBATCH --qos=a100_intern
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 2-00:00:00

set -euo pipefail

module load slurm 2>/dev/null || true
for m in cuda12.1/toolkit/12.1.1 cuda11.8/toolkit/11.8.0; do
    module load "$m" 2>/dev/null && { echo "Loaded module: $m"; break; }
done

VENV="${VENV:-$HOME/envs/brainflow}"
if [ -f "${VENV}/bin/activate" ]; then
    source "${VENV}/bin/activate"; echo "Activated venv: ${VENV}"
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

SUBJECTS="${SUBJECTS:-1 2 5 7}"
DATA_DIR="${DATA_DIR:-./mindeyev2_cache}"
TARGET_DIR="${TARGET_DIR:-${DATA_DIR}}"
OUT="${OUT:-outputs/step1b_ms}"

EVAL_N="${EVAL_N:-128}"

FULL_EVAL_N="${FULL_EVAL_N:-982}"

mkdir -p "${TARGET_DIR}"
N_SUBJ=$(echo ${SUBJECTS} | wc -w)
NEED_GB=$(( N_SUBJ * 26 )); HAVE_GB=0
for s in ${SUBJECTS}; do
    f="${TARGET_DIR}/step1b_bigg_s${s}.pt"
    [ -f "$f" ] && HAVE_GB=$(( HAVE_GB + $(du -m "$f" 2>/dev/null | cut -f1) / 1024 ))
done
NEED_GB=$(( NEED_GB > HAVE_GB ? NEED_GB - HAVE_GB : 0 ))
FREE_GB=$(df -BG "${TARGET_DIR}" 2>/dev/null | tail -1 | awk '{print $4}' | tr -dc '0-9'); FREE_GB=${FREE_GB:-0}
echo "── disk preflight: subjects=[${SUBJECTS}] to-build≈${NEED_GB}G have≈${HAVE_GB}G free=${FREE_GB}G on ${TARGET_DIR}"
if [ "${FREE_GB}" -lt "$(( NEED_GB + 20 ))" ]; then
    echo "ERROR: need ~${NEED_GB}G (+20G headroom) but only ${FREE_GB}G free on ${TARGET_DIR}." >&2
    echo "       Point TARGET_DIR at scratch or train fewer subjects." >&2
    exit 1
fi

echo "════════════════ STAGE 1/2: TRAIN ════════════════"
# TRAIN_EXTRA lets you launch the ablation arm, e.g.
#   TRAIN_EXTRA="--geometry euclidean --no-two-head" OUT=outputs/step1c_ecfm sbatch ...
python -m scripts.train_step1b \
    --data-dir "${DATA_DIR}" \
    --target-dir "${TARGET_DIR}" \
    --subjects ${SUBJECTS} \
    --mindeye-src "${MINDEYE_SRC}" \
    --ckpt-path "${UNCLIP_CKPT}" \
    --epochs 150 --batch-size 48 --lr 3e-4 \
    --eval-freq 10 --decode-eval --decode-n "${EVAL_N}" \
    --out "${OUT}" ${TRAIN_EXTRA:-}

echo "════════════════ STAGE 2/2: FULL EVAL ════════════════"
# Both candidates are now selected on the leakage-free VAL split, but on
# DIFFERENT signals, and they disagree: best.pt maximises token cosine (peaks
# early, favours PixCorr), best_clip2way.pt maximises decoded CLIP-2way (peaks
# late). SELECT picks which one stage 2 reports; EVAL_CKPT overrides outright.
SELECT="${SELECT:-clip}"
case "${SELECT}" in
    clip) CAND="${OUT}/best_clip2way.pt ${OUT}/best.pt ${OUT}/last.pt" ;;
    cos)  CAND="${OUT}/best.pt ${OUT}/best_clip2way.pt ${OUT}/last.pt" ;;
    last) CAND="${OUT}/last.pt" ;;
    *)    echo "ERROR: SELECT must be clip|cos|last (got '${SELECT}')" >&2; exit 1 ;;
esac
if [ -z "${EVAL_CKPT:-}" ]; then
    for c in ${CAND}; do
        [ -f "$c" ] && { EVAL_CKPT="$c"; break; }
    done
fi
[ -n "${EVAL_CKPT:-}" ] || { echo "ERROR: no checkpoint found in ${OUT}" >&2; exit 1; }
echo "Evaluating ${EVAL_CKPT} (SELECT=${SELECT}) over ${FULL_EVAL_N} test images"
python -m scripts.eval_step1b \
    --ckpt "${EVAL_CKPT}" \
    --data-dir "${DATA_DIR}" \
    --target-dir "${TARGET_DIR}" \
    --mindeye-src "${MINDEYE_SRC}" \
    --ckpt-path "${UNCLIP_CKPT}" \
    --cond-source prior --cfg-scale 3.0 --steps 50 \
    --max-images "${FULL_EVAL_N}" \
    --out "${OUT}/eval_full"

echo "Done at $(date)."
echo "  checkpoints -> ${OUT}"
echo "  full-set metrics -> ${OUT}/eval_full/metrics.json"
