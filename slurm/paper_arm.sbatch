#!/bin/bash -l

#SBATCH -J paper_arm
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -p gpu-A100
#SBATCH --account=a100_intern
#SBATCH --qos=a100_intern
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH -c 8
#SBATCH --mem=200G
#SBATCH -t 3-00:00:00

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
mkdir -p logs outputs
[ -f "${REPO_DIR}/.env" ] && { set -a; source "${REPO_DIR}/.env"; set +a; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
echo "Node: $(hostname)"; nvidia-smi || true

MINDEYE_SRC="${MINDEYE_SRC:-third_party/MindEyeV2/src}"
UNCLIP_CKPT="${UNCLIP_CKPT:-third_party/unclip6_epoch0_step110000.ckpt}"
DATA_DIR="${DATA_DIR:-./mindeyev2_cache}"
TARGET_DIR="${TARGET_DIR:-${DATA_DIR}}"

GEOMETRY="${GEOMETRY:-sphere}"
ARM="${ARM:-${GEOMETRY}}"
SEED="${SEED:-0}"
ARM_EXTRA="${ARM_EXTRA:-}"

SUBJECTS="${SUBJECTS:-1 2 5 7}"
EPOCHS="${EPOCHS:-150}"
BATCH="${BATCH:-48}"
LR="${LR:-3e-4}"
ENC_HIDDEN="${ENC_HIDDEN:-2048}"
VAL_FRAC="${VAL_FRAC:-0.1}"
EVAL_N="${EVAL_N:-64}"

OUT="${OUT:-outputs/paper_${ARM}_s${SEED}}"
EVAL_SUBJECTS="${EVAL_SUBJECTS:-1}"     # per-subject eval is ~1.2 h EACH
FULL_EVAL_N="${FULL_EVAL_N:-982}"
SELECT="${SELECT:-clip}"
COND_SOURCES="${COND_SOURCES:-regression blend}"
CFG_SCALES="${CFG_SCALES:-1.0 3.0}"
BLEND_WS="${BLEND_WS:-0.5}"

echo "════════ ARM=${ARM} GEOMETRY=${GEOMETRY} SEED=${SEED} -> ${OUT} ════════"
echo "  fixed: subjects=[${SUBJECTS}] epochs=${EPOCHS} batch=${BATCH} lr=${LR} enc=${ENC_HIDDEN}"
echo "  arm extra: '${ARM_EXTRA}'"

N_SUBJ=$(echo ${SUBJECTS} | wc -w)
NEED_GB=$(( N_SUBJ * 26 )); HAVE_GB=0
for s in ${SUBJECTS}; do
    f="${TARGET_DIR}/step1b_bigg_s${s}.pt"
    [ -f "$f" ] && HAVE_GB=$(( HAVE_GB + $(du -m "$f" 2>/dev/null | cut -f1) / 1024 ))
done
NEED_GB=$(( NEED_GB > HAVE_GB ? NEED_GB - HAVE_GB : 0 ))
FREE_GB=$(df -BG "${TARGET_DIR}" 2>/dev/null | tail -1 | awk '{print $4}' | tr -dc '0-9'); FREE_GB=${FREE_GB:-0}
echo "── disk preflight: to-build≈${NEED_GB}G have≈${HAVE_GB}G free=${FREE_GB}G"
if [ "${FREE_GB}" -lt "$(( NEED_GB + 20 ))" ]; then
    echo "ERROR: need ~${NEED_GB}G (+20G headroom), only ${FREE_GB}G free." >&2
    exit 1
fi

echo "════════════════ STAGE 1/3: TRAIN ════════════════"
python -u -m scripts.train \
    --data-dir "${DATA_DIR}" --target-dir "${TARGET_DIR}" \
    --mindeye-src "${MINDEYE_SRC}" --ckpt-path "${UNCLIP_CKPT}" \
    --subjects ${SUBJECTS} \
    --geometry "${GEOMETRY}" \
    --seed "${SEED}" \
    --epochs "${EPOCHS}" --batch-size "${BATCH}" --lr "${LR}" \
    --enc-hidden "${ENC_HIDDEN}" --val-frac "${VAL_FRAC}" \
    --eval-freq 10 --decode-eval --decode-n "${EVAL_N}" \
    --out "${OUT}" ${ARM_EXTRA}

case "${SELECT}" in
    clip) CAND="${OUT}/best_clip2way.pt ${OUT}/best.pt ${OUT}/last.pt" ;;
    cos)  CAND="${OUT}/best.pt ${OUT}/best_clip2way.pt ${OUT}/last.pt" ;;
    last) CAND="${OUT}/last.pt" ;;
    *)    echo "ERROR: SELECT must be clip|cos|last" >&2; exit 1 ;;
esac
CKPT=""
for c in ${CAND}; do [ -f "$c" ] && { CKPT="$c"; break; }; done
[ -n "${CKPT}" ] || { echo "ERROR: no checkpoint in ${OUT}" >&2; exit 1; }

echo "════════════════ STAGE 2/3: FULL EVAL (${CKPT}) ════════════════"
python -u -m scripts.evaluate \
    --ckpt "${CKPT}" \
    --data-dir "${DATA_DIR}" --target-dir "${TARGET_DIR}" \
    --mindeye-src "${MINDEYE_SRC}" --ckpt-path "${UNCLIP_CKPT}" \
    --cond-source blend --cfg-scale 3.0 --steps 50 \
    --eval-subjects ${EVAL_SUBJECTS} \
    --max-images "${FULL_EVAL_N}" \
    --out "${OUT}/eval_full"

echo "════════════════ STAGE 3/3: OPERATING POINTS ════════════════"
python -u -m scripts.sweep \
    --ckpt "${CKPT}" \
    --data-dir "${DATA_DIR}" --target-dir "${TARGET_DIR}" \
    --mindeye-src "${MINDEYE_SRC}" --ckpt-path "${UNCLIP_CKPT}" \
    --n-images "${FULL_EVAL_N}" \
    --cond-sources ${COND_SOURCES} \
    --cfg-scales ${CFG_SCALES} \
    --blend-ws ${BLEND_WS} \
    --out "${OUT}/sweep"

echo "════════════════ REPORT ════════════════"
python -u -m scripts.report --root outputs --out outputs/paper || true

echo "Done at $(date)."
echo "  checkpoints -> ${OUT}"
echo "  metrics     -> ${OUT}/eval_full/metrics.json"
echo "  sweep       -> ${OUT}/sweep/sweep.json"
echo "  grids       -> ${OUT}/eval_full/recon_grid_s*.png , ${OUT}/sweep/grid_*.png"
echo "  paper table -> outputs/paper/results.md"
