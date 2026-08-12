#!/bin/bash -l
# ─────────────────────────────────────────────────────────────────────────────
# Gate 1 — does R-XFM beat the anchor it starts from?
#
#   sbatch --array=0-3 slurm/gate1_rxfm.sbatch
#
# Four arms on subject 1, pilot length. The question is single and cheap:
# does the flow ADD anything over its own starting point?
#
#   A0  noise        legacy step1c. z0 ~ Uniform(S^1663): ~90 deg from every
#                    target, carries no fMRI information.
#   A1  anchor_det   z0 = the regression anchor mu.
#   A2  anchor_var   z0 ~ q(.|fMRI), learned per-token sigma + KL floor.
#   A3  anchor_var   A2 + the cleanups: no CFG, no dead lambda_reg, cls
#                    conditioning de-teacher-forced, token-level SoftCLIP.
#
# PASS: val/delta_cos > 0.03 for A2 or A3, AND decoded PixCorr >= the
#       regression baseline (0.164 on the full 982).
# FAIL: fall back to A1 -- still bounded below by regression, because v == 0 is
#       the identity flow. If A1 is also flat, the DiT is under-trained rather
#       than the paradigm being wrong: cut prior_depth and raise lambda_flow
#       before touching anything else.
#
# Watch `Dcos=` in the per-epoch line. It is signed. Negative means the prior is
# actively hurting, which is what step1c was doing unlogged.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH -J bf_gate1
#SBATCH -o logs/gate1_%A_%a.out
#SBATCH -e logs/gate1_%A_%a.err
#SBATCH -p gpu-all
#SBATCH --gres=gpu:a100_80GB:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 12:00:00

set -euo pipefail

module load slurm 2>/dev/null || true
for m in cuda12.1/toolkit/12.1.1 cuda11.8/toolkit/11.8.0; do
    module load "$m" 2>/dev/null && { echo "module: $m"; break; }
done

ENV_NAME="${ENV_NAME:-brainflow}"
[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ] && source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

REPO_DIR="${REPO_DIR:-$HOME/BrainFlow}"
cd "${REPO_DIR}"
mkdir -p logs outputs
[ -f .env ] && { set -a; source .env; set +a; }

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Preflight: Gate 0 must have been answered ────────────────────────────────
# Every ablation below is uninterpretable if the fMRI<->image pairing is shifted.
if [ ! -f "outputs/gate0_offset.txt" ]; then
    echo "═══════════════════════════════════════════════════════════════════"
    echo " REFUSING TO RUN: Gate 0 has not been recorded."
    echo " brainflow/data.py indexes betas with (all_trial - 1) while indexing"
    echo " the image hdf5 unshifted. MindEye2 defines behav column 5 as"
    echo "   global_trial = (SESSION-1)*750 + jj      # 0-based"
    echo " and consumes it with NO offset. If ours is wrong, every arm below"
    echo " measures a scrambled pairing."
    echo ""
    echo "   python -m scripts.check_offset --subject 1 | tee outputs/gate0_offset.txt"
    echo ""
    echo " ~15 min, CPU, no training. Then resubmit."
    echo "═══════════════════════════════════════════════════════════════════"
    exit 1
fi
echo "── Gate 0 on record ──"; cat outputs/gate0_offset.txt

# ── Arms ─────────────────────────────────────────────────────────────────────
ARM_ID="${SLURM_ARRAY_TASK_ID:-3}"
case "${ARM_ID}" in
  0) NAME="A0_noise"
     EXTRA="--flow-source noise --cfg-scale 3.0 --cls-cond-mode teacher --lambda-reg 1.0" ;;
  1) NAME="A1_anchor_det"
     EXTRA="--flow-source anchor_det --cfg-scale 1.0 --cls-cond-mode teacher" ;;
  2) NAME="A2_anchor_var"
     EXTRA="--flow-source anchor_var --cfg-scale 1.0 --cls-cond-mode teacher \
            --anchor-jitter-rad 0.15 --lambda-kl 0.1" ;;
  3) NAME="A3_anchor_var_clean"
     EXTRA="--flow-source anchor_var --cfg-scale 1.0 --cls-cond-mode jitter \
            --anchor-jitter-rad 0.15 --lambda-kl 0.1 --lambda-clip-tok 0.1" ;;
  *) echo "unknown arm ${ARM_ID}"; exit 1 ;;
esac

SUBJECTS="${SUBJECTS:-1}"
EPOCHS="${EPOCHS:-40}"
OUT="outputs/gate1_${NAME}"

echo "── arm ${ARM_ID}: ${NAME} → ${OUT}"
echo "   ${EXTRA}"

srun python -m scripts.train_step1b \
    --subjects ${SUBJECTS} \
    --epochs "${EPOCHS}" \
    --batch-size 48 \
    --enc-hidden 2048 \
    --geometry sphere \
    --val-frac 0.1 \
    --eval-freq 5 \
    --diag-batches 8 \
    --gradnorm-every 200 \
    --no-low-level \
    --decode-eval --decode-n 64 \
    --out "${OUT}" \
    ${EXTRA} \
    ${TRAIN_EXTRA:-}

echo "── done $(date). delta_cos trace:"
python - <<'PY'
import json, pathlib
p = pathlib.Path("${OUT}") / "diagnostics.jsonl"
if not p.exists():
    raise SystemExit("no diagnostics.jsonl")
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
rows = [r for r in rows if "val/delta_cos" in r]
print(f"{'epoch':>6} {'anchor':>8} {'flow':>8} {'DELTA':>8} {'retr':>7} {'gapCos':>8}")
for r in rows:
    print(f"{r.get('epoch',0):6.0f} {r['val/anchor_cos']:8.4f} {r['val/flow_cos']:8.4f} "
          f"{r['val/delta_cos']:+8.4f} {r.get('val/retr_fwd_300',float('nan')):7.3f} "
          f"{r.get('gap/flow_cos',float('nan')):+8.4f}")
if rows:
    best = max(r["val/delta_cos"] for r in rows)
    print(f"\nbest val/delta_cos = {best:+.4f}   "
          f"({'PASS' if best > 0.03 else 'FAIL'} vs the 0.03 gate)")
PY
