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
#                    target, carries no fMRI information. Velocity param + CFG,
#                    i.e. the exact configuration step1c ran.
#   A1  anchor_det   z0 = the regression anchor mu, velocity parameterisation.
#                    Isolates the SOURCE change from the PARAMETERISATION change.
#   A2  anchor_var   A1 + the variational source (learned per-token sigma, KL
#                    floor) + endpoint parameterisation.
#   A3  anchor_var   A2 + the rest of tier 1: token-level SoftCLIP on the
#                    flattened 256x1664 grid, cls conditioning de-teacher-forced
#                    in radians, per-position centring.
#
# A1 vs A2 is the only pair that isolates flow_param, so do not read A0->A3 as a
# single ladder.
#
# PASS: val/delta_cos > 0.03 for A2 or A3, AND decoded PixCorr >= the
#       regression baseline (0.164 on the full 982).
# FAIL: fall back to A1 -- still bounded below by regression, because v == 0 is
#       the identity flow. If A1 is also flat, the DiT is under-trained rather
#       than the paradigm being wrong: cut prior_depth and raise lambda_flow
#       before touching anything else.
#
# Watch `Dcos=` in the per-epoch line. It is signed. Negative means the prior is
# actively hurting, which is what step1c was doing unlogged. `2way=` is the
# checkpoint-selection signal and is computed on the flow output, not the anchor.
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
# Existence is not a verdict: the script emits a machine-readable GATE0: line and
# can legitimately conclude INCONCLUSIVE (both offsets near chance). Refuse those
# too, or every arm below measures a pairing nobody actually established.
if ! grep -q "GATE0: verdict=CONCLUSIVE" outputs/gate0_offset.txt; then
    echo "═══════════════════════════════════════════════════════════════════"
    echo " REFUSING TO RUN: Gate 0 exists but did not reach a verdict."
    grep -E "GATE0:|\[warn\]" outputs/gate0_offset.txt || true
    echo ""
    echo "   python -m scripts.check_offset --subject 1 --n-eval 2000 \\"
    echo "       | tee outputs/gate0_offset.txt"
    echo "═══════════════════════════════════════════════════════════════════"
    exit 1
fi

# ── Arms ─────────────────────────────────────────────────────────────────────
ARM_ID="${SLURM_ARRAY_TASK_ID:-3}"
case "${ARM_ID}" in
  0) NAME="A0_noise"
     EXTRA="--flow-source noise --flow-param velocity --cfg-scale 3.0 \
            --cls-cond-mode teacher --lambda-reg 1.0 --select-on cos" ;;
  1) NAME="A1_anchor_det"
     EXTRA="--flow-source anchor_det --flow-param velocity --cfg-scale 1.0 \
            --cls-cond-mode teacher" ;;
  2) NAME="A2_anchor_var_endpoint"
     EXTRA="--flow-source anchor_var --flow-param endpoint --cfg-scale 1.0 \
            --cls-cond-mode teacher --anchor-jitter-rad 0.15 --lambda-kl 0.1" ;;
  3) NAME="A3_tier1"
     EXTRA="--flow-source anchor_var --flow-param endpoint --cfg-scale 1.0 \
            --cls-cond-mode jitter --cls-jitter-rad 0.30 \
            --anchor-jitter-rad 0.15 --lambda-kl 0.1 --lambda-clip-tok 0.1 \
            --center-mode per_position" ;;
  *) echo "unknown arm ${ARM_ID}"; exit 1 ;;
esac

SUBJECTS="${SUBJECTS:-1}"
EPOCHS="${EPOCHS:-40}"

# The subject set is part of the run identity, not a detail: a multi-subject arm
# must not land on top of the single-subject one it is meant to be compared with.
SUBJ_TAG="s$(echo ${SUBJECTS} | tr -s ' ' '-')"
OUT="${OUT:-outputs/gate1_${NAME}_${SUBJ_TAG}}"
if [ -e "${OUT}" ] && [ -z "${OVERWRITE:-}" ]; then
    echo "REFUSING: ${OUT} already exists. Set OVERWRITE=1 or pass OUT=<dir>."
    exit 1
fi

echo "── arm ${ARM_ID}: ${NAME} → ${OUT}"
echo "   subjects: ${SUBJECTS}   epochs: ${EPOCHS}"
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
# NOTE: the heredoc delimiter is QUOTED, so the shell does not expand anything
# inside it -- "${OUT}" would arrive at Python as six literal characters and the
# trace silently reported "no diagnostics.jsonl" for every arm. Pass it through
# the environment instead.
OUT="${OUT}" python - <<'PY'
import json, os, pathlib
p = pathlib.Path(os.environ["OUT"]) / "diagnostics.jsonl"
if not p.exists():
    raise SystemExit(f"no diagnostics.jsonl under {p.parent}")
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
rows = [r for r in rows if "val/delta_cos" in r]
hdr = ("epoch", "anchor", "flow", "DELTA", "2way", "retr", "gapCos")
print(f"{hdr[0]:>6} {hdr[1]:>8} {hdr[2]:>8} {hdr[3]:>8} {hdr[4]:>7} {hdr[5]:>7} {hdr[6]:>8}")
nan = float("nan")
for r in rows:
    print(f"{r.get('epoch',0):6.0f} {r['val/anchor_cos']:8.4f} {r['val/flow_cos']:8.4f} "
          f"{r['val/delta_cos']:+8.4f} {r.get('val/two_way',nan):7.3f} "
          f"{r.get('val/retr_fwd_300',nan):7.3f} {r.get('gap/flow_cos',nan):+8.4f}")
if rows:
    best = max(r["val/delta_cos"] for r in rows)
    print(f"\nbest val/delta_cos = {best:+.4f}   "
          f"({'PASS' if best > 0.03 else 'FAIL'} vs the 0.03 gate)")
PY
