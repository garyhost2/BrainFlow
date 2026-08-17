#!/bin/bash -l
#SBATCH -J bf_ll_latent
#SBATCH -o logs/ll_latent_%j.out
#SBATCH -e logs/ll_latent_%j.err
#SBATCH -p gpu-A100
#SBATCH --account=a100_intern
#SBATCH --qos=a100_intern
#SBATCH --gres=gpu:A100_80GB:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 20:00:00

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

PRIOR="${PRIOR:-outputs/gate1_A3_tier1/best.pt}"
SUBJECTS="${SUBJECTS:-1}"
EPOCHS="${EPOCHS:-120}"
LL_SIZE="${LL_SIZE:-64}"
OUT="${OUT:-outputs/ll_latent}"

if [ ! -f "${PRIOR}" ]; then
    echo "missing prior checkpoint ${PRIOR}; set PRIOR=<path>"
    exit 1
fi

SMOKE="${SMOKE:-outputs/step1c_smoke_latent/smoke_latent.json}"
if [ ! -f "${SMOKE}" ]; then
    echo "======================================================================"
    echo " REFUSING TO RUN: the latent smoke gate has not been recorded."
    echo " Arm C (the constant per-channel mean latent) decides whether any"
    echo " PixCorr gain is decoded signal or just 'any smooth init helps'."
    echo ""
    echo "   SUBJECT=1 N=16 sbatch slurm/smoke_latent.sbatch"
    echo "======================================================================"
    exit 1
fi
echo "-- smoke gate on record --"; cat "${SMOKE}"

CACHE_OK=1
for s in ${SUBJECTS}; do
    [ -f "mindeyev2_cache/step1b_latent_s${s}.pt" ] || CACHE_OK=0
done
if [ "${CACHE_OK}" -eq 0 ]; then
    echo "building blurry-latent targets (needs the A100 VAE)"
    srun python -m scripts.build_latent_targets --subjects ${SUBJECTS} --ll-size "${LL_SIZE}"
fi

srun python -m scripts.train \
    --subjects ${SUBJECTS} \
    --epochs "${EPOCHS}" \
    --batch-size 48 \
    --init-from "${PRIOR}" \
    --freeze-token \
    --ll-target latent \
    --ll-size "${LL_SIZE}" \
    --ll-loss l1 \
    --val-frac 0.1 \
    --eval-freq 5 \
    --decode-eval --decode-n 64 \
    --diag-batches 8 \
    --out "${OUT}" \
    ${TRAIN_EXTRA:-}

echo "-- low-level training done $(date) --"

OUT="${OUT}" python - <<'PY'
import json, os, pathlib
p = pathlib.Path(os.environ["OUT"]) / "diagnostics.jsonl"
if not p.exists():
    raise SystemExit(f"no diagnostics.jsonl under {p.parent}")
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
rows = [r for r in rows if "epoch" in r]
print(f"{'epoch':>6} {'2way':>8} {'retr':>8} {'anchor':>8} {'flow':>8}")
for r in rows:
    nan = float("nan")
    print(f"{r.get('epoch',0):6.0f} {r.get('val/two_way',nan):8.4f} "
          f"{r.get('val/retr_fwd_300',nan):8.4f} {r.get('val/anchor_cos',nan):8.4f} "
          f"{r.get('val/flow_cos',nan):8.4f}")
print()
print("collapse check: the low head is collapsed if the training 'low' term sits at")
print("0.798 (= E|N(0,1)|, the score of predicting the per-channel mean latent).")
PY
