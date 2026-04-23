#!/bin/bash -l
#SBATCH -J brainflow_viz
#SBATCH -p gpu-all
#SBATCH -A qcri
#SBATCH --qos=gpulimit
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 0-00:30:00
#SBATCH -o visualize_%j.out
#SBATCH -e visualize_%j.err

set -euo pipefail

# ── Modules ──
module load slurm/slurm/21.08.8 2>/dev/null || module load slurm 2>/dev/null || true
for m in cuda12.1/toolkit/12.1.1 cuda11.8/toolkit/11.8.0 cuda12.4/toolkit/12.4.1 cuda12.6/toolkit/12.6.2; do
    if module load "$m" 2>/dev/null; then
        echo "Loaded module: $m"; break
    fi
done

# ── Conda env ──
ENV_NAME="${ENV_NAME:-brainflow}"
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate "${ENV_NAME}"

# ── Repo ──
REPO_DIR="${REPO_DIR:-$HOME/BrainFlow}"
cd "${REPO_DIR}"

# ── Secrets ──
if [ -f "${REPO_DIR}/.env" ]; then
    set -a; source "${REPO_DIR}/.env"; set +a
fi

# Visualization settings (env-overridable)
CHECKPOINT=${CHECKPOINT:-"outputs/best_combined_v5.pt"}
OUTPUT=${OUTPUT:-"visualizations.png"}
N_SAMPLES=${N_SAMPLES:-32}
CFG_SCALE=${CFG_SCALE:-2.0}
ODE_STEPS=${ODE_STEPS:-20}

echo "=== Generating Visualizations ==="
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT"
echo "Samples: $N_SAMPLES"
echo "CFG scale: $CFG_SCALE"
echo "ODE steps: $ODE_STEPS"
echo ""

python -m scripts.visualize \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT" \
    --n_samples "$N_SAMPLES" \
    --cfg_scale "$CFG_SCALE" \
    --ode_steps "$ODE_STEPS" \
    --device cuda

echo ""
echo "✓ Done! Download to view:"
echo "  scp panther:~/BrainFlow/$OUTPUT ."
