#!/bin/bash
#SBATCH -p gpu-all
#SBATCH -A qcri
#SBATCH --qos=gpulimit
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH -t 0-00:30:00
#SBATCH -o visualize_%j.out
#SBATCH -e visualize_%j.err

module load cuda12.1/toolkit/12.1.1
cd ~/BrainFlow

# Visualization settings
CHECKPOINT=${CHECKPOINT:-"outputs/best_combined_v5.pt"}
OUTPUT=${OUTPUT:-"visualizations.png"}
N_SAMPLES=${N_SAMPLES:-32}
CFG_SCALE=${CFG_SCALE:-1.0}
ODE_STEPS=${ODE_STEPS:-1}

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
