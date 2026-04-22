#!/bin/bash
# Quick evaluation script for baseline model

cd ~/BrainFlow

module load cuda12.1/toolkit/12.1.1

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate brainflow

echo "=== Evaluating Baseline Model ==="
echo "Checkpoint: outputs/best_combined_v5.pt"
echo "CFG scale: 1.0"
echo "ODE steps: 1"
echo ""

python -m scripts.eval \
    --checkpoint outputs/best_combined_v5.pt \
    --cfg_scale 1.0 \
    --ode_steps 1 \
    --device cuda
