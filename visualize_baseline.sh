#!/bin/bash
# Visualize baseline model reconstructions

cd ~/BrainFlow

module load cuda12.1/toolkit/12.1.1

echo "=== Visualizing Baseline Reconstructions ==="
echo "Checkpoint: outputs/best_combined_v5.pt"
echo "Output: baseline_viz.png"
echo ""

python -m scripts.visualize \
    --checkpoint outputs/best_combined_v5.pt \
    --output baseline_viz.png \
    --n_samples 32 \
    --cfg_scale 1.0 \
    --ode_steps 1 \
    --device cuda

echo ""
echo "✓ Done! Download baseline_viz.png to view"
echo "  scp panther:~/BrainFlow/baseline_viz.png ."
