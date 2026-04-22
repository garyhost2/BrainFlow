#!/bin/bash
# Experiment launcher for BrainFlow v5 variants
# Usage: ./launch_experiment.sh <experiment_name>
# Where experiment_name is: baseline | lpips | l1 | v6 | v7

set -e

EXPERIMENT=$1
if [ -z "$EXPERIMENT" ]; then
    echo "Usage: $0 <experiment_name>"
    echo "Available experiments: baseline, lpips, l1, v6, v7"
    exit 1
fi

# Validate experiment name
case $EXPERIMENT in
    baseline|lpips|l1|v6|v7)
        ;;
    *)
        echo "Error: Unknown experiment '$EXPERIMENT'"
        echo "Available experiments: baseline, lpips, l1, v6, v7"
        exit 1
        ;;
esac

echo "========================================="
echo "Launching experiment: $EXPERIMENT"
echo "========================================="

# Set experiment-specific environment variables
export EXPERIMENT_NAME=$EXPERIMENT

# Configure perceptual loss based on experiment
case $EXPERIMENT in
    lpips)
        export PERCEP_LOSS="lpips"
        export LAMBDA_PERCEP="0.1"
        ;;
    l1)
        export PERCEP_LOSS="l1"
        export LAMBDA_PERCEP="0.05"
        ;;
    v6)
        export PERCEP_LOSS="lpips"
        export LAMBDA_PERCEP="0.1"
        export USE_V6="1"
        ;;
    v7)
        export PERCEP_LOSS="lpips"
        export LAMBDA_PERCEP="0.1"
        export USE_V7="1"
        ;;
    baseline)
        export PERCEP_LOSS="none"
        export LAMBDA_PERCEP="0.0"
        ;;
esac

# Submit job
sbatch slurm/train.sbatch
