#!/bin/bash
# Launch experiment on P100 GPUs
# Usage: ./launch_experiment_p100.sh <experiment_name>

set -e

EXPERIMENT=$1
if [ -z "$EXPERIMENT" ]; then
    echo "Usage: $0 <experiment_name>"
    echo "Available experiments: baseline, lpips, l1, v6"
    exit 1
fi

case $EXPERIMENT in
    baseline|lpips|l1|v6)
        ;;
    *)
        echo "Error: Unknown experiment '$EXPERIMENT'"
        exit 1
        ;;
esac

echo "========================================="
echo "Launching experiment on P100: $EXPERIMENT"
echo "========================================="

export EXPERIMENT_NAME=$EXPERIMENT

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
    baseline)
        export PERCEP_LOSS="none"
        export LAMBDA_PERCEP="0.0"
        ;;
esac

sbatch slurm/train_p100.sbatch
