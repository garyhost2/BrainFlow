# BrainFlow v5: Multi-Experiment Guide

This guide explains how to run multiple BrainFlow experiments in parallel to compare different loss functions and architectures.

## Available Experiments

1. **baseline** - Current configuration (CFM + InfoNCE only)
2. **lpips** - Baseline + LPIPS perceptual loss (weight=0.1)
3. **l1** - Baseline + L1 pixel loss (weight=0.05)
4. **v6** - Enhanced architecture with LPIPS + stronger alignment

## Prerequisites

### 1. Install Additional Dependencies

For LPIPS loss support:
```bash
conda activate brainflow
pip install lpips
```

### 2. Check Cluster Resources

```bash
# Check your job limits
sacctmgr show assoc where user=$USER format=account,qos,grptresmins,grptres

# Check available GPUs
squeue -u $USER
sinfo -p gpu-all -o "%20N %10c %10m %25f %10G %10a"
```

Typical limit: 4-8 GPUs total (2 experiments running simultaneously on 2×V100 each)

## Quick Start

### Option A: Use Launch Script (Recommended)

```bash
cd ~/BrainFlow

# Launch each experiment
./launch_experiment.sh baseline
./launch_experiment.sh lpips
./launch_experiment.sh l1
./launch_experiment.sh v6
```

The script automatically:
- Sets experiment-specific environment variables
- Configures perceptual loss parameters
- Submits the job to Slurm
- Creates isolated output directories

### Option B: Manual Launch

```bash
# Set environment variables for each experiment
export EXPERIMENT_NAME=lpips
export PERCEP_LOSS=lpips
export LAMBDA_PERCEP=0.1
sbatch slurm/train.sbatch

export EXPERIMENT_NAME=l1
export PERCEP_LOSS=l1
export LAMBDA_PERCEP=0.05
sbatch slurm/train.sbatch
```

## Experiment Details

### Baseline (Current Running)
- **Status**: Already running (job 277804 or later)
- **Config**: CFM + InfoNCE, no perceptual loss
- **Expected**: PC ~0.12-0.15, SSIM ~0.35-0.40
- **Action**: Let it continue running

### LPIPS Experiment
- **Loss**: CFM + InfoNCE + LPIPS (0.1×)
- **Goal**: Improve perceptual quality (colors, textures)
- **Expected improvement**: +0.02-0.05 PC/SSIM, better visual quality
- **Memory**: ~10% more GPU memory for LPIPS network
- **Speed**: ~15% slower per epoch

### L1 Experiment  
- **Loss**: CFM + InfoNCE + L1 pixel (0.05×)
- **Goal**: Simple color accuracy, sharper edges
- **Expected improvement**: +0.01-0.03 PC/SSIM
- **Memory**: Minimal overhead
- **Speed**: ~5% slower per epoch

### V6 Experiment
- **Loss**: CFM + InfoNCE (1.0×) + LPIPS (0.15×), reduced mixup
- **Architecture**: Same as V5 with enhanced loss balance
- **Goal**: Maximum reconstruction quality with current architecture
- **Changes from baseline**:
  - λ_align: 0.5 → 1.0 (stronger semantic alignment)
  - λ_percep: 0.1 → 0.15 (stronger perceptual quality)
  - mixup_alpha: 0.2 → 0.1 (cleaner gradients)
- **Note**: This is "V6-lite" - full SDXL-unclip would require major rewrite
- **Expected**: Best overall quality, potential +0.03-0.07 improvement over LPIPS

## Monitoring Experiments

### Check Running Jobs

```bash
# List your jobs
squeue -u $USER -o "%.10i %.9P %.20j %.8T %.10M %.6D %.20R %.10b"

# Check specific job output
tail -f brainflow_v5_<JOBID>.out

# Cancel a job if needed
scancel <JOBID>
```

### Monitor Training Progress

Each experiment writes to:
- **Logs**: `brainflow_v5_<JOBID>.out`
- **Checkpoints**: `outputs/<experiment_name>/`
- **W&B**: https://wandb.ai/jedbjh-essai/brainflow-v5

W&B run names:
- `v5-multisubject` (baseline)
- `v5-multisubject-lpips`
- `v5-multisubject-l1`
- `v5-multisubject-v6`

### Compare Results

After training completes, compare metrics:

```bash
# Extract metrics from logs
grep "Ep.*PC=" brainflow_v5_*.out | sort

# Or use W&B dashboard for visual comparison
```

## Expected Timeline

Assuming 2× V100 32GB per experiment:

| Experiment | Epochs | Time/Epoch | Total Time |
|-----------|--------|------------|------------|
| Baseline  | 150    | ~5 min     | ~12.5 hrs  |
| LPIPS     | 150    | ~5.7 min   | ~14.3 hrs  |
| L1        | 150    | ~5.2 min   | ~13.0 hrs  |
| V6        | 150    | ~6 min     | ~15 hrs    |

All experiments can run in parallel if you have 8 GPUs available.

## Checkpoint Management

Each experiment saves independently:

```
outputs/
├── baseline/
│   ├── best_pc_v5.pt
│   ├── best_clip_v5.pt
│   ├── best_combined_v5.pt
│   └── v5_ema_final.pt
├── lpips/
│   └── ... (same structure)
├── l1/
│   └── ... (same structure)
└── v6/
    └── ... (same structure)
```

## Resource Management

### If You Hit GPU Limits

Run experiments sequentially instead of parallel:

```bash
# Wait for baseline to reach epoch 30, then launch LPIPS
# Check current epoch:
tail brainflow_v5_<BASELINE_JOBID>.out | grep "Ep"

# Once baseline is stable, launch next:
./launch_experiment.sh lpips
```

### If Jobs Are Queued

Check queue status:
```bash
squeue -p gpu-all | grep v100nv

# See estimated start time
squeue -u $USER --start
```

## Troubleshooting

### LPIPS Import Error

```bash
conda activate brainflow
pip install lpips
```

### Out of Memory

Reduce batch size in `config.yaml`:
```yaml
dataloader:
  batch_size_per_gpu: 40  # was 48
```

### Experiments Writing to Same Directory

The launcher script automatically isolates outputs. If you edited config.yaml manually, ensure:
```yaml
output:
  experiment_name: "baseline"  # Change per experiment
```

## Post-Training Analysis

### Load Checkpoints

```python
from brainflow.models import BrainFlowV5
from brainflow.config import load_config

cfg = load_config()
cfg.experiment_name = "lpips"  # or l1, v6
model = BrainFlowV5(cfg, voxels)
model.load_state_dict(torch.load(f"outputs/{cfg.experiment_name}/best_combined_v5.pt"))
```

### Compare Metrics

Best approach: Use W&B "Compare Runs" feature
1. Go to https://wandb.ai/jedbjh-essai/brainflow-v5
2. Select all experiment runs
3. Click "Compare" → "Charts"
4. Plot: PC, SSIM, CLIP over epochs

## Next Steps

After all experiments complete:

1. **Compare final metrics** - Which experiment achieved best PC/SSIM/CLIP?
2. **Visual inspection** - Generate samples from each model, compare perceptual quality
3. **Ensemble** - Combine predictions from multiple experiments
4. **Iterate** - Based on results, tune λ_percep or architecture

## Contact

For issues, check:
- Training logs: `brainflow_v5_<JOBID>.out`
- Error logs: `brainflow_v5_<JOBID>.err`
- W&B dashboard for live metrics
