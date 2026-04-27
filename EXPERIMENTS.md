# BrainFlow v5: Multi-Experiment Guide

This guide explains how to run multiple BrainFlow experiments in parallel to compare different loss functions and architectures.

## Available Experiments

1. **baseline** - Current configuration (CFM + InfoNCE only)
2. **lpips** - Baseline + LPIPS perceptual loss (weight=0.1)
3. **l1** - Baseline + L1 pixel loss (weight=0.05)
4. **v6** - Enhanced architecture: 64 tokens + LPIPS (weight=0.15)
5. **v7** - Compact model: 64 tokens + smaller UNet (192 base_ch, 6 enc blocks)
6. **v8** - MindEye-v2 quality target: 32 tokens + CLIPPriorHead + pixel L1 (target: PC ≥ 0.32)
7. **v9** - Phase 1+2 SOTA push: 64 tokens + CLIPPriorHead + rebalanced losses + LPIPS + accurate eval (target: PC ~0.18–0.22, CLIP ~0.85–0.90)

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
./launch_experiment.sh v7
./launch_experiment.sh v8   # new: targets MindEye-v2 level quality
./launch_experiment.sh v9   # new: Phase 1+2 SOTA push (64 tokens + rebalanced losses)
```

The script automatically:
- Sets experiment-specific environment variables
- Configures perceptual loss parameters
- Sets `OMP_NUM_THREADS=1` for optimal throughput
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
- **Loss**: CFM + InfoNCE (0.1×) + LPIPS (0.15×), reduced mixup
- **Architecture**: 64 tokens (4× baseline), same UNet
- **Goal**: Richer representation with more tokens + perceptual quality
- **Changes from baseline**:
  - n_tokens: 16 → 64 (4× more expressive capacity)
  - λ_align: 0.1 (baseline level, not increased)
  - λ_percep: 0.15 (strong perceptual guidance)
  - mixup_alpha: 0.2 → 0.1 (cleaner gradients)
- **Expected**: Best quality with current data, +0.03-0.07 improvement

### V7 Experiment
- **Loss**: CFM + InfoNCE (0.1×) + LPIPS (0.1×)
- **Architecture**: Compact model with richer representation
- **Goal**: Faster training with efficient architecture, test if smaller UNet works
- **Changes from baseline**:
  - n_tokens: 16 → 64 (4× more tokens)
  - unet_base_ch: 256 → 192 (25% fewer UNet params)
  - n_enc_blocks: 8 → 6 (25% fewer encoder layers)
  - batch_size_per_gpu: 48 → 32 (fits better on single GPU)
  - λ_align: 0.1 (baseline level)
  - λ_percep: 0.1 (moderate perceptual guidance)
- **Params**: ~180M (vs 231M baseline, 22% smaller)
- **Expected**: Faster training (~20% speedup), comparable or better quality if tokens are key
- **Strategy**: Test hypothesis that more tokens > bigger UNet

### V8 Experiment (NEW — targets MindEye-v2)
- **Loss**: CFM + InfoNCE (0.2×) + CLIPPrior (0.3×) + Pixel-L1 (0.3×, t>0.85 only)
- **Architecture**: 32 tokens + CLIPPriorHead (two-stage brain→CLIP prior head)
- **Method**: `baseline` (standard Gaussian source, no HRF)
- **Goal**: Match or beat MindEye v2 (PC ≥ 0.32, SSIM ≥ 0.42, CLIP_Sim ≥ 0.94 on subj-1)
- **Changes from baseline**:
  - n_tokens: 16 → 32
  - use_clip_prior: True (CLIPPriorHead appended as extra context token)
  - λ_prior: 0.3 (CLIP prior supervision)
  - λ_pixel: 0.3 (pixel-space L1, only for t > 0.85 steps)
  - λ_align: 0.2 (balanced alignment)
- **Expected**: After 60 epochs on subj-1: PC ≥ 0.32, SSIM ≥ 0.42, CLIP ≥ 0.94

### V9 Experiment (Phase 1+2 SOTA Push)

V9 incorporates all Phase 1 and Phase 2 improvements identified in the MC experiment audit.
The baseline MC run at epoch 45 showed PC=0.090, SSIM=0.334, CLIP=0.740 — all significantly
below SOTA (~0.35, ~0.42, ~0.94 respectively). V9 fixes the root causes systematically.

- **Loss**: CFM (0.5×) + InfoNCE (0.8×) + CLIPPrior (0.3×) + LPIPS (0.15×)
- **Architecture**: 64 tokens + deeper encoder (6 blocks) + CLIPPriorHead
- **Method**: `baseline` (UNet backbone + CLIPPriorHead — not full DiT)
- **Goal**: PC ~0.18–0.22, SSIM ~0.38–0.42, CLIP ~0.85–0.90

#### What each change does and why

| Change | Was | Now | Why |
|--------|-----|-----|-----|
| `use_clip_prior` | `False` | `True` | THE most important fix — forces semantic alignment via CLIPPriorHead |
| `lambda_prior` | — | `0.3` | Supervision weight for CLIPPriorHead cosine loss |
| `lambda_cfm` | `1.0` | `0.5` | CFM dominated at 9:1 over InfoNCE; must be balanced |
| `lambda_align` | `0.2` | `0.8` | InfoNCE must compete with CFM to learn semantics first |
| `infonce_temp` | `0.07` | `0.04` | Harder negatives = stronger contrastive signal |
| `cfg_scale` | `2.0` | `6.0` | Critical for CLIP similarity at inference (SOTA uses 5–10) |
| `cfg_drop_prob` | `0.10` | `0.15` | More unconditional training to support high CFG scale |
| `n_tokens` | `16` | `64` | 4× richer brain representation; was a severe bottleneck |
| `enc_blocks` | `4` | `6` | Deeper encoder extracts richer fMRI features |
| `percep_loss` | `"none"` | `"lpips"` | Pixel-quality gradient signal; explains stagnant SSIM/PC |
| `lambda_percep` | — | `0.15` | LPIPS weight |
| `eval_ode_steps` | `10` | `25` | 10-step Euler systematically underestimates all metrics |
| `eval_solver` | `"euler"` | `"midpoint"` | Strictly better accuracy at same NFE |
| `ode_steps` | `20` | `30` | Better inference quality |

#### How to launch

```bash
./launch_experiment.sh v9
```

Or directly via Slurm:

```bash
sbatch slurm/train_mc_xl_v9_v100_32gb.sbatch
```

#### Checkpoint continuity

Existing baseline/v6/v7/v8 checkpoints can be continued with `strict=False`.
`CLIPPriorHead` and `clip_to_brain` Linear are new parameters (not in old checkpoints),
but all other weights are compatible. The rest of the encoder/UNet architecture is unchanged.

```python
model.load_state_dict(torch.load("outputs/baseline/best_combined_v5.pt"), strict=False)
```

#### Expected metric trajectory

| Epoch | PC (expected) | SSIM (expected) | CLIP (expected) |
|-------|---------------|-----------------|-----------------|
| Baseline EP45 | 0.090 | 0.334 | 0.740 |
| V9 EP15 | ~0.10–0.13 | ~0.34–0.37 | ~0.79–0.83 |
| V9 EP30 | ~0.14–0.17 | ~0.36–0.40 | ~0.83–0.87 |
| V9 EP60 | ~0.18–0.22 | ~0.38–0.42 | ~0.85–0.90 |



The `hrf` method (`METHOD=hrf`) requires several fixes that are now applied to
prevent metric collapse (cfm loss → 0 while PC stagnates or falls):

1. **Source detach (B.1)**: `_source_from_cls` now detaches `cls_emb` so the encoder
   cannot trivially collapse `x0 → latent`. Source noise is unit-variance (was 0.1×).
2. **Non-zero init (B.2)**: `cls_to_latent` is initialised with `std=0.02` weights
   instead of all-zeros. This prevents immediate degenerate convergence.
3. **Causal HRF kernel (B.3)**: `_hrf_bias` now uses left-only padding
   (future tokens no longer leak into past — physiologically correct).
4. **No CFG at HRF eval (B.4)**: `cfg_scale=1.0` forced during sampling for HRF
   (the source is already conditional; CFG correction is meaningless).
5. **Mixup disabled for HRF (B.6)**: Mixed fMRI → mixed `x0` but un-mixed latent target
   creates inconsistent (x0, latent) pairs. Mixup is a no-op when `method=hrf`.
6. **Source collapse guard (B.7)**: A warning is printed when `(latent - x0).mse < 0.05`
   for >50% of a batch, making future regressions visible.

## Speed Improvements (All Experiments)

All experiments now benefit from the following throughput improvements:

| Change | Expected Speedup |
|--------|-----------------|
| Batch size doubled (48→96), grad_accum=1 (was 2) | ~2× |
| Shared input projection (ModuleDict → single Linear) | 15-30% on multi-GPU |
| No per-step `.item()` syncs (detach + epoch-end materialise) | 20-40% |
| `torch.compile(flow_unet, mode="reduce-overhead")` | 10-25% |
| `eval_batches=32`, 10-step Euler eval | Eval: 20× faster |
| `OMP_NUM_THREADS=1` | Minor |
| WandB offline mode | Minor |

**Throughput target**: ≤ 30 min/epoch on 2× V100 32GB (was ~84 min).

## Time/Epoch Estimates

| Experiment | Epochs | Time/Epoch (2×V100) | Total Time |
|-----------|--------|---------------------|------------|
| Baseline  | 150    | ~25 min             | ~62.5 hrs  |
| LPIPS     | 150    | ~28 min             | ~70 hrs    |
| L1        | 150    | ~26 min             | ~65 hrs    |
| V6        | 150    | ~30 min             | ~75 hrs    |
| V8        | 60     | ~27 min             | ~27 hrs    |
| HRF (fixed)| 150  | ~25 min             | ~62.5 hrs  |

## Monitoring Experiments

### Check Running Jobs

```bash
# List your jobs
squeue -u $USER -o "%.10i %.9P %.20j %.8T %.10M %.6D %.20R %.10b"

# Check specific job output
tail -f brainflow_v5_<JOBID>.out

# Check all experiments
ls -lh outputs/*/best_*.pt
```

### Monitor Training Progress

Each experiment writes to:
- **Logs**: `brainflow_v5_<JOBID>.out`
- **Checkpoints**: `outputs/<experiment_name>/`
- **W&B**: `wandb sync <run_dir>` after training (now offline by default)

W&B run names:
- `v5-multisubject` (baseline)
- `v5-multisubject-lpips`
- `v5-multisubject-l1`
- `v5-multisubject-v6`
- `v5-multisubject-v8`

### Compare Results

After training completes, compare metrics:

```bash
# Extract metrics from logs
grep "Ep.*PC=" brainflow_v5_*.out | sort

# Or use W&B dashboard for visual comparison
# First sync offline runs:
wandb sync outputs/baseline/wandb/
```

## Expected Timeline

Assuming 2× V100 32GB per experiment:

| Experiment | Epochs | Time/Epoch | Total Time |
|-----------|--------|------------|------------|
| Baseline  | 150    | ~25 min    | ~62.5 hrs  |
| LPIPS     | 150    | ~28 min    | ~70 hrs    |
| L1        | 150    | ~26 min    | ~65 hrs    |
| V6        | 150    | ~30 min    | ~75 hrs    |
| V8        | 60     | ~27 min    | ~27 hrs    |

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
├── v6/
│   └── ... (same structure)
└── v8/
    └── ... (same structure)
```

### Loading Old Checkpoints (per-subject input_proj → shared Linear)

If you have a checkpoint from before this PR (with per-subject `input_proj`),
use the migration helper:

```python
from brainflow.models import migrate_input_proj, BrainFlowV5
from brainflow.config import load_config
import torch

cfg = load_config()
model = BrainFlowV5(cfg, voxels)
old_sd = torch.load("outputs/old_checkpoint.pt")
new_sd = migrate_input_proj(old_sd, new_max_vox=model.brain_enc.max_vox)
model.load_state_dict(new_sd, strict=False)  # strict=False in case of other shape mismatches
```

A warning is printed during migration. A hard crash is never raised.

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
  batch_size_per_gpu: 80  # was 96
```

### Experiments Writing to Same Directory

The launcher script automatically isolates outputs. If you edited config.yaml manually, ensure:
```yaml
output:
  experiment_name: "baseline"  # Change per experiment
```

### HRF Metric Collapse (`cfm < 0.1` while PC stalls)

The root cause fixes are in B.1–B.3 above. Check:
1. Is `cls_to_latent` still zero-init? → should use `std=0.02`
2. Are you running with `mixup_alpha > 0`? → set to 0.0 for HRF
3. Is the HRF kernel non-causal? → check `_hrf_bias` uses left-pad only

If `cfm < 0.1` while PC is still stuck after the fixes, bump noise scale:
```python
# In _source_from_cls: change randn_like to:
return proj + 1.5 * torch.randn_like(proj)
```

## Post-Training Analysis

### Load Checkpoints

```python
from brainflow.models import BrainFlowV5, migrate_input_proj
from brainflow.config import load_config

cfg = load_config()
cfg.experiment_name = "v8"  # or baseline, lpips, etc.
model = BrainFlowV5(cfg, voxels)
sd = torch.load(f"outputs/{cfg.experiment_name}/best_combined_v5.pt")
sd = migrate_input_proj(sd, model.brain_enc.max_vox)  # no-op for new checkpoints
model.load_state_dict(sd)
```

### Compare Metrics

Best approach: Use W&B "Compare Runs" feature (after `wandb sync`)
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
./launch_experiment.sh v7
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
- **Loss**: CFM + InfoNCE (0.1×) + LPIPS (0.15×), reduced mixup
- **Architecture**: 64 tokens (4× baseline), same UNet
- **Goal**: Richer representation with more tokens + perceptual quality
- **Changes from baseline**:
  - n_tokens: 16 → 64 (4× more expressive capacity)
  - λ_align: 0.1 (baseline level, not increased)
  - λ_percep: 0.15 (strong perceptual guidance)
  - mixup_alpha: 0.2 → 0.1 (cleaner gradients)
- **Expected**: Best quality with current data, +0.03-0.07 improvement

### V7 Experiment
- **Loss**: CFM + InfoNCE (0.1×) + LPIPS (0.1×)
- **Architecture**: Compact model with richer representation
- **Goal**: Faster training with efficient architecture, test if smaller UNet works
- **Changes from baseline**:
  - n_tokens: 16 → 64 (4× more tokens)
  - unet_base_ch: 256 → 192 (25% fewer UNet params)
  - n_enc_blocks: 8 → 6 (25% fewer encoder layers)
  - batch_size_per_gpu: 48 → 32 (fits better on single GPU)
  - λ_align: 0.1 (baseline level)
  - λ_percep: 0.1 (moderate perceptual guidance)
- **Params**: ~180M (vs 231M baseline, 22% smaller)
- **Expected**: Faster training (~20% speedup), comparable or better quality if tokens are key
- **Strategy**: Test hypothesis that more tokens > bigger UNet

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
