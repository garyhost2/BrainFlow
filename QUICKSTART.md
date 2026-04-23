# Quick Start: Running BrainFlow Experiments

## Step 1: Install Dependencies (One Time)

```bash
conda activate brainflow
pip install lpips
```

## Step 2: Make Launch Script Executable

```bash
cd ~/BrainFlow
chmod +x launch_experiment.sh
git pull  # Get latest changes
```

## Step 3: Launch All Experiments

**Important**: Your baseline experiment is already running. Don't relaunch it!

Launch the experiments:

```bash
# LPIPS experiment (perceptual loss)
./launch_experiment.sh lpips

# L1 experiment (pixel loss)
./launch_experiment.sh l1

# V6 experiment (enhanced baseline+LPIPS)
./launch_experiment.sh v6

# V8 experiment — targets MindEye-v2 level quality (PC ≥ 0.32)
./launch_experiment.sh v8
```

Each command submits a Slurm job. You'll see output like:
```
Submitted batch job 277900
```

## Step 4: Monitor Jobs

```bash
# Check running jobs
squeue -u $USER

# Watch training progress (replace JOBID)
tail -f brainflow_v5_<JOBID>.out

# Check all experiments
ls -lh outputs/*/best_*.pt
```

## Step 5: Compare Results

After training completes, check final metrics:

```bash
# Extract final epoch metrics from all logs
grep "Ep.*150" brainflow_v5_*.out

# Or sync W&B offline runs and visit dashboard:
wandb sync outputs/baseline/wandb/
# https://wandb.ai/jedbjh-essai/brainflow-v5
```

## Expected Outcomes

| Experiment | PC Goal  | SSIM Goal | CLIP Goal | Best For |
|-----------|----------|-----------|-----------|----------|
| Baseline  | 0.12-0.15| 0.35-0.40 | —         | Speed baseline |
| LPIPS     | 0.14-0.17| 0.38-0.42 | —         | Perceptual quality |
| L1        | 0.13-0.16| 0.37-0.41 | —         | Color accuracy |
| V6        | 0.15-0.19| 0.40-0.44 | —         | Overall best |
| **V8**    | **≥ 0.32**| **≥ 0.42**| **≥ 0.94**| **MindEye-v2 target** |

### V8 Notes

V8 uses a two-stage CLIP prior head (`CLIPPriorHead`) appended as a single extra
cross-attention token, plus pixel-space L1 loss for high-t timesteps. Train for
60 epochs on subj-1 to reach the MindEye-v2 quality target.

## Troubleshooting

**Job pending**: Wait or cancel baseline after epoch 30
```bash
# Check when jobs will start
squeue -u $USER --start

# Cancel baseline to free GPUs (optional)
scancel <BASELINE_JOBID>
```

**Import error**: Install lpips
```bash
pip install lpips
```

**Out of memory**: Edit config.yaml:
```yaml
batch_size_per_gpu: 80  # reduce from 96
```

**HRF metric collapse** (cfm < 0.1, PC falling):
The fixes in B.1–B.3 should prevent this. If it still occurs, bump noise scale in
`_source_from_cls`:
```python
return proj + 1.5 * torch.randn_like(proj)  # was 1.0
```

**WandB not uploading**: WandB is now offline by default. Sync after training:
```bash
wandb sync outputs/<experiment>/wandb/
```

## What Changed (This PR)

Speed improvements (all experiments):
- ✅ `config.yaml`: `batch_size_per_gpu: 96` (was 48), `grad_accum: 1` (was 2)
- ✅ `scripts/train.py`: No per-step `.item()` syncs; throttled tqdm postfix
- ✅ `scripts/train.py`: `torch.compile(flow_unet)` for kernel fusion
- ✅ `scripts/train.py`: 10-step Euler eval (was 20-step midpoint)
- ✅ `brainflow/models.py`: Shared `Linear(max_vox, enc_hidden)` (was per-subject ModuleDict)
- ✅ `config.yaml`: `eval_batches: 32` (was 9999); `wandb_mode: offline`
- ✅ Launch scripts: `OMP_NUM_THREADS=1`

HRF metric collapse fixes:
- ✅ `brainflow/models.py`: `_source_from_cls` detaches cls + unit-variance noise
- ✅ `brainflow/models.py`: `cls_to_latent` init `std=0.02` (was zeros)
- ✅ `brainflow/models.py`: Causal HRF convolution (left-pad only)
- ✅ `brainflow/models.py`: `cfg_scale=1.0` forced at HRF eval time
- ✅ `brainflow/models.py`: Mixup disabled for HRF method

New architectural features:
- ✅ `brainflow/models.py`: `CLIPPriorHead` (C.1), gated by `cfg.use_clip_prior`
- ✅ `brainflow/models.py`: Pixel-L1 loss for `t > 0.85` (C.2), `cfg.lambda_pixel`
- ✅ `brainflow/config_overrides.py`: V8 preset (`USE_V8=1`)
- ✅ `launch_experiment.sh`: Added `v8` experiment
- ✅ `brainflow/models.py`: `migrate_input_proj` helper for old checkpoint migration
- ✅ Combined eval metric includes SSIM: `PC + 0.5*SSIM + 0.3*CLIP`

## Next: Pull Changes from GitHub

```bash
cd ~/BrainFlow
git status
git pull
./launch_experiment.sh v8
```
