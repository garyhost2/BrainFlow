# Quick Start: Running 4 Parallel Experiments

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

Launch the 3 new experiments:

```bash
# LPIPS experiment (perceptual loss)
./launch_experiment.sh lpips

# L1 experiment (pixel loss)
./launch_experiment.sh l1

# V6 experiment (enhanced baseline+LPIPS)
./launch_experiment.sh v6
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

After ~12-15 hours, check final metrics:

```bash
# Extract final epoch metrics from all logs
grep "Ep.*150" brainflow_v5_*.out

# Or visit W&B dashboard:
# https://wandb.ai/jedbjh-essai/brainflow-v5
```

## Expected Outcomes

| Experiment | PC Goal  | SSIM Goal | Best For |
|-----------|----------|-----------|----------|
| Baseline  | 0.12-0.15| 0.35-0.40 | Speed baseline |
| LPIPS     | 0.14-0.17| 0.38-0.42 | Perceptual quality |
| L1        | 0.13-0.16| 0.37-0.41 | Color accuracy |
| V6        | 0.15-0.19| 0.40-0.44 | Overall best |

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
batch_size_per_gpu: 40  # reduce from 48
```

## What Changed?

All changes are in your cloned repo:
- ✅ `config.yaml`: Added `percep_loss`, `lambda_percep`, `experiment_name`
- ✅ `brainflow/perceptual_loss.py`: New LPIPS/L1 loss modules
- ✅ `brainflow/config_overrides.py`: Environment variable support
- ✅ `brainflow/models.py`: Updated `training_step` for perceptual loss
- ✅ `scripts/train.py`: Added VAE + perceptual loss to training loop
- ✅ `launch_experiment.sh`: Easy experiment launcher
- ✅ `EXPERIMENTS.md`: Full documentation

## Next: Pull Changes from GitHub

```bash
cd ~/BrainFlow
git status
git add -A
git commit -m "feat: multi-experiment framework with LPIPS/L1/V6"
git push
```

Then on your cluster:
```bash
cd ~/BrainFlow
git pull
./launch_experiment.sh lpips
./launch_experiment.sh l1  
./launch_experiment.sh v6
```
