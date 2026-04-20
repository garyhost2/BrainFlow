# BrainFlow v5 — fMRI → Image Reconstruction (Multi-subject CFM)

End-to-end pipeline for reconstructing seen images from fMRI on the
**MindEyeV2 / NSD** dataset using **Conditional Flow Matching** in the
SD-VAE latent space.

```
fMRI (per-subject voxels)
   │  per-subject Linear projection
   ▼
BrainEncoder (MLP + stochastic depth)
   │  → 16 context tokens (B, 16, 768)  +  CLS (B, 768)
   ▼
FlowUNet velocity field (~120M)  ◄─── Euler integration (20 steps, CFG=2.0)
   │
   ▼
frozen SD-VAE decoder  →  256×256 image
```

## Highlights — v5

- **Multi-subject training** (1..8) via per-subject input projection
  (`Linear(n_voxels[s], 1024)`) routed by `subject` id.
- **Subject-grouped batch sampler** so every minibatch is voxel-shape
  homogeneous (works with `ConcatDataset` + DDP).
- **Anti-overfitting bag**: stronger dropout, weight decay, voxel masking,
  token dropout, mixup, stochastic depth, early stopping.
- **DDP-aware** training & evaluation; checkpoints/wandb logging on rank-0.
- **Tensor / CLIP / VAE-latent caches** so you only download once.
- **Slurm scripts** prewired for the QCRI **Panther** cluster.

---

## Repository layout

```
BrainFlow/
├── config.yaml                 # single source of truth
├── requirements.txt
├── .env.example                # HF_TOKEN, WANDB_API_KEY
├── brainflow/
│   ├── config.py               # YAML → dataclass
│   ├── data.py                 # multi-subject dataloader + caches + DDP sampler
│   ├── models.py               # BrainEncoder + FlowUNet + BrainFlowV5
│   ├── ema.py
│   ├── metrics.py              # PixCorr / SSIM / CLIP-Sim
│   └── vae.py                  # frozen SD-VAE-ft-mse
├── scripts/
│   ├── train.py                # DDP training entrypoint
│   └── inference.py            # checkpoint eval + CFG / NFE sweeps
├── slurm/
│   ├── setup_env.sh            # one-time conda env build
│   ├── prepare_data.sbatch     # download + CLIP/VAE precompute (1 GPU)
    ├── train.sbatch            # multi-GPU DDP training
    └── inference.sbatch        # 1-GPU evaluation
```

---

## Quick start (local)

```bash
git clone https://github.com/garyhost2/BrainFlow.git
cd BrainFlow

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env → set HF_TOKEN (required) and optionally WANDB_API_KEY

# Single-GPU training
python -m scripts.train

# Inference
python -m scripts.inference --ckpt outputs/best_pc_v5.pt
```

---

## Running on the Panther cluster

### 0. One-time: install Anaconda + create env

```bash
ssh <user>@panther-login.qcri.org
git clone https://github.com/garyhost2/BrainFlow.git ~/BrainFlow
cd ~/BrainFlow
cp .env.example .env && nano .env       # add HF_TOKEN + WANDB_API_KEY

# (skip if Anaconda already installed)
bash slurm/setup_env.sh
```

### 1. Pre-cache the dataset (once, on a GPU node)

```bash
sbatch slurm/prepare_data.sbatch
```

This downloads ≈12 GB of NSD shards for all 8 subjects, then encodes
images through CLIP + SD-VAE and writes `mindeyev2_cache/`:
- `all_subjects_tensors.pt`
- `all_subjects_clip.pt`
- `all_subjects_latents.pt`

### 2. Multi-GPU DDP training

```bash
sbatch slurm/train.sbatch
```

Defaults to **4× V100 16GB** on `gpu-all`. Edit the header to change:
- `--gres=gpu:v100_16GB:4`  →  e.g. `--gres=gpu:T4_16GB:4`
- `--gres=gpu:a100_80GB:2`  for the A100 nodes (only 10 cards cluster-wide)
- max **8 simultaneous GPUs per user** per Panther policy.

The script auto-detects `$NGPUS` and launches `torchrun --standalone
--nproc_per_node=$NGPUS -m scripts.train`.

### 3. Inference

```bash
sbatch slurm/inference.sbatch                 # uses outputs/best_pc_v5.pt
# or
CKPT=outputs/best_combined_v5.pt sbatch slurm/inference.sbatch
```

---

## Configuration

Everything tunable lives in [`config.yaml`](config.yaml):

| section       | important keys                                                |
|---------------|---------------------------------------------------------------|
| `data`        | `subjects`, `data_dir`, `force_rebuild`                       |
| `dataloader`  | `batch_size_per_gpu`, `num_workers`                           |
| `model`       | `enc_hidden`, `n_tokens`, `unet_base_ch`, `attn_heads`        |
| `training`    | `num_epochs`, `lr`, `cfg_scale`, `patience`, regularisation   |
| `inference`   | `ode_steps`                                                   |
| `output`      | `output_dir`                                                  |
| `wandb`       | `wandb_project`, `wandb_run_name`, `wandb_mode`               |

Effective batch size = `batch_size_per_gpu × num_gpus × grad_accum`.

---

## Multi-subject design notes

NSD subjects have different voxel counts (≈15k–18k). We handle this by:

1. **Per-subject input projection** in `BrainEncoder`:
   `Linear(voxels[s] → ENC_HIDDEN=1024)`.
2. **`SubjectBatchSampler`** in `brainflow/data.py` ensures every batch
   contains samples from a single subject, so we can stack `fmri` tensors
   into a uniform `(B, V_s)` shape and route through the right projection.
3. **Shared trunk + UNet** so all data jointly trains the bulk of the model.

Voxel counts are auto-detected from each subject's HDF5 and persisted in
the tensor cache as `voxels[subject]`.

---

## Outputs

`outputs/` contains:
- `best_pc_v5.pt` — best PixCorr checkpoint
- `best_clip_v5.pt` — best CLIP-Sim checkpoint
- `best_combined_v5.pt` — best of `PC + 0.3 × CLIP_Sim`
- `v5_ema_final.pt`, `v5_raw_final.pt` — final
- `inference_v5.png` — qualitative grid (after `scripts/inference.py`)

Logs (rank-0): `logs/train_<jobid>.out`.
