# BrainFlow v5 — fMRI → Image Reconstruction (Multi-subject CFM + VFM)

End-to-end pipeline for reconstructing seen images from fMRI on the
**MindEyeV2 / NSD** dataset using **Conditional Flow Matching** (v5) and
**Variational Flow Matching** (v5.2 / Phase 2) in the SD-VAE latent space.

### Phase 1 Architecture (v5)

```
fMRI (per-subject voxels)
   │  per-subject Linear projection
   ▼
BrainEncoder (MLP + stochastic depth)
   │  → 16 context tokens (B, 16, 768)  +  CLS (B, 768)
   ▼
FlowUNet velocity field (see startup param-count banner)  ◄─── Euler integration (20 steps, CFG=2.0)
   │
   ▼
frozen SD-VAE decoder  →  256×256 image
```

### Phase 2 Architecture (v5.2 — Hierarchical VFM)

```
fMRI (per-subject voxels)
   │  per-subject Linear projection
   ▼
BrainEncoder (MLP + stochastic depth)
   │  → N context tokens (B, N, 768)  +  CLS (B, 768)
   ├──────────────────────────────────────────┐
   ▼                                          ▼
Flow_CLIP DiT  ←──── VFM objective ────   brain tokens
   │  16×16×1024 CLIP patch-token grid
   │  (ViT-L/14, CLS dropped from flow)
   ▼
CLIP patch tokens  (B, 256, 1024)
   │  clip_context_proj → brain_dim
   ▼
Flow_VAE UNet  ◄──── VFM or CFM ────  brain tokens + CLIP tokens
   │  parallel brain + CLIP cross-attention
   │  backward-compat: clip_ctx=None → original behavior
   ▼
frozen SD-VAE decoder  →  256×256 image
```

**`flow_objective` flag** (in `config.yaml` or any phase2 YAML):
- `flow_objective: "cfm"` — standard CFM MSE loss (v5 baseline, default)
- `flow_objective: "vfm"` — Variational Flow Matching (rg-vfm, ICLR 2026)

```yaml
training:
  flow_objective: "vfm"   # or "cfm" for ablation
  lambda_cos: 0.1         # cosine loss weight (Flow_CLIP only)
```

### New Options (Items 1–4)

#### Item 1 — Spherical geometry for `ClipPrior` (`clip_prior_geometry`)

CLIP embeddings are L2-normalized and live on the unit hypersphere S^(d-1).
Enabling `geometry: "sphere"` switches `ClipPrior` from straight-line Euclidean
CFM to Riemannian flow matching:

- Source: uniform on S^(d-1) (Gaussian → L2-normalize)
- Interpolant: SLERP geodesic (stays on the sphere)
- Target velocity: analytic time-derivative of the geodesic (tangent)
- Sampler: exponential-map retraction keeps iterates on the sphere at every step

*Math correctness note:* The straight-line interpolant `x_t = (1-t)·x_0 + t·x_1`
leaves the sphere for `t ∈ (0,1)` and the velocity `u_t = x_1 - x_0` is not
tangent to S^(d-1).  The SLERP interpolant and exp-map retraction correct both
failures, keeping the entire trajectory on the manifold.

```yaml
geometry:
  clip_prior_geometry: "euclidean"   # "euclidean" (default) | "sphere"
```

The flag is backward-compatible: `"euclidean"` reproduces the original behavior
exactly, including mean/std normalization of the GT embedding.  In `"sphere"`
mode, mean/std normalization is **not** applied (unit-norm vectors already fix
the scale); this is documented in `brainflow/clip_prior.py`.

#### Item 2 — Time schedule for ODE integration (`t_schedule`)

`make_t_grid` in `brainflow/solvers.py` now supports three schedules:

```yaml
schedule:
  t_schedule: "linear"       # "linear" (default) | "cosine" | "logit_normal"
  logit_normal_m: 0.0        # mean for logit_normal (ignored otherwise)
  logit_normal_s: 1.0        # sigma for logit_normal (ignored otherwise)
```

- **`"linear"`** — uniform spacing; byte-for-byte identical to previous behavior.
- **`"cosine"`** — `t_i = (1 - cos(i·π/N)) / 2`; concentrates steps near `t=0`
  and `t=1` where flow curvature is highest.
- **`"logit_normal"`** — maps evenly spaced quantiles through `sigmoid(m + s·Φ⁻¹)`,
  concentrating steps in the interior of `[0,1]` (useful for heavy-tailed flows).

#### Item 3 — Minibatch OT source coupling (`use_ot_coupling`)

Random pairing of noise `x0` to VAE latents `x1` in Stage 2B yields near-
orthogonal, long trajectories.  Enabling minibatch optimal-transport coupling
reorders the noise batch to minimize total squared-Euclidean transport cost,
shortening trajectories and reducing velocity-estimation variance.

```yaml
ot:
  use_ot_coupling: false   # true to enable (default: false — current behavior)
  ot_reg: 0.05             # Sinkhorn entropic regularization
  ot_iters: 50             # Sinkhorn iterations
```

Implementation: `brainflow/ot_coupling.py::ot_minibatch_coupling` uses a pure-
PyTorch log-domain Sinkhorn (no hard dependency on `pot`/`geomloss`), resolved
to a hard row-wise argmax assignment.

#### Item 4 — Full 8-metric eval harness (`scripts/eval_full.py`)

```bash
# Full GPU run (requires checkpoint + NSD data)
python -m scripts.eval_full --ckpt outputs/best.pt --subject 1

# CI / offline self-test (PixCorr + SSIM only, no GPU/data needed)
python -m scripts.eval_full --self-test
```

Flags: `--ckpt`, `--subject`, `--ode-steps`, `--solver`, `--schedule`,
`--cfg-scale`, `--out-dir`.  Results are printed as a table and written to
`outputs/<experiment>/full_metrics.json`.

#### Item 5 — SDPrior + frozen diffusion decoder (`scripts/train_sdprior.py`)

Additive Phase-3 path:

`fMRI -> BrainEncoder tokens -> ClipPrior -> CLIP image embedding -> FrozenImageEmbeddingDecoder -> image`

- Trained: `BrainEncoder` and `ClipPrior`
- Frozen: diffusion decoder (`decoder_model_id`, loaded lazily)
- Eval: `python -m scripts.eval_sdprior` routes outputs into `brainflow.metrics_full.evaluate_full`

```bash
# train
BRAINFLOW_CONFIG=configs/phase3_sdprior.yaml \
INIT_FROM=outputs/best_pc_v5.pt \
python -m scripts.train_sdprior

# eval (8 metrics + cosine + *_2way keys)
BRAINFLOW_CONFIG=configs/phase3_sdprior.yaml \
CHECKPOINT=outputs/phase3_sdprior/best_clip_cos.pt \
python -m scripts.eval_sdprior
```

Decoder weights download on first use into `data_dir/hf_cache`; offline runs must pre-cache that model.

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

### Phase 1 bugfix patch (expected: PC 0.10→~0.18, SSIM 0.30→~0.40, latency 1.5s→~250ms)

Seven critical pipeline bugs identified and fixed:

| ID | Bug | Fix |
|----|-----|-----|
| B1 | Test fMRI z-scored with test-set statistics (data leak) | `NSDDataset` now accepts `fmri_mu`/`fmri_std`; train stats passed to test set via `build_dataloaders` |
| B3 | `find_unused_parameters=True` in DDP (throughput/correctness) | Set `False` + `broadcast_buffers=False` + `_set_static_graph()` |
| B4 | No NSD trial averaging on test set (3 reps treated separately) | Average betas across repetitions of same `coco_id` in `load_subject` |
| B5 | Token dropout zeros tokens; conditional path noisier than unconditional | Replaced with `null_tokens` substitution, moved to `training_step` after CFG drop |
| B6 | Forward Euler evaluates velocity at interval start | Midpoint Euler default; optional Heun solver (`solver="heun"`) |
| B7 | `cfg_scale=1.0` (CFG off) and `ode_steps=1` as defaults | `cfg_scale: 2.0`, `ode_steps: 20` |
| — | Mixup mixes VAE latents (off-manifold) | Mix fMRI only; latent target unchanged |

Additional fixes: `lambda_align` 0.1→0.5, `grad_clip` 0.5→1.0, `eval_batches` 8→9999,
unsharded eval loader for accurate rank-0 metrics, CLIP singleton (no reload per eval),
VAE fp16 inference, flash-attention via `F.scaled_dot_product_attention`,
fused AdamW, bf16 autocast on Ampere+, stochastic depth on-device bernoulli.

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
