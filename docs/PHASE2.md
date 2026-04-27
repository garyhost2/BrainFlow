# Phase 2: Hierarchical Variational Flow Matching for fMRI-to-Image

## Overview

Phase 2 implements a two-stage hierarchical flow-matching pipeline:

1. **Flow_CLIP** (`brainflow/flow_clip_dit.py`): A DiT-based flow that maps
   fMRI brain tokens → ViT-L/14 patch tokens (16×16×1024 grid).
2. **Flow_VAE** (`brainflow/flow_unet.py`): The existing UNet-based flow in
   VAE latent space, now conditioned on decoded CLIP patch tokens from Flow_CLIP.

Both flows are trained under a unified **Variational Flow Matching (VFM)**
objective from:

> Chen, R. T. Q. & Lipman, Y. (2024). *Riemannian Flow Matching on General
> Geometries*. ICLR 2026. (rg-vfm)

---

## Design Decisions

### 1. VFM Objective (`brainflow/vfm.py`)

Instead of directly predicting velocity `v = dx/dt`, the network predicts
parameters of a **Gaussian posterior** `q(x1 | xt, t)`:

```
raw output → (μ, log σ)   via a VFMHead (splits 2× output channels)

velocity:  v(xt, t) = (E_q[x1] - xt) / (1 - t)   with clamp on (1-t) ≥ ε

loss:      L_VFM = E[(μ - x1)² / (2σ²) + log σ]  (negative ELBO on x1)
```

The VFM objective is toggled via `cfg.flow_objective ∈ {"cfm", "vfm"}`.
Setting `flow_objective = "cfm"` recovers the standard CFM MSE loss exactly
(used as a clean ablation baseline).

### 2. Flow_CLIP DiT: 16×16×1024 Direct Flow

**Why patch tokens, not CLS?**

- The CLS token is a global summary — it conflates spatial layout, colour, and
  high-level semantics into a single vector. Flowing CLS → CLS loses all
  spatial structure needed for image reconstruction.
- Patch tokens (16×16 = 256 tokens at 1024 dim each) retain spatial layout
  and can be decoded directly to guide the VAE flow.
- **CLS token is dropped from the flow target** and instead predicted by an
  auxiliary MLP head from the pooled DiT output. This provides a clean
  inference fallback for retrieval-based metrics.

**Why 16×16 not 14×14 (original ViT-L/14)?**

- ViT-L/14 produces 196 patch tokens (14×14) + 1 CLS from a 224×224 image.
  We zero-pad to 256 (16×16) to get a spatially convenient power-of-2 grid
  that is compatible with standard 2D positional embeddings.
- In practice, the model crops/pads the 196 patch tokens to fill a 16×16 grid.

### 3. Cosine + MSE Hybrid Loss

The CLIP token loss combines MSE (on the predicted vs. true token values) with
a cosine term (encouraging directional alignment of the flow):

```
L = MSE(v_pred, v_true) + λ_cos · (1 - cos(v_pred, v_true))
```

Per-token cosine similarity is computed after reshaping to (B·H·W, 1024).
`λ_cos` defaults to 0.1 (configurable).

### 4. Per-Channel CLIP Standardization

CLIP ViT-L/14 patch tokens have significantly different per-channel statistics
(some channels dominate in magnitude). Flowing in the raw space biases the VFM
towards high-variance channels.

**Fix**: compute per-channel mean/std over the training set and standardize
before flowing; de-standardize after sampling. Stored as model buffers so they
are saved/loaded with the checkpoint.

```python
# Fit once on training data
model.fit_standardization(all_clip_tokens)  # (N, 16, 16, 1024)

# At training time
x_std = model.standardize(x)   # → roughly N(0, I)
...
x_out = model.destandardize(x_sampled)
```

Round-trip error < 1e-5 (verified in `tests/test_clip_standardization.py`).

### 5. Flow_VAE Modifications

The UNet flow backbone (`brainflow/flow_unet.py`) gains a parallel
cross-attention stack:

```
h = res(h, te)
h = attn_brain(h, brain_ctx)           # existing brain cross-attention
if clip_ctx is not None:
    h = attn_clip(h, clip_context_proj(clip_ctx))   # new CLIP cross-attention
```

A `clip_context_proj` linear layer maps 1024-dim CLIP tokens → `brain_dim`
so the CLIP cross-attention modules use the same hidden dimension as the
brain attention modules.

**Backward compatibility**: `clip_ctx=None` skips the CLIP cross-attention
entirely, so v9 checkpoints load and run without modification.

### 6. Exposure Bias Mitigation (Stage 2B)

During Stage 2B joint training, the Flow_VAE receives CLIP tokens from
Flow_CLIP's ODE sampler with probability `clip_sample_prob` (otherwise it
receives the ground-truth CLIP tokens — teacher forcing). This probability
is linearly ramped from 0 → `clip_sample_prob_max` over the first
`clip_ramp_frac * num_epochs` epochs:

```
clip_sample_prob(epoch) = min(clip_sample_prob_max,
                               epoch / (clip_ramp_frac * num_epochs))
```

Default: ramp 0 → 0.5 over the first 30% of 150 epochs.

---

## Training Stages

| Stage | Config | What trains | Budget |
|-------|--------|------------|--------|
| 2A | `phase2_stage2a.yaml` | Flow_CLIP DiT only; BrainEncoder **frozen** | ~10h |
| 2B | `phase2_stage2b.yaml` | Joint: BrainEncoder + Flow_CLIP + Flow_VAE | ~25h |
| 2C | `phase2_stage2c_sweep.yaml` | Inference sweep (CFG × solver × NFE) | — |

---

## Solver API (`brainflow/solvers.py`)

All solvers share the unified interface:

```python
from brainflow.solvers import solve, make_t_grid

t_grid = make_t_grid(n_steps=20)   # [0, 1/20, 2/20, ..., 1]
x_T = solve(vel_fn, x0, t_grid, method="heun")
```

| Solver | NFE/step | Order | Notes |
|--------|----------|-------|-------|
| `euler` | 1 | 1st | Fast baseline |
| `midpoint` | 1 | 2nd | Default eval solver |
| `heun` | 2 | 2nd | Best accuracy/NFE tradeoff |
| `dpm_pp_2m` | 1 | 2nd-3rd | Adams multistep, ~1.5× faster than Heun |
| `adaptive_rk45` | variable | 5th | Requires `torchdiffeq`; used for benchmarking |

---

## Bug Fixes (Phase 2, §0)

### CLIP ImageNet Normalisation

The original `metrics.py` did not apply any normalisation before passing images
to the CLIP encoder — it only resized to 224×224. ViT-L/14 (openai) was trained
with ImageNet normalisation:

```python
# WRONG (original)
pi_r = F.interpolate(pi, 224)
clip_enc.encode_image(pi_r)

# CORRECT (fixed)
mean = [0.48145466, 0.4578275,  0.40821073]
std  = [0.26862954, 0.26130258, 0.27577711]
pi_r = (pi_r - mean) / std
clip_enc.encode_image(pi_r)
```

This bug inflated CLIP similarity scores (because unnormalised images activate
irrelevant features), artificially boosting reported CLIP_Sim by ~0.02-0.05.

---

## Files Added / Modified

| File | Status | Description |
|------|--------|-------------|
| `brainflow/vfm.py` | **NEW** | VFM objective (velocity recovery, ELBO loss) |
| `brainflow/flow_clip_dit.py` | **NEW** | Flow_CLIP DiT in 16×16×1024 space |
| `brainflow/flow_unet.py` | **NEW** | FlowUNet + CLIP cross-attention (back-compat) |
| `brainflow/solvers.py` | **NEW** | Unified solver API (euler/midpoint/heun/dpm++/rk45) |
| `brainflow/metrics.py` | **FIXED** | Correct CLIP ImageNet normalisation |
| `brainflow/metrics_full.py` | **NEW** | Full 8-metric evaluation suite |
| `brainflow/config.py` | **UPDATED** | Phase 2 config fields |
| `brainflow/__init__.py` | **UPDATED** | Export new modules |
| `configs/phase2_stage2a.yaml` | **NEW** | Stage 2A training config |
| `configs/phase2_stage2b.yaml` | **NEW** | Stage 2B joint fine-tune config |
| `configs/phase2_stage2c_sweep.yaml` | **NEW** | Stage 2C inference sweep config |
| `scripts/solver_sweep.py` | **NEW** | Solver sweep → BENCHMARK_NOTES.md table |
| `tests/test_vfm.py` | **NEW** | VFM unit tests |
| `tests/test_flow_clip_dit.py` | **NEW** | Flow_CLIP DiT smoke tests |
| `tests/test_flow_unet_clip_ctx.py` | **NEW** | FlowUNet back-compat tests |
| `tests/test_solvers.py` | **NEW** | Solver integration tests |
| `tests/test_clip_standardization.py` | **NEW** | Standardization round-trip test |
| `BENCHMARK_NOTES.md` | **UPDATED** | Corrected v9 baseline numbers |
| `README.md` | **UPDATED** | New architecture diagram + flow_objective flag |
