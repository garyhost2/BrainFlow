# BrainFlow Benchmark Notes

## 1. HRF Source Collapse — Root Cause and Fixes

### Failure Mode

In the `v5-multisubject-mb_hrf_full` run, training loss `cfm` dropped
`0.70 → 0.095` over 5 epochs while `PixCorr` *fell* from `0.097 → 0.041`.
The model was solving a degenerate objective.

### Root Cause

The HRF method's `_source_from_cls` had three simultaneous failures:

1. **No gradient detach on `cls`**: The encoder could drive
   `cls_to_latent(enc(fmri)) → latent`, making `x0 ≈ latent` and `v ≈ 0`.
   CFM loss → 0 trivially, but the flow learns nothing useful.
2. **Tiny source noise (0.1×)**: The source distribution was effectively a
   delta mass around `proj(cls)` rather than a proper distribution. No flow
   was needed to integrate over a trivial trajectory.
3. **Non-causal HRF kernel**: Symmetric padding in `_hrf_bias` leaked "future"
   tokens into "past" — physiologically incorrect and an additional cheat path.
4. **Mixup contamination**: Mixed fMRI → mixed `x0`, but latent target
   unaffected → 30% of steps had inconsistent `(x0, latent)` pairs.

### Fixes Applied (B.1–B.3, B.6)

```python
# B.1: Detach cls + unit-variance noise
def _source_from_cls(self, cls):
    cls_d = cls.detach()           # stop encoder from collapsing source
    proj = self.cls_to_latent(cls_d).view(...)
    return proj + torch.randn_like(proj)   # unit-variance (was 0.1×)

# B.2: Non-zero init for cls_to_latent
nn.init.normal_(self.cls_to_latent.weight, std=0.02)
nn.init.zeros_(self.cls_to_latent.bias)

# B.3: Causal HRF padding (left-only)
x = F.pad(x, (kernel.shape[-1] - 1, 0))
x = F.conv1d(x, kernel, groups=D)

# B.6: Disable mixup for HRF
if self.training and cfg.mixup_alpha > 0 and method != "hrf" and ...:
    ...  # mixup only for non-HRF methods
```

**Expected outcome after fixes**: At epoch 5, `cfm` should sit in `[0.30, 0.55]`
(not `< 0.10`). PC should be monotonically increasing for the first 20 epochs,
exceeding `0.12` by epoch 30.

**Regression indicator**: If `cfm < 0.1` while PC stalls after these fixes, the
source is still degenerate — bump noise to `1.5×` or add a KL/variance penalty
on `proj(cls)`.

---

## 2. The v9 Recipe (Phase 1+2 SOTA Push)

**Motivation**: The baseline MC experiment (`brainflow_mc_xl_v100_32_282325.out`) plateaued
at epoch 45 with PC=0.090, SSIM=0.334, CLIP=0.740 — far below SOTA (~0.35, ~0.42, ~0.94).
The audit identified seven root causes, all addressed in v9.

**Target**: PC ~0.18–0.22, SSIM ~0.38–0.42, CLIP_Sim ~0.85–0.90 (after 60 epochs).

### Root Causes Fixed

1. **CLIP prior disabled** (`use_clip_prior: false`) — no mechanism forces semantic alignment.
   V9 enables `CLIPPriorHead`, the single most impactful change (+0.08–0.15 CLIP expected).
2. **InfoNCE too weak vs CFM** (λ_align=0.2 vs λ_cfm=1.0) — 9:1 gradient imbalance meant
   the model learned pixel texture before semantics. V9 sets λ_align=0.8, λ_cfm=0.5.
3. **Only 16 brain tokens** — severe representational bottleneck. V9 uses 64 tokens (4×).
4. **CFG scale too low** (`cfg_scale=2.0`) — SOTA uses 5–10. V9 sets 6.0.
5. **Eval uses 10-step Euler** — systematically underestimates all metrics. V9 uses
   25-step midpoint solver for accurate benchmarking.
6. **Perceptual loss disabled** — no pixel-quality gradient. V9 adds LPIPS (λ=0.15).
7. **InfoNCE temperature too warm** (`infonce_temp=0.07`) — V9 uses 0.04 for harder negatives.

### Configuration

```bash
USE_V9=1 ./launch_experiment.sh v9
# or:
sbatch slurm/train_mc_xl_v9_v100_32gb.sbatch
```

### Expected Metric Trajectory vs Baseline

| Epoch | Baseline PC | V9 PC (est.) | Baseline CLIP | V9 CLIP (est.) |
|-------|-------------|--------------|---------------|----------------|
| 15 | 0.091 | ~0.10–0.13 | 0.731 | ~0.79–0.83 |
| 30 | 0.088 | ~0.14–0.17 | 0.738 | ~0.83–0.87 |
| 45 | 0.087 | ~0.16–0.20 | 0.740 | ~0.85–0.89 |
| 60 | — | ~0.18–0.22 | — | ~0.85–0.90 |

The CLIP gain is driven primarily by enabling `CLIPPriorHead` and raising `cfg_scale`.
The PC/SSIM gain is driven by rebalancing λ_align/λ_cfm, adding LPIPS, and 64 tokens.

---

## 3. The v8 Recipe (MindEye-v2 Level Quality)

### Configuration

```bash
USE_V8=1 ./launch_experiment.sh v8
```

This sets:
- `n_tokens = 32` (2× baseline — richer brain representation)
- `use_clip_prior = True` — enables `CLIPPriorHead` (two-stage brain→CLIP prior)
- `lambda_prior = 0.3` — supervision on predicted CLIP embedding
- `lambda_pixel = 0.3` — pixel-space L1 loss (only applied for `t > 0.85` steps)
- `lambda_align = 0.2` — balanced InfoNCE alignment
- `method = "baseline"` — standard Gaussian source (simpler than HRF)

### CLIPPriorHead Architecture (C.1)

```python
class CLIPPriorHead(nn.Module):
    """Two-stage brain→CLIP prior head."""
    def __init__(self, brain_dim, clip_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(brain_dim, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(1024, 1024), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(1024, clip_dim),
        )
    def forward(self, cls):
        return F.normalize(self.net(cls), dim=-1)
```

The predicted CLIP embedding is projected back to `brain_dim` and appended as a
**single extra token** to the cross-attention context (not broadcast over all N
tokens — that would erase brain signal). At inference, the head's predicted
embedding is used (not ground-truth CLIP).

### Pixel-Space L1 Loss (C.2)

Applied only when `t > 0.85`, where the velocity approximation
`pred_lat ≈ xt + vt*(1-t)` is valid. This avoids training on grossly wrong
"predicted images" during mid-trajectory steps.

---

## 4. Checkpoint Migration (per-subject input_proj → shared Linear)

Old checkpoints used a `nn.ModuleDict` with one `Linear(voxels_s, enc_hidden)`
per subject. The new architecture uses a single `Linear(max_vox, enc_hidden)`
with zero-padding for shorter subjects.

### Why the change?

With 8 subjects × ~15k voxels × 1024 hidden ≈ 120M parameters, 7/8 receive
zero gradient every step but are still allreduced by DDP — wasting ~100M
all-reduce bandwidth per step.

### Migration helper

```python
from brainflow.models import migrate_input_proj, BrainFlowV5
import torch

cfg = load_config()
model = BrainFlowV5(cfg, voxels)
old_sd = torch.load("outputs/old_checkpoint.pt")
new_sd = migrate_input_proj(old_sd, model.brain_enc.max_vox)
model.load_state_dict(new_sd, strict=False)
```

**Behaviour**:
- If the checkpoint already has `brain_enc.input_proj.weight` (new format): no-op.
- If it has `brain_enc.input_proj.<sid>.weight` (old format): the largest
  subject's weight is zero-padded to `max_vox` and used to initialise the new
  shared projection. Old per-subject keys are removed.
- A warning is always printed. A hard crash is never raised.

---

## 5. Throughput Improvements Summary

| Change | File | Expected Speedup |
|--------|------|-----------------|
| A.1: No per-step `.item()` syncs | `scripts/train.py` | 20-40% |
| A.2: bs=96, grad_accum=1 (halve iterations) | `config.yaml` | ~2× |
| A.4: Shared input_proj (no wasted DDP allreduce) | `brainflow/models.py` | 15-30% (multi-GPU) |
| A.5: `torch.compile(flow_unet)` | `scripts/train.py` | 10-25% |
| A.6: `eval_batches=32` | `config.yaml` | Eval 20× faster |
| A.7: 10-step Euler eval | `config.yaml` | Eval 2× faster |
| A.8: `OMP_NUM_THREADS=1`, matmul precision, WandB offline | multiple | Minor |

**Overall target**: ≤ 30 min/epoch on 2× V100 32GB (was ~84 min).

---

## 6. Regression Guard

The following observations indicate a degenerate training run:

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `cfm < 0.1` by epoch 5 while PC stalls | Source collapse (B.1 not applied) | Ensure `cls.detach()` in `_source_from_cls` |
| PC monotonically decreasing | Mixup corrupting HRF (B.6 not applied) | Set `mixup_alpha=0.0` or add `method != "hrf"` guard |
| `cfm` still low after B.1-B.3 | Source noise too small | Bump to `1.5 * torch.randn_like(proj)` |
| Source collapse warning printed | `(latent - x0).mse < 0.05` for >50% batch | Check `cls_to_latent` init and noise scale |
