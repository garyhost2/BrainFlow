# BrainFlow — step1c status & to-do

**Project.** fMRI → image reconstruction (NSD, subjects 1/2/5/7). Pipeline:
fMRI → brain tokens → **rectified-flow prior over ViT-bigG tokens** → SDXL-unCLIP (`unclip6`) decoder.
**Contribution (the paper):** the prior is a *spherical / oblique-manifold* RCFM (product-of-spheres over the
256 bigG token directions + radius head). Everything below the prior is standard MindEye2 machinery.

Branch `step1c-sphere-fm`. Prior checkpoint: `outputs/step1c_sphere/best.pt` (frozen reference for all
low-level experiments).

---

## Current numbers (subjects 1/2/5/7, honest)

| metric | our prior | GT-token ceiling | GT-blurry+img2img ceiling | MindEye2 SOTA |
|---|---|---|---|---|
| **CLIP-2way** | **~0.80** | — | — | ~0.87 |
| PixCorr | ~0.13 | 0.64 | 0.91 @ str 0.5 | 0.32 |
| SSIM | ~0.22 | 0.25 | 0.48 @ str 0.5 | 0.43 |
| token_cos | ~0.37 | 1.0 | — | — |

Reading: **CLIP is competitive.** PixCorr/SSIM are prior-limited (token_cos 0.37 ≪ 1) *and* lack a working
low-level pathway. The decoder itself is not the bottleneck (GT tokens → 0.64 PixCorr; GT-blurry init → 0.91).

---

## Done

- [x] Spherical/oblique-manifold RCFM prior + radius head; two-head (CLS + patch); geodesic Heun sampler.
- [x] Stable training, leakage-free val selection (`--val-frac`), honest full 982-img eval.
- [x] **Sphere beats Euclidean** on CLIP-2way at matched epochs *and* is NaN-stable where Euclidean collapses.
- [x] **Decoder-ceiling gate (G0):** GT tokens → PixCorr 0.64 / SSIM 0.25; GT-blurry img2img → 0.91 / 0.48.
      ⇒ headroom is real; the ceiling is the *prior* and the *missing low-level path*, not the decoder.
- [x] **Low-level pathway investigation (negative result, clean):** MindEye2-style blurry-image → img2img.
      Three heads — mean-pool tokens, spatial-grid tokens, raw-voxel encoder — **all** plateaued at
      `blur_mse ≈ 0.07` (= predicting the mean colour), no PixCorr gain. Isolated the cause to the
      **loss + target** (MSE is mean-seeking; a 128²·3 = 49k-dim RGB target is unfittable from voxels).
- [x] `--freeze-token` warm-start (train only the low head, prior frozen → CLIP protected).

## In progress / next (ranked)

- [ ] **#1 — Sampling sweep (FREE, run first).** We only ever measured PixCorr at `cond_source=prior`, the
      *worst* setting for it. `regression` + low CFG trades CLIP for PixCorr (standard).
      `CKPT=outputs/step1c_sphere/best.pt sbatch slurm/sweep_step1b.sbatch` → read PixCorr-optimal config.
      *Expected: PixCorr 0.13 → ~0.18–0.25, free.* (Both earlier sweeps crashed at startup — never seen.)
- [ ] **#2a — Low head, learnable version (BUILT, commit `f73fb0f`).** L1 loss (not MSE) + 64² target.
      CPU test proves it now overfits two distinct layouts (the mean-collapse could not).
      Launch: `TRAIN_EXTRA="--init-from outputs/step1c_sphere/best.pt --freeze-token --ll-loss l1 --ll-size 64
      --ll-strength 0.6" OUT=outputs/step1c_ll_l1 EVAL_N=64 sbatch slurm/step1b_full_a100.sbatch`.
      **Decision:** watch the training `low` loss in the first ~10 epochs — if it *drops* (vs the old flat
      ~0.065), the head is finally learning layout → check Ep10 `blur_mse`/PixCorr. If still flat → **#2b**.
- [ ] **#2b — Full SD-VAE-latent low head (escalation, only if #2a underfits).** Predict the *VAE latent* of a
      blurred image, not raw RGB (this is exactly MindEye2's mechanism and why theirs works). Spec:
      head outputs a 4×96×96 latent (no sigmoid); target = `VAE.encode(blur(GT))` (precompute per subject,
      or on-the-fly + RAM memo keyed by trial); loss = L1 in latent space; decoder gets `init_latent` directly
      (skip the RGB re-encode). **Gate with a smoke first** (à la `smoke_lowlevel`: GT-latent → img2img on
      N=8) before any 150-epoch run. Realistic target PixCorr ~0.20–0.30.
- [ ] **#3 — Stronger prior (BUILT via existing flag).** enc_hidden 2048 → 4096 (MindEye2 capacity) lifts
      token_cos → raises *both* PixCorr and CLIP toward the 0.64 ceiling. Fresh prior retrain:
      `TRAIN_EXTRA="--enc-hidden 4096 --no-low-level" OUT=outputs/step1c_sphere_4096 sbatch
      slurm/step1b_full_a100.sbatch` (drop `--batch-size 32` if it OOMs).

## Open / eval-correctness (M0 — needed before any leaderboard claim)

- [ ] Repeat-averaging over the 3 test presentations; 300-way retrieval metric.
- [ ] EffNet/SwAV as `1 − corr` distance (currently coded as similarity).
- [ ] Wire `metrics_full.evaluate_full`; report all 8 NSD metrics on the full 982.

## Operational notes

- Home NFS is near-full (was 96%). Each run writes ~14 GB; `rm -rf` superseded `outputs/step1c_ll_*` dirs,
  but **keep `outputs/step1c_sphere`** (the `--init-from` prior). Preflight needs ~free ≥ NEED+20 GB.
- Trains on QCRI Panther (A100-only). `squeue` is unreliable here — use `sacct -j <id> --format=State`.
