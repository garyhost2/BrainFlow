# BrainFlow — step1c status & to-do

**Project.** fMRI → image reconstruction (NSD, subjects 1/2/5/7). Pipeline:
fMRI → brain tokens → **rectified-flow prior over ViT-bigG tokens** → SDXL-unCLIP (`unclip6`) decoder.
**Contribution (the paper):** the prior is a *spherical / oblique-manifold* RCFM (product-of-spheres over the
256 bigG token directions + radius head). Everything below the prior is standard MindEye2 machinery.

Branch `step1c-sphere-fm`. Prior checkpoint: `outputs/step1c_sphere/best.pt` (frozen reference for all
low-level experiments).

---

## Current numbers — full 982-image eval, subject 1, trained on 1/2/5/7

Every row is a *decode config* on the same checkpoint (`step1c_sphere/best.pt`, job `347832`).
There is no single operating point that is good at both ends.

| config | PixCorr | SSIM | CLIP-2way |
|---|---|---|---|
| regression, cfg 1.0 | **0.164** | **0.348** | 0.703 |
| blend (w=0.5), cfg 1.0 | 0.131 | 0.276 | **0.738** |
| MindEye2 | 0.322 | 0.431 | 0.930 |
| Takagi & Nishimoto 2023 | 0.246 | 0.410 | 0.821 |

Ceilings (job `331952`, GT inputs): GT tokens → PixCorr 0.64 / SSIM 0.25; GT-blurry img2img @ str 0.5 →
0.91 / 0.48. So the decoder is not the bottleneck — the prior is (token_cos ~0.37 ≪ 1), and the low-level
pathway never worked.

**Honest standing: below Takagi on all three axes at a single operating point.** MindEye2 wins both ends at
once because it *fuses* a high-level and a low-level pathway; we only have the high-level one.

### Runs that did NOT help (2026-07-21/22, 33.5 GPU-h)

- `347830` low-level retry #2a (voxel head, L1, ll_size 64, frozen prior) → 0.142 / 0.220 / 0.696.
  Strictly dominated by the free `regression` config above. `blur_mse` *rose* monotonically
  0.0833 → 0.1044 over 150 epochs: the 137M-param head overfits. Voxels do carry the signal (unlike the
  frozen-backbone attempts, which could not fit even the train set) — it just does not generalise.
- `347831` enc_hidden 4096, no low-level → 0.091 / 0.233 / 0.748, and **best.pt is epoch 10 of 150**:
  val_cos falls monotonically 0.3485 → 0.3053 from the first eval. **Width is the wrong lever; the model
  is data-starved.**

---

## Done

- [x] Spherical/oblique-manifold RCFM prior + radius head; two-head (CLS + patch); geodesic Heun sampler.
- [x] Stable training, leakage-free val selection (`--val-frac`), honest full 982-img eval.
- [x] **Sphere beats Euclidean** on CLIP-2way at matched epochs *and* is NaN-stable where Euclidean collapses.
- [x] **Decoder-ceiling gate (G0):** GT tokens → PixCorr 0.64 / SSIM 0.25; GT-blurry img2img → 0.91 / 0.48.
      ⇒ headroom is real; the ceiling is the *prior* and the *missing low-level path*, not the decoder.
- [x] **Low-level pathway investigation (negative result, clean):** MindEye2-style blurry-image → img2img.
      Four heads — mean-pool tokens, spatial-grid tokens, raw-voxel encoder, and the L1 / 64² retry
      (`347830`) — none produced a PixCorr gain. The first three plateaued at `blur_mse ≈ 0.07` (= predicting
      the mean colour) because MSE is mean-seeking on a 128²·3 = 49k-dim target; the fourth fixed that and
      then **overfit** instead (`blur_mse` rising 0.083 → 0.104 while val PixCorr stayed flat). Underfit *and*
      overfit both fail ⇒ the RGB-blur target itself is the wrong intermediate. Escalation is #6 (VAE latent).
- [x] `--freeze-token` warm-start (train only the low head, prior frozen → CLIP protected).

## In progress / next (ranked)

- [ ] **#1 — Evaluate `step1c_sphere_4096/last.pt` (FREE, ~1.2 h).** In `347831`, `best.pt` was picked on
      `sel_cos`, which fell from epoch 10 — while the decoded CLIP-2way *rose* to 0.824 at epoch 110.
      `last.pt` (epoch 150) was saved every eval epoch and has never been scored.
      `CKPT=outputs/step1c_sphere_4096/last.pt COND_SOURCES="regression blend" N_IMAGES=982 sbatch slurm/sweep_step1b.sbatch`
- [ ] **#2 — Single-subject control (decisive, ~6 h).** Every step1c run is a 4-subject joint model at
      0.70–0.75 CLIP; the old step1b *single*-subject baseline hit ~0.82. With naive per-subject `input_proj`
      + a shared trunk (no SubjectResidualAdapter — deferred as M3/M8), extra subjects may be *hurting*.
      `SUBJECTS="1" TRAIN_EXTRA="--no-low-level" OUT=outputs/step1c_sphere_s1 sbatch slurm/step1b_full_a100.sbatch`
      **Decision:** s1-only ≥ 0.80 CLIP ⇒ multi-subject sharing is the bug, and the adapter refactor becomes
      the top priority. s1-only ≈ 0.75 ⇒ subject count is not the problem; go to #3.
- [ ] **#3 — BiMixCo (BUILT, `--mixup-pct 0.33`).** The measured failure in `347831` is overfitting from the
      first eval, and the only augmentation we had was Gaussian voxel noise. MixCo mixes voxel patterns and
      swaps the contrastive target for the matching two-hot distribution (MindEye2's schedule: first ⅓ of
      epochs, then clean SoftCLIP). Flow/regression/radius targets stay unmixed — a slerp between two images'
      token sequences is not a valid point on the target manifold.
      `TRAIN_EXTRA="--mixup-pct 0.33 --no-low-level" OUT=outputs/step1c_sphere_mixco sbatch slurm/step1b_full_a100.sbatch`
- [ ] **#4 — Widen the decode grid.** `blend_w` was hardcoded at 0.5 and never swept; it is the direct
      PixCorr↔CLIP dial. `prior` and cfg > 1.0 are still unmeasured at 982.
      `BLEND_WS="0.25 0.5 0.75" COND_SOURCES="blend" N_IMAGES=982 sbatch slurm/sweep_step1b.sbatch`
- [ ] **#5 — Regularisation sweep (now exposed).** `--enc-drop` (was hardcoded 0.15), `--weight-decay`,
      `--fmri-noise-std`, plus `--lambda-cos` / `--lambda-reg`. Cheaper than any architecture change.
- [ ] **#6 — SD-VAE-latent low head (BUILT 2026-08-07, `--ll-target latent`).** Predicts `E(blur(GT))` in
      R^{4×96×96} — the img2img init the sampler actually starts from — instead of a blurry RGB image that
      then has to survive a sigmoid and a VAE re-encode. `LatentLowLevelHead` is linear (no sigmoid: latents
      are unbounded and ~unit-Gaussian per channel), the target is per-channel standardised from train-split
      statistics, and `decoder.decode(init_latent=...)` takes it directly. Latents are cached per subject
      (~74 KB/image fp16 ≈ 2 GB/subject) so the VAE never runs during training.

      Useful property: with a zero-init head the L1 starts at `E|N(0,1)| = 0.798`, which **is** the
      "predicted the channel mean" score. So the logged `low` is self-calibrating — below 0.798 the head has
      learned layout, at 0.798 it has collapsed. The four RGB attempts needed an external reference
      (`blur_mse ≈ 0.07`) to see this at all.

      **Order of operations — the smoke gate is not optional:**
      1. `SUBJECT=1 N=16 sbatch slurm/smoke_latent.sbatch`
      2. `python -m scripts.build_latent_targets --subjects 1 2 3 4 5 6 7 8 --ll-size 64` (needs the A100)
      3. `TRAIN_EXTRA="--init-from <prior>/best.pt --freeze-token --ll-target latent --ll-strength <from smoke>"`

      The smoke runs three arms and prints two gate lines. `latent` must match `rgb` (same blur, so
      disagreement means the new code path is wrong). More importantly it adds an arm the earlier smokes
      never had: **`mean`**, the constant per-channel mean latent — what a *collapsed* head predicts. Four
      low heads have now collapsed toward the mean, so if `latent − mean` is small, most of the img2img
      PixCorr gain is "any smooth init" rather than decoded brain signal, and a PixCorr number obtained that
      way would not mean what it appears to mean. `latent − mean` is the honest headroom, not `latent`.

## Eval correctness (M0)

- [x] Repeat-averaging over the 3 test presentations — **already correct**: `brainflow/data.py:160` averages
      all repeats per unique COCO id (982 unique), and `config.yaml` sets `max_test: 999999` so nothing is
      truncated before averaging.
- [x] EffNet/SwAV as `1 − corr` **distance** (were coded as cosine similarity — inverted vs every published
      number). Now `metrics_full.correlation_distance`; `LOWER_IS_BETTER` names them.
- [x] 2-way identification now uses **Pearson** correlation (the `np.corrcoef` convention), not raw cosine.
- [x] SSIM now matches the leaderboard definition: **grayscale** (`rgb2gray`, Rec. 709) and a **valid**
      window, mirroring skimage's border crop. The old per-channel zero-padded number is still reported as
      `legacy_SSIM` so historical runs stay comparable.
- [x] `metrics_full.evaluate_full` wired into `eval_step1b` **and** `sweep_step1b` — all 8 NSD metrics on the
      full 982, printed next to the published table (`--no-full-metrics` opts out).
- [x] 300-way image/brain retrieval (`retrieval_fwd` / `retrieval_bwd`), computed on the predicted embedding,
      so it needs no diffusion pass. Short trailing pools are dropped rather than scored with fewer foils.
- [ ] Report per-subject rather than subject-1-only (`--max-images 982` currently evaluates subject 1; eval
      warns if several subjects get pooled, which would put the *same image* in the foil set).

## Checkpoint selection (fixed 2026-08-06)

`best_clip2way.pt` used to be written only when `--val-frac 0`, so every recent run — all of which set
`--val-frac` — produced **no CLIP-selected checkpoint at all**, while `best.pt` tracked `sel_cos`, a signal
that in `347831` moved *opposite* to decoded CLIP. Now both are written, both selected on the leakage-free
VAL split (the per-epoch decode moved off TEST), and `SELECT=clip|cos|last` picks which one stage 2 reports.

## Operational notes

- Home NFS is near-full (was 96%). Each run writes ~14 GB; `rm -rf` superseded `outputs/step1c_ll_*` dirs,
  but **keep `outputs/step1c_sphere`** (the `--init-from` prior). Preflight needs ~free ≥ NEED+20 GB.
- Trains on QCRI Panther (A100-only). `squeue` is unreliable here — use `sacct -j <id> --format=State`.
