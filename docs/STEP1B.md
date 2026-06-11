# Step 1b — bigG token grid + SDXL-unCLIP decoder (the MindEye2-competitive path)

Step 1 (`docs/STEP1.md`) proves the pipeline with a clean diffusers decoder.
**Step 1b swaps in MindEye2's actual decoder** so the CLIP number is competitive:

| | Step 1 | Step 1b |
|---|---|---|
| Target | ViT-H/14 **pooled** (1024-d vector) | ViT-bigG/14 **token grid** (256×1664) |
| Prior | MLP rectified flow | **DiT** rectified flow over 256 tokens |
| Decoder | SD-2.1-unCLIP (`diffusers`) | **SDXL-unCLIP** (MindEye2's `sgm` engine) |
| CLIP ceiling | moderate | **MindEye2-class** |

Everything carries over from Step 1 — backbone, flow math (v-pred, logit-normal t,
CFG), EMA, metrics — only the target shape, the prior (DiT), and the decoder change.

## One-time setup
```bash
bash scripts/setup_step1b.sh
```
This (1) clones `MedARC-AI/MindEyeV2` into `third_party/` for the **vendored `sgm`**
and `unclip6.yaml`, (2) installs the pinned deps (`open_clip_torch==2.24.0`,
`kornia==0.7.1`, `omegaconf==2.3.0`, `pytorch-lightning==2.0.1`, …), and (3)
downloads the decoder checkpoint `unclip6_epoch0_step110000.ckpt` (~5 GB) from
`pscotti/mindeyev2`.

## Train + evaluate
```bash
# Train (subject 1). The bigG target cache builds automatically on first run.
python -m scripts.train_step1b \
    --data-dir ./mindeyev2_cache --subjects 1 \
    --mindeye-src third_party/MindEyeV2/src \
    --ckpt-path third_party/unclip6_epoch0_step110000.ckpt \
    --epochs 150 --batch-size 48 --lr 3e-4 --out outputs/step1b

# Full eval: decode the test set through SDXL-unCLIP, save grid + metrics.json
python -m scripts.eval_step1b \
    --ckpt outputs/step1b/best_cos.pt --data-dir ./mindeyev2_cache \
    --mindeye-src third_party/MindEyeV2/src \
    --ckpt-path third_party/unclip6_epoch0_step110000.ckpt \
    --cond-source prior --cfg-scale 3.0 --steps 50 --out outputs/step1b/eval
```

Start with `--cond-source regression` to confirm clear images end-to-end (this is
the ridge→decoder baseline), then switch to `prior` for the flow-refined tokens.

## Correctness details baked in (verified verbatim against MindEye2 source)
- **Targets extracted with the vendored `sgm` bigG embedder itself** — exact space
  the decoder was trained on. The embedder does its **own** preprocessing
  (resize 224 bicubic, `(x+1)/2`, CLIP-normalize); we feed `[0,1]` images directly
  as MindEye2 does — **do not pre-normalize**.
- **`unclip_recon` is a verbatim port**: latent `(B,4,96,96)`, scale 0.13025,
  EulerEDMSampler `num_steps=38`, VanillaCFG `scale=5.0`, offset-noise 0.04, the
  **random-token** unconditional branch, `clamp(samples_x*0.8+0.2, 0, 1)`.
- **`vector_suffix`** = `conditioner(dummy jpg batch)["vector"]` (size/crop
  embedding, constant across images) — exactly as MindEye2 builds it.
- DiffusionEngine built from the individual configs (no `ckpt_config`); weights
  come entirely from `ckpt["state_dict"]`. `first_stage` target overridden to
  `sgm.models.autoencoder.AutoencoderKL`.

## Resource notes
- **bigG target cache is large**: ~`N×256×1664×2 B` ≈ **7–8 GB fp16 per subject**
  (train). Start with one subject; for all 8 you need ~70 GB disk.
- **Decoding is slow**: 38 diffusion steps at 768² per image, one at a time. Use
  `--decode-n 16` during training and a capped `--max-images` for eval sweeps.
- Batch 48 fits comfortably on an A100 80GB; the (B,256,1664) targets dominate
  memory. Reduce to 32 if you scale the prior up.

## This is your ICLR contribution
The DiT **flow prior** (`TokenFlowPrior`) is the part that replaces MindEye2's DDPM
diffusion prior. To make the *variational / one-to-many* story from earlier
(diversity + calibration + uncertainty-gated decoding), extend the prior to emit a
posterior (predict per-token σ) and sample multiple plausible token grids per fMRI —
the decoder then renders diverse, calibrated reconstructions that point-estimate
methods structurally cannot produce.

## Low-level branch (PixCorr/SSIM) — still to add
The token decoder is semantic; for PixCorr/SSIM add MindEye2's blurry-init path:
predict an SD-VAE latent, decode a blurry image, and start `unclip_recon` from a
noised version of it (img2img) instead of pure noise (`z` in `_recon_one`).
