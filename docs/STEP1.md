# Step 1 — fMRI → CLIP embedding → frozen unCLIP decoder

The SOTA brain-decoding recipe: **do not generate pixels from scratch.** Predict
the conditioning embedding a *pretrained* unCLIP decoder already knows how to
render, and let that decoder draw the picture. This is what MindEye2 /
Brain-Diffuser / Brain-IT all do; the from-scratch FlowUNet path cannot match it.

```
fMRI ──Backbone──▶ brain features
                      ├── reg head ──▶ point estimate of CLIP embedding (clear-image baseline)
                      └── FlowPrior ─▶ refined, in-distribution embedding (your contribution)
                                          │ un-standardize
                                          ▼
                     frozen SD-2.1-unCLIP (image_embeds=) ──▶ image
```

**Target = OpenCLIP ViT-H/14 *pooled* embedding (1024-d)**, the space
`stabilityai/stable-diffusion-2-1-unclip` ingests directly.

## Why this gets clear images + strong CLIP
The unCLIP decoder is trained to *invert* CLIP: `decode(e)` yields an image whose
re-encoded CLIP embedding ≈ `e`. So if the predicted embedding is close to the
true one, `CLIP(pred) ≈ CLIP(gt)` — the decoder closes the loop in CLIP space.
The only thing you have to learn is the (low-dimensional, learnable) fMRI→CLIP map.

## Correctness details handled (verified against library source)
- **Targets extracted with the decoder's own `image_encoder`** so they match its
  internal `image_normalizer`. Embeddings are passed **raw** (no L2-norm, no
  pre-standardization) — the pipeline standardizes internally; pre-normalizing
  would double-apply. (`targets.py`, `decoder.py`)
- **Flow matching in a standardized space** (train mean/std), un-standardized
  before decode — well-conditioned training, correct decoder input.
- **v-prediction**, **logit-normal t sampling** with **uniform loss weight** (SD3).
- **CFG** via condition-dropout + learned null condition.
- **bf16 autocast, no GradScaler** (A100); TF32 on; fused AdamW; `set_to_none`;
  **EMA over float params only**, swapped in-place (torch.compile-safe).
- **VAE upcast to fp32** to avoid the known fp16 decode NaN/black-frame bug.

## Run

Prereqs: the existing tensor cache `mindeyev2_cache/all_subjects_tensors.pt`
(built by the normal data pipeline), plus `diffusers`, `transformers`,
`open_clip_torch`, `torchvision`.

```bash
# 1) Train (subject 1 first; embedding cache builds automatically on first run)
python -m scripts.train_step1 \
    --data-dir ./mindeyev2_cache --subjects 1 \
    --epochs 150 --batch-size 256 --lr 3e-4 --out outputs/step1

# add --decode-eval to also decode a 64-image subset every 5 epochs (slow but
# shows real PixCorr/SSIM/CLIP during training)

# 2) Full evaluation (decode the whole test set, save grid + metrics.json)
python -m scripts.eval_step1 \
    --ckpt outputs/step1/best_cos.pt --data-dir ./mindeyev2_cache \
    --cond-source prior --cfg-scale 3.0 --steps 50 \
    --decode-steps 25 --guidance 10 --out outputs/step1/eval
```

`--cond-source`:
- `regression` — decode the point estimate. **Fastest path to clear images**;
  run this first to confirm the decoder + pipeline work end-to-end.
- `prior` — decode the flow-prior sample (the refined, in-distribution embedding).
- `blend` — weighted mix (`blend_w`).

## Expected trajectory
- `regression` alone should already produce **clear, semantically correct images**
  (this is the Brain-Diffuser result: ridge→VD ≈ 90% CLIP). If it doesn't, the
  decoder/target wiring is wrong — debug there before touching the prior.
- The flow `prior` should match or beat `regression` on CLIP by emitting sharper,
  in-distribution embeddings — this is the piece that becomes your paper.

## Upgrade path to max CLIP (Step 1b — beat MindEye2)
Swap the **target + decoder** to the MindEye2 stack:
- Target: OpenCLIP **ViT-bigG/14**, the **256×1664 token grid** (`output_tokens=True`).
- Decoder: MindEye2's **SDXL-unCLIP** (`unclip6_epoch0_step110000.ckpt`, requires
  Stability's `sgm` codebase — not `diffusers`).
- Make `FlowPrior` emit a token *sequence* (add positional embeddings + a few
  self-attention layers) instead of a single vector.

The backbone, training loop, EMA, metrics, and CFG/flow math here all carry over
unchanged — only the embedding dim/shape and the decoder call change.

## Low-level branch (for PixCorr/SSIM) — Step 1c
A pooled-embedding decoder discards spatial detail, so PixCorr/SSIM stay modest.
Add a head that predicts a blurry SD-VAE latent → init image → img2img init for
the decoder (MindEye2's `enhanced_recon` / Brain-Diffuser's VDVAE init).
