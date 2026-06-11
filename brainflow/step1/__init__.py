"""Step 1 — fMRI → CLIP image embedding → frozen unCLIP decoder.

A clean, math-correct, A100-optimized reimplementation of the *winning* brain
decoding recipe (MindEye2 / Brain-Diffuser family): do NOT generate pixels from
scratch.  Instead predict the conditioning embedding a *pretrained* unCLIP
decoder already knows how to render, and let that decoder draw the picture.

Pipeline
--------
    fMRI ──Backbone──▶ brain features
                          ├──reg head──▶ point estimate of CLIP embedding
                          └──FlowPrior──▶ refined, in-distribution embedding
                                              │ (un-standardize)
                                              ▼
                         frozen SD-2.1-unCLIP (image_embeds=) ──▶ image

The decoder target is the OpenCLIP ViT-H/14 *pooled* image embedding (1024-d),
which `stabilityai/stable-diffusion-2-1-unclip` ingests directly.  Everything is
swappable to the bigG / SDXL-unCLIP target for higher CLIP later.
"""

from .model import Backbone, FlowPrior, Step1Model
from .targets import build_or_load_targets, TargetStats

__all__ = ["Backbone", "FlowPrior", "Step1Model", "build_or_load_targets", "TargetStats"]
