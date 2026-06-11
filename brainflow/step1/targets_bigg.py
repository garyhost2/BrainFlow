"""bigG token-grid targets (Step 1b) — extracted with MindEye2's *exact* embedder.

We use the vendored sgm `FrozenOpenCLIPImageEmbedder` (ViT-bigG/14,
laion2b_s39b_b160k, output_tokens=True, only_tokens=True) so the (256, 1664)
token grid lands in exactly the space the SDXL-unCLIP decoder was trained on.

CRITICAL (verified against sgm source): the embedder does its OWN preprocessing
(resize 224 bicubic, then `(x+1)/2`, then CLIP-normalize).  MindEye2 feeds images
in [0,1] directly — so do we (do NOT pre-normalize).  The `(x+1)/2` maps [0,1] ->
[0.5,1]; this is the (quirky but self-consistent) distribution the decoder learned.
"""
from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .targets import TargetStats

TOKEN_LEN = 256
TOKEN_DIM = 1664


def _add_sgm_to_path(mindeye_src: Path):
    mindeye_src = Path(mindeye_src)
    gm = mindeye_src / "generative_models"
    if not gm.exists():
        raise FileNotFoundError(
            f"Vendored sgm not found at {gm}. Run scripts/setup_step1b.sh first "
            f"(it clones MedARC-AI/MindEyeV2) or set --mindeye-src to its `src/` dir.")
    for p in (str(mindeye_src), str(gm)):
        if p not in sys.path:
            sys.path.insert(0, p)


@torch.no_grad()
def _load_embedder(device, mindeye_src: Path):
    _add_sgm_to_path(mindeye_src)
    from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder
    emb = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14", version="laion2b_s39b_b160k",
        output_tokens=True, only_tokens=True,
    )
    emb = emb.eval().to(device)
    for p in emb.parameters():
        p.requires_grad_(False)
    return emb


@torch.no_grad()
def _encode(emb, imgs, device, bs=16):
    """imgs (N,3,H,W) uint8/[0,1] -> (N,256,1664) fp16. Embedder preprocesses internally."""
    outs = []
    for i in tqdm(range(0, len(imgs), bs), desc="bigG tokens", leave=False):
        x = imgs[i:i + bs]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        x = x.clamp(0, 1).to(device)
        tok = emb(x)                       # (B, 256, 1664) — embedder does resize+normalize
        outs.append(tok.float().cpu().half())
    return torch.cat(outs)


def build_or_load_bigg_targets(tensors, subjects, cache_path, device, mindeye_src,
                               hf_cache=None, force_rebuild=False):
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_rebuild:
        print(f"✓ Loading bigG target cache: {cache_path}")
        blob = torch.load(cache_path, map_location="cpu")
        blob["_stats"] = TargetStats.from_dict(blob["_stats"])
        return blob

    emb = _load_embedder(device, mindeye_src)
    out = {}
    cnt = 0
    s_sum = torch.zeros(TOKEN_DIM, dtype=torch.float64)
    s_sqsum = torch.zeros(TOKEN_DIM, dtype=torch.float64)
    for subj in subjects:
        print(f"  bigG encode subj{subj:02d}")
        e_tr = _encode(emb, tensors[f"imgs_train_{subj}"], device)
        e_te = _encode(emb, tensors[f"imgs_test_{subj}"], device)
        out[f"emb_train_{subj}"] = e_tr
        out[f"emb_test_{subj}"] = e_te
        # accumulate per-channel stats over TRAIN tokens (flatten N and 256)
        flat = e_tr.float().reshape(-1, TOKEN_DIM)
        cnt += flat.shape[0]
        s_sum += flat.sum(0).double()
        s_sqsum += (flat * flat).sum(0).double()

    mean = (s_sum / cnt).float()
    var = (s_sqsum / cnt - (s_sum / cnt) ** 2).clamp_min(1e-12).float()
    stats = TargetStats(mean=mean, std=var.sqrt().clamp_min(1e-6))
    out["_stats"] = stats.to_dict()
    out["_meta"] = {"embedder": "ViT-bigG-14/laion2b_s39b_b160k",
                    "token_len": TOKEN_LEN, "token_dim": TOKEN_DIM, "norm": "raw_tokens"}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, cache_path)
    print(f"✓ bigG target cache saved: {cache_path}  ({cache_path.stat().st_size/1e9:.1f} GB)")
    del emb
    gc.collect(); torch.cuda.empty_cache()
    out["_stats"] = stats
    return out
