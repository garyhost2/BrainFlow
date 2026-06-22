"""Decoder-target CLIP embeddings + standardization stats.

We predict the embedding that the frozen unCLIP decoder consumes.  To be exact
we extract it with the *decoder's own* image encoder (the CLIPVisionModelWithProjection
that ships inside `stabilityai/stable-diffusion-2-1-unclip`), so the embedding
distribution matches the decoder's internal `image_normalizer` buffers.

CRITICAL correctness notes (verified against diffusers source):
  * The pipeline's `noise_image_embeddings` standardizes the embedding with its
    OWN learned mean/std and then un-standardizes.  Therefore the value we feed
    to `image_embeds=` must be the **raw pooled projection** — do NOT L2-normalize
    and do NOT pre-standardize it.
  * Preprocessing must match the pipeline's CLIPImageProcessor: resize to 224
    (bicubic), scale to [0,1], normalize with the OpenAI/OpenCLIP mean/std.

Separately, for the *flow-matching prior* we fit a per-dim mean/std on the
TRAIN embeddings and train in that standardized space (well-conditioned), then
un-standardize the sampled embedding before handing it to the decoder.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

# OpenAI/OpenCLIP image normalization (identical for ViT-L/H/bigG at 224).
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

# Decoder checkpoint whose image_encoder defines the target space.
UNCLIP_REPO = "stabilityai/stable-diffusion-2-1-unclip"
TARGET_EMB_DIM = 1024  # OpenCLIP ViT-H/14 pooled projection


@dataclass
class TargetStats:
    """Per-dim standardization for the flow prior's working space."""
    mean: torch.Tensor  # (D,)
    std: torch.Tensor   # (D,)

    def standardize(self, e: torch.Tensor) -> torch.Tensor:
        return (e - self.mean.to(e.device, e.dtype)) / self.std.to(e.device, e.dtype)

    def unstandardize(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.std.to(z.device, z.dtype) + self.mean.to(z.device, z.dtype)

    def to_dict(self) -> dict:
        return {"mean": self.mean.cpu(), "std": self.std.cpu()}

    @classmethod
    def from_dict(cls, d: dict) -> "TargetStats":
        return cls(mean=d["mean"], std=d["std"])


def _preprocess(imgs_uint8: torch.Tensor, device: torch.device) -> torch.Tensor:
    """(N,3,H,W) uint8/float[0,1] -> (N,3,224,224) CLIP-normalized float."""
    x = imgs_uint8
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    x = x.clamp(0, 1)
    if x.shape[-1] != 224 or x.shape[-2] != 224:
        # bicubic to match CLIPImageProcessor (NSD images are square -> resize==crop)
        x = F.interpolate(x, 224, mode="bicubic", align_corners=False).clamp(0, 1)
    x = (x.to(device) - _CLIP_MEAN.to(device)) / _CLIP_STD.to(device)
    return x


@torch.no_grad()
def _load_image_encoder(device: torch.device, cache_dir: Path | None):
    """The exact CLIPVisionModelWithProjection used by the unCLIP decoder."""
    from transformers import CLIPVisionModelWithProjection

    enc = CLIPVisionModelWithProjection.from_pretrained(
        UNCLIP_REPO, subfolder="image_encoder",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    enc = enc.eval().to(device)
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


@torch.no_grad()
def _encode(enc, imgs_uint8: torch.Tensor, device: torch.device, bs: int = 64) -> torch.Tensor:
    """Return RAW pooled image embeddings (N, D). No L2-norm, no standardization."""
    outs = []
    for i in tqdm(range(0, len(imgs_uint8), bs), desc="ViT-H target", leave=False):
        x = _preprocess(imgs_uint8[i:i + bs], device)
        # image_embeds = pooled CLS projection (what SD-2.1-unCLIP expects)
        e = enc(pixel_values=x).image_embeds.float().cpu()
        outs.append(e)
    return torch.cat(outs)


def build_or_load_targets(
    tensors: dict,
    subjects: list[int],
    cache_path: Path,
    device: torch.device,
    hf_cache: Path | None = None,
    force_rebuild: bool = False,
) -> dict:
    """Build (or load) the decoder-target embeddings for every NSD image.

    `tensors` is the dict produced by `brainflow.data.build_or_load_tensors`
    (keys: imgs_train_{subj}, imgs_test_{subj}, ...).

    Returns dict with:
        emb_train_{subj}, emb_test_{subj}   # (N, D) float, RAW (un-normalized)
        _stats : TargetStats fit on the pooled TRAIN embeddings (all subjects)
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_rebuild:
        print(f"✓ Loading target-embedding cache: {cache_path}")
        blob = torch.load(cache_path, map_location="cpu")
        blob["_stats"] = TargetStats.from_dict(blob["_stats"])
        return blob

    print("Building decoder-target embeddings (ViT-H/14 pooled, raw)…")
    enc = _load_image_encoder(device, hf_cache)
    out: dict = {}
    train_for_stats = []
    for subj in subjects:
        print(f"  target encode subj{subj:02d}")
        e_tr = _encode(enc, tensors[f"imgs_train_{subj}"], device)
        e_te = _encode(enc, tensors[f"imgs_test_{subj}"], device)
        out[f"emb_train_{subj}"] = e_tr
        out[f"emb_test_{subj}"] = e_te
        train_for_stats.append(e_tr)

    all_train = torch.cat(train_for_stats)
    stats = TargetStats(
        mean=all_train.mean(0),
        std=all_train.std(0).clamp_min(1e-6),
    )
    out["_stats"] = stats.to_dict()
    out["_meta"] = {"repo": UNCLIP_REPO, "dim": TARGET_EMB_DIM, "norm": "raw_pooled"}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, cache_path)
    print(f"✓ Target cache saved: {cache_path}")

    del enc
    gc.collect()
    torch.cuda.empty_cache()
    out["_stats"] = stats
    return out
