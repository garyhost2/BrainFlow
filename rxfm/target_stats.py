from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

UNCLIP_REPO = "stabilityai/stable-diffusion-2-1-unclip"
TARGET_EMB_DIM = 1024

@dataclass
class TargetStats:
    mean: torch.Tensor
    std: torch.Tensor

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
    x = imgs_uint8
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    x = x.clamp(0, 1)
    if x.shape[-1] != 224 or x.shape[-2] != 224:

        x = F.interpolate(x, 224, mode="bicubic", align_corners=False).clamp(0, 1)
    x = (x.to(device) - _CLIP_MEAN.to(device)) / _CLIP_STD.to(device)
    return x

@torch.no_grad()
def _load_image_encoder(device: torch.device, cache_dir: Path | None):
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
    outs = []
    for i in tqdm(range(0, len(imgs_uint8), bs), desc="ViT-H target", leave=False):
        x = _preprocess(imgs_uint8[i:i + bs], device)

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
