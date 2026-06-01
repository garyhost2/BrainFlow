from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brainflow.clip_prior import ClipPrior
from brainflow.config import load_config
from brainflow.data import build_dataloaders
from brainflow.decoder import FrozenImageEmbeddingDecoder
from brainflow.metrics_full import evaluate_full
from brainflow.models import BrainEncoder

load_dotenv()


@torch.no_grad()
def evaluate_sdprior(brain_enc, prior, decoder, loader, cfg, device):
    pred_images = []
    target_images = []
    cos_vals = []

    brain_enc.eval()
    prior.eval()
    for batch in loader:
        fmri = batch["fmri"].to(device)
        subject = batch["subject"].to(device)
        clip_gt = F.normalize(batch["clip_emb"].to(device).float(), dim=-1)
        img_gt = batch["image"].to(device).float()

        brain_out = brain_enc(fmri, subject)
        clip_pred = prior.sample(
            brain_out.tokens,
            n_steps=int(getattr(cfg, "prior_ode_steps", 20)),
            cfg_scale=float(getattr(cfg, "cfg_scale", 1.0)),
            normalize_output=True,
        )
        if getattr(prior, "prior_target", "cls") == "patches":
            if "clip_patches" not in batch:
                raise RuntimeError("prior_target='patches' requires clip_patches in eval batches.")
            clip_metric = F.normalize(prior.patches_to_embedding(clip_pred), dim=-1)
            clip_gt_metric = F.normalize(
                prior.patches_to_embedding(batch["clip_patches"].to(device).float()),
                dim=-1,
            )
        else:
            clip_metric = clip_pred
            clip_gt_metric = clip_gt
        cos_vals.append(F.cosine_similarity(clip_metric, clip_gt_metric, dim=-1).cpu())

        pred = decoder.generate(
            clip_image_embedding=None if getattr(prior, "prior_target", "cls") == "patches" else clip_pred,
            clip_patch_tokens=clip_pred if getattr(prior, "prior_target", "cls") == "patches" else None,
            num_inference_steps=int(getattr(cfg, "decoder_num_inference_steps", 30)),
            guidance_scale=float(getattr(cfg, "decoder_guidance_scale", 7.5)),
        )
        gt = img_gt.clamp(0, 1)
        pred_images.append(pred.cpu())
        target_images.append(gt.cpu())

    pred_all = torch.cat(pred_images, dim=0)
    target_all = torch.cat(target_images, dim=0)

    metrics = evaluate_full(pred_all, target_all, device=device)
    metrics["cosine"] = torch.cat(cos_vals, dim=0).mean().item()
    return metrics


@torch.no_grad()
def main():
    cfg = load_config()
    if cfg.training_stage != "sdprior":
        raise ValueError("eval_sdprior.py expects training_stage='sdprior'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, eval_loader, _, voxels = build_dataloaders(cfg)

    brain_enc = BrainEncoder(cfg, voxels).to(device)
    prior = ClipPrior(
        clip_dim=cfg.clip_dim,
        ctx_dim=cfg.brain_dim,
        dim=getattr(cfg, "prior_dim", 512),
        depth=getattr(cfg, "prior_blocks", 6),
        heads=getattr(cfg, "attn_heads", 8),
        dropout=cfg.enc_drop,
        cfg_drop=cfg.cfg_drop_prob,
        geometry=getattr(cfg, "geometry", getattr(cfg, "clip_prior_geometry", "euclidean")),
        prior_target=getattr(cfg, "prior_target", "cls"),
    ).to(device)
    decoder = FrozenImageEmbeddingDecoder(
        model_id=cfg.decoder_model_id,
        cache_dir=Path(cfg.data_dir) / "hf_cache",
    ).to(device)

    ckpt_path = Path(os.environ.get("CHECKPOINT", str(cfg.output_dir / "best_clip_cos.pt")))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "prior" in ckpt:
        prior.load_state_dict(ckpt["prior"], strict=False)
        if "brain_encoder" in ckpt:
            brain_enc.load_state_dict(ckpt["brain_encoder"], strict=False)
        if "clip_mean" in ckpt and "clip_std" in ckpt:
            prior.clip_mean.copy_(ckpt["clip_mean"])
            prior.clip_std.copy_(ckpt["clip_std"])
            prior._stats_fitted = True
    else:
        prior.load_state_dict(ckpt, strict=False)

    metrics = evaluate_sdprior(brain_enc, prior, decoder, eval_loader, cfg, device)

    print("\n=== SDPRIOR FULL BENCHMARK (test split) ===")
    for k in sorted(metrics):
        print(f"{k:16s}: {metrics[k]:.4f}")


if __name__ == "__main__":
    main()
