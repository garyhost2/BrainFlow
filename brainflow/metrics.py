"""Evaluation metrics: PixCorr, SSIM, CLIP similarity."""
from __future__ import annotations
import gc
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr


def pixel_correlation(pred, target):
    p = pred.flatten(1).cpu().numpy()
    t = target.flatten(1).cpu().numpy()
    return float(np.nanmean([pearsonr(pi, ti)[0] for pi, ti in zip(p, t)]))


def ssim_pytorch(pred, target, ws=11, sigma=1.5):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    B, C, H, W = pred.shape
    g = torch.arange(ws, dtype=torch.float32) - ws // 2
    g = torch.exp(-g ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    k = g.outer(g).unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1).to(pred.device)
    pad = ws // 2
    def mu(x): return F.conv2d(x, k, padding=pad, groups=C)
    mx = mu(pred); my = mu(target)
    sx = mu(pred * pred) - mx ** 2
    sy = mu(target * target) - my ** 2
    sxy = mu(pred * target) - mx * my
    return float(((2 * mx * my + C1) * (2 * sxy + C2) /
                  ((mx ** 2 + my ** 2 + C1) * (sx + sy + C2))).mean())


@torch.no_grad()
def evaluate(model, vae, loader, device, cfg, n_batches=8):
    import open_clip
    model.eval(); vae.to(device)
    pcs, sss, css = [], [], []
    clip_enc, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    clip_enc = clip_enc.eval().to(device)
    for p in clip_enc.parameters(): p.requires_grad_(False)

    for i, batch in enumerate(loader):
        if i >= n_batches: break
        fmri = batch["fmri"].to(device)
        images = batch["image"].to(device)
        subject = batch["subject"].to(device)
        tokens, _ = (model.module if hasattr(model, "module") else model).encode_fmri(fmri, subject)
        pl = (model.module if hasattr(model, "module") else model).sample(
            tokens, n_steps=cfg.ode_steps, cfg_scale=cfg.cfg_scale)
        pi = vae.decode(pl)
        pcs.append(pixel_correlation(pi, images))
        sss.append(ssim_pytorch(pi.cpu(), images.cpu()))
        pi_r = F.interpolate(pi.cpu(), 224, mode="bilinear", align_corners=False).to(device)
        gt_r = F.interpolate(images.cpu(), 224, mode="bilinear", align_corners=False).to(device)
        ep = F.normalize(clip_enc.encode_image(pi_r).float(), dim=-1)
        et = F.normalize(clip_enc.encode_image(gt_r).float(), dim=-1)
        css.append(float((ep * et).sum(-1).mean()))

    del clip_enc; vae.cpu(); gc.collect(); torch.cuda.empty_cache()
    model.train()
    return {
        "PixCorr": float(np.mean(pcs)) if pcs else 0.0,
        "SSIM": float(np.mean(sss)) if sss else 0.0,
        "CLIP_Sim": float(np.mean(css)) if css else 0.0,
    }
