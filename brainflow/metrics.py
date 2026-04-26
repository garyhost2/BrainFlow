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


# Module-level CLIP singleton — loaded once and kept on GPU for all eval calls
_CLIP_ENC = None


def _get_clip(device):
    global _CLIP_ENC
    if _CLIP_ENC is None:
        import open_clip
        m, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        m = m.eval().to(device)
        for p in m.parameters():
            p.requires_grad_(False)
        _CLIP_ENC = m
    return _CLIP_ENC


@torch.no_grad()
def evaluate(model, vae, loader, device, cfg, n_batches=9999,
             n_steps=None, solver=None):
    """Evaluate model on loader.

    n_steps: ODE steps (defaults to cfg.ode_steps)
    solver:  ODE solver (defaults to "midpoint")
    """
    model.eval(); vae.to(device)
    clip_enc = _get_clip(device)
    if n_steps is None:
        n_steps = cfg.ode_steps
    if solver is None:
        solver = "midpoint"
    pcs, sss, css = [], [], []

    for i, batch in enumerate(loader):
        if i >= n_batches: break
        fmri = batch["fmri"].to(device)
        images = batch["image"].to(device)
        subject = batch["subject"].to(device)
        raw = model.module if hasattr(model, "module") else model
        tokens, cls = raw.encode_fmri(fmri, subject)
        pl = raw.sample(tokens, n_steps=n_steps, cfg_scale=cfg.cfg_scale,
                        solver=solver, cls=cls)
        pi = vae.decode(pl)
        pi = (pi * 0.5 + 0.5).clamp(0, 1)   # VAE outputs [-1,1]; convert to [0,1] for metrics
        pcs.append(pixel_correlation(pi, images))
        sss.append(ssim_pytorch(pi.cpu(), images.cpu()))
        pi_r = F.interpolate(pi.cpu(), 224, mode="bilinear", align_corners=False).to(device)
        gt_r = F.interpolate(images.cpu(), 224, mode="bilinear", align_corners=False).to(device)
        ep = F.normalize(clip_enc.encode_image(pi_r).float(), dim=-1)
        et = F.normalize(clip_enc.encode_image(gt_r).float(), dim=-1)
        css.append(float((ep * et).sum(-1).mean()))

    # Keep clip_enc and vae on GPU for next eval call (no del / vae.cpu())
    model.train()
    return {
        "PixCorr": float(np.mean(pcs)) if pcs else 0.0,
        "SSIM": float(np.mean(sss)) if sss else 0.0,
        "CLIP_Sim": float(np.mean(css)) if css else 0.0,
    }
