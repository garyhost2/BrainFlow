"""Development metrics + EMA for Step 1.

Metrics: PixCorr, SSIM, CLIP cosine + 2-way identification.
The CLIP metric uses ViT-L/14 — a DIFFERENT encoder than the ViT-H/14 we
*condition* on — so we measure semantic fidelity, not the conditioning space.

These are correct for tracking/development.  For final paper numbers, run the
official MindEye2 eval harness (AlexNet/Inception/EffNet/SwAV) on the saved grid.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


# ── PixCorr ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def pixcorr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Mean per-image Pearson correlation over flattened pixels. pred/gt in [0,1]."""
    p = pred.flatten(1).float()
    g = gt.flatten(1).float()
    p = p - p.mean(1, keepdim=True)
    g = g - g.mean(1, keepdim=True)
    num = (p * g).sum(1)
    den = p.norm(dim=1) * g.norm(dim=1) + 1e-8
    return (num / den).mean().item()


# ── SSIM (grayscale, Gaussian window, skimage-style) ─────────────────────────
def _gaussian_window(ch: int, ws: int = 11, sigma: float = 1.5):
    coords = torch.arange(ws).float() - (ws - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    w2d = (g[:, None] * g[None, :])
    return w2d.expand(ch, 1, ws, ws).contiguous()


@torch.no_grad()
def ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """SSIM on luminance. pred/gt in [0,1], shape (N,3,H,W)."""
    def lum(x):
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        return (0.2989 * r + 0.5870 * g + 0.1140 * b)[:, None]
    p, g = lum(pred.float()), lum(gt.float())
    win = _gaussian_window(1, 11, 1.5).to(p.device)
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_p = F.conv2d(p, win, padding=5)
    mu_g = F.conv2d(g, win, padding=5)
    mu_p2, mu_g2, mu_pg = mu_p ** 2, mu_g ** 2, mu_p * mu_g
    sp = F.conv2d(p * p, win, padding=5) - mu_p2
    sg = F.conv2d(g * g, win, padding=5) - mu_g2
    spg = F.conv2d(p * g, win, padding=5) - mu_pg
    s = ((2 * mu_pg + C1) * (2 * spg + C2)) / ((mu_p2 + mu_g2 + C1) * (sp + sg + C2) + 1e-12)
    return s.mean().item()


# ── CLIP (held-out ViT-L/14) ─────────────────────────────────────────────────
class CLIPMetric:
    def __init__(self, device, hf_cache=None):
        import open_clip
        m, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        self.m = m.eval().to(device)
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def _embed(self, imgs):  # imgs (N,3,H,W) in [0,1]
        x = F.interpolate(imgs.to(self.device), 224, mode="bicubic", align_corners=False).clamp(0, 1)
        x = (x - _CLIP_MEAN.to(self.device)) / _CLIP_STD.to(self.device)
        return F.normalize(self.m.encode_image(x).float(), dim=-1)

    @torch.no_grad()
    def score(self, pred, gt):
        ep, eg = self._embed(pred), self._embed(gt)
        cos = (ep * eg).sum(-1).mean().item()
        # 2-way identification: for each pred, is it closer to its gt than a random other gt?
        sim = ep @ eg.T                       # (N,N)
        n = sim.shape[0]
        diag = sim.diag()[:, None]
        two_way = ((diag > sim).float().sum(1) / max(1, n - 1)).mean().item()
        return {"CLIP_cos": cos, "CLIP_2way": two_way}


# ── EMA (float params only; in-place swap, torch.compile-safe) ────────────────
class EMA:
    """Tracks an EMA of floating-point params/buffers.

    Eval pattern (no deepcopy -> works with torch.compile):
        ema.store(model); ema.copy_to(model)
        ... evaluate ...
        ema.restore(model)
    """
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if v.is_floating_point()}
        self._backup = None

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def store(self, model):
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()
                        if k in self.shadow}

    @torch.no_grad()
    def copy_to(self, model):
        sd = model.state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_(v)          # in-place mutate of live params/buffers

    @torch.no_grad()
    def restore(self, model):
        if self._backup is None:
            return
        sd = model.state_dict()
        for k, v in self._backup.items():
            sd[k].copy_(v)
        self._backup = None
