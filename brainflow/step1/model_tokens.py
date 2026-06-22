from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ResMLP, SinusoidalTime

def soft_clip_loss(preds: torch.Tensor, targs: torch.Tensor, temp: float) -> torch.Tensor:
    clip_clip = (targs @ targs.T) / temp
    brain_clip = (preds @ targs.T) / temp
    soft = clip_clip.softmax(dim=-1)
    loss1 = -(brain_clip.log_softmax(dim=-1) * soft).sum(dim=-1).mean()
    loss2 = -(brain_clip.T.log_softmax(dim=-1) * soft).sum(dim=-1).mean()
    return (loss1 + loss2) / 2

@dataclass
class TokenStep1Config:
    token_len: int = 256
    token_dim: int = 1664
    brain_dim: int = 1024
    n_brain_tokens: int = 64

    enc_hidden: int = 2048
    enc_blocks: int = 4
    enc_drop: float = 0.15

    reg_depth: int = 2
    reg_heads: int = 8

    prior_width: int = 1024
    prior_depth: int = 8
    prior_heads: int = 8
    time_dim: int = 256

    cfg_drop_prob: float = 0.1
    logit_normal_m: float = 0.0
    logit_normal_s: float = 1.0
    lambda_flow: float = 1.0
    lambda_reg: float = 1.0
    lambda_cos: float = 0.5

    lambda_clip: float = 1.0
    clip_temp: float = 0.006

    n_steps: int = 50
    cfg_scale: float = 3.0
    solver: str = "heun"
    cond_source: str = "prior"
    blend_w: float = 0.5
    subjects: list[int] = field(default_factory=lambda: [1])

class SelfAttn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads; self.dh = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def rsh(t): return t.view(B, L, self.h, self.dh).transpose(1, 2)
        o = F.scaled_dot_product_attention(rsh(q), rsh(k), rsh(v))
        return self.proj(o.transpose(1, 2).reshape(B, L, C))

class CrossAttn(nn.Module):
    def __init__(self, dim, ctx_dim, heads):
        super().__init__()
        self.h = heads; self.dh = dim // heads
        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(ctx_dim, dim, bias=False)
        self.to_v = nn.Linear(ctx_dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, ctx):
        B, L, C = x.shape
        xn = self.norm(x)
        q = self.to_q(xn); k = self.to_k(ctx); v = self.to_v(ctx)
        def rsh(t, n): return t.view(B, n, self.h, self.dh).transpose(1, 2)
        o = F.scaled_dot_product_attention(rsh(q, L), rsh(k, ctx.shape[1]), rsh(v, ctx.shape[1]))
        return x + self.proj(o.transpose(1, 2).reshape(B, L, C))

class TokenBackbone(nn.Module):
    def __init__(self, cfg: TokenStep1Config, voxels_per_subject: dict[int, int]):
        super().__init__()
        self.cfg = cfg
        h = cfg.enc_hidden
        self.input_proj = nn.ModuleDict({
            str(s): nn.Linear(v, h) for s, v in voxels_per_subject.items()})
        self.stem = nn.Sequential(nn.LayerNorm(h), nn.GELU(), nn.Dropout(cfg.enc_drop))
        self.blocks = nn.ModuleList([ResMLP(h, 4, cfg.enc_drop) for _ in range(cfg.enc_blocks)])
        self.to_brain = nn.Sequential(
            nn.Linear(h, cfg.n_brain_tokens * cfg.brain_dim),
            nn.Unflatten(-1, (cfg.n_brain_tokens, cfg.brain_dim)),
            nn.LayerNorm(cfg.brain_dim),
        )

    def forward(self, fmri, subject: int):
        x = self.input_proj[str(int(subject))](fmri)
        x = self.stem(x)
        for b in self.blocks:
            x = b(x)
        return self.to_brain(x)

class TokenRegHead(nn.Module):
    def __init__(self, cfg: TokenStep1Config):
        super().__init__()
        d = cfg.brain_dim
        self.queries = nn.Parameter(torch.randn(1, cfg.token_len, d) * 0.02)
        self.cross = nn.ModuleList([CrossAttn(d, d, cfg.reg_heads) for _ in range(cfg.reg_depth)])
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))
            for _ in range(cfg.reg_depth)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, cfg.token_dim))

    def forward(self, brain):
        B = brain.shape[0]
        x = self.queries.expand(B, -1, -1)
        for cr, mlp in zip(self.cross, self.mlps):
            x = cr(x, brain)
            x = x + mlp(x)
        return self.out(x)

class DiTBlock(nn.Module):
    def __init__(self, w, heads, brain_dim, time_dim):
        super().__init__()
        self.n1 = nn.LayerNorm(w, elementwise_affine=False)
        self.attn = SelfAttn(w, heads)
        self.cross = CrossAttn(w, brain_dim, heads)
        self.n2 = nn.LayerNorm(w, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(w, w * 4), nn.GELU(), nn.Linear(w * 4, w))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, 6 * w))
        nn.init.zeros_(self.ada[-1].weight); nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, t_emb, brain):
        s1, g1, gate1, s2, g2, gate2 = self.ada(t_emb).chunk(6, dim=-1)
        h = self.n1(x) * (1 + g1[:, None]) + s1[:, None]
        x = x + gate1[:, None] * self.attn(h)
        x = self.cross(x, brain)
        h = self.n2(x) * (1 + g2[:, None]) + s2[:, None]
        x = x + gate2[:, None] * self.mlp(h)
        return x

class TokenFlowPrior(nn.Module):
    def __init__(self, cfg: TokenStep1Config):
        super().__init__()
        self.cfg = cfg
        w = cfg.prior_width
        self.time = SinusoidalTime(cfg.time_dim, w)
        self.in_proj = nn.Linear(cfg.token_dim, w)
        self.pos = nn.Parameter(torch.zeros(1, cfg.token_len, w))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([
            DiTBlock(w, cfg.prior_heads, cfg.brain_dim, w) for _ in range(cfg.prior_depth)])
        self.out_norm = nn.LayerNorm(w, elementwise_affine=False)
        self.out = nn.Linear(w, cfg.token_dim)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.null_brain = nn.Parameter(torch.zeros(1, cfg.n_brain_tokens, cfg.brain_dim))

    def forward(self, xt, t, brain):
        te = self.time(t)
        h = self.in_proj(xt) + self.pos
        for blk in self.blocks:
            h = blk(h, te, brain)
        return self.out(self.out_norm(h))

class TokenStep1Model(nn.Module):
    def __init__(self, cfg: TokenStep1Config, voxels_per_subject: dict[int, int]):
        super().__init__()
        self.cfg = cfg
        self.backbone = TokenBackbone(cfg, voxels_per_subject)
        self.reg_head = TokenRegHead(cfg)
        self.prior = TokenFlowPrior(cfg)

        self.clip_proj = nn.Sequential(
            nn.LayerNorm(cfg.brain_dim),
            nn.Linear(cfg.brain_dim, cfg.brain_dim), nn.GELU(),
            nn.Linear(cfg.brain_dim, cfg.token_dim),
        )

    def _sample_t(self, B, device):
        z = torch.randn(B, device=device) * self.cfg.logit_normal_s + self.cfg.logit_normal_m
        return torch.sigmoid(z)

    def training_step(self, fmri, subject, target_std):
        cfg = self.cfg
        B = fmri.shape[0]; device = fmri.device
        brain = self.backbone(fmri, subject)

        reg = self.reg_head(brain)
        loss_reg = F.mse_loss(reg, target_std)

        loss_cos = 1.0 - F.cosine_similarity(reg.float(), target_std.float(), dim=-1).mean()

        if cfg.lambda_clip > 0:
            b = F.normalize(self.clip_proj(brain.mean(dim=1)).float(), dim=-1)
            z = F.normalize(target_std.mean(dim=1).float(), dim=-1)
            loss_clip = soft_clip_loss(b, z, cfg.clip_temp)
        else:
            loss_clip = torch.zeros((), device=device)

        x1 = target_std
        x0 = torch.randn_like(x1)
        t = self._sample_t(B, device)
        tb = t[:, None, None]
        xt = (1 - tb) * x0 + tb * x1
        ut = x1 - x0
        drop = (torch.rand(B, device=device) < cfg.cfg_drop_prob)[:, None, None]
        brain_in = torch.where(drop, self.prior.null_brain.to(brain.dtype), brain)
        v = self.prior(xt, t, brain_in)
        loss_flow = F.mse_loss(v, ut)

        total = (cfg.lambda_flow * loss_flow + cfg.lambda_reg * loss_reg
                 + cfg.lambda_cos * loss_cos + cfg.lambda_clip * loss_clip)
        return {"loss": total, "flow": loss_flow.detach(),
                "reg": loss_reg.detach(), "cos": loss_cos.detach(),
                "clip": loss_clip.detach()}

    @torch.no_grad()
    def _sample_prior(self, brain, n_steps, cfg_scale, solver):
        cfg = self.cfg
        B = brain.shape[0]; device = brain.device
        x = torch.randn(B, cfg.token_len, cfg.token_dim, device=device)
        null = self.prior.null_brain.to(brain.dtype).expand(B, -1, -1)
        t_grid = torch.linspace(0.0, 1.0, n_steps + 1, device=device)

        def vel(xt, tval):
            t = torch.full((B,), float(tval), device=device)
            if cfg_scale != 1.0:
                vc = self.prior(xt, t, brain); vu = self.prior(xt, t, null)
                return vu + cfg_scale * (vc - vu)
            return self.prior(xt, t, brain)

        for i in range(n_steps):
            t0 = float(t_grid[i]); t1 = float(t_grid[i + 1]); dt = t1 - t0
            v1 = vel(x, t0)
            if solver == "heun":
                x = x + 0.5 * (v1 + vel(x + v1 * dt, t1)) * dt
            else:
                x = x + v1 * dt
        return x

    @torch.no_grad()
    def predict_tokens(self, fmri, subject, stats, *, n_steps=None, cfg_scale=None,
                       solver=None, cond_source=None):
        cfg = self.cfg
        n_steps = n_steps or cfg.n_steps
        cfg_scale = cfg.cfg_scale if cfg_scale is None else cfg_scale
        solver = solver or cfg.solver
        cond_source = cond_source or cfg.cond_source
        brain = self.backbone(fmri, subject)
        if cond_source == "regression":
            z = self.reg_head(brain)
        elif cond_source == "prior":
            z = self._sample_prior(brain, n_steps, cfg_scale, solver)
        elif cond_source == "blend":
            z = cfg.blend_w * self._sample_prior(brain, n_steps, cfg_scale, solver) \
                + (1 - cfg.blend_w) * self.reg_head(brain)
        else:
            raise ValueError(f"Unknown cond_source: {cond_source!r}")
        return stats.unstandardize(z)
