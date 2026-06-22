from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class Step1Config:
    emb_dim: int = 1024

    enc_hidden: int = 2048
    enc_blocks: int = 4
    enc_drop: float = 0.15
    cond_dim: int = 1024

    prior_width: int = 2048
    prior_depth: int = 6
    time_dim: int = 256

    cfg_drop_prob: float = 0.1
    logit_normal_m: float = 0.0
    logit_normal_s: float = 1.0
    lambda_flow: float = 1.0
    lambda_reg: float = 1.0
    lambda_cos: float = 0.5

    n_steps: int = 50
    cfg_scale: float = 3.0
    solver: str = "heun"
    cond_source: str = "prior"
    blend_w: float = 0.5
    subjects: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8])

class ResMLP(nn.Module):
    def __init__(self, dim: int, mult: int = 4, drop: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult), nn.GELU(), nn.Dropout(drop),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return x + self.net(self.norm(x))

class SinusoidalTime(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half).float() / max(1, half - 1))
        self.register_buffer("freqs", freqs, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))

    def forward(self, t):
        a = t[:, None] * self.freqs[None, :]
        emb = torch.cat([a.sin(), a.cos()], dim=-1)
        return self.mlp(emb)

class Backbone(nn.Module):
    def __init__(self, cfg: Step1Config, voxels_per_subject: dict[int, int]):
        super().__init__()
        self.cfg = cfg
        h = cfg.enc_hidden

        self.input_proj = nn.ModuleDict({
            str(s): nn.Linear(v, h) for s, v in voxels_per_subject.items()
        })
        self.stem = nn.Sequential(nn.LayerNorm(h), nn.GELU(), nn.Dropout(cfg.enc_drop))
        self.blocks = nn.ModuleList([ResMLP(h, 4, cfg.enc_drop) for _ in range(cfg.enc_blocks)])
        self.reg_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, cfg.emb_dim))
        self.cond_head = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, cfg.cond_dim))

    def forward(self, fmri: torch.Tensor, subject: int):
        h = self.input_proj[str(int(subject))](fmri)
        h = self.stem(h)
        for b in self.blocks:
            h = b(h)
        reg = self.reg_head(h)
        cond = self.cond_head(h)
        return reg, cond

class _PriorBlock(nn.Module):
    def __init__(self, width: int, cond_dim: int, mult: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.net = nn.Sequential(nn.Linear(width, width * mult), nn.GELU(), nn.Linear(width * mult, width))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 3 * width))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, c):
        shift, scale, gate = self.ada(c).chunk(3, dim=-1)
        h = self.norm(x) * (1 + scale) + shift
        return x + gate * self.net(h)

class FlowPrior(nn.Module):
    def __init__(self, cfg: Step1Config):
        super().__init__()
        self.cfg = cfg
        d, w = cfg.emb_dim, cfg.prior_width
        self.time = SinusoidalTime(cfg.time_dim, cfg.cond_dim)
        self.cond_proj = nn.Linear(cfg.cond_dim, cfg.cond_dim)
        self.in_proj = nn.Linear(d, w)
        self.blocks = nn.ModuleList([_PriorBlock(w, cfg.cond_dim) for _ in range(cfg.prior_depth)])
        self.out_norm = nn.LayerNorm(w)
        self.out = nn.Linear(w, d)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        self.null_cond = nn.Parameter(torch.zeros(1, cfg.cond_dim))

    def forward(self, xt: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        c = self.cond_proj(cond) + self.time(t)
        h = self.in_proj(xt)
        for blk in self.blocks:
            h = blk(h, c)
        return self.out(self.out_norm(h))

class Step1Model(nn.Module):
    def __init__(self, cfg: Step1Config, voxels_per_subject: dict[int, int]):
        super().__init__()
        self.cfg = cfg
        self.backbone = Backbone(cfg, voxels_per_subject)
        self.prior = FlowPrior(cfg)

    def _sample_t(self, B: int, device) -> torch.Tensor:
        z = torch.randn(B, device=device) * self.cfg.logit_normal_s + self.cfg.logit_normal_m
        return torch.sigmoid(z)

    def training_step(self, fmri, subject, target_emb_std):
        cfg = self.cfg
        B = fmri.shape[0]
        device = fmri.device
        reg, cond = self.backbone(fmri, subject)

        loss_reg = F.mse_loss(reg, target_emb_std)
        loss_cos = 1.0 - F.cosine_similarity(reg, target_emb_std, dim=-1).mean()

        x1 = target_emb_std
        x0 = torch.randn_like(x1)
        t = self._sample_t(B, device)
        xt = (1 - t)[:, None] * x0 + t[:, None] * x1
        ut = x1 - x0

        drop = (torch.rand(B, device=device) < cfg.cfg_drop_prob)[:, None]
        cond_in = torch.where(drop, self.prior.null_cond.to(cond.dtype), cond)
        v = self.prior(xt, t, cond_in)
        loss_flow = F.mse_loss(v, ut)

        total = (cfg.lambda_flow * loss_flow
                 + cfg.lambda_reg * loss_reg
                 + cfg.lambda_cos * loss_cos)
        return {"loss": total, "flow": loss_flow.detach(),
                "reg": loss_reg.detach(), "cos": loss_cos.detach()}

    @torch.no_grad()
    def _sample_prior(self, cond: torch.Tensor, n_steps: int, cfg_scale: float, solver: str):
        B = cond.shape[0]
        device = cond.device
        x = torch.randn(B, self.cfg.emb_dim, device=device)
        null = self.prior.null_cond.to(cond.dtype).expand(B, -1)
        t_grid = torch.linspace(0.0, 1.0, n_steps + 1, device=device)

        def vel(xt, t_scalar):
            t = torch.full((B,), float(t_scalar), device=device)
            if cfg_scale != 1.0:
                vc = self.prior(xt, t, cond)
                vu = self.prior(xt, t, null)
                return vu + cfg_scale * (vc - vu)
            return self.prior(xt, t, cond)

        for i in range(n_steps):
            t0 = float(t_grid[i]); t1 = float(t_grid[i + 1]); dt = t1 - t0
            v1 = vel(x, t0)
            if solver == "heun" and i < n_steps:
                x_pred = x + v1 * dt
                v2 = vel(x_pred, t1)
                x = x + 0.5 * (v1 + v2) * dt
            else:
                x = x + v1 * dt
        return x

    @torch.no_grad()
    def predict_embedding(self, fmri, subject, stats, *, n_steps=None, cfg_scale=None,
                          solver=None, cond_source=None) -> torch.Tensor:
        cfg = self.cfg
        n_steps = n_steps or cfg.n_steps
        cfg_scale = cfg.cfg_scale if cfg_scale is None else cfg_scale
        solver = solver or cfg.solver
        cond_source = cond_source or cfg.cond_source

        reg, cond = self.backbone(fmri, subject)
        if cond_source == "regression":
            z = reg
        elif cond_source == "prior":
            z = self._sample_prior(cond, n_steps, cfg_scale, solver)
        elif cond_source == "blend":
            zp = self._sample_prior(cond, n_steps, cfg_scale, solver)
            z = cfg.blend_w * zp + (1 - cfg.blend_w) * reg
        else:
            raise ValueError(f"Unknown cond_source: {cond_source!r}")
        return stats.unstandardize(z)
