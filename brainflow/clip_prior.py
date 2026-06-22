from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_SLERP_EPS = 1e-6

def random_sphere(shape: tuple[int, ...], device: torch.device | str) -> torch.Tensor:
    x = torch.randn(*shape, device=device)
    return F.normalize(x, dim=-1)

def slerp(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    dot = (x0 * x1).sum(dim=-1, keepdim=True).clamp(-1 + _SLERP_EPS, 1 - _SLERP_EPS)
    theta = torch.acos(dot)
    sin_theta = theta.sin()

    t_col = t[:, None]
    s0 = torch.sin((1.0 - t_col) * theta)
    s1 = torch.sin(t_col * theta)

    slerp_val = (s0 * x0 + s1 * x1) / sin_theta.clamp(min=_SLERP_EPS)
    lerp_val = F.normalize((1.0 - t_col) * x0 + t_col * x1, dim=-1)

    near_collinear = (sin_theta.abs() < _SLERP_EPS)
    return torch.where(near_collinear, lerp_val, slerp_val)

def slerp_tangent(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    dot = (x0 * x1).sum(dim=-1, keepdim=True).clamp(-1 + _SLERP_EPS, 1 - _SLERP_EPS)
    theta = torch.acos(dot)
    sin_theta = theta.sin()

    t_col = t[:, None]

    scale = theta / sin_theta.clamp(min=_SLERP_EPS)
    c0 = -torch.cos((1.0 - t_col) * theta)
    c1 = torch.cos(t_col * theta)
    tangent_val = scale * (c0 * x0 + c1 * x1)

    near_collinear = (sin_theta.abs() < _SLERP_EPS)
    return torch.where(near_collinear, x1 - x0, tangent_val)

def exp_map(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:

    v_tan = project_tangent(x, v)
    norm_v = v_tan.norm(dim=-1, keepdim=True)
    v_hat = v_tan / norm_v.clamp(min=_SLERP_EPS)
    result = torch.cos(norm_v) * x + torch.sin(norm_v) * v_hat

    small = (norm_v < _SLERP_EPS)
    fallback = F.normalize(x + v_tan, dim=-1)
    return torch.where(small, fallback, result)

def log_map(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dot = (x * y).sum(dim=-1, keepdim=True).clamp(-1 + _SLERP_EPS, 1 - _SLERP_EPS)
    theta = torch.acos(dot)
    sin_theta = theta.sin()
    scale = theta / sin_theta.clamp(min=_SLERP_EPS)
    tangent_val = scale * (y - dot * x)
    near_collinear = (sin_theta.abs() < _SLERP_EPS)
    return torch.where(near_collinear, y - x, tangent_val)

def project_tangent(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    dot = (v * x).sum(dim=-1, keepdim=True)
    return v - dot * x

class _SinEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half) / max(half - 1, 1))
        self.register_buffer("freqs", freqs)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:

        x = t[:, None] * self.freqs[None, :]
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        return self.proj(x)

class _Block(nn.Module):
    def __init__(self, dim: int, heads: int, ctx_dim: int, drop: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True,
                                                kdim=ctx_dim, vdim=ctx_dim)

        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim * 4, dim),
            nn.Dropout(drop),
        )

        self.ada_norm = nn.Linear(dim, 6 * dim, bias=True)
        nn.init.zeros_(self.ada_norm.weight)
        nn.init.zeros_(self.ada_norm.bias)

    def _modulate(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor,
                t_emb: torch.Tensor) -> torch.Tensor:

        ada = self.ada_norm(t_emb).clamp(-6, 6)
        s1, g1, s2, g2, s3, g3 = ada.chunk(6, dim=-1)

        h = self._modulate(self.norm1(x), s1, g1)
        h, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + h

        h = self._modulate(self.norm2(x), s2, g2)
        h, _ = self.cross_attn(h, ctx, ctx, need_weights=False)
        x = x + h

        h = self._modulate(self.norm3(x), s3, g3)
        h = self.mlp(h)
        x = x + h

        return x

class ClipPrior(nn.Module):
    def __init__(
        self,
        clip_dim: int = 768,
        ctx_dim: int = 768,
        dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        dropout: float = 0.1,
        cfg_drop: float = 0.10,
        geometry: str = "euclidean",
        prior_target: str = "cls",
        patch_tokens: int = 256,
        patch_dim: int = 1024,
        patch_valid_tokens: int = 196,
    ):
        super().__init__()
        if prior_target not in ("cls", "patches"):
            raise ValueError(
                f"prior_target must be 'cls' or 'patches', got {prior_target!r}"
            )
        if geometry not in ("euclidean", "sphere"):
            raise ValueError(
                f"geometry must be 'euclidean' or 'sphere', got {geometry!r}"
            )
        if prior_target == "patches":

            geometry = "euclidean"
        self.clip_dim = clip_dim
        self.ctx_dim = ctx_dim
        self.cfg_drop = cfg_drop
        self.geometry = geometry
        self.prior_target = prior_target
        self.patch_tokens = patch_tokens
        self.patch_dim = patch_dim
        self.patch_valid_tokens = min(patch_tokens, patch_valid_tokens)
        self.target_dim = clip_dim if prior_target == "cls" else patch_dim

        self.x_proj = nn.Linear(self.target_dim, dim)

        self.t_emb = _SinEmb(dim)

        self.null_ctx = nn.Parameter(torch.randn(1, 1, ctx_dim) * 0.02)

        self.blocks = nn.ModuleList([
            _Block(dim, heads, ctx_dim, drop=dropout)
            for _ in range(depth)
        ])

        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, self.target_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.register_buffer("clip_mean", torch.zeros(clip_dim))
        self.register_buffer("clip_std", torch.ones(clip_dim))
        self._stats_fitted = False

    def fit_stats(self, clip_embs: torch.Tensor):
        if self.prior_target != "cls":
            return
        clip_embs = clip_embs.float()
        self.clip_mean.copy_(clip_embs.mean(0))
        self.clip_std.copy_(clip_embs.std(0).clamp(min=1e-2))
        self._stats_fitted = True

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self._stats_fitted:
            return x
        return (x - self.clip_mean) / self.clip_std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self._stats_fitted:
            return x
        return x * self.clip_std + self.clip_mean

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        ctx: torch.Tensor,
        force_drop: bool = False,
    ) -> torch.Tensor:
        B = x.shape[0]

        if self.training and self.cfg_drop > 0.0:
            drop_mask = torch.rand(B, device=x.device) < self.cfg_drop
            null = self.null_ctx.expand(B, ctx.shape[1], self.ctx_dim)
            ctx = torch.where(drop_mask[:, None, None], null, ctx)
        elif force_drop:
            ctx = self.null_ctx.expand(B, ctx.shape[1], self.ctx_dim)

        if self.prior_target == "cls":
            h = self.x_proj(x).unsqueeze(1)
        else:
            h = self.x_proj(x)
        t_emb = self.t_emb(t)

        for block in self.blocks:
            h = block(h, ctx, t_emb)

        if self.prior_target == "cls":
            h = self.out_norm(h.squeeze(1))
            return self.out_proj(h)

        h = self.out_norm(h)
        return self.out_proj(h)

    def patches_to_embedding(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        return patch_tokens[:, :self.patch_valid_tokens].mean(dim=1)

    def flow_loss(
        self,
        clip_emb: torch.Tensor,
        ctx: torch.Tensor,
        sigma_min: float = 1e-4,
    ) -> torch.Tensor:
        B = clip_emb.shape[0]
        if self.prior_target == "patches":

            with torch.cuda.amp.autocast(enabled=False):
                x1 = clip_emb.float()
                eps = torch.randn_like(x1)
                t = torch.rand(B, device=x1.device)
                xt = (1 - (1 - sigma_min) * t[:, None, None]) * eps + t[:, None, None] * x1
                ut = x1 - (1 - sigma_min) * eps
            v_pred = self.forward(xt.to(clip_emb.dtype), t, ctx)
            return F.mse_loss(v_pred.float(), ut)

        with torch.cuda.amp.autocast(enabled=False):
            if self.geometry == "sphere":

                x1 = F.normalize(clip_emb.float(), dim=-1)
                x0 = random_sphere((B, self.clip_dim), device=x1.device)
                t = torch.rand(B, device=x1.device)
                xt = slerp(x0, x1, t)
                ut = slerp_tangent(x0, x1, t)
                v_pred = self.forward(xt.to(clip_emb.dtype), t, ctx)

                v_proj = project_tangent(xt, v_pred.float())
                return F.mse_loss(v_proj, ut)
            else:

                x1 = self.normalize(clip_emb.float())
                eps = torch.randn_like(x1)
                t = torch.rand(B, device=x1.device)

                xt = (1 - (1 - sigma_min) * t[:, None]) * eps + t[:, None] * x1

                ut = x1 - (1 - sigma_min) * eps

        v_pred = self.forward(xt.to(clip_emb.dtype), t, ctx)
        return F.mse_loss(v_pred.float(), ut)

    @torch.no_grad()
    def sample(
        self,
        ctx: torch.Tensor,
        n_steps: int = 50,
        cfg_scale: float = 1.5,
        sigma_min: float = 1e-4,
        normalize_output: bool = True,
    ) -> torch.Tensor:
        B = ctx.shape[0]
        device = ctx.device
        dt = 1.0 / n_steps

        if self.geometry == "sphere":
            x = random_sphere((B, self.clip_dim), device=device)

            for i in range(n_steps):
                t_val = i / n_steps
                t = torch.full((B,), t_val, device=device)

                v_cond = self.forward(x, t, ctx, force_drop=False)
                if cfg_scale > 1.0:
                    v_uncond = self.forward(x, t, ctx, force_drop=True)
                    v = v_uncond + cfg_scale * (v_cond - v_uncond)
                else:
                    v = v_cond

                v_proj = project_tangent(x, v)
                x = exp_map(x, v_proj * dt)

            return F.normalize(x, dim=-1)

        else:

            if self.prior_target == "cls":
                x = torch.randn(B, self.clip_dim, device=device)
            else:
                x = torch.randn(B, self.patch_tokens, self.patch_dim, device=device)

            for i in range(n_steps):
                t_val = i / n_steps
                t = torch.full((B,), t_val, device=device)

                v_cond = self.forward(x, t, ctx, force_drop=False)
                if cfg_scale > 1.0:
                    v_uncond = self.forward(x, t, ctx, force_drop=True)
                    v = v_uncond + cfg_scale * (v_cond - v_uncond)
                else:
                    v = v_cond

                x = x + v * dt

            if self.prior_target == "cls":

                x = self.denormalize(x)

                if normalize_output:
                    x = F.normalize(x, dim=-1)

            return x
