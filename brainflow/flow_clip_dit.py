from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .vfm import flow_loss as vfm_flow_loss

CLIP_GRID_H = 16
CLIP_GRID_W = 16
CLIP_TOKEN_DIM = 1024

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half).float() / (half - 1))
        self.register_buffer("freqs", freqs)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(-1) * self.freqs
        return self.mlp(torch.cat([t.sin(), t.cos()], dim=-1))

class CrossAttention(nn.Module):
    def __init__(self, qd: int, cd: int, nh: int = 8, hd: int = 64):
        super().__init__()
        inner = nh * hd
        self.nh = nh
        self.hd = hd
        self.to_q = nn.Linear(qd, inner, bias=False)
        self.to_k = nn.Linear(cd, inner, bias=False)
        self.to_v = nn.Linear(cd, inner, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, qd), nn.Dropout(0.05))
        self.norm = nn.LayerNorm(qd)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        res = x
        x = self.norm(x)

        def rsh(t, s):
            return t.view(B, s, self.nh, self.hd).transpose(1, 2)

        Q = rsh(self.to_q(x), L)
        K = rsh(self.to_k(ctx), ctx.shape[1])
        V = rsh(self.to_v(ctx), ctx.shape[1])
        out = F.scaled_dot_product_attention(Q, K, V)
        return res + self.to_out(out.transpose(1, 2).reshape(B, L, -1))

class DiTBlock(nn.Module):
    def __init__(self, d: int, nh: int, brain_dim: int, td: int):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(d, nh, batch_first=True, dropout=0.0)
        self.cross_attn = CrossAttention(d, brain_dim, nh)
        self.n3 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d),
        )

        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(td, 6 * d))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x: torch.Tensor, te: torch.Tensor,
                brain_ctx: torch.Tensor) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = \
            self.ada(te).chunk(6, dim=-1)

        h = self.n1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        h_out, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + gate_a[:, None] * h_out

        x = self.cross_attn(x, brain_ctx)

        h = self.n3(x) * (1 + scale_m[:, None]) + shift_m[:, None]
        x = x + gate_m[:, None] * self.mlp(h)
        return x

class CLSHead(nn.Module):
    def __init__(self, in_dim: int, cls_dim: int = CLIP_TOKEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(in_dim * 2, cls_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1)
        return self.net(pooled)

class FlowCLIPDiT(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()

        d = getattr(cfg, "clip_dit_dim", cfg.dit_dim)
        nh = getattr(cfg, "clip_dit_heads", cfg.dit_heads)
        depth = getattr(cfg, "clip_dit_depth", cfg.dit_depth)
        bd = cfg.brain_dim
        td = cfg.time_emb_dim
        self.d = d
        self.grid_h = CLIP_GRID_H
        self.grid_w = CLIP_GRID_W
        self.token_dim = CLIP_TOKEN_DIM
        self.flow_objective = getattr(cfg, "flow_objective", "vfm")
        self.lambda_cos = getattr(cfg, "lambda_cos", 0.1)
        self.lambda_cls = getattr(cfg, "lambda_cls", 0.1)
        self.brain_dim = bd

        out_mult = 2 if self.flow_objective == "vfm" else 1

        self.te = SinusoidalTimeEmbedding(td)

        self.patch_embed = nn.Conv2d(CLIP_TOKEN_DIM, d, kernel_size=1)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, CLIP_GRID_H * CLIP_GRID_W, d)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.brain_proj = nn.Linear(bd, bd)

        self.blocks = nn.ModuleList([
            DiTBlock(d, nh, bd, td) for _ in range(depth)
        ])

        self.final_norm = nn.LayerNorm(d, elementwise_affine=False)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(td, 2 * d))
        nn.init.zeros_(self.final_ada[-1].weight)
        nn.init.zeros_(self.final_ada[-1].bias)

        self.final_proj = nn.Linear(d, CLIP_TOKEN_DIM * out_mult)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

        self.cls_head = CLSHead(d, cfg.clip_dim)

        self.null_tokens = nn.Parameter(
            torch.randn(1, cfg.n_tokens, bd) * 0.01
        )

        self.register_buffer(
            "clip_token_mean",
            torch.zeros(CLIP_TOKEN_DIM),
        )
        self.register_buffer(
            "clip_token_std",
            torch.ones(CLIP_TOKEN_DIM),
        )
        self._standardization_fitted = False

    def fit_standardization(self, token_grid_samples: torch.Tensor) -> None:
        x = token_grid_samples.float().reshape(-1, CLIP_TOKEN_DIM)
        self.clip_token_mean.copy_(x.mean(0))
        self.clip_token_std.copy_(x.std(0).clamp(min=1e-6))
        self._standardization_fitted = True

    def standardize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.clip_token_mean) / self.clip_token_std

    def destandardize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.clip_token_std + self.clip_token_mean

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                brain_ctx: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        te = self.te(t)

        h = self.patch_embed(x)

        h = h.flatten(2).transpose(1, 2) + self.pos_embed

        for blk in self.blocks:
            h = blk(h, te, brain_ctx)

        shift, scale = self.final_ada(te).chunk(2, dim=-1)
        h_norm = self.final_norm(h) * (1 + scale[:, None]) + shift[:, None]

        out_tokens = self.final_proj(h_norm)

        out_mult = 2 if self.flow_objective == "vfm" else 1
        out = out_tokens.transpose(1, 2).reshape(
            B, CLIP_TOKEN_DIM * out_mult, self.grid_h, self.grid_w
        )
        return out

    def flow_loss(self, clip_grid_gt: torch.Tensor,
                  brain_ctx: torch.Tensor,
                  clip_cls_gt: torch.Tensor,
                  x0: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        B = clip_grid_gt.shape[0]
        device = clip_grid_gt.device
        if x0 is None:
            x0 = torch.randn_like(clip_grid_gt)
        t = torch.rand(B, device=device)
        t_exp = t[:, None, None, None]
        xt = (1 - t_exp) * x0 + t_exp * clip_grid_gt
        ut = clip_grid_gt - x0

        raw = self.forward(xt, t, brain_ctx)

        if self.flow_objective == "vfm":
            from .vfm import vfm_loss, velocity_from_posterior, VFMHead
            head = VFMHead(spatial=True)
            mu, log_sigma = head(raw)
            loss_mse = vfm_loss(mu, log_sigma, clip_grid_gt)
            v_pred = velocity_from_posterior(mu, xt, t)
        else:
            v_pred = raw
            loss_mse = F.mse_loss(v_pred, ut)

        def to_tokens(z):
            return z.permute(0, 2, 3, 1).reshape(-1, CLIP_TOKEN_DIM)

        loss_cos = (1.0 - F.cosine_similarity(
            to_tokens(v_pred), to_tokens(ut), dim=-1
        ).mean())

        loss_cls = self._cls_head_loss(clip_grid_gt, brain_ctx, clip_cls_gt, t)

        total = loss_mse + self.lambda_cos * loss_cos + self.lambda_cls * loss_cls
        return {
            "loss": total,
            "loss_mse": loss_mse,
            "loss_cos": loss_cos,
            "loss_cls": loss_cls,
        }

    def _cls_head_loss(self, clip_grid_gt: torch.Tensor,
                       brain_ctx: torch.Tensor,
                       clip_cls_gt: torch.Tensor,
                       t: torch.Tensor) -> torch.Tensor:
        B = clip_grid_gt.shape[0]
        te = self.te(t)
        cls_target = F.normalize(clip_cls_gt, dim=-1)

        x0 = torch.randn_like(clip_grid_gt)
        t_exp = t[:, None, None, None]
        xt = (1 - t_exp) * x0 + t_exp * clip_grid_gt

        h = self.patch_embed(xt).flatten(2).transpose(1, 2) + self.pos_embed
        for blk in self.blocks:
            h = blk(h, te, brain_ctx)

        cls_pred = self.cls_head(h)
        loss_mse = F.mse_loss(cls_pred, cls_target)
        loss_cos = 1.0 - F.cosine_similarity(cls_pred, cls_target, dim=-1).mean()
        return loss_mse + loss_cos

    @torch.no_grad()
    def sample(self, brain_ctx: torch.Tensor, n_steps: int = 20,
               cfg_scale: float = 1.0,
               solver: str = "euler") -> torch.Tensor:
        B = brain_ctx.shape[0]
        device = brain_ctx.device
        x = torch.randn(B, CLIP_TOKEN_DIM, self.grid_h, self.grid_w, device=device)
        null_ctx = self.null_tokens.expand(B, -1, -1)
        dt = 1.0 / n_steps

        def _vel(xt, t_val):
            t = torch.full((B,), t_val, device=device)
            raw_cond = self.forward(xt, t, brain_ctx)
            if cfg_scale > 1.0:
                raw_uncond = self.forward(xt, t, null_ctx)

                if self.flow_objective == "vfm":
                    mu_c = raw_cond.chunk(2, dim=1)[0]
                    mu_u = raw_uncond.chunk(2, dim=1)[0]
                    from .vfm import velocity_from_posterior
                    v_c = velocity_from_posterior(mu_c, xt, t)
                    v_u = velocity_from_posterior(mu_u, xt, t)
                else:
                    v_c = raw_cond
                    v_u = raw_uncond
                return v_u + cfg_scale * (v_c - v_u)

            if self.flow_objective == "vfm":
                mu = raw_cond.chunk(2, dim=1)[0]
                from .vfm import velocity_from_posterior
                return velocity_from_posterior(mu, xt, t)
            return raw_cond

        for i in range(n_steps):
            if solver == "euler":
                v = _vel(x, i * dt)
                x = x + v * dt
            elif solver == "midpoint":
                v = _vel(x, (i + 0.5) * dt)
                x = x + v * dt
            elif solver == "heun":
                t_cur, t_next = i * dt, (i + 1) * dt
                v1 = _vel(x, t_cur)
                x_euler = x + v1 * dt
                v2 = _vel(x_euler, t_next)
                x = x + (v1 + v2) * (dt / 2)
            else:
                raise ValueError(f"Unknown solver: {solver!r}")

        return x

    @torch.no_grad()
    def predict_cls(self, brain_ctx: torch.Tensor,
                    n_steps: int = 20) -> torch.Tensor:
        patch_tokens = self.sample(brain_ctx, n_steps=n_steps)
        B = brain_ctx.shape[0]
        t = torch.zeros(B, device=brain_ctx.device)
        te = self.te(t)
        h = self.patch_embed(patch_tokens).flatten(2).transpose(1, 2) + self.pos_embed
        for blk in self.blocks:
            h = blk(h, te, brain_ctx)
        cls = self.cls_head(h)
        return F.normalize(cls, dim=-1)
