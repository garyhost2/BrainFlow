from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config

def _double_gamma_hrf(length: int, peak: float = 6.0, undershoot: float = 16.0,
                      ratio: float = 6.0, dt: float = 1.0) -> torch.Tensor:
    import numpy as np
    t = np.arange(length, dtype=np.float64) * dt + dt / 2
    def gamma_pdf(t, k):
        return np.exp((k - 1) * np.log(t + 1e-12) - t - math.lgamma(k))
    h = gamma_pdf(t, peak) - gamma_pdf(t, undershoot) / ratio
    h = h / (abs(h).sum() + 1e-12)
    return torch.tensor(h, dtype=torch.float32)

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

class UNetResBlock(nn.Module):
    def __init__(self, ic: int, oc: int, td: int):
        super().__init__()
        self.n1 = nn.GroupNorm(min(32, ic), ic)
        self.c1 = nn.Conv2d(ic, oc, 3, padding=1)
        self.n2 = nn.GroupNorm(min(32, oc), oc)
        self.c2 = nn.Conv2d(oc, oc, 3, padding=1)
        self.act = nn.SiLU()
        self.tp = nn.Linear(td, oc * 2)
        self.skip = nn.Conv2d(ic, oc, 1) if ic != oc else nn.Identity()
        self.dropout = nn.Dropout2d(0.05)

    def forward(self, x: torch.Tensor, te: torch.Tensor) -> torch.Tensor:
        h = self.act(self.n1(x))
        h = self.c1(h)
        sc, sh = self.act(self.tp(te))[:, :, None, None].chunk(2, dim=1)
        h = self.n2(h) * (1 + sc) + sh
        h = self.act(h)
        h = self.dropout(h)
        h = self.c2(h)
        return h + self.skip(x)

class FlowUNet(nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        ic = cfg.latent_ch
        bc = cfg.unet_base_ch
        bd = cfg.brain_dim
        td = cfg.time_emb_dim
        nh = cfg.attn_heads
        chs = [bc, bc * 2, bc * 4, bc * 4]
        clip_token_dim = 1024

        self.te = SinusoidalTimeEmbedding(td)
        self.ip = nn.Conv2d(ic, bc, 3, padding=1)
        self.null_tokens = nn.Parameter(torch.randn(1, cfg.n_tokens, bd) * 0.01)

        self.eb = nn.ModuleList()
        self.ea = nn.ModuleList()
        self.ea_clip = nn.ModuleList()
        self.ed = nn.ModuleList()

        ch = bc
        for i, co in enumerate(chs):
            self.eb.append(UNetResBlock(ch, co, td))
            self.ea.append(CrossAttention(co, bd, nh))
            self.ea_clip.append(CrossAttention(co, bd, nh))
            self.ed.append(
                nn.Conv2d(co, co, 4, stride=2, padding=1)
                if i < len(chs) - 1 else nn.Identity()
            )
            ch = co

        self.m1 = UNetResBlock(chs[-1], chs[-1], td)
        self.ma = CrossAttention(chs[-1], bd, nh)
        self.ma_clip = CrossAttention(chs[-1], bd, nh)
        self.m2 = UNetResBlock(chs[-1], chs[-1], td)

        self.du = nn.ModuleList()
        self.db = nn.ModuleList()
        self.da = nn.ModuleList()
        self.da_clip = nn.ModuleList()

        for i, co in enumerate(reversed(chs)):
            self.du.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(ch, ch, 3, padding=1),
            ) if i > 0 else nn.Identity())
            self.db.append(UNetResBlock(ch + chs[len(chs) - 1 - i], co, td))
            self.da.append(CrossAttention(co, bd, nh))
            self.da_clip.append(CrossAttention(co, bd, nh))
            ch = co

        self.on = nn.GroupNorm(min(32, bc), bc)
        self.op = nn.Conv2d(bc, ic, 3, padding=1)
        nn.init.zeros_(self.op.weight)
        nn.init.zeros_(self.op.bias)

        self.clip_context_proj = nn.Linear(clip_token_dim, bd)

        self._hrf_enabled = (getattr(cfg, "method", "baseline") == "hrf")
        if self._hrf_enabled:
            hrf_len = int(getattr(cfg, "hrf_len", cfg.n_tokens))
            self._latent_res = cfg.latent_res
            self.register_buffer(
                "hrf_kernel",
                _double_gamma_hrf(hrf_len).view(1, 1, hrf_len),
                persistent=False,
            )
            self.hrf_proj = nn.Sequential(
                nn.LayerNorm(bd),
                nn.Linear(bd, ic * cfg.latent_res),
                nn.GELU(),
            )
            self.hrf_to_map = nn.Linear(cfg.n_tokens, cfg.latent_res, bias=False)
            nn.init.zeros_(self.hrf_to_map.weight)

    def _hrf_bias(self, ctx: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, N, D = ctx.shape
        x = ctx.transpose(1, 2)
        kernel = self.hrf_kernel.expand(D, 1, -1)
        x = F.pad(x, (kernel.shape[-1] - 1, 0))
        x = F.conv1d(x, kernel, groups=D)
        x = x.transpose(1, 2)
        rows = self.hrf_proj(x)
        ic_x_H = rows.shape[-1]
        H = self._latent_res
        ic = ic_x_H // H
        rows = rows.view(B, N, ic, H).permute(0, 2, 3, 1)
        bias = self.hrf_to_map(rows)
        gate = torch.sin(math.pi * t).clamp(min=0.0)[:, None, None, None]
        return gate * bias

    def _apply_attn(self, h: torch.Tensor, attn: CrossAttention,
                    ctx: torch.Tensor) -> torch.Tensor:
        B, C, H, W = h.shape
        h_flat = h.permute(0, 2, 3, 1).reshape(B, H * W, C)
        h_flat = attn(h_flat, ctx)
        return h_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                brain_ctx: torch.Tensor,
                clip_ctx: torch.Tensor | None = None) -> torch.Tensor:

        if clip_ctx is not None:
            clip_ctx_proj = self.clip_context_proj(clip_ctx)
        else:
            clip_ctx_proj = None

        te = self.te(t)
        h = self.ip(x)
        sk = []

        for res, attn_b, attn_c, dn in zip(self.eb, self.ea, self.ea_clip, self.ed):
            h = res(h, te)
            h = self._apply_attn(h, attn_b, brain_ctx)
            if clip_ctx_proj is not None:
                h = self._apply_attn(h, attn_c, clip_ctx_proj)
            sk.append(h)
            h = dn(h)

        h = self.m1(h, te)
        h = self._apply_attn(h, self.ma, brain_ctx)
        if clip_ctx_proj is not None:
            h = self._apply_attn(h, self.ma_clip, clip_ctx_proj)
        h = self.m2(h, te)

        for i, (up, res, attn_b, attn_c) in enumerate(
            zip(self.du, self.db, self.da, self.da_clip)
        ):
            h = up(h)
            h = torch.cat([h, sk[-(i + 1)]], dim=1)
            h = res(h, te)
            h = self._apply_attn(h, attn_b, brain_ctx)
            if clip_ctx_proj is not None:
                h = self._apply_attn(h, attn_c, clip_ctx_proj)

        out = self.op(F.silu(self.on(h)))
        if self._hrf_enabled:
            out = out + self._hrf_bias(brain_ctx, t)
        return out
