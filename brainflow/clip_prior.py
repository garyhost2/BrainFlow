"""ClipPrior — CFM DiT that maps fMRI brain tokens → CLIP CLS embedding.

Phase 3 SDPrior experiment:
  BrainEncoder (frozen) → brain_tokens (B, N, D)
  ClipPrior (trained)   → sampled CLIP CLS embedding (B, 768)
  Karlo decoder (frozen) → image (256×256)

Architecture: ~20M-param DiT with cross-attention conditioning on brain tokens.
Training: conditional flow matching (CFM) in CLIP CLS space.
"""
from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Sinusoidal time embedding ──────────────────────────────────────────────────

class _SinEmb(nn.Module):
    """Sinusoidal embedding for scalar time t ∈ [0, 1] → (B, dim)."""
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half) / max(half - 1, 1))
        self.register_buffer("freqs", freqs)  # (half,)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) float in [0, 1]
        x = t[:, None] * self.freqs[None, :]  # (B, half)
        x = torch.cat([x.sin(), x.cos()], dim=-1)  # (B, dim)
        return self.proj(x)


# ── Single DiT block (self-attn + cross-attn + MLP) ───────────────────────────

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

        # AdaLN-Zero modulation from time embedding
        self.ada_norm = nn.Linear(dim, 6 * dim, bias=True)
        nn.init.zeros_(self.ada_norm.weight)
        nn.init.zeros_(self.ada_norm.bias)

    def _modulate(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor,
                t_emb: torch.Tensor) -> torch.Tensor:
        # AdaLN modulation: 6 chunks → shift/scale for self, cross, mlp
        # Clamp to prevent fp16 overflow (values > ~60k overflow float16)
        ada = self.ada_norm(t_emb).clamp(-6, 6)  # (B, 6*dim)
        s1, g1, s2, g2, s3, g3 = ada.chunk(6, dim=-1)

        # Self-attention
        h = self._modulate(self.norm1(x), s1, g1)
        h, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + h

        # Cross-attention on brain tokens
        h = self._modulate(self.norm2(x), s2, g2)
        h, _ = self.cross_attn(h, ctx, ctx, need_weights=False)
        x = x + h

        # MLP
        h = self._modulate(self.norm3(x), s3, g3)
        h = self.mlp(h)
        x = x + h

        return x


# ── ClipPrior ─────────────────────────────────────────────────────────────────

class ClipPrior(nn.Module):
    """CFM DiT: brain_tokens → CLIP CLS embedding.

    Args:
        clip_dim:   Dimension of the CLIP CLS embedding (ViT-L/14 = 768).
        ctx_dim:    Dimension of brain tokens (brain_dim = 768 by default).
        dim:        Internal hidden dimension of the DiT.
        depth:      Number of DiT blocks.
        heads:      Attention heads per block.
        dropout:    Dropout probability.
        cfg_drop:   Probability of dropping context (CFG null conditioning).
    """
    def __init__(
        self,
        clip_dim: int = 768,
        ctx_dim: int = 768,
        dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        dropout: float = 0.1,
        cfg_drop: float = 0.10,
    ):
        super().__init__()
        self.clip_dim = clip_dim
        self.ctx_dim = ctx_dim
        self.cfg_drop = cfg_drop

        # Project noisy CLIP embedding to internal dim
        self.x_proj = nn.Linear(clip_dim, dim)
        # Time embedding
        self.t_emb = _SinEmb(dim)
        # Null context token (for CFG dropout)
        self.null_ctx = nn.Parameter(torch.randn(1, 1, ctx_dim) * 0.02)

        self.blocks = nn.ModuleList([
            _Block(dim, heads, ctx_dim, drop=dropout)
            for _ in range(depth)
        ])

        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, clip_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Running statistics for CLIP embedding normalization
        self.register_buffer("clip_mean", torch.zeros(clip_dim))
        self.register_buffer("clip_std", torch.ones(clip_dim))
        self._stats_fitted = False

    # ── Normalization helpers ──────────────────────────────────────────────────

    def fit_stats(self, clip_embs: torch.Tensor):
        """Fit mean/std from a (N, clip_dim) tensor of training CLIP embeddings."""
        clip_embs = clip_embs.float()
        self.clip_mean.copy_(clip_embs.mean(0))
        self.clip_std.copy_(clip_embs.std(0).clamp(min=1e-6))
        self._stats_fitted = True

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self._stats_fitted:
            return x
        return (x - self.clip_mean) / self.clip_std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if not self._stats_fitted:
            return x
        return x * self.clip_std + self.clip_mean

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,           # (B, clip_dim) — noisy CLIP embedding
        t: torch.Tensor,           # (B,) — flow time in [0, 1]
        ctx: torch.Tensor,         # (B, N, ctx_dim) — brain tokens
        force_drop: bool = False,  # force CFG null condition (for CFG eval)
    ) -> torch.Tensor:
        B = x.shape[0]

        # CFG context dropout during training
        if self.training and self.cfg_drop > 0.0:
            drop_mask = torch.rand(B, device=x.device) < self.cfg_drop
            null = self.null_ctx.expand(B, ctx.shape[1], self.ctx_dim)
            ctx = torch.where(drop_mask[:, None, None], null, ctx)
        elif force_drop:
            ctx = self.null_ctx.expand(B, ctx.shape[1], self.ctx_dim)

        h = self.x_proj(x).unsqueeze(1)  # (B, 1, dim)
        t_emb = self.t_emb(t)            # (B, dim)

        for block in self.blocks:
            h = block(h, ctx, t_emb)

        h = self.out_norm(h.squeeze(1))  # (B, dim)
        return self.out_proj(h)          # (B, clip_dim) — predicted velocity

    # ── CFM loss ──────────────────────────────────────────────────────────────

    def flow_loss(
        self,
        clip_emb: torch.Tensor,    # (B, clip_dim) — GT CLIP embedding
        ctx: torch.Tensor,         # (B, N, ctx_dim) — brain tokens
        sigma_min: float = 1e-4,
    ) -> torch.Tensor:
        """Conditional flow matching loss: MSE between predicted and target velocity.

        Straight-line (optimal transport) interpolant:
            x_t = (1 - (1 - σ_min) * t) * ε + t * x_1
            u_t = x_1 - (1 - σ_min) * ε    (constant velocity field)
        """
        B = clip_emb.shape[0]
        x1 = self.normalize(clip_emb)
        eps = torch.randn_like(x1)
        t = torch.rand(B, device=x1.device)

        # Interpolate
        xt = (1 - (1 - sigma_min) * t[:, None]) * eps + t[:, None] * x1
        # Target velocity
        ut = x1 - (1 - sigma_min) * eps

        v_pred = self.forward(xt, t, ctx)
        return F.mse_loss(v_pred, ut)

    # ── ODE sampling ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        ctx: torch.Tensor,             # (B, N, ctx_dim) — brain tokens
        n_steps: int = 50,
        cfg_scale: float = 1.5,
        sigma_min: float = 1e-4,
        normalize_output: bool = True,
    ) -> torch.Tensor:
        """Euler ODE sampler with optional CFG.

        Returns CLIP embeddings (B, clip_dim) — NOT normalized (denormalized to
        original CLIP space, then L2-normalized for cosine similarity).
        """
        B = ctx.shape[0]
        device = ctx.device
        x = torch.randn(B, self.clip_dim, device=device)
        dt = 1.0 / n_steps

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

        # Denormalize from training space → original CLIP space
        x = self.denormalize(x)

        if normalize_output:
            x = F.normalize(x, dim=-1)

        return x
