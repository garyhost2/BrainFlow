"""Flow_CLIP DiT: flow-matching in ViT-L/14 patch-token space.

Target space: ViT-L/14 patch tokens reshaped to a 16×16×1024 2D grid
(CLS token dropped). The model learns to map from Gaussian noise → CLIP
patch tokens conditioned on brain tokens (B, 64, brain_dim).

Architecture:
  - Conv2d patch embed (1×1 → projects 1024-dim tokens to dit_dim)
  - Learned 2D positional embedding
  - N DiT blocks with adaLN-zero + cross-attention to brain tokens
  - Auxiliary CLS MLP head (small MLP predicting the CLIP CLS token)
  - Per-channel CLIP standardization (training-set mean/std as buffers)
  - Learned null context embedding for classifier-free guidance

VFM is the default objective (CFM toggleable for ablation via
cfg.flow_objective ∈ {"vfm", "cfm"}).

Cosine + MSE hybrid loss:
    L = MSE + λ_cos · (1 - cos(pred, target))   per token
    λ_cos configurable (default 0.1).

Reference: rg-vfm (ICLR 2026) for the VFM formulation.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .vfm import flow_loss as vfm_flow_loss


# ────────────────────────────────────────────────────────────────────────────
# Constants for ViT-L/14 patch token grid
# ────────────────────────────────────────────────────────────────────────────
CLIP_GRID_H = 16    # patch rows
CLIP_GRID_W = 16    # patch cols
CLIP_TOKEN_DIM = 1024   # ViT-L/14 embedding dim


# ────────────────────────────────────────────────────────────────────────────
# Building blocks
# ────────────────────────────────────────────────────────────────────────────

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
    """Cross-attention from flow tokens to brain conditioning tokens."""
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
    """DiT block: adaLN-zero self-attn + cross-attn to brain tokens + MLP."""
    def __init__(self, d: int, nh: int, brain_dim: int, td: int):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(d, nh, batch_first=True, dropout=0.0)
        self.cross_attn = CrossAttention(d, brain_dim, nh)
        self.n3 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d),
        )
        # adaLN-zero: 6 modulation params (shift/scale/gate for attn+mlp)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(td, 6 * d))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x: torch.Tensor, te: torch.Tensor,
                brain_ctx: torch.Tensor) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = \
            self.ada(te).chunk(6, dim=-1)
        # Self-attention with adaLN-zero
        h = self.n1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        h_out, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + gate_a[:, None] * h_out
        # Cross-attention to brain tokens (has its own LayerNorm)
        x = self.cross_attn(x, brain_ctx)
        # MLP with adaLN-zero
        h = self.n3(x) * (1 + scale_m[:, None]) + shift_m[:, None]
        x = x + gate_m[:, None] * self.mlp(h)
        return x


class CLSHead(nn.Module):
    """Auxiliary MLP head predicting the CLIP CLS token from pooled DiT output.

    Provides a clean inference fallback for the CLS token when the flow
    target is the patch-token grid only (CLS dropped from flow target).
    Supervised separately with MSE + cosine loss.
    """
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
        """x: (B, N, D) — pool over tokens then project."""
        pooled = x.mean(dim=1)  # (B, D)
        return self.net(pooled)  # (B, cls_dim)


# ────────────────────────────────────────────────────────────────────────────
# Main Flow_CLIP DiT
# ────────────────────────────────────────────────────────────────────────────

class FlowCLIPDiT(nn.Module):
    """Flow-matching DiT operating in ViT-L/14 patch-token space.

    Target: (B, 16, 16, 1024) CLIP patch-token grid (CLS token dropped).
    Condition: brain tokens (B, n_tokens, brain_dim) from BrainEncoder.

    The 16×16×1024 grid is treated as a spatial feature map:
      - A 1×1 Conv2d maps 1024 → dit_dim at each spatial position.
      - N DiT blocks with 2D positional embedding + brain cross-attention.
      - VFM objective (default) or CFM (ablation).
    """

    def __init__(self, cfg: Config):
        super().__init__()
        # Use clip_dit_* fields; fall back to dit_* for backward compat
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
        self.brain_dim = bd

        # Number of output channels: 2× for VFM (mu + log_sigma), 1× for CFM
        out_mult = 2 if self.flow_objective == "vfm" else 1

        # Time embedding
        self.te = SinusoidalTimeEmbedding(td)

        # 1×1 Conv2d patch embed: maps 1024 → dit_dim at each (16×16) position
        self.patch_embed = nn.Conv2d(CLIP_TOKEN_DIM, d, kernel_size=1)

        # Learned 2D positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, CLIP_GRID_H * CLIP_GRID_W, d)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Brain context projector: project brain_dim → dit_dim for cross-attn
        self.brain_proj = nn.Linear(bd, bd)  # passthrough, cross-attn uses bd directly

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(d, nh, bd, td) for _ in range(depth)
        ])

        # Final norm + adaLN + output projection
        self.final_norm = nn.LayerNorm(d, elementwise_affine=False)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(td, 2 * d))
        nn.init.zeros_(self.final_ada[-1].weight)
        nn.init.zeros_(self.final_ada[-1].bias)

        # Output projection: dit_dim → CLIP_TOKEN_DIM (×out_mult for VFM)
        self.final_proj = nn.Linear(d, CLIP_TOKEN_DIM * out_mult)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

        # Auxiliary CLS head
        self.cls_head = CLSHead(d, CLIP_TOKEN_DIM)

        # Learned null context embedding for classifier-free guidance
        self.null_tokens = nn.Parameter(
            torch.randn(1, cfg.n_tokens, bd) * 0.01
        )

        # Per-channel CLIP standardization buffers (filled during training)
        # Shape: (CLIP_TOKEN_DIM,) — one mean/std per channel of the 1024-dim space
        self.register_buffer(
            "clip_token_mean",
            torch.zeros(CLIP_TOKEN_DIM),
        )
        self.register_buffer(
            "clip_token_std",
            torch.ones(CLIP_TOKEN_DIM),
        )
        self._standardization_fitted = False

    # ── Per-channel standardization ──────────────────────────────────────────

    def fit_standardization(self, token_grid_samples: torch.Tensor) -> None:
        """Compute and store per-channel mean/std from training set samples.

        Args:
            token_grid_samples: (N, 16, 16, 1024) or (N, 256, 1024) CLIP tokens
        """
        x = token_grid_samples.float().reshape(-1, CLIP_TOKEN_DIM)
        self.clip_token_mean.copy_(x.mean(0))
        self.clip_token_std.copy_(x.std(0).clamp(min=1e-6))
        self._standardization_fitted = True

    def standardize(self, x: torch.Tensor) -> torch.Tensor:
        """Standardize (B, H, W, 1024) or (B, 256, 1024) tokens."""
        return (x - self.clip_token_mean) / self.clip_token_std

    def destandardize(self, x: torch.Tensor) -> torch.Tensor:
        """Invert standardization."""
        return x * self.clip_token_std + self.clip_token_mean

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                brain_ctx: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x:          noisy CLIP token grid (B, 1024, 16, 16)  in CHW layout
            t:          time values (B,)
            brain_ctx:  brain conditioning tokens (B, n_tokens, brain_dim)

        Returns:
            raw output (B, 1024*out_mult, 16, 16) — split by caller or vfm.flow_loss
        """
        B = x.shape[0]
        te = self.te(t)  # (B, td)

        # Patch embed: (B, 1024, 16, 16) → (B, d, 16, 16)
        h = self.patch_embed(x)
        # Flatten spatial: (B, d, 16, 16) → (B, 256, d)
        h = h.flatten(2).transpose(1, 2) + self.pos_embed  # (B, 256, d)

        # DiT blocks
        for blk in self.blocks:
            h = blk(h, te, brain_ctx)

        # Final adaLN
        shift, scale = self.final_ada(te).chunk(2, dim=-1)  # (B, d) each
        h_norm = self.final_norm(h) * (1 + scale[:, None]) + shift[:, None]

        # Project to output token dim
        out_tokens = self.final_proj(h_norm)  # (B, 256, 1024*out_mult)

        # Reshape back to spatial: (B, 1024*out_mult, 16, 16)
        out_mult = 2 if self.flow_objective == "vfm" else 1
        out = out_tokens.transpose(1, 2).reshape(
            B, CLIP_TOKEN_DIM * out_mult, self.grid_h, self.grid_w
        )
        return out

    # ── Loss computation ──────────────────────────────────────────────────────

    def flow_loss(self, clip_grid_gt: torch.Tensor,
                  brain_ctx: torch.Tensor,
                  x0: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Compute VFM/CFM loss for a batch of CLIP token grids.

        Args:
            clip_grid_gt:  ground-truth CLIP patch tokens (B, 1024, 16, 16)
            brain_ctx:     brain conditioning tokens (B, n_tokens, brain_dim)
            x0:            optional source noise; if None, drawn from N(0,I)

        Returns:
            dict with 'loss', 'loss_mse', 'loss_cos', 'loss_cls' keys
        """
        B = clip_grid_gt.shape[0]
        device = clip_grid_gt.device
        if x0 is None:
            x0 = torch.randn_like(clip_grid_gt)
        t = torch.rand(B, device=device)
        t_exp = t[:, None, None, None]  # broadcast over (C, H, W)
        xt = (1 - t_exp) * x0 + t_exp * clip_grid_gt
        ut = clip_grid_gt - x0  # velocity target for CFM

        raw = self.forward(xt, t, brain_ctx)  # (B, C*out_mult, 16, 16)

        if self.flow_objective == "vfm":
            from .vfm import vfm_loss, velocity_from_posterior, VFMHead
            head = VFMHead(spatial=True)
            mu, log_sigma = head(raw)
            loss_mse = vfm_loss(mu, log_sigma, clip_grid_gt)
            v_pred = velocity_from_posterior(mu, xt, t)
        else:
            v_pred = raw
            loss_mse = F.mse_loss(v_pred, ut)

        # Cosine loss: per-token, averaged
        # Reshape to (B*16*16, 1024) for cosine similarity
        def to_tokens(z):
            return z.permute(0, 2, 3, 1).reshape(-1, CLIP_TOKEN_DIM)

        loss_cos = (1.0 - F.cosine_similarity(
            to_tokens(v_pred), to_tokens(ut), dim=-1
        ).mean())

        # Auxiliary CLS head loss — needs the hidden representation.
        # We do a second forward to get the hidden state for the CLS head.
        # For efficiency, re-use the patch embed output (computed inside forward).
        # Here we compute it separately since forward() only returns raw output.
        loss_cls = self._cls_head_loss(clip_grid_gt, brain_ctx, t)

        total = loss_mse + self.lambda_cos * loss_cos + 0.1 * loss_cls
        return {
            "loss": total,
            "loss_mse": loss_mse,
            "loss_cos": loss_cos,
            "loss_cls": loss_cls,
        }

    def _cls_head_loss(self, clip_grid_gt: torch.Tensor,
                       brain_ctx: torch.Tensor,
                       t: torch.Tensor) -> torch.Tensor:
        """Auxiliary CLS head loss (MSE + cosine on predicted CLS token).

        The CLS head predicts from the DiT's pooled hidden state, not from
        the output tokens. We run a partial forward (up to final_norm) to
        obtain the hidden state, then apply the CLS head.
        """
        B = clip_grid_gt.shape[0]
        te = self.te(t)
        # We need a ground-truth CLS target. When the input is the full grid,
        # we use the L2-norm of the mean token as a proxy CLS target.
        # In practice the caller can pass the real CLS token separately, but
        # for the loss computation we use the mean token as target.
        cls_target = F.normalize(
            clip_grid_gt.permute(0, 2, 3, 1).reshape(B, -1, CLIP_TOKEN_DIM).mean(1),
            dim=-1,
        )

        # Partial forward to get hidden tokens
        x0 = torch.randn_like(clip_grid_gt)
        t_exp = t[:, None, None, None]
        xt = (1 - t_exp) * x0 + t_exp * clip_grid_gt

        h = self.patch_embed(xt).flatten(2).transpose(1, 2) + self.pos_embed
        for blk in self.blocks:
            h = blk(h, te, brain_ctx)

        cls_pred = self.cls_head(h)  # (B, 1024)
        loss_mse = F.mse_loss(cls_pred, cls_target)
        loss_cos = 1.0 - F.cosine_similarity(cls_pred, cls_target, dim=-1).mean()
        return loss_mse + loss_cos

    # ── Sampling ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(self, brain_ctx: torch.Tensor, n_steps: int = 20,
               cfg_scale: float = 1.0,
               solver: str = "euler") -> torch.Tensor:
        """ODE integration from Gaussian noise → CLIP patch-token grid.

        Args:
            brain_ctx:  brain conditioning tokens (B, n_tokens, brain_dim)
            n_steps:    number of ODE steps
            cfg_scale:  classifier-free guidance scale (1.0 = no CFG)
            solver:     "euler" | "midpoint" | "heun"

        Returns:
            CLIP patch-token grid (B, 1024, 16, 16)
        """
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
                # For VFM extract mu; for CFM use raw directly
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
            # No CFG
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
        """Sample CLIP patch tokens and predict CLS via the auxiliary head.

        Args:
            brain_ctx:  (B, n_tokens, brain_dim)
            n_steps:    ODE steps for sampling

        Returns:
            predicted CLS token (B, 1024) — L2-normalized
        """
        patch_tokens = self.sample(brain_ctx, n_steps=n_steps)  # (B, 1024, 16, 16)
        B = brain_ctx.shape[0]
        t = torch.zeros(B, device=brain_ctx.device)
        te = self.te(t)
        h = self.patch_embed(patch_tokens).flatten(2).transpose(1, 2) + self.pos_embed
        for blk in self.blocks:
            h = blk(h, te, brain_ctx)
        cls = self.cls_head(h)
        return F.normalize(cls, dim=-1)
