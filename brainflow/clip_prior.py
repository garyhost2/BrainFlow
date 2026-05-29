"""ClipPrior — CFM DiT that maps fMRI brain tokens → CLIP CLS embedding.

Phase 3 SDPrior experiment:
  BrainEncoder (frozen) → brain_tokens (B, N, D)
  ClipPrior (trained)   → sampled CLIP CLS embedding (B, 768)
  Karlo decoder (frozen) → image (256×256)

Architecture: ~20M-param DiT with cross-attention conditioning on brain tokens.
Training: conditional flow matching (CFM) in CLIP CLS space.

Geometry modes
--------------
``geometry="euclidean"`` (default)
    Standard straight-line CFM in R^d with Gaussian source and mean/std
    normalization of the GT embedding.  This is the original behavior and
    is preserved unchanged so existing checkpoints are fully compatible.

``geometry="sphere"``
    Riemannian flow matching on S^(d-1).  The GT CLIP embedding is
    L2-normalized and used *directly* — no mean/std standardization —
    because the sphere geometry already fixes the scale.  The source is
    uniform on S^(d-1) (Gaussian then L2-normalize).  Interpolation
    uses the SLERP geodesic; the target velocity is the analytic
    time-derivative of the geodesic (tangent to the sphere).  During
    sampling, each Euler step is replaced by the exponential map so
    that the trajectory stays on S^(d-1) at every step.
"""
from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Sphere geometry helpers ────────────────────────────────────────────────────

_SLERP_EPS = 1e-6  # avoid division by sin(θ) ≈ 0


def random_sphere(shape: tuple[int, ...], device: torch.device | str) -> torch.Tensor:
    """Sample uniformly on S^(d-1): draw N(0,I) then L2-normalize.

    Args:
        shape: Output shape ``(B, d)``.
        device: Target device.

    Returns:
        Tensor of shape ``shape`` with unit L2 norm along the last dimension.
    """
    x = torch.randn(*shape, device=device)
    return F.normalize(x, dim=-1)


def slerp(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Geodesic (SLERP) interpolation between unit vectors on S^(d-1).

    For each sample i:
        phi_t = sin((1-t_i)θ_i)/sin(θ_i) · x0_i  +  sin(t_i·θ_i)/sin(θ_i) · x1_i

    where θ_i = arccos(clamp(⟨x0_i, x1_i⟩, -1+ε, 1-ε)).

    Near-collinear pairs (|θ| < _SLERP_EPS) fall back to normalized linear
    interpolation to avoid division by ~0.

    Args:
        x0: (B, d) unit vectors (source).
        x1: (B, d) unit vectors (target).
        t:  (B,)  interpolation time in [0, 1].

    Returns:
        (B, d) unit vectors at time t.
    """
    dot = (x0 * x1).sum(dim=-1, keepdim=True).clamp(-1 + _SLERP_EPS, 1 - _SLERP_EPS)
    theta = torch.acos(dot)  # (B, 1)
    sin_theta = theta.sin()  # (B, 1)

    t_col = t[:, None]  # (B, 1)
    s0 = torch.sin((1.0 - t_col) * theta)
    s1 = torch.sin(t_col * theta)

    # SLERP formula; fall back to normalized lerp where sin_theta ≈ 0
    slerp_val = (s0 * x0 + s1 * x1) / sin_theta.clamp(min=_SLERP_EPS)
    lerp_val = F.normalize((1.0 - t_col) * x0 + t_col * x1, dim=-1)

    near_collinear = (sin_theta.abs() < _SLERP_EPS)  # (B, 1) bool
    return torch.where(near_collinear, lerp_val, slerp_val)


def slerp_tangent(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Time-derivative of the geodesic (target velocity for Riemannian CFM).

    d/dt phi_t = (θ/sin θ) · (-cos((1-t)θ) · x0  +  cos(t·θ) · x1)

    This vector lies in the tangent space T_{phi_t} S^(d-1).  For near-
    collinear pairs the limit is simply (x1 - x0), which is already
    tangent when x0 ≈ x1.

    Args:
        x0: (B, d) unit source vectors.
        x1: (B, d) unit target vectors.
        t:  (B,)  time in [0, 1].

    Returns:
        (B, d) tangent velocity at time t.
    """
    dot = (x0 * x1).sum(dim=-1, keepdim=True).clamp(-1 + _SLERP_EPS, 1 - _SLERP_EPS)
    theta = torch.acos(dot)  # (B, 1)
    sin_theta = theta.sin()

    t_col = t[:, None]
    # Analytic derivative: θ/sinθ · (-cos((1-t)θ)·x0 + cos(t·θ)·x1)
    scale = theta / sin_theta.clamp(min=_SLERP_EPS)
    c0 = -torch.cos((1.0 - t_col) * theta)
    c1 = torch.cos(t_col * theta)
    tangent_val = scale * (c0 * x0 + c1 * x1)

    # Fallback for near-collinear: limit is x1 - x0
    near_collinear = (sin_theta.abs() < _SLERP_EPS)
    return torch.where(near_collinear, x1 - x0, tangent_val)


def exp_map(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Exponential map: retract point x along tangent v onto S^(d-1).

    exp_x(v) = cos(||v_⊥||) · x  +  sin(||v_⊥||) · (v_⊥ / ||v_⊥||)

    where v_⊥ = project_tangent(x, v) is the component of v orthogonal to x.
    Projecting onto the tangent space first ensures the output is always on
    S^(d-1), even when the caller passes a non-tangent vector.

    For small ||v_⊥|| falls back to ``normalize(x + v_⊥)``.

    Args:
        x: (B, d) unit vectors (base point).
        v: (B, d) tangent (or approximately tangent) vectors at x.

    Returns:
        (B, d) unit vectors on S^(d-1).
    """
    # Project onto tangent space to guarantee correct output norm
    v_tan = project_tangent(x, v)
    norm_v = v_tan.norm(dim=-1, keepdim=True)  # (B, 1)
    v_hat = v_tan / norm_v.clamp(min=_SLERP_EPS)
    result = torch.cos(norm_v) * x + torch.sin(norm_v) * v_hat
    # Fallback for nearly-zero tangent
    small = (norm_v < _SLERP_EPS)
    fallback = F.normalize(x + v_tan, dim=-1)
    return torch.where(small, fallback, result)


def log_map(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Logarithmic map: tangent vector at x pointing toward y on S^(d-1).

    log_x(y) = (θ / sin θ) · (y - cos(θ) · x)

    where θ = arccos(⟨x, y⟩).

    Args:
        x: (B, d) base point (unit).
        y: (B, d) target point (unit).

    Returns:
        (B, d) tangent vector v such that exp_x(v) ≈ y.
    """
    dot = (x * y).sum(dim=-1, keepdim=True).clamp(-1 + _SLERP_EPS, 1 - _SLERP_EPS)
    theta = torch.acos(dot)
    sin_theta = theta.sin()
    scale = theta / sin_theta.clamp(min=_SLERP_EPS)
    tangent_val = scale * (y - dot * x)
    near_collinear = (sin_theta.abs() < _SLERP_EPS)
    return torch.where(near_collinear, y - x, tangent_val)


def project_tangent(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Project v onto the tangent space T_x S^(d-1): v - ⟨v, x⟩·x.

    Args:
        x: (B, d) unit vectors (base point).
        v: (B, d) arbitrary vectors.

    Returns:
        (B, d) tangent vectors at x.
    """
    dot = (v * x).sum(dim=-1, keepdim=True)
    return v - dot * x


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
        geometry:   ``"euclidean"`` (default, current behavior) or ``"sphere"``.
                    In sphere mode the flow operates on S^(d-1) using SLERP
                    interpolation and exponential-map retractions; mean/std
                    normalization is **not** applied (the sphere already fixes
                    the scale).  See module docstring for details.
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
        geometry: str = "euclidean",
    ):
        super().__init__()
        if geometry not in ("euclidean", "sphere"):
            raise ValueError(
                f"geometry must be 'euclidean' or 'sphere', got {geometry!r}"
            )
        self.clip_dim = clip_dim
        self.ctx_dim = ctx_dim
        self.cfg_drop = cfg_drop
        self.geometry = geometry

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

        Euclidean mode (geometry='euclidean'):
            Straight-line (optimal transport) interpolant:
                x_t = (1 - (1 - σ_min) * t) * ε + t * x_1
                u_t = x_1 - (1 - σ_min) * ε    (constant velocity field)
            x_1 is mean/std normalized; ε ~ N(0, I).

        Sphere mode (geometry='sphere'):
            Riemannian flow matching on S^(d-1):
                x_0 = random_sphere(...)            (uniform on sphere)
                x_1 = L2_normalize(clip_emb)        (unit vector, no mean/std)
                x_t = slerp(x_0, x_1, t)           (geodesic interpolant)
                u_t = slerp_tangent(x_0, x_1, t)   (tangent velocity)
            The network prediction is projected onto T_{x_t} S^(d-1) before
            computing MSE, ensuring the loss is purely in tangent space.
        """
        B = clip_emb.shape[0]
        # Force float32 for CFM math — fp16 overflows when std is near zero
        with torch.cuda.amp.autocast(enabled=False):
            if self.geometry == "sphere":
                # Sphere mode: operate on unit vectors, no mean/std normalization
                x1 = F.normalize(clip_emb.float(), dim=-1)
                x0 = random_sphere((B, self.clip_dim), device=x1.device)
                t = torch.rand(B, device=x1.device)
                xt = slerp(x0, x1, t)
                ut = slerp_tangent(x0, x1, t)
                v_pred = self.forward(xt.to(clip_emb.dtype), t, ctx)
                # Project prediction onto tangent space at xt
                v_proj = project_tangent(xt, v_pred.float())
                return F.mse_loss(v_proj, ut)
            else:
                # Euclidean mode (original behavior — unchanged)
                x1 = self.normalize(clip_emb.float())
                eps = torch.randn_like(x1)
                t = torch.rand(B, device=x1.device)
                # Interpolate
                xt = (1 - (1 - sigma_min) * t[:, None]) * eps + t[:, None] * x1
                # Target velocity
                ut = x1 - (1 - sigma_min) * eps

        v_pred = self.forward(xt.to(clip_emb.dtype), t, ctx)
        return F.mse_loss(v_pred.float(), ut)

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

        Euclidean mode:
            Returns CLIP embeddings (B, clip_dim) — denormalized to original
            CLIP space, then L2-normalized for cosine similarity.

        Sphere mode:
            Initializes from the uniform distribution on S^(d-1), integrates
            using the exponential map to keep the trajectory on the sphere at
            every step, and returns L2-normalized embeddings directly (no
            denormalization — sphere mode does not use mean/std statistics).
        """
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

                # Project velocity onto tangent space at x, then retract via exp map
                v_proj = project_tangent(x, v)
                x = exp_map(x, v_proj * dt)

            # Enforce unit norm for safety (already on sphere, but clamp numerics)
            return F.normalize(x, dim=-1)

        else:
            # Euclidean mode (original behavior — unchanged)
            x = torch.randn(B, self.clip_dim, device=device)

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
