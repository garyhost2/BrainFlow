"""BrainIPAdapter — uncertainty-gated IP-Adapter for FLUX.1-dev.

Injects brain conditioning (fMRI tokens + VFM posterior) into FLUX.1-dev's
double-stream transformer blocks via lightweight cross-attention layers.

Key design choices
------------------
* Flash attention via F.scaled_dot_product_attention — no custom kernels needed.
* Zero-init gate (tanh(gate_scale), init=0) — adapter output starts at exactly
  zero, so early training does not corrupt FLUX's pretrained image distribution.
* Uncertainty gate from VFM log_sigma: sigmoid(-mean(log_sigma)) per sample.
  High VFM confidence  → gate ≈ 1 → strong brain conditioning.
  Low  VFM confidence  → gate ≈ 0 → fall back on FLUX's learned prior.
* All adapter weights in fp32; brain conditioning cast to FLUX dtype at call.
* torch.compile-safe: no host-side Python branching in forward hot path.
* Gradient checkpointing available via use_gradient_checkpointing=True.
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── constants ────────────────────────────────────────────────────────────────

FLUX_HIDDEN_DIM   = 3072   # FLUX.1 transformer hidden dim
FLUX_NUM_HEADS    = 24     # FLUX.1 number of attention heads
FLUX_HEAD_DIM     = FLUX_HIDDEN_DIM // FLUX_NUM_HEADS   # 128
FLUX_DOUBLE_BLOCKS = 19   # number of double-stream blocks in FLUX.1


# ── helpers ──────────────────────────────────────────────────────────────────

def _heads_reshape(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """(B, L, D) → (B, H, L, D//H)  for multi-head attention."""
    B, L, D = x.shape
    return x.view(B, L, num_heads, D // num_heads).transpose(1, 2)


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    """(B, H, L, Dh) → (B, L, H*Dh)."""
    B, H, L, Dh = x.shape
    return x.transpose(1, 2).contiguous().view(B, L, H * Dh)


# ── single adapter attention block ───────────────────────────────────────────

class BrainIPAdapterAttention(nn.Module):
    """Cross-attention from FLUX image tokens → brain conditioning tokens.

    Injected into one FLUX double-stream block.  The output is scaled by:
        tanh(gate_scale) * uncertainty_gate
    where gate_scale is a learned scalar (init=0) and uncertainty_gate is a
    per-sample scalar derived from the VFM posterior log_sigma.

    Parameters
    ----------
    ip_dim : int
        Dimension of the projected brain conditioning tokens.
    flux_hidden_dim : int
        FLUX transformer hidden dimension (3072).
    num_heads : int
        Number of attention heads (must divide flux_hidden_dim).
    dropout : float
        Dropout on the attention weights (training only).
    """

    def __init__(
        self,
        ip_dim: int = 1024,
        flux_hidden_dim: int = FLUX_HIDDEN_DIM,
        num_heads: int = FLUX_NUM_HEADS,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert flux_hidden_dim % num_heads == 0, \
            f"flux_hidden_dim {flux_hidden_dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim  = flux_hidden_dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.dropout   = dropout

        # Q comes from FLUX img tokens (projected to flux_hidden_dim already).
        # We only add K and V projections from the brain conditioning.
        self.to_q_norm = nn.LayerNorm(flux_hidden_dim, elementwise_affine=False)
        self.to_k      = nn.Linear(ip_dim, flux_hidden_dim, bias=False)
        self.to_v      = nn.Linear(ip_dim, flux_hidden_dim, bias=False)
        self.to_out    = nn.Linear(flux_hidden_dim, flux_hidden_dim, bias=False)

        # Learned output gate — tanh(gate_scale) starts at tanh(0) = 0.
        # Adapter output is exactly zero at init; FLUX distribution preserved.
        self.gate_scale = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        # Small normal init for K, V projectors; zero init for output.
        nn.init.normal_(self.to_k.weight, std=0.02)
        nn.init.normal_(self.to_v.weight, std=0.02)
        nn.init.zeros_(self.to_out.weight)

    def forward(
        self,
        img_tokens: torch.Tensor,          # (B, L_img, flux_hidden_dim)
        brain_cond: torch.Tensor,          # (B, L_brain, ip_dim)
        uncertainty_gate: torch.Tensor,    # (B, 1, 1) — from VFM sigma
    ) -> torch.Tensor:
        """Returns residual to add to FLUX img_tokens (same shape)."""
        dtype = img_tokens.dtype
        B = img_tokens.shape[0]

        # Q from normalised image tokens
        q = self.to_q_norm(img_tokens.float()).to(dtype)
        q = _heads_reshape(q, self.num_heads)                  # (B,H,L,Dh)

        # K, V from brain conditioning (cast to match FLUX dtype)
        k = self.to_k(brain_cond.to(dtype))                    # (B,Lb,D)
        v = self.to_v(brain_cond.to(dtype))
        k = _heads_reshape(k, self.num_heads)                  # (B,H,Lb,Dh)
        v = _heads_reshape(v, self.num_heads)

        # Flash attention (uses CUDA FlashAttn or efficient math kernel)
        drop_p = self.dropout if self.training else 0.0
        attn = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=drop_p,
            scale=self.scale,
        )                                                       # (B,H,L,Dh)

        out = self.to_out(_merge_heads(attn))                  # (B,L,D)

        # Gating: tanh(gate_scale) * uncertainty_gate
        # Both factors start near 0 → safe ramp-up during training.
        gate = torch.tanh(self.gate_scale) * uncertainty_gate  # (B,1,1)
        return out * gate


# ── full adapter ─────────────────────────────────────────────────────────────

class BrainIPAdapter(nn.Module):
    """Complete IP-Adapter for FLUX.1-dev.

    Holds:
    - brain_proj  : project BrainEncoder tokens  (brain_dim → ip_dim)
    - clip_proj   : project VFM mu_clip          (clip_dim  → ip_dim)
    - attn_layers : one BrainIPAdapterAttention per FLUX double-stream block

    The uncertainty gate is computed from VFM log_sigma:
        gate = sigmoid(-mean(log_sigma, dim=-1))  →  (B,) ∈ (0,1)
    which is 1 when the VFM posterior is sharp (low sigma) and 0 when
    diffuse (high sigma), automatically falling back on FLUX's prior.

    Parameters
    ----------
    brain_dim : int
        BrainEncoder output token dimension.
    clip_dim : int
        ClipPrior output (CLIP embedding) dimension.
    ip_dim : int
        Internal projection dimension for conditioning tokens.
    n_blocks : int
        Number of FLUX double-stream blocks to inject into.
    flux_hidden_dim : int
        FLUX transformer hidden dimension.
    num_heads : int
        Attention heads in each adapter block.
    dropout : float
        Attention dropout (training only).
    use_gradient_checkpointing : bool
        Recompute activations during backward to save VRAM.
    """

    def __init__(
        self,
        brain_dim: int = 1024,
        clip_dim: int = 768,
        ip_dim: int = 1024,
        n_blocks: int = FLUX_DOUBLE_BLOCKS,
        flux_hidden_dim: int = FLUX_HIDDEN_DIM,
        num_heads: int = FLUX_NUM_HEADS,
        dropout: float = 0.0,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.ip_dim  = ip_dim
        self.n_blocks = n_blocks
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Project brain tokens and CLIP mean into shared ip_dim space.
        self.brain_proj = nn.Sequential(
            nn.LayerNorm(brain_dim),
            nn.Linear(brain_dim, ip_dim, bias=False),
        )
        self.clip_proj = nn.Sequential(
            nn.LayerNorm(clip_dim),
            nn.Linear(clip_dim, ip_dim, bias=False),
        )

        self.attn_layers = nn.ModuleList([
            BrainIPAdapterAttention(
                ip_dim=ip_dim,
                flux_hidden_dim=flux_hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(n_blocks)
        ])

        self._init_weights()

    def _init_weights(self):
        for proj in (self.brain_proj, self.clip_proj):
            linear = proj[1]
            nn.init.normal_(linear.weight, std=0.02)

    # ── conditioning ─────────────────────────────────────────────────────────

    def build_conditioning(
        self,
        brain_tokens: torch.Tensor,   # (B, N, brain_dim)
        mu_clip: torch.Tensor,         # (B, clip_dim)
        log_sigma: torch.Tensor,       # (B, clip_dim)  — from VFM head
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project and concatenate brain + CLIP tokens; compute uncertainty gate.

        Returns
        -------
        cond_tokens : (B, N+1, ip_dim)
            Concatenation of projected mu_clip (1 token) and projected
            brain_tokens (N tokens).
        uncertainty_gate : (B, 1, 1)
            Per-sample scalar in (0, 1): high = confident, low = uncertain.
        """
        # Uncertainty gate: low sigma → gate ≈ 1 (strong conditioning)
        # Shape: (B,) → (B, 1, 1) for broadcasting over (B, L, D)
        gate = torch.sigmoid(-log_sigma.mean(dim=-1))          # (B,)
        gate = gate.view(-1, 1, 1)                             # (B, 1, 1)

        # Project brain tokens: (B, N, brain_dim) → (B, N, ip_dim)
        b_tokens = self.brain_proj(brain_tokens)

        # Project CLIP mean: (B, clip_dim) → (B, 1, ip_dim)
        c_token = self.clip_proj(mu_clip).unsqueeze(1)         # (B, 1, ip_dim)

        # CLIP token prepended so it's always in the adapter's receptive field
        cond_tokens = torch.cat([c_token, b_tokens], dim=1)    # (B, N+1, ip_dim)
        return cond_tokens, gate

    # ── per-block forward (called by FLUX hook) ───────────────────────────────

    def forward_block(
        self,
        block_idx: int,
        img_tokens: torch.Tensor,          # (B, L, flux_hidden_dim) — FLUX img stream
        cond_tokens: torch.Tensor,         # (B, N+1, ip_dim)
        uncertainty_gate: torch.Tensor,    # (B, 1, 1)
    ) -> torch.Tensor:
        """Compute and return the adapter residual for one FLUX double-stream block.

        The caller adds this residual to FLUX's img_tokens after the block's
        own attention outputs.
        """
        if block_idx >= self.n_blocks:
            return torch.zeros_like(img_tokens)

        layer = self.attn_layers[block_idx]

        if self.use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                layer, img_tokens, cond_tokens, uncertainty_gate,
                use_reentrant=False,
            )
        return layer(img_tokens, cond_tokens, uncertainty_gate)

    # ── convenience: all blocks at once (for unit tests / offline use) ────────

    def forward(
        self,
        img_tokens_per_block: List[torch.Tensor],   # list of length n_blocks
        brain_tokens: torch.Tensor,
        mu_clip: torch.Tensor,
        log_sigma: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Apply all adapter blocks. Returns list of residuals (same shapes)."""
        cond, gate = self.build_conditioning(brain_tokens, mu_clip, log_sigma)
        return [
            self.forward_block(i, img_tokens_per_block[i], cond, gate)
            for i in range(self.n_blocks)
        ]

    # ── parameter / memory helpers ────────────────────────────────────────────

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters() if not trainable_only else (
            p for p in self.parameters() if p.requires_grad
        )
        return sum(p.numel() for p in params)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad_(True)
