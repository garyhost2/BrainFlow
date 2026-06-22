from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

FLUX_HIDDEN_DIM   = 3072
FLUX_NUM_HEADS    = 24
FLUX_HEAD_DIM     = FLUX_HIDDEN_DIM // FLUX_NUM_HEADS
FLUX_DOUBLE_BLOCKS = 19

def _heads_reshape(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    B, L, D = x.shape
    return x.view(B, L, num_heads, D // num_heads).transpose(1, 2)

def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    B, H, L, Dh = x.shape
    return x.transpose(1, 2).contiguous().view(B, L, H * Dh)

class BrainIPAdapterAttention(nn.Module):

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

        self.to_q_norm = nn.LayerNorm(flux_hidden_dim, elementwise_affine=False)
        self.to_k      = nn.Linear(ip_dim, flux_hidden_dim, bias=False)
        self.to_v      = nn.Linear(ip_dim, flux_hidden_dim, bias=False)
        self.to_out    = nn.Linear(flux_hidden_dim, flux_hidden_dim, bias=False)

        self.gate_scale = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):

        nn.init.normal_(self.to_k.weight, std=0.02)
        nn.init.normal_(self.to_v.weight, std=0.02)
        nn.init.zeros_(self.to_out.weight)

    def forward(
        self,
        img_tokens: torch.Tensor,
        brain_cond: torch.Tensor,
        uncertainty_gate: torch.Tensor,
    ) -> torch.Tensor:
        dtype = img_tokens.dtype
        B = img_tokens.shape[0]

        q = self.to_q_norm(img_tokens.float()).to(dtype)
        q = _heads_reshape(q, self.num_heads)

        k = self.to_k(brain_cond.to(dtype))
        v = self.to_v(brain_cond.to(dtype))
        k = _heads_reshape(k, self.num_heads)
        v = _heads_reshape(v, self.num_heads)

        drop_p = self.dropout if self.training else 0.0
        attn = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=drop_p,
            scale=self.scale,
        )

        out = self.to_out(_merge_heads(attn))

        gate = torch.tanh(self.gate_scale) * uncertainty_gate
        return out * gate

class BrainIPAdapter(nn.Module):

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

    def build_conditioning(
        self,
        brain_tokens: torch.Tensor,
        mu_clip: torch.Tensor,
        log_sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        gate = torch.sigmoid(-log_sigma.mean(dim=-1))
        gate = gate.view(-1, 1, 1)

        b_tokens = self.brain_proj(brain_tokens)

        c_token = self.clip_proj(mu_clip).unsqueeze(1)

        cond_tokens = torch.cat([c_token, b_tokens], dim=1)
        return cond_tokens, gate

    def forward_block(
        self,
        block_idx: int,
        img_tokens: torch.Tensor,
        cond_tokens: torch.Tensor,
        uncertainty_gate: torch.Tensor,
    ) -> torch.Tensor:
        if block_idx >= self.n_blocks:
            return torch.zeros_like(img_tokens)

        layer = self.attn_layers[block_idx]

        if self.use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                layer, img_tokens, cond_tokens, uncertainty_gate,
                use_reentrant=False,
            )
        return layer(img_tokens, cond_tokens, uncertainty_gate)

    def forward(
        self,
        img_tokens_per_block: List[torch.Tensor],
        brain_tokens: torch.Tensor,
        mu_clip: torch.Tensor,
        log_sigma: torch.Tensor,
    ) -> List[torch.Tensor]:
        cond, gate = self.build_conditioning(brain_tokens, mu_clip, log_sigma)
        return [
            self.forward_block(i, img_tokens_per_block[i], cond, gate)
            for i in range(self.n_blocks)
        ]

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
