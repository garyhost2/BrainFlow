"""BrainFlow v5 model: shared encoder, subject adapters, and flow backbones."""
from __future__ import annotations
from dataclasses import dataclass
import math, random
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .ridge_init import load_ridge_init_state


def _double_gamma_hrf(length: int, peak: float = 6.0, undershoot: float = 16.0,
                      ratio: float = 6.0, dt: float = 1.0) -> torch.Tensor:
    """Standard SPM-style double-gamma HRF, normalized to unit L1.
    Used by the HRF brain-manifold flow (cfg.method == "hrf")."""
    t = np.arange(length, dtype=np.float64) * dt + dt / 2
    def gamma_pdf(t, k):
        return np.exp((k - 1) * np.log(t + 1e-12) - t - math.lgamma(k))
    h = gamma_pdf(t, peak) - gamma_pdf(t, undershoot) / ratio
    h = h / (np.abs(h).sum() + 1e-12)
    return torch.tensor(h, dtype=torch.float32)


def _sample_t(B, device, cfg) -> torch.Tensor:
    """Sample t ∈ [0,1] per cfg.t_schedule: 'linear' | 'cosine' | 'logit_normal'."""
    sched = getattr(cfg, "t_schedule", "linear")
    if sched == "cosine":
        u = torch.rand(B, device=device)
        return (1 - torch.cos(math.pi * u)) / 2
    if sched == "logit_normal":
        m = getattr(cfg, "logit_normal_m", 0.0)
        s = getattr(cfg, "logit_normal_s", 1.0)
        z = torch.randn(B, device=device) * s + m
        return torch.sigmoid(z)
    return torch.rand(B, device=device)


def _sinkhorn_ot(C, reg=0.05, n_iter=50) -> torch.Tensor:
    """Minibatch OT via Sinkhorn log-domain.
    C: (B,B) cost matrix.
    Returns col_idx (B,): x1[i] is paired with x0[col_idx[i]]."""
    B = C.shape[0]
    log_M = -C / reg
    log_a = torch.full((B,), -math.log(B), device=C.device)
    log_b = torch.full((B,), -math.log(B), device=C.device)
    log_u = torch.zeros(B, device=C.device)
    for _ in range(n_iter):
        log_v = log_b - torch.logsumexp(log_M + log_u[:, None], dim=0)
        log_u = log_a - torch.logsumexp(log_M + log_v[None, :], dim=1)
    log_pi = log_M + log_u[:, None] + log_v[None, :]
    pi = (log_pi - log_pi.max(dim=1, keepdim=True).values).exp()
    return torch.multinomial(pi + 1e-10, num_samples=1).squeeze(-1)


def _slerp(x0, x1, t) -> torch.Tensor:
    """SLERP geodesic on S^(d-1). x0, x1: (B,d) unit vectors. t: (B,)."""
    cos_theta = (x0 * x1).sum(-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(cos_theta)
    sin_theta = theta.sin().clamp(min=1e-6)
    w0 = torch.sin((1 - t[:, None]) * theta) / sin_theta
    w1 = torch.sin(t[:, None] * theta) / sin_theta
    return w0 * x0 + w1 * x1


def _slerp_velocity(x0, x1, t) -> torch.Tensor:
    """Tangent vector (d/dt SLERP) at time t. Lives in T_{x_t}S^(d-1)."""
    cos_theta = (x0 * x1).sum(-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(cos_theta)
    sin_theta = theta.sin().clamp(min=1e-6)
    scale = theta / sin_theta
    dw0 = -torch.cos((1 - t[:, None]) * theta)
    dw1 = torch.cos(t[:, None] * theta)
    return scale * (dw0 * x0 + dw1 * x1)


def _exp_map(x, v, dt) -> torch.Tensor:
    """Exponential map retraction on S^(d-1): Exp_x(dt·v).
    Projects v to tangent space then retracts. Always stays on manifold."""
    v_tan = v - (v * x).sum(-1, keepdim=True) * x
    norm_v = v_tan.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    step = dt * norm_v
    return torch.cos(step) * x + torch.sin(step) * (v_tan / norm_v)


class ResBlockMLP(nn.Module):
    def __init__(self, dim, drop=0.1, mult=4, stoch_depth=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult), nn.GELU(), nn.Dropout(drop),
            nn.Linear(dim * mult, dim), nn.Dropout(drop),
        )
        self.drop_prob = stoch_depth

    def forward(self, x):
        # Use on-device bernoulli so this is compile-safe (no host-side random.random())
        if self.training and self.drop_prob > 0:
            keep = torch.empty(x.shape[0], 1, device=x.device).bernoulli_(1 - self.drop_prob)
            return x + self.net(self.norm(x)) * keep / (1 - self.drop_prob)
        return x + self.net(self.norm(x))


@dataclass
class BrainEncoderOutput:
    tokens: torch.Tensor
    cls: torch.Tensor
    cls_align: torch.Tensor

    def __iter__(self):
        yield self.tokens
        yield self.cls


class SubjectResidualAdapter(nn.Module):
    """Per-subject low-rank residual on shared stem features.

    Uses embedding lookups so only the active subject's adapter weights receive
    gradients each step, avoiding the DDP allreduce waste from full per-subject
    projection layers.
    """
    def __init__(self, n_subjects: int, hidden_dim: int, rank: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = max(1, int(rank))
        self.down = nn.Embedding(n_subjects, hidden_dim * self.rank)
        self.up = nn.Embedding(n_subjects, self.rank * hidden_dim)
        self.bias = nn.Embedding(n_subjects, hidden_dim)
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.bias.weight)

    def forward(self, h: torch.Tensor, subject_idx: torch.Tensor) -> torch.Tensor:
        down = self.down(subject_idx).view(-1, self.hidden_dim, self.rank)
        up = self.up(subject_idx).view(-1, self.rank, self.hidden_dim)
        delta = torch.bmm(torch.bmm(h.unsqueeze(1), down), up).squeeze(1)
        return h + delta + self.bias(subject_idx)


class BrainEncoder(nn.Module):
    """fMRI → (B, N_TOKENS, BRAIN_DIM) tokens + (B, BRAIN_DIM) cls.
    Shared input projection with zero-padding for variable-voxel subjects.
    A single Linear(max_vox → enc_hidden) replaces per-subject ModuleDict
    to eliminate the DDP allreduce overhead from 7/8 zero-gradient projections."""
    def __init__(self, cfg: Config, voxels_per_subject: Dict[int, int]):
        super().__init__()
        self.cfg = cfg
        self.subjects = sorted(voxels_per_subject.keys())
        self.use_subject_heads = bool(getattr(cfg, "use_subject_heads", True))
        self.use_ridge_init = bool(getattr(cfg, "use_ridge_init", True))
        self.subject_head_rank = int(getattr(cfg, "subject_head_rank", 16))
        # A.4: shared projection; pad shorter subjects' fmri with zeros on-the-fly
        self.voxels_per_subject = dict(voxels_per_subject)
        self.max_vox = max(voxels_per_subject.values())
        self.input_proj = nn.Linear(self.max_vox, cfg.enc_hidden)
        max_subject_id = max(self.subjects) if self.subjects else 0
        subject_lookup = torch.full((max_subject_id + 1,), -1, dtype=torch.long)
        for i, sid in enumerate(self.subjects):
            subject_lookup[sid] = i
        self.register_buffer("subject_lookup", subject_lookup, persistent=False)
        if self.use_subject_heads and self.subjects:
            self.subject_adapter = SubjectResidualAdapter(
                n_subjects=len(self.subjects),
                hidden_dim=cfg.enc_hidden,
                rank=self.subject_head_rank,
            )
        else:
            self.subject_adapter = None
        self.stem_norm = nn.LayerNorm(cfg.enc_hidden)
        self.stem_act = nn.GELU()
        self.stem_drop = nn.Dropout(float(getattr(cfg, "stem_drop", 0.15)))

        self.blocks = nn.ModuleList([
            ResBlockMLP(
                cfg.enc_hidden, cfg.enc_drop, mult=4,
                stoch_depth=cfg.stochastic_depth * (i + 1) / cfg.enc_blocks,
            )
            for i in range(cfg.enc_blocks)
        ])
        self.to_tokens = nn.Sequential(
            nn.Linear(cfg.enc_hidden, cfg.n_tokens * cfg.brain_dim),
            nn.Unflatten(-1, (cfg.n_tokens, cfg.brain_dim)),
            nn.LayerNorm(cfg.brain_dim),
            nn.Dropout(0.1),
        )
        self.cls_head = nn.Linear(cfg.enc_hidden, cfg.brain_dim)
        self.align_proj = nn.Linear(cfg.brain_dim, cfg.clip_dim)
        self.ridge_skip = nn.Linear(self.max_vox, cfg.clip_dim)
        nn.init.zeros_(self.ridge_skip.weight)
        nn.init.zeros_(self.ridge_skip.bias)
        self._maybe_load_ridge_init()

    def _subject_indices(self, subject: torch.Tensor | int, batch_size: int, device) -> torch.Tensor:
        if isinstance(subject, int):
            subject = torch.full((batch_size,), subject, device=device, dtype=torch.long)
        elif not torch.is_tensor(subject):
            subject = torch.as_tensor(subject, device=device, dtype=torch.long)
        else:
            subject = subject.to(device=device, dtype=torch.long).view(-1)
        if subject.numel() == 1 and batch_size != 1:
            subject = subject.expand(batch_size)
        if subject.numel() != batch_size:
            raise ValueError(f"Expected {batch_size} subject ids, got {subject.numel()}.")
        if subject.max().item() >= self.subject_lookup.numel():
            raise KeyError(f"Unknown subject id(s): {subject.tolist()}")
        subject_idx = self.subject_lookup[subject]
        if (subject_idx < 0).any():
            raise KeyError(f"Unknown subject id(s): {subject.tolist()}")
        return subject_idx

    def _maybe_load_ridge_init(self) -> None:
        if not self.use_ridge_init:
            return
        ridge_state = load_ridge_init_state(self.cfg, self.max_vox, self.cfg.clip_dim)
        if ridge_state is None:
            return
        with torch.no_grad():
            self.ridge_skip.weight.copy_(ridge_state["weight"])
            self.ridge_skip.bias.copy_(ridge_state["bias"])

    def forward(self, fmri: torch.Tensor, subject: torch.Tensor | int):
        if fmri.shape[-1] > self.max_vox:
            raise ValueError(f"fMRI width {fmri.shape[-1]} exceeds max_vox {self.max_vox}.")
        if fmri.shape[-1] < self.max_vox:
            fmri = F.pad(fmri, (0, self.max_vox - fmri.shape[-1]))
        subject_idx = self._subject_indices(subject, fmri.shape[0], fmri.device)
        h = self.input_proj(fmri)
        if self.subject_adapter is not None:
            h = self.subject_adapter(h, subject_idx)
        h = self.stem_drop(self.stem_act(self.stem_norm(h)))
        for b in self.blocks:
            h = b(h)
        tokens = self.to_tokens(h)
        cls = F.normalize(self.cls_head(h), dim=-1)
        cls_align = self.align_proj(cls)
        if self.use_ridge_init:
            cls_align = cls_align + self.ridge_skip(fmri)
        cls_align = F.normalize(cls_align, dim=-1)
        return BrainEncoderOutput(tokens=tokens, cls=cls, cls_align=cls_align)



class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half).float() / (half - 1))
        self.register_buffer("freqs", freqs)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        t = t.unsqueeze(-1) * self.freqs
        return self.mlp(torch.cat([t.sin(), t.cos()], dim=-1))


class CrossAttention(nn.Module):
    def __init__(self, qd, cd, nh=8, hd=64):
        super().__init__()
        inner = nh * hd
        self.nh = nh; self.hd = hd
        self.to_q = nn.Linear(qd, inner, bias=False)
        self.to_k = nn.Linear(cd, inner, bias=False)
        self.to_v = nn.Linear(cd, inner, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, qd), nn.Dropout(0.05))
        self.norm = nn.LayerNorm(qd)

    def forward(self, x, ctx):
        B, L, _ = x.shape
        res = x; x = self.norm(x)
        def rsh(t, s): return t.view(B, s, self.nh, self.hd).transpose(1, 2)
        Q = rsh(self.to_q(x), L)
        K = rsh(self.to_k(ctx), ctx.shape[1])
        V = rsh(self.to_v(ctx), ctx.shape[1])
        # Use flash-attention / memory-efficient kernels automatically
        out = F.scaled_dot_product_attention(Q, K, V)
        return res + self.to_out(out.transpose(1, 2).reshape(B, L, -1))


class UNetResBlock(nn.Module):
    def __init__(self, ic, oc, td):
        super().__init__()
        self.n1 = nn.GroupNorm(min(32, ic), ic)
        self.c1 = nn.Conv2d(ic, oc, 3, padding=1)
        self.n2 = nn.GroupNorm(min(32, oc), oc)
        self.c2 = nn.Conv2d(oc, oc, 3, padding=1)
        self.act = nn.SiLU()
        self.tp = nn.Linear(td, oc * 2)
        self.skip = nn.Conv2d(ic, oc, 1) if ic != oc else nn.Identity()
        self.dropout = nn.Dropout2d(0.05)

    def forward(self, x, te):
        h = self.act(self.n1(x)); h = self.c1(h)
        sc, sh = self.act(self.tp(te))[:, :, None, None].chunk(2, dim=1)
        h = self.n2(h) * (1 + sc) + sh
        h = self.act(h); h = self.dropout(h); h = self.c2(h)
        return h + self.skip(x)


class FlowUNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        ic = cfg.latent_ch; bc = cfg.unet_base_ch
        bd = cfg.brain_dim; td = cfg.time_emb_dim; nh = cfg.attn_heads
        chs = [bc, bc * 2, bc * 4, bc * 4]
        self.te = SinusoidalTimeEmbedding(td)
        self.ip = nn.Conv2d(ic, bc, 3, padding=1)
        self.null_tokens = nn.Parameter(torch.randn(1, cfg.n_tokens, bd) * 0.01)

        self.eb = nn.ModuleList(); self.ea = nn.ModuleList(); self.ed = nn.ModuleList()
        ch = bc
        for i, co in enumerate(chs):
            self.eb.append(UNetResBlock(ch, co, td))
            self.ea.append(CrossAttention(co, bd, nh))
            self.ed.append(nn.Conv2d(co, co, 4, stride=2, padding=1)
                           if i < len(chs) - 1 else nn.Identity())
            ch = co

        self.m1 = UNetResBlock(chs[-1], chs[-1], td)
        self.ma = CrossAttention(chs[-1], bd, nh)
        self.m2 = UNetResBlock(chs[-1], chs[-1], td)

        self.du = nn.ModuleList(); self.db = nn.ModuleList(); self.da = nn.ModuleList()
        for i, co in enumerate(reversed(chs)):
            self.du.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(ch, ch, 3, padding=1),
            ) if i > 0 else nn.Identity())
            self.db.append(UNetResBlock(ch + chs[len(chs) - 1 - i], co, td))
            self.da.append(CrossAttention(co, bd, nh))
            ch = co

        self.on = nn.GroupNorm(min(32, bc), bc)
        self.op = nn.Conv2d(bc, ic, 3, padding=1)
        nn.init.zeros_(self.op.weight); nn.init.zeros_(self.op.bias)

        # ── Optional HRF velocity bias (cfg.method == "hrf") ──
        # Projects HRF-convolved brain tokens → per-channel velocity bias map,
        # gated by sin(pi*t). Enabled lazily so baseline checkpoints stay compatible.
        self._hrf_enabled = (getattr(cfg, "method", "baseline") == "hrf")
        if self._hrf_enabled:
            hrf_len = int(getattr(cfg, "hrf_len", cfg.n_tokens))
            self._latent_res = cfg.latent_res
            self.register_buffer(
                "hrf_kernel",
                _double_gamma_hrf(hrf_len).view(1, 1, hrf_len),
                persistent=False,
            )
            # ctx: (B, n_tokens, brain_dim) → bias map: (B, ic, latent_res, latent_res)
            self.hrf_proj = nn.Sequential(
                nn.LayerNorm(bd),
                nn.Linear(bd, ic * cfg.latent_res),  # rows
                nn.GELU(),
            )
            self.hrf_to_map = nn.Linear(cfg.n_tokens, cfg.latent_res, bias=False)
            nn.init.zeros_(self.hrf_to_map.weight)  # zero-init → identity at start

    def _hrf_bias(self, ctx: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute sin(pi*t) * HRF-convolved velocity bias. ctx: (B, N, D)."""
        B, N, D = ctx.shape
        # HRF-convolve along token axis (treat tokens as a 1D temporal signal).
        # B.3: causal (left-only) padding so future tokens don't leak into past.
        x = ctx.transpose(1, 2)                                  # (B, D, N)
        kernel = self.hrf_kernel.expand(D, 1, -1)                # (D, 1, L)
        x = F.pad(x, (kernel.shape[-1] - 1, 0))                 # left-pad only
        x = F.conv1d(x, kernel, groups=D)                       # (B, D, N)
        x = x.transpose(1, 2)                                    # (B, N, D)
        rows = self.hrf_proj(x)                                  # (B, N, ic*H)
        ic_x_H = rows.shape[-1]
        H = self._latent_res
        ic = ic_x_H // H
        rows = rows.view(B, N, ic, H).permute(0, 2, 3, 1)        # (B, ic, H, N)
        bias = self.hrf_to_map(rows)                             # (B, ic, H, W)
        gate = torch.sin(math.pi * t).clamp(min=0.0)[:, None, None, None]
        return gate * bias

    def forward(self, x, t, ctx):
        te = self.te(t); h = self.ip(x); sk = []
        for res, attn, dn in zip(self.eb, self.ea, self.ed):
            h = res(h, te)
            B, C, H, W = h.shape
            h = attn(h.permute(0, 2, 3, 1).reshape(B, H * W, C), ctx) \
                .reshape(B, H, W, C).permute(0, 3, 1, 2)
            sk.append(h); h = dn(h)
        h = self.m1(h, te)
        B, C, H, W = h.shape
        h = self.ma(h.permute(0, 2, 3, 1).reshape(B, H * W, C), ctx) \
            .reshape(B, H, W, C).permute(0, 3, 1, 2)
        h = self.m2(h, te)
        for i, (up, res, attn) in enumerate(zip(self.du, self.db, self.da)):
            h = up(h)
            h = torch.cat([h, sk[-(i + 1)]], dim=1)
            h = res(h, te)
            B, C, H, W = h.shape
            h = attn(h.permute(0, 2, 3, 1).reshape(B, H * W, C), ctx) \
                .reshape(B, H, W, C).permute(0, 3, 1, 2)
        out = self.op(F.silu(self.on(h)))
        if self._hrf_enabled:
            out = out + self._hrf_bias(ctx, t)
        return out


# ════════════════════════════════════════════════════════════════════════════
# DiT backbone + unCLIP-style diffusion prior (cfg.method == "dit")
# ════════════════════════════════════════════════════════════════════════════

class DiTBlock(nn.Module):
    """DiT block with adaLN-zero time conditioning, self-attn, cross-attn to
    brain tokens (+ optional CLIP prior token), and MLP. Zero-init gates so
    the block starts as identity → stable training from scratch."""
    def __init__(self, d: int, nh: int, brain_dim: int, td: int):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d, nh, batch_first=True, dropout=0.0)
        self.cattn = CrossAttention(d, brain_dim, nh)
        self.n3 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d),
        )
        # adaLN-zero: 6 modulation params (shift/scale for attn+mlp, gate for each)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(td, 6 * d))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, te, ctx):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = \
            self.ada(te).chunk(6, dim=-1)
        # self-attention with adaLN-zero
        h = self.n1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        h_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_a[:, None] * h_out
        # cross-attention to brain tokens (has its own LayerNorm)
        x = self.cattn(x, ctx)
        # MLP with adaLN-zero
        h = self.n3(x) * (1 + scale_m[:, None]) + shift_m[:, None]
        x = x + gate_m[:, None] * self.mlp(h)
        return x


class FlowDiT(nn.Module):
    """Diffusion Transformer flow denoiser. Replaces FlowUNet.

    Input:  x (B, C, H, W) latent + t (B,) + ctx (B, N, brain_dim)
    Output: velocity v (B, C, H, W)

    Note: the cross-attention context carries (n_tokens + 1) entries when
    method == 'dit' — the +1 is the CLIP prior token. null_tokens matches."""
    def __init__(self, cfg: Config):
        super().__init__()
        ic = cfg.latent_ch
        res = cfg.latent_res
        p = cfg.dit_patch
        assert res % p == 0, f"latent_res {res} not divisible by patch {p}"
        gs = res // p                   # grid side, e.g. 32/4 = 8
        self.patch = p
        self.grid = gs
        self.num_patches = gs * gs
        d = cfg.dit_dim
        nh = cfg.dit_heads
        depth = cfg.dit_depth
        bd = cfg.brain_dim
        td = cfg.time_emb_dim
        ic_ = ic

        self.te = SinusoidalTimeEmbedding(td)
        self.patch_embed = nn.Conv2d(ic, d, p, stride=p)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([DiTBlock(d, nh, bd, td) for _ in range(depth)])

        self.final_norm = nn.LayerNorm(d, elementwise_affine=False)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(td, 2 * d))
        nn.init.zeros_(self.final_ada[-1].weight)
        nn.init.zeros_(self.final_ada[-1].bias)
        self.final_proj = nn.Linear(d, ic_ * p * p)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

        # Context length = n_tokens + 1 (last slot is the CLIP prior token)
        self.null_tokens = nn.Parameter(torch.randn(1, cfg.n_tokens + 1, bd) * 0.01)

        # Stash sizes for unpatchify
        self._ic = ic_

    def forward(self, x, t, ctx):
        B = x.shape[0]
        te = self.te(t)
        h = self.patch_embed(x)                   # (B, d, gs, gs)
        h = h.flatten(2).transpose(1, 2)          # (B, P, d)
        h = h + self.pos_embed
        for blk in self.blocks:
            h = blk(h, te, ctx)
        shift, scale = self.final_ada(te).chunk(2, dim=-1)
        h = self.final_norm(h) * (1 + scale[:, None]) + shift[:, None]
        h = self.final_proj(h)                    # (B, P, ic*p*p)
        p = self.patch; ic = self._ic; gs = self.grid
        h = h.view(B, gs, gs, ic, p, p)
        h = h.permute(0, 3, 1, 4, 2, 5).contiguous()
        h = h.view(B, ic, gs * p, gs * p)
        return h


class CLIPPrior(nn.Module):
    """Flow-matching diffusion prior in CLIP-embedding space (unCLIP-style).
    Given fMRI CLS, learns to denoise CLIP embeddings via CFM.
    At training, also provides the predicted CLIP token to condition the
    latent-space flow (true CLIP is teacher-forced; prior supervises itself).
    At inference, ODE-samples a predicted CLIP embedding.

    Sized via cfg.prior_dim and cfg.prior_blocks for scaling experiments."""
    def __init__(self, cfg: Config):
        super().__init__()
        cd = cfg.clip_dim
        bd = cfg.brain_dim
        d = int(getattr(cfg, "prior_dim", 512))
        n_blocks = int(getattr(cfg, "prior_blocks", 3))
        td = 128
        self.cd = cd
        self.cfg = cfg
        self.geometry = getattr(cfg, "clip_prior_geometry", "euclidean")
        self.flow_obj = getattr(cfg, "flow_objective", "cfm")
        self.te = SinusoidalTimeEmbedding(td)
        self.in_proj = nn.Linear(cd, d)
        self.cond_proj = nn.Linear(bd, d)
        self.time_proj = nn.Linear(td, d)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, d * 2),
                nn.GELU(),
                nn.Dropout(0.05),
                nn.Linear(d * 2, d),
            )
            for _ in range(n_blocks)
        ])
        self.out_norm = nn.LayerNorm(d)
        self.out_proj = nn.Linear(d, cd)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.sigma_head = None
        if self.flow_obj == "vfm":
            self.sigma_head = nn.Linear(d, cd)
            nn.init.zeros_(self.sigma_head.weight)
            nn.init.constant_(self.sigma_head.bias, -2.0)

    def _forward_features(self, clip_t: torch.Tensor, t: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
        """Return shared trunk features before output projections."""
        te = self.te(t)
        h = self.in_proj(clip_t) + self.cond_proj(cls) + self.time_proj(te)
        for blk in self.blocks:
            h = h + blk(h)
        return self.out_norm(h)

    def forward(self, clip_t: torch.Tensor, t: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
        """Predict velocity v = (clip_clean - clip_noise). Shapes: (B, cd)."""
        h = self._forward_features(clip_t, t, cls)
        return self.out_proj(h)

    def forward_with_sigma(self, clip_t: torch.Tensor, t: torch.Tensor, cls: torch.Tensor):
        """Predict velocity mean and standard deviation for VFM training."""
        if self.sigma_head is None:
            raise RuntimeError("forward_with_sigma is only available when flow_objective='vfm'.")
        h = self._forward_features(clip_t, t, cls)
        mu = self.out_proj(h)
        log_sigma = self.sigma_head(h)
        sigma = torch.exp(log_sigma).clamp(1e-4, 10.0)
        return mu, sigma, log_sigma

    def flow_loss(self, clip_gt: torch.Tensor, cls: torch.Tensor, vfm_kl_weight: float = 0.0) -> torch.Tensor:
        """Compute CLIP prior flow-matching loss for euclidean or spherical geometry."""
        B = clip_gt.shape[0]
        t = _sample_t(B, clip_gt.device, self.cfg)

        if self.geometry == "sphere":
            x1 = F.normalize(clip_gt, dim=-1)
            x0 = F.normalize(torch.randn_like(x1), dim=-1)
            xt = _slerp(x0, x1, t)
            ut = _slerp_velocity(x0, x1, t)
        else:
            x0 = torch.randn_like(clip_gt)
            xt = (1 - t)[:, None] * x0 + t[:, None] * clip_gt
            ut = clip_gt - x0

        if self.flow_obj == "vfm" and self.sigma_head is not None:
            mu, sigma, log_sigma = self.forward_with_sigma(xt, t, cls)
            recon = (mu - ut).pow(2)
            kl = sigma.pow(2) - 1 - 2 * log_sigma
            return (recon + vfm_kl_weight * kl).mean()

        vt_pred = self.forward(xt, t, cls)
        return F.mse_loss(vt_pred, ut)

    @torch.no_grad()
    def sample(self, cls: torch.Tensor, n_steps: int = 10, solver: str = "dpm2m") -> torch.Tensor:
        """Sample CLIP embeddings with Euler, midpoint, or DPM-Solver++(2M)."""
        B = cls.shape[0]
        if self.geometry == "sphere":
            x = F.normalize(torch.randn(B, self.cd, device=cls.device), dim=-1)
        else:
            x = torch.randn(B, self.cd, device=cls.device)
        dt = 1.0 / max(1, n_steps)
        v_prev = None
        h_prev = None

        def _vel(xt: torch.Tensor, t_val: float) -> torch.Tensor:
            t = torch.full((B,), t_val, device=xt.device, dtype=xt.dtype)
            return self.forward(xt, t, cls)

        for i in range(max(1, n_steps)):
            t_cur = i * dt
            t_next = (i + 1) * dt
            h_cur = t_next - t_cur

            if solver == "euler":
                v = _vel(x, t_cur)
                v_step = v
            elif solver == "midpoint":
                v = _vel(x, t_cur)
                t_mid = 0.5 * (t_cur + t_next)
                if self.geometry == "sphere":
                    x_mid = _exp_map(x, v, h_cur / 2)
                else:
                    x_mid = x + v * (h_cur / 2)
                v_step = _vel(x_mid, t_mid)
            elif solver == "dpm2m":
                v_cur = _vel(x, t_cur)
                if v_prev is None or h_prev is None:
                    v_step = v_cur
                else:
                    r = h_prev / h_cur
                    v_step = (1 + r / 2) * v_cur - (r / 2) * v_prev
                v_prev = v_cur
                h_prev = h_cur
            else:
                raise ValueError(f"Unknown solver: {solver!r}. Use 'euler', 'midpoint', or 'dpm2m'.")

            if self.geometry == "sphere":
                x = _exp_map(x, v_step, h_cur)
            else:
                x = x + v_step * h_cur
        return F.normalize(x, dim=-1)

    @torch.no_grad()
    def get_uncertainty(self, clip_t: torch.Tensor, t: torch.Tensor, cls: torch.Tensor) -> torch.Tensor:
        """Return per-sample mean predictive sigma for VFM priors."""
        if self.sigma_head is None:
            return torch.zeros(clip_t.shape[0], device=clip_t.device, dtype=clip_t.dtype)
        _, sigma, _ = self.forward_with_sigma(clip_t, t, cls)
        return sigma.mean(dim=-1)



def migrate_input_proj(state_dict: dict, new_max_vox: int) -> dict:
    """Migrate a checkpoint with per-subject ModuleDict input_proj to the new
    shared zero-padded Linear format (A.4).

    If the state_dict already has the new format (brain_enc.input_proj.weight),
    this is a no-op. If it has the old format (brain_enc.input_proj.<sid>.weight),
    the largest subject's weight is used to initialise the new projection (other
    subjects' weights have different voxel counts and cannot be reused directly).

    A warning is always printed when migration occurs so future regressions are
    visible. A hard crash is never raised — worst case the projection is random
    and training resumes from that starting point. New subject-adapter / ridge
    skip parameters are intentionally left absent so strict=False loading keeps
    their zero-init no-op behaviour.
    """
    new_key = "brain_enc.input_proj.weight"
    if new_key in state_dict:
        return state_dict  # already new format

    # Find old per-subject keys and pick the largest weight
    import re
    old_keys = {k for k in state_dict if re.match(r"brain_enc\.input_proj\.\d+\.(weight|bias)", k)}
    if not old_keys:
        return state_dict  # no input_proj at all — return unchanged

    print("[migrate_input_proj] Detected old per-subject input_proj. Migrating to shared Linear…")
    # Find subject with most voxels (largest weight matrix)
    weight_keys = sorted(
        [k for k in old_keys if k.endswith(".weight")],
        key=lambda k: state_dict[k].shape[1],  # sort by in_features
        reverse=True,
    )
    best_w_key = weight_keys[0]
    best_b_key = best_w_key.replace(".weight", ".bias")
    best_w = state_dict[best_w_key]   # (enc_hidden, old_max_vox)
    old_vox = best_w.shape[1]

    out = {k: v for k, v in state_dict.items() if k not in old_keys}
    # Pad or truncate to new_max_vox
    if old_vox <= new_max_vox:
        padded = F.pad(best_w, (0, new_max_vox - old_vox))
    else:
        padded = best_w[:, :new_max_vox]
    out[new_key] = padded
    out["brain_enc.input_proj.bias"] = state_dict.get(best_b_key, torch.zeros(best_w.shape[0]))
    print(f"[migrate_input_proj] Done. Initialised from '{best_w_key}' (vox={old_vox} → {new_max_vox}).")
    return out


class CLIPPriorHead(nn.Module):
    """Two-stage CLIP prior head (C.1, MindEye-v2 style).
    Maps brain CLS embedding → predicted L2-normalised CLIP embedding.
    Used when cfg.use_clip_prior is True. Added as a single extra token
    to the cross-attention context — avoids broadcasting over all N tokens."""
    def __init__(self, brain_dim: int, clip_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(brain_dim, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(1024, 1024), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(1024, clip_dim),
        )

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(cls), dim=-1)


class BrainFlowV5(nn.Module):
    def __init__(self, cfg: Config, voxels_per_subject: Dict[int, int]):
        super().__init__()
        self.cfg = cfg
        self.brain_enc = BrainEncoder(cfg, voxels_per_subject)
        method = getattr(cfg, "method", "baseline")
        # Flow backbone: DiT when method=="dit", else UNet
        if method == "dit":
            self.flow_dit = FlowDiT(cfg)
            self.clip_prior = CLIPPrior(cfg)
            # Project (predicted or true) CLIP embedding → brain-token dim so
            # it can be concatenated to the cross-attention context.
            self.clip_to_token = nn.Sequential(
                nn.LayerNorm(cfg.clip_dim),
                nn.Linear(cfg.clip_dim, cfg.brain_dim),
            )
            self.flow_unet = None
        else:
            self.flow_unet = FlowUNet(cfg)
        # Brain-manifold source projector (cfg.method == "hrf"):
        # maps the CLS embedding into the VAE latent space to serve as a
        # structured starting point for the flow.
        # B.2: initialise with small non-zero weights (not zeros) so the source
        # distribution is not trivially degenerate from the first step.
        if method == "hrf":
            self.cls_to_latent = nn.Linear(
                cfg.brain_dim, cfg.latent_ch * cfg.latent_res * cfg.latent_res
            )
            nn.init.normal_(self.cls_to_latent.weight, std=0.02)
            nn.init.zeros_(self.cls_to_latent.bias)
        else:
            self.cls_to_latent = None

        # C.1: optional CLIP prior head — maps CLS → predicted CLIP embedding,
        # appended as a single extra token to the cross-attention context.
        if getattr(cfg, "use_clip_prior", False):
            self.clip_prior_head = CLIPPriorHead(cfg.brain_dim, cfg.clip_dim)
            self.clip_to_brain = nn.Linear(cfg.clip_dim, cfg.brain_dim)
        else:
            self.clip_prior_head = None
            self.clip_to_brain = None

        # Learned null embedding for the prior token slot — used instead of
        # zero-padding when n_ctx > base_null.shape[1] (zero is out-of-distribution
        # for the attention key space and makes CFG unconditional path degenerate).
        if getattr(cfg, "use_clip_prior", False) or getattr(cfg, "method", "baseline") == "dit":
            self.null_prior_token = nn.Parameter(torch.randn(1, 1, cfg.brain_dim) * 0.01)
        else:
            self.null_prior_token = None

    def encode_fmri(self, fmri, subject):
        return self.brain_enc(fmri, subject)

    def _flow(self):
        """Dispatch to the active flow backbone (DiT or UNet)."""
        if self.cfg.method == "dit":
            return self.flow_dit
        return self.flow_unet

    def _expected_context_tokens(self, has_cls: bool) -> int:
        n = int(self.cfg.n_tokens)
        if getattr(self.cfg, "method", "baseline") == "dit":
            n += 1
        if self.clip_prior_head is not None and has_cls:
            n += 1
        return n

    def _source_from_cls(self, cls: torch.Tensor) -> torch.Tensor:
        """Brain-manifold starting point: projected CLS + unit-variance noise.

        B.1: detach cls so the encoder cannot cheat by driving x0 → latent
        (which would collapse CFM loss to 0 trivially). Unit-variance noise
        keeps the source distribution well-spread, making the flow non-trivial.
        """
        cfg = self.cfg
        B = cls.shape[0]
        cls_d = cls.detach()  # B.1: stop encoder from collapsing the source
        proj = self.cls_to_latent(cls_d).view(B, cfg.latent_ch, cfg.latent_res, cfg.latent_res)
        return proj + torch.randn_like(proj)  # B.1: unit-variance noise (was 0.1×)

    def _get_null_tokens(self, flow, n_ctx: int, B: int, device, dtype):
        """Build unconditional context tokens with optional learned prior-token pad."""
        base_null = flow.null_tokens.to(device=device, dtype=dtype)
        if n_ctx > base_null.shape[1]:
            if self.null_prior_token is not None:
                null_pad = self.null_prior_token.to(device=device, dtype=dtype)
            else:
                null_pad = torch.zeros(1, n_ctx - base_null.shape[1], base_null.shape[2],
                                       device=device, dtype=dtype)
            return torch.cat([base_null, null_pad], dim=1).expand(B, -1, -1)
        return base_null.expand(B, -1, -1)

    @torch.no_grad()
    def sample(self, tokens, n_steps=None, cfg_scale=None, solver: str = "dpm2m",
               cls: torch.Tensor | None = None):
        """ODE integration from source distribution → VAE latent.

        For cfg.method == "hrf", `cls` MUST be provided so the source distribution
        can be initialized as projected_CLS + noise (brain-manifold flow).
        For cfg.method == "dit", `cls` MUST be provided to drive the CLIP prior,
        whose sampled embedding is appended to the cross-attention context.
        For baseline/sb, source = pure Gaussian noise.

        solver: "euler" | "midpoint" | "heun" | "dpm2m"
        """
        cfg = self.cfg
        if n_steps is None: n_steps = cfg.ode_steps
        if cfg_scale is None: cfg_scale = cfg.cfg_scale
        n_steps = max(1, int(n_steps))
        B = tokens.shape[0]
        method = getattr(cfg, "method", "baseline")

        # B.4: For HRF, the source x_0 is already conditional on cls — using CFG
        # on top of that is meaningless (and hurts PC). Force cfg_scale=1.0.
        if method == "hrf":
            cfg_scale = 1.0

        # ── Build cross-attention context (+ prior token for DiT / CLIPPriorHead) ──
        if method == "dit":
            assert cls is not None, "DiT method requires cls for the CLIP prior"
            clip_pred = self.clip_prior.sample(cls, n_steps=cfg.prior_ode_steps, solver="dpm2m")
            prior_tok = self.clip_to_token(clip_pred).unsqueeze(1)      # (B,1,bd)
            tokens = torch.cat([tokens, prior_tok], dim=1)

        # C.1: CLIPPriorHead — append predicted CLIP token to cross-attention context
        if self.clip_prior_head is not None and cls is not None:
            clip_pred = self.clip_prior_head(cls)                        # (B, clip_dim)
            prior_tok = self.clip_to_brain(clip_pred).unsqueeze(1)       # (B, 1, brain_dim)
            tokens = torch.cat([tokens, prior_tok], dim=1)
        assert tokens.shape[1] == self._expected_context_tokens(cls is not None), (
            f"context token mismatch at sampling: got {tokens.shape[1]}, "
            f"expected {self._expected_context_tokens(cls is not None)}"
        )
        # ── Source distribution ──
        if method == "hrf":
            assert cls is not None, "HRF method requires cls embedding for sampling"
            x = self._source_from_cls(cls)
        else:
            x = torch.randn(B, cfg.latent_ch, cfg.latent_res, cfg.latent_res,
                            device=tokens.device)
        flow = self._flow()
        n_ctx = tokens.shape[1]
        null = self._get_null_tokens(flow, n_ctx, B, tokens.device, tokens.dtype)

        if getattr(cfg, "t_schedule", "linear") == "cosine":
            idx = torch.arange(n_steps + 1, device=tokens.device, dtype=tokens.dtype)
            t_grid = (1 - torch.cos(math.pi * idx / n_steps)) / 2
        else:
            t_grid = torch.linspace(0.0, 1.0, steps=n_steps + 1, device=tokens.device, dtype=tokens.dtype)

        def _vel(xt, t_val):
            t = torch.full((B,), t_val, device=xt.device, dtype=xt.dtype)
            if cfg_scale > 1.0:
                v_cond = flow(xt, t, tokens)
                v_uncond = flow(xt, t, null)
                return v_uncond + cfg_scale * (v_cond - v_uncond)
            return flow(xt, t, tokens)

        v_prev = None
        h_prev = None
        for i in range(n_steps):
            t_cur = float(t_grid[i].item())
            t_next = float(t_grid[i + 1].item())
            h_cur = t_next - t_cur
            if solver == "euler":
                v = _vel(x, t_cur)
                x = x + v * h_cur
            elif solver == "midpoint":
                t_mid = 0.5 * (t_cur + t_next)
                v = _vel(x, t_mid)
                x = x + v * h_cur
            elif solver == "heun":
                v1 = _vel(x, t_cur)
                x_euler = x + v1 * h_cur
                v2 = _vel(x_euler, t_next)
                x = x + (v1 + v2) * (h_cur / 2)
            elif solver == "dpm2m":
                v_cur = _vel(x, t_cur)
                if v_prev is None or h_prev is None:
                    v_step = v_cur
                else:
                    r = h_prev / h_cur
                    v_step = (1 + r / 2) * v_cur - (r / 2) * v_prev
                x = x + v_step * h_cur
                v_prev = v_cur
                h_prev = h_cur
            else:
                raise ValueError(f"Unknown solver: {solver!r}. Use 'euler', 'midpoint', 'heun', or 'dpm2m'.")

            # SB stochastic sampler: add σ(t)·√dt·ε at each step (Euler-Maruyama).
            # σ(t) = σ_max·√(t(1-t)) peaks at t=0.5, vanishes at endpoints so the
            # final sample is deterministic given the velocity field.
            if getattr(cfg, "method", "baseline") == "sb":
                sigma_t = cfg.sb_sigma_max * math.sqrt(max(t_next * (1 - t_next), 0.0))
                if sigma_t > 0:
                    x = x + sigma_t * math.sqrt(h_cur) * torch.randn_like(x)
        return x

    def training_step(self, batch, device, vae=None, percep_loss_fn=None, epoch: int = 0):
        cfg = self.cfg
        fmri = batch["fmri"].to(device, non_blocking=True)
        latent = batch["latent"].to(device, non_blocking=True)
        clip_emb = batch["clip_emb"].to(device, non_blocking=True)
        subject = batch["subject"].to(device, non_blocking=True)
        B = fmri.shape[0]
        method = getattr(cfg, "method", "baseline")

        # Mixup: mix only the fMRI input (not the latent target).
        # VAE latent space is not linear — mixed latents are off-manifold.
        # Mixing fMRI still acts as input augmentation without corrupting targets.
        # B.6: Disable mixup entirely for HRF method — mixed fMRI → mixed cls_emb
        # → mixed x0, but latent target is NOT mixed → inconsistent (x0, latent)
        # pairs on 30% of steps sabotage the flow objective.
        if self.training and cfg.mixup_alpha > 0 and method != "hrf" and random.random() < 0.3:
            lam = float(np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha))
            lam = max(lam, 1 - lam)
            perm = torch.randperm(B, device=device)
            fmri_mixed = lam * fmri + (1 - lam) * fmri[perm]
        else:
            fmri_mixed = fmri
            lam = 1.0

        enc_out = self.encode_fmri(fmri_mixed, subject)
        tokens, cls_emb = enc_out
        cls_align = enc_out.cls_align

        # DiT: teacher-force the CLIP prior token onto the context.
        # Curriculum exposure bias fix: ramp up prior-sample probability from 0 → 0.3
        # over the first 30 epochs so the DiT learns to handle imperfect CLIP tokens
        # that it will see at inference time (train/eval mismatch fix).
        if method == "dit":
            prior_sample_prob = min(0.3, (epoch / 30) * 0.3)
            if self.training and random.random() < prior_sample_prob:
                with torch.no_grad():
                    clip_token_for_ctx = self.clip_prior.sample(cls_emb.detach(), n_steps=5)
            else:
                clip_token_for_ctx = clip_emb  # teacher-force true CLIP embedding
            prior_tok = self.clip_to_token(clip_token_for_ctx).unsqueeze(1)        # (B,1,bd)
            tokens = torch.cat([tokens, prior_tok], dim=1)

        # C.1: CLIPPriorHead — teacher-force true CLIP embedding at train time
        # as a single extra token (don't broadcast over all N tokens).
        loss_clip_prior = torch.tensor(0.0, device=device)
        if self.clip_prior_head is not None:
            clip_pred = self.clip_prior_head(cls_emb)                   # (B, clip_dim)
            prior_tok = self.clip_to_brain(clip_pred).unsqueeze(1)       # (B, 1, brain_dim)
            tokens = torch.cat([tokens, prior_tok], dim=1)
            # Prior loss: 1 - cosine similarity with true CLIP embedding
            loss_clip_prior = 1.0 - (clip_pred * clip_emb).sum(-1).mean()
        assert tokens.shape[1] == self._expected_context_tokens(True), (
            f"context token mismatch at train: got {tokens.shape[1]}, "
            f"expected {self._expected_context_tokens(True)}"
        )
        # Source distribution: brain-manifold (HRF) or pure noise (baseline/sb/dit)
        if method == "hrf":
            x0 = self._source_from_cls(cls_emb)
            # B.7: Sanity guard — warn if source has collapsed (x0 ≈ latent)
            with torch.no_grad():
                src_mse = (latent - x0).pow(2).mean(dim=[1, 2, 3])
                collapse_frac = (src_mse < 0.05).float().mean().item()
                if collapse_frac > 0.5:
                    print(
                        f"[HRF WARNING] source_collapse_warn=1 "
                        f"(collapse_frac={collapse_frac:.2f}, src_mse={src_mse.mean().item():.4f}). "
                        f"Consider bumping noise scale or checking cls_to_latent init."
                    )
        else:
            x0 = torch.randn_like(latent)
            if getattr(cfg, "use_ot_coupling", False):
                with torch.no_grad():
                    C = torch.cdist(latent.view(B, -1), x0.view(B, -1)).pow(2)
                    col_idx = _sinkhorn_ot(C, reg=cfg.ot_reg, n_iter=cfg.ot_iters)
                x0 = x0[col_idx]
        t = _sample_t(B, device, cfg)
        t_expand = t[:, None, None, None]
        xt = (1 - t_expand) * x0 + t_expand * latent
        ut = latent - x0

        # SB noisy-bridge interpolant: add σ(t)·ε to xt (velocity target unchanged)
        if method == "sb":
            sigma_t = cfg.sb_sigma_max * torch.sqrt((t_expand * (1 - t_expand)).clamp(min=0))
            xt = xt + sigma_t * torch.randn_like(xt)

        flow = self._flow()
        n_ctx = tokens.shape[1]
        null_tokens = self._get_null_tokens(flow, n_ctx, B, tokens.device, tokens.dtype)

        drop_mask = torch.rand(B, device=device) < cfg.cfg_drop_prob
        ctx = torch.where(drop_mask[:, None, None], null_tokens, tokens)

        # Token dropout — replace dropped tokens with null_tokens so the
        # UNet always sees a valid distribution (not zeros).
        # Applied AFTER CFG drop so we don't double-apply to already-nulled path.
        if self.training and cfg.token_drop_prob > 0:
            drop = torch.rand(B, ctx.shape[1], 1, device=device) < cfg.token_drop_prob
            ctx = torch.where(drop, null_tokens, ctx)

        vt = flow(xt, t, ctx)
        loss_cfm = F.mse_loss(vt, ut)

        if lam < 1.0:
            clean_out = self.encode_fmri(fmri, subject)
            _, cls_clean = clean_out
            cls_clean_align = clean_out.cls_align
        else:
            cls_clean = cls_emb
            cls_clean_align = cls_align
        labels = torch.arange(B, device=device)
        use_mixco = bool(getattr(cfg, "use_mixco", True))
        mixco_alpha = float(getattr(cfg, "mixco_alpha", 0.3))
        if self.training and use_mixco and mixco_alpha > 0:
            lam = float(np.random.beta(mixco_alpha, mixco_alpha))
            lam = max(lam, 1.0 - lam)
            perm = torch.randperm(B, device=device)
            cls_anchor = F.normalize(
                lam * cls_clean_align + (1.0 - lam) * cls_clean_align[perm], dim=-1
            )
            logits = (cls_anchor @ clip_emb.T) / cfg.infonce_temp
            loss_i = lam * F.cross_entropy(logits, labels) + \
                (1.0 - lam) * F.cross_entropy(logits, perm)
            loss_t = lam * F.cross_entropy(logits.T, labels) + \
                (1.0 - lam) * F.cross_entropy(logits.T, perm)
            loss_align = (loss_i + loss_t) / 2
        else:
            logits = (cls_clean_align @ clip_emb.T) / cfg.infonce_temp
            loss_align = (F.cross_entropy(logits, labels) +
                          F.cross_entropy(logits.T, labels)) / 2

        # DiT: diffusion-prior flow-matching loss on CLIP embeddings
        loss_prior = torch.tensor(0.0, device=device)
        if method == "dit":
            vfm_kl_w = getattr(cfg, "vfm_kl_weight", 0.0)
            vfm_anneal = getattr(cfg, "vfm_kl_anneal_epochs", 50)
            kl_weight = vfm_kl_w * min(1.0, epoch / max(1, vfm_anneal))
            loss_prior = self.clip_prior.flow_loss(clip_emb, cls_clean, vfm_kl_weight=kl_weight)

        # Compute low-level losses from one-step endpoint estimate:
        #   pred_latent_hat = xt + vt * (1 - t)
        # Apply a soft t-weight (higher t gets more weight) with an optional
        # hard floor to avoid decoding near-pure-noise latents at very low t.
        percep_t_min = float(getattr(cfg, "percep_t_min", 0.5))
        percep_t_power = float(getattr(cfg, "percep_t_power", 1.0))
        pred_latent_hat = xt + vt * (1 - t_expand)

        def _weighted_mean(vals: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            denom = weights.sum().clamp_min(1e-8)
            return (vals * weights).sum() / denom

        def _per_sample_percep(pred_imgs: torch.Tensor, gt_imgs: torch.Tensor) -> torch.Tensor:
            raw = percep_loss_fn(pred_imgs, gt_imgs)
            if raw.ndim == 0:
                if hasattr(percep_loss_fn, "lpips"):
                    vals = percep_loss_fn.lpips(pred_imgs, gt_imgs)
                    return vals.view(vals.shape[0], -1).mean(dim=1)
                return (pred_imgs - gt_imgs).abs().flatten(1).mean(dim=1)
            return raw.view(raw.shape[0], -1).mean(dim=1)

        loss_percep = torch.tensor(0.0, device=device)
        if percep_loss_fn is not None and vae is not None and cfg.lambda_percep > 0:
            mask = t >= percep_t_min
            if mask.any():
                pred_imgs = vae.decode_grad(pred_latent_hat[mask]) * 2.0 - 1.0
                with torch.no_grad():
                    gt_imgs_masked = batch["image"].to(device, non_blocking=True)[mask]
                    gt_imgs_masked = gt_imgs_masked * 2.0 - 1.0  # [0,1] → [-1,1] for LPIPS
                w = t[mask].pow(percep_t_power)
                per_sample = _per_sample_percep(pred_imgs, gt_imgs_masked)
                loss_percep = _weighted_mean(per_sample, w)

        # C.2: Pixel-space L1 loss with the same t-floor + soft weighting.
        loss_pixel = torch.tensor(0.0, device=device)
        lambda_pixel = getattr(cfg, "lambda_pixel", 0.0)
        if lambda_pixel > 0 and vae is not None:
            mask = t >= percep_t_min
            if mask.any():
                pred_imgs_px = vae.decode_grad(pred_latent_hat[mask]) * 2.0 - 1.0
                with torch.no_grad():
                    gt_imgs_px = batch["image"].to(device, non_blocking=True) * 2 - 1
                    gt_imgs_px = gt_imgs_px[mask]
                per_sample_l1 = (pred_imgs_px - gt_imgs_px).abs().flatten(1).mean(dim=1)
                w = t[mask].pow(percep_t_power)
                loss_pixel = _weighted_mean(per_sample_l1, w)

        lambda_prior_w = getattr(cfg, "lambda_prior", 0.0)
        total = (cfg.lambda_cfm * loss_cfm
                 + cfg.lambda_align * loss_align
                 + cfg.lambda_percep * loss_percep
                 + lambda_prior_w * (loss_prior + loss_clip_prior)
                 + lambda_pixel * loss_pixel)
        return {"loss": total, "cfm": loss_cfm, "align": loss_align,
                "percep": loss_percep, "prior": loss_prior, "pixel": loss_pixel}
