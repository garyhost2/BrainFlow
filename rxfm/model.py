from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import ResMLP, SinusoidalTime
from .sphere import (random_sphere, project_tangent, exp_map, log_map,
                     slerp, slerp_velocity, polar_encode, polar_decode,
                     tangent_noise_rad)


def soft_clip_loss(preds: torch.Tensor, targs: torch.Tensor, temp: float) -> torch.Tensor:
    clip_clip = (targs @ targs.T) / temp
    brain_clip = (preds @ targs.T) / temp
    soft = clip_clip.softmax(dim=-1)
    loss1 = -(brain_clip.log_softmax(dim=-1) * soft).sum(dim=-1).mean()
    loss2 = -(brain_clip.T.log_softmax(dim=-1) * soft).sum(dim=-1).mean()
    return (loss1 + loss2) / 2


def mixco(x: torch.Tensor, beta: float = 0.15, s_thresh: float = 0.5):
    B = x.shape[0]
    device = x.device
    perm = torch.randperm(B, device=device)
    betas = torch.distributions.Beta(beta, beta).sample([B]).to(device, x.dtype)
    select = torch.rand(B, device=device) <= s_thresh
    betas = torch.where(select, betas, torch.ones_like(betas))
    shape = [B] + [1] * (x.dim() - 1)
    bview = betas.view(shape)
    return x * bview + x[perm] * (1 - bview), perm, betas, select


def mixco_nce(preds: torch.Tensor, targs: torch.Tensor, temp: float,
              perm: torch.Tensor, betas: torch.Tensor,
              select: torch.Tensor) -> torch.Tensor:
    sim = (preds @ targs.T) / temp
    B = preds.shape[0]
    idx = torch.arange(B, device=preds.device)
    probs = torch.zeros_like(sim)
    probs[idx, idx] = betas.to(sim.dtype)
    sel = idx[select]
    probs[sel, perm[select]] = probs[sel, perm[select]] + (1 - betas[select]).to(sim.dtype)
    loss1 = -(sim.log_softmax(dim=-1) * probs).sum(dim=-1).mean()
    loss2 = -(sim.T.log_softmax(dim=-1) * probs.T).sum(dim=-1).mean()
    return (loss1 + loss2) / 2


@dataclass
class TokenStep1Config:
    token_len: int = 256
    token_dim: int = 1664
    cls_dim: int = 1280
    brain_dim: int = 1024
    n_brain_tokens: int = 64

    enc_hidden: int = 2048
    enc_blocks: int = 4
    enc_drop: float = 0.15

    reg_depth: int = 2
    reg_heads: int = 8
    radius_depth: int = 2

    prior_width: int = 1024
    prior_depth: int = 8
    prior_heads: int = 8
    time_dim: int = 256

    two_head: bool = True
    cls_width: int = 512
    cls_depth: int = 4
    cls_heads: int = 8
    cls_cfg_scale: float = 3.0

    cfg_drop_prob: float = 0.1
    logit_normal_m: float = 0.0
    logit_normal_s: float = 1.0
    uniform_t_prob: float = 0.1
    t_min: float = 0.02
    t_max: float = 0.98

    flow_source: str = "anchor_var"
    flow_param: str = "auto"

    def resolved_flow_param(self) -> str:
        if self.flow_param != "auto":
            return self.flow_param
        return "endpoint" if self.geometry == "sphere" else "velocity"
    anchor_jitter_rad: float = 0.15
    cls_cond_mode: str = "jitter"
    cls_jitter_rad: float = 0.30
    cls_ss_prob: float = 0.5
    cls_ss_steps: int = 8

    lambda_flow: float = 1.0
    lambda_rcfm: float = 1.0
    lambda_reg: float = 0.0
    lambda_cos: float = 0.5
    lambda_radius: float = 1.0
    lambda_clip: float = 0.033
    lambda_clip_tok: float = 0.0
    lambda_kl: float = 0.1
    clip_temp: float = 0.006

    mixup_pct: float = 0.0
    mixco_beta: float = 0.15
    mixco_s_thresh: float = 0.5

    low_level: bool = True
    ll_target: str = "rgb"
    ll_size: int = 64
    ll_base: int = 256
    ll_hidden: int = 1024
    ll_loss: str = "l1"
    lambda_low: float = 1.0
    ll_strength: float = 0.7

    n_steps: int = 50
    cfg_scale: float = 3.0
    solver: str = "heun"
    cond_source: str = "prior"
    blend_w: float = 0.5
    stochastic_source: bool = False

    geometry: str = "sphere"
    center_tokens: bool = True
    center_mode: str = "global"
    subjects: list[int] = field(default_factory=lambda: [1])

    shared_encoder: bool = True
    subject_rank: int = 16


class SelfAttn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads; self.dh = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def rsh(t): return t.view(B, L, self.h, self.dh).transpose(1, 2)
        o = F.scaled_dot_product_attention(rsh(q), rsh(k), rsh(v))
        return self.proj(o.transpose(1, 2).reshape(B, L, C))


class CrossAttn(nn.Module):

    def __init__(self, dim, ctx_dim, heads):
        super().__init__()
        self.h = heads; self.dh = dim // heads
        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(ctx_dim, dim, bias=False)
        self.to_v = nn.Linear(ctx_dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, ctx):
        B, L, C = x.shape
        M = ctx.shape[1]
        xn = self.norm(x)
        q = self.to_q(xn); k = self.to_k(ctx); v = self.to_v(ctx)
        def rsh(t, n): return t.view(B, n, self.h, self.dh).transpose(1, 2)
        o = F.scaled_dot_product_attention(rsh(q, L), rsh(k, M), rsh(v, M))
        return x + self.proj(o.transpose(1, 2).reshape(B, L, C))


class SubjectResidualAdapter(nn.Module):

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


class TokenBackbone(nn.Module):
    def __init__(self, cfg: TokenStep1Config, voxels_per_subject: dict[int, int]):
        super().__init__()
        self.cfg = cfg
        h = cfg.enc_hidden
        self.subject_ids = sorted(int(s) for s in voxels_per_subject)
        self.voxels_per_subject = {int(s): int(v) for s, v in voxels_per_subject.items()}
        self.max_vox = max(self.voxels_per_subject.values())
        self.shared_encoder = bool(getattr(cfg, "shared_encoder", False))

        if self.shared_encoder:
            self.input_proj = nn.Linear(self.max_vox, h)
            self.subject_adapter = SubjectResidualAdapter(
                n_subjects=len(self.subject_ids), hidden_dim=h,
                rank=int(getattr(cfg, "subject_rank", 16)))
            lookup = torch.full((max(self.subject_ids) + 1,), -1, dtype=torch.long)
            for i, sid in enumerate(self.subject_ids):
                lookup[sid] = i
            self.register_buffer("subject_lookup", lookup, persistent=False)
        else:
            self.input_proj = nn.ModuleDict({
                str(s): nn.Linear(v, h) for s, v in self.voxels_per_subject.items()})
            self.subject_adapter = None

        self.stem = nn.Sequential(nn.LayerNorm(h), nn.GELU(), nn.Dropout(cfg.enc_drop))
        self.blocks = nn.ModuleList([ResMLP(h, 4, cfg.enc_drop) for _ in range(cfg.enc_blocks)])
        self.to_brain = nn.Sequential(
            nn.Linear(h, cfg.n_brain_tokens * cfg.brain_dim),
            nn.Unflatten(-1, (cfg.n_brain_tokens, cfg.brain_dim)),
            nn.LayerNorm(cfg.brain_dim),
        )

    def _subject_indices(self, subject, batch_size: int, device) -> torch.Tensor:
        if torch.is_tensor(subject):
            sub = subject.to(device=device, dtype=torch.long).reshape(-1)
        else:
            sub = torch.as_tensor([int(subject)], device=device, dtype=torch.long)
        if sub.numel() == 1 and batch_size != 1:
            sub = sub.expand(batch_size)
        if sub.numel() != batch_size:
            raise ValueError(f"expected {batch_size} subject ids, got {sub.numel()}")
        if int(sub.max()) >= self.subject_lookup.numel() or int(sub.min()) < 0:
            raise KeyError(f"unknown subject id(s): {sub.unique().tolist()}")
        idx = self.subject_lookup[sub]
        if bool((idx < 0).any()):
            raise KeyError(f"unknown subject id(s): {sub.unique().tolist()}")
        return idx

    def forward(self, fmri, subject):
        if self.shared_encoder:
            if fmri.shape[-1] > self.max_vox:
                raise ValueError(
                    f"fMRI width {fmri.shape[-1]} exceeds max_vox {self.max_vox}")
            if fmri.shape[-1] < self.max_vox:
                fmri = F.pad(fmri, (0, self.max_vox - fmri.shape[-1]))
            x = self.input_proj(fmri)
            x = self.subject_adapter(
                x, self._subject_indices(subject, fmri.shape[0], fmri.device))
        else:
            x = self.input_proj[str(int(subject))](fmri)
        x = self.stem(x)
        for b in self.blocks:
            x = b(x)
        return self.to_brain(x)


class TokenRegHead(nn.Module):

    def __init__(self, cfg: TokenStep1Config):
        super().__init__()
        d = cfg.brain_dim
        self.queries = nn.Parameter(torch.randn(1, cfg.token_len, d) * 0.02)
        self.cross = nn.ModuleList([CrossAttn(d, d, cfg.reg_heads) for _ in range(cfg.reg_depth)])
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))
            for _ in range(cfg.reg_depth)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, cfg.token_dim))

    def forward(self, brain):
        B = brain.shape[0]
        x = self.queries.expand(B, -1, -1)
        for cr, mlp in zip(self.cross, self.mlps):
            x = cr(x, brain)
            x = x + mlp(x)
        return self.out(x)


class RadiusHead(nn.Module):

    def __init__(self, cfg: TokenStep1Config):
        super().__init__()
        d = cfg.brain_dim
        self.queries = nn.Parameter(torch.randn(1, cfg.token_len, d) * 0.02)
        self.cross = nn.ModuleList([CrossAttn(d, d, cfg.reg_heads) for _ in range(cfg.radius_depth)])
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))
            for _ in range(cfg.radius_depth)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, brain):
        B = brain.shape[0]
        x = self.queries.expand(B, -1, -1)
        for cr, mlp in zip(self.cross, self.mlps):
            x = cr(x, brain)
            x = x + mlp(x)
        return self.out(x).squeeze(-1)


class LowLevelHead(nn.Module):

    def __init__(self, cfg: "TokenStep1Config", voxels_per_subject: dict[int, int]):
        super().__init__()
        self.base = cfg.ll_base
        self.grid = 8
        h = cfg.ll_hidden
        self.input_proj = nn.ModuleDict({
            str(s): nn.Linear(v, h) for s, v in voxels_per_subject.items()})
        self.stem = nn.Sequential(nn.LayerNorm(h), nn.GELU(), nn.Dropout(cfg.enc_drop))
        self.to_grid = nn.Sequential(
            nn.Linear(h, self.base * self.grid * self.grid), nn.GELU())
        self.mix = nn.Sequential(
            nn.Conv2d(self.base, self.base, 3, padding=1),
            nn.GroupNorm(8, self.base), nn.SiLU())
        n_up = max(1, int(math.log2(max(cfg.ll_size, self.grid) // self.grid)))
        blocks, c = [], self.base
        for _ in range(n_up):
            nc = max(c // 2, 32)
            blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(c, nc, 3, padding=1), nn.GroupNorm(8, nc), nn.SiLU(),
                nn.Conv2d(nc, nc, 3, padding=1), nn.SiLU()))
            c = nc
        self.ups = nn.Sequential(*blocks)
        self.out = nn.Conv2d(c, 3, 3, padding=1)

    def forward(self, fmri, subject):
        B = fmri.shape[0]
        x = self.stem(self.input_proj[str(int(subject))](fmri))
        h = self.to_grid(x).view(B, self.base, self.grid, self.grid)
        h = self.mix(h)
        return torch.sigmoid(self.out(self.ups(h)))


class LatentLowLevelHead(nn.Module):

    def __init__(self, cfg: "TokenStep1Config", voxels_per_subject: dict[int, int]):
        super().__init__()
        self.base = cfg.ll_base
        self.grid = 12
        h = cfg.ll_hidden
        self.input_proj = nn.ModuleDict({
            str(s): nn.Linear(v, h) for s, v in voxels_per_subject.items()})
        self.stem = nn.Sequential(nn.LayerNorm(h), nn.GELU(), nn.Dropout(cfg.enc_drop))
        self.to_grid = nn.Sequential(
            nn.Linear(h, self.base * self.grid * self.grid), nn.GELU())
        self.mix = nn.Sequential(
            nn.Conv2d(self.base, self.base, 3, padding=1),
            nn.GroupNorm(8, self.base), nn.SiLU())
        blocks, c = [], self.base
        for _ in range(3):
            nc = max(c // 2, 32)
            blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(c, nc, 3, padding=1), nn.GroupNorm(8, nc), nn.SiLU(),
                nn.Conv2d(nc, nc, 3, padding=1), nn.SiLU()))
            c = nc
        self.ups = nn.Sequential(*blocks)
        self.out = nn.Conv2d(c, 4, 3, padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.register_buffer("lat_mean", torch.zeros(4))
        self.register_buffer("lat_std", torch.ones(4))

    def set_latent_stats(self, mean, std):
        self.lat_mean.copy_(mean.to(self.lat_mean.device, self.lat_mean.dtype))
        self.lat_std.copy_(std.to(self.lat_std.device, self.lat_std.dtype))

    def standardize(self, lat):
        return (lat - self.lat_mean.view(1, -1, 1, 1)) / self.lat_std.view(1, -1, 1, 1)

    def unstandardize(self, z):
        return z * self.lat_std.view(1, -1, 1, 1) + self.lat_mean.view(1, -1, 1, 1)

    def forward(self, fmri, subject):
        B = fmri.shape[0]
        x = self.stem(self.input_proj[str(int(subject))](fmri))
        h = self.to_grid(x).view(B, self.base, self.grid, self.grid)
        return self.out(self.ups(self.mix(h)))


def sigma_from_jitter(jitter_rad: float, k: int) -> float:
    return jitter_rad / math.sqrt(max(1, k))


class LogSigmaHead(nn.Module):

    def __init__(self, cfg: "TokenStep1Config"):
        super().__init__()
        d = cfg.brain_dim
        self.queries = nn.Parameter(torch.randn(1, cfg.token_len, d) * 0.02)
        self.cross = nn.ModuleList([CrossAttn(d, d, cfg.reg_heads)
                                    for _ in range(cfg.radius_depth)])
        self.mlps = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))
            for _ in range(cfg.radius_depth)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        nn.init.zeros_(self.out[-1].weight)
        sigma_p = sigma_from_jitter(cfg.anchor_jitter_rad, cfg.token_dim - 1)
        nn.init.constant_(self.out[-1].bias, math.log(sigma_p))

    def forward(self, brain):
        B = brain.shape[0]
        x = self.queries.expand(B, -1, -1)
        for cr, mlp in zip(self.cross, self.mlps):
            x = cr(x, brain)
            x = x + mlp(x)
        return self.out(x).squeeze(-1)


def wrapped_gaussian_kl(logs: torch.Tensor, sigma_p: float) -> torch.Tensor:
    var_ratio = (2 * logs).exp() / (sigma_p ** 2)
    return 0.5 * (var_ratio - 1.0 - 2 * logs + 2 * math.log(sigma_p))


class DiTBlock(nn.Module):
    def __init__(self, w, heads, brain_dim, time_dim):
        super().__init__()
        self.n1 = nn.LayerNorm(w, elementwise_affine=False)
        self.attn = SelfAttn(w, heads)
        self.cross = CrossAttn(w, brain_dim, heads)
        self.n2 = nn.LayerNorm(w, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(w, w * 4), nn.GELU(), nn.Linear(w * 4, w))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, 6 * w))
        nn.init.zeros_(self.ada[-1].weight); nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, t_emb, brain):
        s1, g1, gate1, s2, g2, gate2 = self.ada(t_emb).chunk(6, dim=-1)
        h = self.n1(x) * (1 + g1[:, None]) + s1[:, None]
        x = x + gate1[:, None] * self.attn(h)
        x = self.cross(x, brain)
        h = self.n2(x) * (1 + g2[:, None]) + s2[:, None]
        x = x + gate2[:, None] * self.mlp(h)
        return x


class TokenFlowPrior(nn.Module):

    def __init__(self, cfg: TokenStep1Config):
        super().__init__()
        self.cfg = cfg
        w = cfg.prior_width
        self.time = SinusoidalTime(cfg.time_dim, w)
        self.in_proj = nn.Linear(cfg.token_dim, w)
        self.pos = nn.Parameter(torch.zeros(1, cfg.token_len, w))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([
            DiTBlock(w, cfg.prior_heads, cfg.brain_dim, w) for _ in range(cfg.prior_depth)])
        self.out_norm = nn.LayerNorm(w, elementwise_affine=False)
        self.out = nn.Linear(w, cfg.token_dim)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.null_brain = nn.Parameter(torch.zeros(1, cfg.n_brain_tokens, cfg.brain_dim))

        if cfg.two_head:
            self.cls_kv = nn.Linear(cfg.cls_dim, cfg.brain_dim)
            self.cls_ada = nn.Linear(cfg.cls_dim, w)
            nn.init.zeros_(self.cls_ada.weight); nn.init.zeros_(self.cls_ada.bias)
            self.null_cls = nn.Parameter(torch.zeros(1, cfg.cls_dim))

    def forward(self, xt, t, brain, cls_emb=None):
        te = self.time(t)
        ctx = brain
        if self.cfg.two_head and cls_emb is not None:
            te = te + self.cls_ada(cls_emb)
            ctx = torch.cat([brain, self.cls_kv(cls_emb).unsqueeze(1)], dim=1)
        h = self.in_proj(xt) + self.pos
        for blk in self.blocks:
            h = blk(h, te, ctx)
        return self.out(self.out_norm(h))


class ClsBlock(nn.Module):

    def __init__(self, w, heads, ctx_dim, time_dim):
        super().__init__()
        self.h = heads; self.dh = w // heads
        self.n1 = nn.LayerNorm(w, elementwise_affine=False)
        self.to_q = nn.Linear(w, w, bias=False)
        self.to_k = nn.Linear(ctx_dim, w, bias=False)
        self.to_v = nn.Linear(ctx_dim, w, bias=False)
        self.cproj = nn.Linear(w, w)
        self.n2 = nn.LayerNorm(w, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(w, w * 4), nn.GELU(), nn.Linear(w * 4, w))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, 6 * w))
        nn.init.zeros_(self.ada[-1].weight); nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, t_emb, ctx):
        s1, g1, gate1, s2, g2, gate2 = self.ada(t_emb).chunk(6, dim=-1)
        B, L, C = x.shape; M = ctx.shape[1]
        h = self.n1(x) * (1 + g1[:, None]) + s1[:, None]
        q = self.to_q(h); k = self.to_k(ctx); v = self.to_v(ctx)
        def rsh(t, n): return t.view(B, n, self.h, self.dh).transpose(1, 2)
        o = F.scaled_dot_product_attention(rsh(q, L), rsh(k, M), rsh(v, M))
        x = x + gate1[:, None] * self.cproj(o.transpose(1, 2).reshape(B, L, C))
        h = self.n2(x) * (1 + g2[:, None]) + s2[:, None]
        x = x + gate2[:, None] * self.mlp(h)
        return x


class ClsFlowPrior(nn.Module):

    def __init__(self, cfg: TokenStep1Config):
        super().__init__()
        self.cfg = cfg
        w = cfg.cls_width
        self.time = SinusoidalTime(cfg.time_dim, w)
        self.in_proj = nn.Linear(cfg.cls_dim, w)
        self.blocks = nn.ModuleList([
            ClsBlock(w, cfg.cls_heads, cfg.brain_dim, w) for _ in range(cfg.cls_depth)])
        self.out_norm = nn.LayerNorm(w, elementwise_affine=False)
        self.out = nn.Linear(w, cfg.cls_dim)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.null_brain = nn.Parameter(torch.zeros(1, cfg.n_brain_tokens, cfg.brain_dim))

    def forward(self, x, t, brain):
        te = self.time(t)
        h = self.in_proj(x).unsqueeze(1)
        for blk in self.blocks:
            h = blk(h, te, brain)
        return self.out(self.out_norm(h.squeeze(1)))


class TokenStep1Model(nn.Module):
    def __init__(self, cfg: TokenStep1Config, voxels_per_subject: dict[int, int]):
        super().__init__()
        if cfg.flow_param not in ("auto", "endpoint", "velocity"):
            raise ValueError(f"Unknown flow_param: {cfg.flow_param!r}")
        if cfg.flow_param == "endpoint" and cfg.geometry != "sphere":
            raise ValueError(
                "flow_param='endpoint' is defined by the geodesic identity "
                "u_t = Log_{z_t}(z1)/(1-t) and is therefore sphere-only; "
                "use geometry='sphere' or flow_param='velocity'.")
        self.flow_param = cfg.resolved_flow_param()
        if cfg.center_mode not in ("global", "per_position"):
            raise ValueError(f"Unknown center_mode: {cfg.center_mode!r}")
        self.cfg = cfg
        self.backbone = TokenBackbone(cfg, voxels_per_subject)
        self.reg_head = TokenRegHead(cfg)
        self.prior = TokenFlowPrior(cfg)
        self.cls_prior = ClsFlowPrior(cfg) if cfg.two_head else None

        clip_out = cfg.cls_dim if cfg.two_head else cfg.token_dim
        self.clip_proj = nn.Sequential(
            nn.LayerNorm(cfg.brain_dim),
            nn.Linear(cfg.brain_dim, cfg.brain_dim), nn.GELU(),
            nn.Linear(cfg.brain_dim, clip_out),
        )
        self.radius_head = RadiusHead(cfg)
        self.logs_head = LogSigmaHead(cfg) if cfg.flow_source == "anchor_var" else None
        self.low_head = None
        if cfg.low_level:
            head_cls = (LatentLowLevelHead if getattr(cfg, "ll_target", "rgb") == "latent"
                        else LowLevelHead)
            self.low_head = head_cls(cfg, voxels_per_subject)
        self.register_buffer(
            "tgt_mean",
            torch.zeros(cfg.token_len, cfg.token_dim)
            if cfg.center_mode == "per_position" else torch.zeros(cfg.token_dim))

    def set_target_mean(self, mean):
        mean = mean.to(self.tgt_mean.device, self.tgt_mean.dtype)
        if mean.shape != self.tgt_mean.shape:
            if self.cfg.center_mode == "per_position" and mean.dim() == 1:
                mean = mean.unsqueeze(0).expand_as(self.tgt_mean).contiguous()
            else:
                raise ValueError(
                    f"tgt_mean shape {tuple(mean.shape)} does not match "
                    f"center_mode={self.cfg.center_mode!r} "
                    f"(expected {tuple(self.tgt_mean.shape)})")
        self.tgt_mean.copy_(mean)


    def anchor(self, brain):
        mu = F.normalize(self.reg_head(brain).float(), dim=-1)
        logr = self.radius_head(brain).float()
        logs = None
        if self.logs_head is not None:
            k = self.cfg.token_dim - 1
            hi = math.log(sigma_from_jitter(math.pi / 2, k))
            logs = self.logs_head(brain).float().clamp(-12.0, hi)
        return mu, logs, logr

    def sample_anchor(self, mu, logs, detach_base: bool = False):
        if logs is None:
            return mu.detach() if detach_base else mu
        base = mu.detach() if detach_base else mu
        eps = torch.randn_like(base) * logs.exp().unsqueeze(-1)
        return exp_map(base, project_tangent(eps, base))

    def _flow_source(self, mu, logs, shape, device, dtype, detach_base: bool = False):
        src = self.cfg.flow_source
        if src == "noise":
            return random_sphere(shape, device, dtype)
        if src == "anchor_det":
            return mu.detach() if detach_base else mu
        if src == "anchor_var":
            return self.sample_anchor(mu, logs, detach_base=detach_base)
        raise ValueError(f"Unknown flow_source: {src!r}")

    def _prior_velocity(self, zt, t, brain, cls_emb):
        raw = self.prior(zt, t, brain, cls_emb).float()
        if self.flow_param != "endpoint":
            return raw
        z1_hat = F.normalize(zt.float() + raw, dim=-1)
        one_minus_t = (1.0 - t).clamp_min(1.0 - self.cfg.t_max)
        while one_minus_t.dim() < zt.dim():
            one_minus_t = one_minus_t.unsqueeze(-1)
        return log_map(zt.float(), z1_hat) / one_minus_t

    def _prior_endpoint(self, zt, t, brain, cls_emb):
        raw = self.prior(zt, t, brain, cls_emb).float()
        return F.normalize(zt.float() + raw, dim=-1)

    def _cls_cond(self, brain, cls_dir):
        mode = self.cfg.cls_cond_mode
        if cls_dir is None or mode == "none":
            return None
        if mode == "teacher":
            return cls_dir
        if mode == "jitter":
            return tangent_noise_rad(cls_dir, self.cfg.cls_jitter_rad)
        if mode == "sampled":
            if not self.training or torch.rand(()) >= self.cfg.cls_ss_prob:
                return cls_dir
            with torch.no_grad():
                return self._sample_cls(brain, self.cfg.cls_ss_steps,
                                        self.cfg.cls_cfg_scale, "euler").detach()
        raise ValueError(f"Unknown cls_cond_mode: {mode!r}")

    def _sample_t(self, B, device):
        z = torch.randn(B, device=device) * self.cfg.logit_normal_s + self.cfg.logit_normal_m
        t = torch.sigmoid(z)
        if self.cfg.uniform_t_prob > 0:
            mix = torch.rand(B, device=device) < self.cfg.uniform_t_prob
            t = torch.where(mix, torch.rand(B, device=device), t)
        return t

    def training_step(self, fmri, subject, target_std, target_raw=None, target_cls=None,
                      target_img=None, low_only=False, use_mixco=False, target_lat=None,
                      keep_term_graph=False):
        cfg = self.cfg
        B = fmri.shape[0]; device = fmri.device
        if low_only:
            out = self._low_level_loss(fmri, subject, target_img, target_lat)
            if "low" not in out:
                raise ValueError(
                    "low_only training needs low_head enabled + the matching target "
                    "(target_img for ll_target=rgb, target_lat for ll_target=latent)")
            out = {k: v.detach() for k, v in out.items()} | {
                "loss": cfg.lambda_low * out["low"]}
            return out
        mix = None
        fmri_clean = fmri
        if use_mixco and cfg.mixup_pct > 0:
            fmri, perm, betas, select = mixco(fmri, cfg.mixco_beta, cfg.mixco_s_thresh)
            mix = (perm, betas, select)
        brain = self.backbone(fmri, subject)
        cls_dir = None
        if cfg.two_head and target_cls is not None:
            cls_dir = F.normalize(target_cls.float(), dim=-1)

        out = self._cls_flow_loss(brain, cls_dir, B, device)
        cls_cond = self._cls_cond(brain, cls_dir)
        if cfg.geometry == "sphere":
            out |= self._patch_step_sphere(brain, target_raw, cls_cond, B, device)
        else:
            out |= self._patch_step_euclidean(brain, target_std, cls_cond, B, device)
        out |= self._contrastive_loss(brain, target_std, cls_dir, device, mix=mix)
        out |= self._low_level_loss(fmri_clean, subject, target_img, target_lat)

        total = (cfg.lambda_flow * out["flow"]
                 + cfg.lambda_cos * out["cos"] + cfg.lambda_clip * out["clip"]
                 + cfg.lambda_rcfm * out["rcfm"])
        if cfg.lambda_reg > 0 and "reg" in out:
            total = total + cfg.lambda_reg * out["reg"]
        if "radius" in out:
            total = total + cfg.lambda_radius * out["radius"]
        if "kl" in out:
            total = total + cfg.lambda_kl * out["kl"]
        if "clip_tok" in out:
            total = total + cfg.lambda_clip_tok * out["clip_tok"]
        if "low" in out:
            total = total + cfg.lambda_low * out["low"]
        if not keep_term_graph:
            out = {k: v.detach() for k, v in out.items()}
        out["loss"] = total
        return out

    @property
    def low_is_latent(self) -> bool:
        return isinstance(self.low_head, LatentLowLevelHead)

    def _low_level_loss(self, fmri, subject, target_img, target_lat=None):
        if self.low_head is None:
            return {}
        if self.low_is_latent:
            if target_lat is None:
                return {}
            target = self.low_head.standardize(target_lat.float())
        elif target_img is not None:
            target = F.interpolate(target_img.float(), self.cfg.ll_size,
                                   mode="bilinear", align_corners=False).clamp(0, 1)
        else:
            return {}
        pred = self.low_head(fmri, subject)
        mode = getattr(self.cfg, "ll_loss", "l1")
        if mode == "mse":
            low = F.mse_loss(pred, target)
        elif mode == "huber":
            low = F.huber_loss(pred, target, delta=0.1)
        else:
            low = F.l1_loss(pred, target)
        return {"low": low}

    @torch.no_grad()
    def predict_lowlevel(self, fmri, subject):
        if self.low_head is None or self.low_is_latent:
            return None
        return self.low_head(fmri, subject)

    @torch.no_grad()
    def predict_low_latent(self, fmri, subject):
        if not self.low_is_latent:
            return None
        return self.low_head.unstandardize(self.low_head(fmri, subject).float())

    def _cls_flow_loss(self, brain, cls_dir, B, device):
        if self.cls_prior is None or cls_dir is None:
            return {"rcfm": torch.zeros((), device=device)}
        cfg = self.cfg
        z0 = random_sphere(cls_dir.shape, device, cls_dir.dtype)
        t = self._sample_t(B, device)
        tb = t[:, None]
        zt = slerp(z0, cls_dir, tb)
        ut = slerp_velocity(z0, cls_dir, tb)
        drop = (torch.rand(B, device=device) < cfg.cfg_drop_prob)[:, None, None]
        brain_in = torch.where(drop, self.cls_prior.null_brain.to(brain.dtype), brain)
        v = project_tangent(self.cls_prior(zt, t, brain_in).float(), zt)
        return {"rcfm": F.mse_loss(v, ut)}

    def _patch_cond(self, brain, cls_dir, B, device):
        cfg = self.cfg
        drop = (torch.rand(B, device=device) < cfg.cfg_drop_prob)
        brain_in = torch.where(drop[:, None, None], self.prior.null_brain.to(brain.dtype), brain)
        cls_in = None
        if cfg.two_head and cls_dir is not None:
            null_cls = self.prior.null_cls.to(cls_dir.dtype)
            cls_in = torch.where(drop[:, None], null_cls, cls_dir)
        return brain_in, cls_in

    def _patch_step_sphere(self, brain, target_raw, cls_dir, B, device):
        cfg = self.cfg
        z1, r1 = polar_encode(target_raw.float(), self.tgt_mean.float())

        mu, logs, logr = self.anchor(brain)
        loss_cos = 1.0 - F.cosine_similarity(mu, z1, dim=-1).mean()
        loss_radius = F.mse_loss(logr, r1.log())
        loss_reg = F.mse_loss(mu, z1)

        out = {"reg": loss_reg, "cos": loss_cos, "radius": loss_radius}
        if logs is not None:
            sigma_p = sigma_from_jitter(cfg.anchor_jitter_rad, cfg.token_dim - 1)
            out["kl"] = wrapped_gaussian_kl(logs, sigma_p).mean()

        z0 = self._flow_source(mu, logs, z1.shape, device, z1.dtype, detach_base=True)

        t = self._sample_t(B, device)
        tb = t[:, None, None]
        zt = slerp(z0, z1, tb)
        brain_in, cls_in = self._patch_cond(brain, cls_dir, B, device)

        if self.flow_param == "endpoint":
            z1_hat = self._prior_endpoint(zt, t, brain_in, cls_in)
            out["flow"] = 1.0 - F.cosine_similarity(z1_hat, z1, dim=-1).mean()
        else:
            ut = slerp_velocity(z0, z1, tb)
            v = project_tangent(self.prior(zt, t, brain_in, cls_in).float(), zt)
            out["flow"] = F.mse_loss(v, ut)

        if cfg.lambda_clip_tok > 0:
            out["clip_tok"] = soft_clip_loss(
                F.normalize(mu.flatten(1), dim=-1),
                F.normalize(z1.flatten(1), dim=-1), cfg.clip_temp)
        return out

    def _patch_step_euclidean(self, brain, target_std, cls_dir, B, device):
        cfg = self.cfg
        reg = self.reg_head(brain)
        loss_reg = F.mse_loss(reg, target_std)
        loss_cos = 1.0 - F.cosine_similarity(reg.float(), target_std.float(), dim=-1).mean()

        x1 = target_std
        x0 = torch.randn_like(x1)
        t = self._sample_t(B, device)
        tb = t[:, None, None]
        xt = (1 - tb) * x0 + tb * x1
        ut = x1 - x0
        brain_in, cls_in = self._patch_cond(brain, cls_dir, B, device)
        v = self.prior(xt, t, brain_in, cls_in)
        loss_flow = F.mse_loss(v, ut)
        return {"flow": loss_flow, "reg": loss_reg, "cos": loss_cos}

    def _contrastive_loss(self, brain, target_std, cls_dir, device, mix=None):
        cfg = self.cfg
        if cfg.lambda_clip <= 0:
            return {"clip": torch.zeros((), device=device)}
        b = F.normalize(self.clip_proj(brain.mean(dim=1)).float(), dim=-1)
        if cfg.two_head:
            if cls_dir is None:
                return {"clip": torch.zeros((), device=device)}
            z = cls_dir
        else:
            z = F.normalize(target_std.mean(dim=1).float(), dim=-1)
        if mix is not None:
            perm, betas, select = mix
            out = {"clip": mixco_nce(b, z, cfg.clip_temp, perm, betas, select)}
        else:
            out = {"clip": soft_clip_loss(b, z, cfg.clip_temp)}

        return out

    def _tq(self, t):
        return min(max(t, self.cfg.t_min), self.cfg.t_max)

    @torch.no_grad()
    def _integrate_sphere(self, z, vel_fn, n_steps, solver):
        device = z.device
        grid = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
        for i in range(n_steps):
            t0 = float(grid[i]); t1 = float(grid[i + 1]); dt = t1 - t0
            w1 = project_tangent(vel_fn(z, self._tq(t0)).float(), z)
            if solver == "heun":
                z_pred = exp_map(z, w1 * dt)
                w2 = project_tangent(vel_fn(z_pred, self._tq(t1)).float(), z_pred)
                w = project_tangent(0.5 * (w1 + w2), z)
                z = exp_map(z, w * dt)
            else:
                z = exp_map(z, w1 * dt)
        return F.normalize(z, dim=-1)

    @torch.no_grad()
    def _sample_cls(self, brain, n_steps, cfg_scale, solver):
        cfg = self.cfg
        B = brain.shape[0]; device = brain.device
        null = self.cls_prior.null_brain.to(brain.dtype).expand(B, -1, -1)

        def vel(z, tq):
            t = torch.full((B,), tq, device=device)
            vc = self.cls_prior(z, t, brain)
            if cfg_scale != 1.0:
                vu = self.cls_prior(z, t, null)
                return vu + cfg_scale * (vc - vu)
            return vc

        z0 = random_sphere((B, cfg.cls_dim), device)
        return self._integrate_sphere(z0, vel, n_steps, solver)

    @torch.no_grad()
    def _sample_prior_sphere(self, brain, cls_hat, n_steps, cfg_scale, solver,
                             z0=None):
        cfg = self.cfg
        B = brain.shape[0]; device = brain.device
        null_b = self.prior.null_brain.to(brain.dtype).expand(B, -1, -1)
        null_c = (self.prior.null_cls.expand(B, -1) if cfg.two_head else None)

        def vel(z, tq):
            t = torch.full((B,), tq, device=device)
            vc = self._prior_velocity(z, t, brain, cls_hat)
            if cfg_scale != 1.0:
                vu = self._prior_velocity(z, t, null_b, null_c)
                return vu + cfg_scale * (vc - vu)
            return vc

        if z0 is None:
            z0 = random_sphere((B, cfg.token_len, cfg.token_dim), device)
        return self._integrate_sphere(z0, vel, n_steps, solver)

    @torch.no_grad()
    def _sample_prior_euclidean(self, brain, cls_hat, n_steps, cfg_scale, solver):
        cfg = self.cfg
        B = brain.shape[0]; device = brain.device
        x = torch.randn(B, cfg.token_len, cfg.token_dim, device=device)
        null_b = self.prior.null_brain.to(brain.dtype).expand(B, -1, -1)
        null_c = (self.prior.null_cls.expand(B, -1) if cfg.two_head else None)
        grid = torch.linspace(0.0, 1.0, n_steps + 1, device=device)

        def vel(xt, tq):
            t = torch.full((B,), tq, device=device)
            vc = self.prior(xt, t, brain, cls_hat)
            if cfg_scale != 1.0:
                vu = self.prior(xt, t, null_b, null_c)
                return vu + cfg_scale * (vc - vu)
            return vc

        for i in range(n_steps):
            t0 = float(grid[i]); t1 = float(grid[i + 1]); dt = t1 - t0
            v1 = vel(x, self._tq(t0))
            if solver == "heun":
                v2 = vel(x + v1 * dt, self._tq(t1))
                x = x + 0.5 * (v1 + v2) * dt
            else:
                x = x + v1 * dt
        return x


    def _infer_z0(self, mu, logs):
        if self.cfg.flow_source == "noise":
            return random_sphere(mu.shape, mu.device, mu.dtype)
        if self.cfg.stochastic_source:
            return self.sample_anchor(mu, logs)
        return mu

    @torch.no_grad()
    def diagnose(self, fmri, subject, target_raw, *, target_cls=None,
                 n_steps=20, solver="heun"):
        cfg = self.cfg
        brain = self.backbone(fmri, subject)
        z1, r1 = polar_encode(target_raw.float(), self.tgt_mean.float())
        mu, logs, logr = self.anchor(brain)

        cls_hat = None
        cls_cos = None
        if cfg.two_head and self.cls_prior is not None:
            cls_hat = self._sample_cls(brain, n_steps, cfg.cls_cfg_scale, solver)
            if target_cls is not None:
                tc = F.normalize(target_cls.to(cls_hat.device).float(), dim=-1)
                cls_cos = F.cosine_similarity(cls_hat, tc, dim=-1)

        z0 = self._infer_z0(mu, logs)
        z_flow = self._sample_prior_sphere(brain, cls_hat, n_steps, cfg.cfg_scale,
                                           solver, z0=z0)
        return {
            "anchor_cos": F.cosine_similarity(z0, z1, dim=-1).mean(-1),
            "flow_cos": F.cosine_similarity(z_flow, z1, dim=-1).mean(-1),
            "cls_cos": cls_cos,
            "logr_pred": logr,
            "logr_true": r1.log(),
            "sigma": (logs.exp() if logs is not None else None),
            "z_flow": z_flow,
        }

    @torch.no_grad()
    def _draw_sample(self, brain, mu, logs, n_steps, cfg_scale, solver):
        cls_k = None
        if self.cfg.two_head and self.cls_prior is not None:
            cls_k = self._sample_cls(brain, n_steps, self.cfg.cls_cfg_scale, solver)
        z0 = (random_sphere(mu.shape, mu.device, mu.dtype)
              if self.cfg.flow_source == "noise" else self.sample_anchor(mu, logs))
        z = self._sample_prior_sphere(brain, cls_k, n_steps, cfg_scale, solver, z0=z0)
        return z, cls_k

    @torch.no_grad()
    def predict_tokens_frontier(self, fmri, subject, stats, *, ks, n_steps=None,
                                cfg_scale=None, solver=None):
        cfg = self.cfg
        if cfg.geometry != "sphere":
            raise ValueError("the multi-sample frontier is defined on the sphere only")
        ks = sorted({int(k) for k in ks if int(k) >= 1})
        if not ks:
            raise ValueError("ks must contain at least one positive integer")
        n_steps = n_steps or cfg.n_steps
        cfg_scale = cfg.cfg_scale if cfg_scale is None else cfg_scale
        solver = solver or cfg.solver

        brain = self.backbone(fmri, subject)
        mu, logs, logr = self.anchor(brain)
        radius = logr.exp()
        mean = self.tgt_mean.float()

        z_acc = torch.zeros_like(mu)
        cls_acc = None
        out: dict[int, dict] = {}
        for k in range(1, ks[-1] + 1):
            z_k, cls_k = self._draw_sample(brain, mu, logs, n_steps, cfg_scale, solver)
            z_acc = z_acc + z_k
            if cls_k is not None:
                cls_acc = cls_k if cls_acc is None else cls_acc + cls_k
            if k in ks:
                bar = z_acc / k
                resultant = bar.norm(dim=-1)
                out[k] = {
                    "tokens": polar_decode(F.normalize(bar, dim=-1), radius, mean),
                    "cls": None if cls_acc is None else F.normalize(cls_acc / k, dim=-1),
                    "resultant": resultant.mean(dim=-1),
                }
        out["anchor"] = {
            "tokens": polar_decode(mu, radius, mean),
            "cls": None if cls_acc is None else F.normalize(cls_acc / ks[-1], dim=-1),
            "resultant": torch.ones(mu.shape[0], device=mu.device),
        }
        return out

    @torch.no_grad()
    def predict_tokens(self, fmri, subject, stats, *, n_steps=None, cfg_scale=None,
                       solver=None, cond_source=None, n_samples=1):
        cfg = self.cfg
        n_steps = n_steps or cfg.n_steps
        cfg_scale = cfg.cfg_scale if cfg_scale is None else cfg_scale
        solver = solver or cfg.solver
        cond_source = cond_source or cfg.cond_source
        n_samples = max(1, int(n_samples))
        brain = self.backbone(fmri, subject)

        cls_hat = None
        if cfg.two_head:
            cls_hat = self._sample_cls(brain, n_steps, cfg.cls_cfg_scale, solver)

        if cfg.geometry == "sphere":
            mu, logs, logr = self.anchor(brain)
            if cond_source == "regression":
                zdir = mu
            elif cond_source == "prior":
                if n_samples > 1:
                    z_acc = torch.zeros_like(mu)
                    cls_acc = None
                    for _ in range(n_samples):
                        z_k, cls_k = self._draw_sample(brain, mu, logs, n_steps,
                                                       cfg_scale, solver)
                        z_acc = z_acc + z_k
                        if cls_k is not None:
                            cls_acc = cls_k if cls_acc is None else cls_acc + cls_k
                    zdir = F.normalize(z_acc / n_samples, dim=-1)
                    if cls_acc is not None:
                        cls_hat = F.normalize(cls_acc / n_samples, dim=-1)
                else:
                    zdir = self._sample_prior_sphere(brain, cls_hat, n_steps, cfg_scale,
                                                     solver, z0=self._infer_z0(mu, logs))
            elif cond_source == "blend":
                sp = self._sample_prior_sphere(brain, cls_hat, n_steps, cfg_scale, solver)
                zdir = F.normalize(cfg.blend_w * sp + (1 - cfg.blend_w) * mu, dim=-1)
            else:
                raise ValueError(f"Unknown cond_source: {cond_source!r}")
            return polar_decode(zdir, logr.exp(), self.tgt_mean.float()), cls_hat

        if cond_source == "regression":
            z = self.reg_head(brain)
        elif cond_source == "prior":
            z = self._sample_prior_euclidean(brain, cls_hat, n_steps, cfg_scale, solver)
        elif cond_source == "blend":
            z = cfg.blend_w * self._sample_prior_euclidean(brain, cls_hat, n_steps, cfg_scale, solver) \
                + (1 - cfg.blend_w) * self.reg_head(brain)
        else:
            raise ValueError(f"Unknown cond_source: {cond_source!r}")
        return stats.unstandardize(z), cls_hat
