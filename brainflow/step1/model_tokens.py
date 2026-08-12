"""BrainFlow step-1 token model: the two-head, oblique-manifold pipeline.

This is the concrete realisation of ``docs/paper`` (sections 7-9). From an fMRI
vector we encode 64 brain tokens ``c`` and model the structured bigG target with
*two* Riemannian flow-matching heads:

  * a **global CLS prior** -- RCFM on the single sphere S^{dcls-1} (dcls=1280),
    a Perceiver-style cross-attention head that transports a uniform base to the
    conditional law p(c_cls | c) (paper Stage 2b);
  * a **patch prior** -- a DiT that transports a uniform base on the oblique
    manifold (S^{d-1})^N (d=1664, N=256) to p(X | c, c_cls), conditioned on the
    sampled global direction so global semantics steer local detail (Stage 3).

The token magnitudes are peeled into a light radius side-model and recombined by
the polar map X_i = mu + r_i z_i (Stage 4). The predicted (c_cls, X) feed the
frozen unCLIP decoder (Stage 5). A SoftCLIP contrastive branch (Stage 6) aligns
the pooled brain vector with the true CLS direction.

The default configuration is the paper's spherical two-head design. The
``geometry="euclidean"`` and ``two_head=False`` switches recover the flat
rectified-flow / patch-only baselines used in the paper's ablations A and D.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import ResMLP, SinusoidalTime
from .sphere import (random_sphere, project_tangent, exp_map,
                     slerp, slerp_velocity, polar_encode, polar_decode)


def soft_clip_loss(preds: torch.Tensor, targs: torch.Tensor, temp: float) -> torch.Tensor:
    """Symmetrised SoftCLIP / InfoNCE on the sphere (paper eq. softclip-loss)."""
    clip_clip = (targs @ targs.T) / temp
    brain_clip = (preds @ targs.T) / temp
    soft = clip_clip.softmax(dim=-1)
    loss1 = -(brain_clip.log_softmax(dim=-1) * soft).sum(dim=-1).mean()
    loss2 = -(brain_clip.T.log_softmax(dim=-1) * soft).sum(dim=-1).mean()
    return (loss1 + loss2) / 2


def mixco(x: torch.Tensor, beta: float = 0.15, s_thresh: float = 0.5):
    """Convex-mix a random subset of the batch with a shuffled copy of itself.

    MindEye2's BiMixCo augmentation. NSD gives ~27k train trials per subject for a
    777M-parameter model, and run 347831 showed val_cos falling from the *first*
    eval -- the model is data-starved, not capacity-starved. Mixing voxel
    patterns manufactures interpolated training points; the contrastive target
    becomes the matching two-hot distribution (see :func:`mixco_nce`).

    Returns ``(mixed, perm, betas, select)``; rows outside ``select`` are
    untouched and carry ``beta = 1``.
    """
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
    """InfoNCE whose target is two-hot: beta on self, 1-beta on the mix partner."""
    sim = (preds @ targs.T) / temp
    B = preds.shape[0]
    idx = torch.arange(B, device=preds.device)
    probs = torch.zeros_like(sim)
    probs[idx, idx] = betas.to(sim.dtype)
    # += (not =) so a row that happened to draw itself as its partner still sums to 1
    sel = idx[select]
    probs[sel, perm[select]] = probs[sel, perm[select]] + (1 - betas[select]).to(sim.dtype)
    loss1 = -(sim.log_softmax(dim=-1) * probs).sum(dim=-1).mean()
    loss2 = -(sim.T.log_softmax(dim=-1) * probs.T).sum(dim=-1).mean()
    return (loss1 + loss2) / 2


@dataclass
class TokenStep1Config:
    # --- target / conditioning dims -------------------------------------
    token_len: int = 256          # N patch tokens
    token_dim: int = 1664         # d  (penultimate bigG token width)
    cls_dim: int = 1280           # dcls (pooled bigG image-embedding width)
    brain_dim: int = 1024         # brain-token width
    n_brain_tokens: int = 64      # number of brain tokens

    # --- encoder --------------------------------------------------------
    enc_hidden: int = 2048
    enc_blocks: int = 4
    enc_drop: float = 0.15

    # --- regression anchor / radius heads -------------------------------
    reg_depth: int = 2
    reg_heads: int = 8
    radius_depth: int = 2          # per-token cross-attention radius head

    # --- patch DiT prior ------------------------------------------------
    prior_width: int = 1024
    prior_depth: int = 8
    prior_heads: int = 8
    time_dim: int = 256

    # --- global CLS (RCFM) prior ----------------------------------------
    two_head: bool = True
    cls_width: int = 512
    cls_depth: int = 4
    cls_heads: int = 8
    cls_cfg_scale: float = 3.0

    # --- flow-matching schedule / CFG -----------------------------------
    cfg_drop_prob: float = 0.1
    logit_normal_m: float = 0.0
    logit_normal_s: float = 1.0
    uniform_t_prob: float = 0.1   # mass at the endpoints for sampler robustness
    t_min: float = 0.02
    t_max: float = 0.98

    # --- loss weights ---------------------------------------------------
    lambda_flow: float = 1.0      # patch flow
    lambda_rcfm: float = 1.0      # global CLS flow
    lambda_reg: float = 1.0       # regression anchor
    lambda_cos: float = 0.5       # per-token angular fidelity
    lambda_radius: float = 1.0    # radius side-model
    lambda_clip: float = 0.033    # SoftCLIP contrastive (blueprint-tuned)
    clip_temp: float = 0.006

    # --- BiMixCo augmentation (anti-overfit; 0 disables) -----------------
    mixup_pct: float = 0.0        # fraction of epochs trained with mixed voxels
    mixco_beta: float = 0.15      # Beta(b, b) mixing coefficient
    mixco_s_thresh: float = 0.5   # fraction of the batch that gets mixed

    # --- low-level pathway (spatial layout -> decoder img2img init) ------
    low_level: bool = True
    ll_target: str = "rgb"        # rgb | latent -- what the low head regresses.
                                  # "latent" predicts E(blur(GT)) in R^{4x96x96},
                                  # the decoder's actual img2img init (no VAE
                                  # round-trip, no sigmoid on an unbounded target).
    ll_size: int = 64             # blurry-image resolution (lower = lower-dim,
                                  # more learnable target; MSE@128 regressed to mean)
    ll_base: int = 256            # channels at the 8x8 stem
    ll_hidden: int = 1024         # low head's OWN per-subject voxel-encoder width
    ll_loss: str = "l1"           # l1 | huber | mse -- L1 is far less mean-seeking
    lambda_low: float = 1.0
    ll_strength: float = 0.7      # img2img strength at decode (calibrate via smoke)

    # --- sampling -------------------------------------------------------
    n_steps: int = 50
    cfg_scale: float = 3.0
    solver: str = "heun"          # geodesic Heun (sphere) / Heun (euclidean)
    cond_source: str = "prior"    # prior | regression | blend
    blend_w: float = 0.5

    # --- geometry / centering -------------------------------------------
    geometry: str = "sphere"      # sphere (paper) | euclidean (ablation A)
    center_tokens: bool = True
    subjects: list[int] = field(default_factory=lambda: [1])


# --------------------------------------------------------------------------
#  Attention primitives
# --------------------------------------------------------------------------
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
    """Pre-norm cross-attention with a residual add (query attends to ``ctx``)."""

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


# --------------------------------------------------------------------------
#  Brain encoder  E_phi : R^{V_s} -> R^{64 x 1024}
# --------------------------------------------------------------------------
class TokenBackbone(nn.Module):
    def __init__(self, cfg: TokenStep1Config, voxels_per_subject: dict[int, int]):
        super().__init__()
        self.cfg = cfg
        h = cfg.enc_hidden
        self.input_proj = nn.ModuleDict({
            str(s): nn.Linear(v, h) for s, v in voxels_per_subject.items()})
        self.stem = nn.Sequential(nn.LayerNorm(h), nn.GELU(), nn.Dropout(cfg.enc_drop))
        self.blocks = nn.ModuleList([ResMLP(h, 4, cfg.enc_drop) for _ in range(cfg.enc_blocks)])
        self.to_brain = nn.Sequential(
            nn.Linear(h, cfg.n_brain_tokens * cfg.brain_dim),
            nn.Unflatten(-1, (cfg.n_brain_tokens, cfg.brain_dim)),
            nn.LayerNorm(cfg.brain_dim),
        )

    def forward(self, fmri, subject: int):
        x = self.input_proj[str(int(subject))](fmri)
        x = self.stem(x)
        for b in self.blocks:
            x = b(x)
        return self.to_brain(x)


class TokenRegHead(nn.Module):
    """Low-variance point-estimate anchor: cross-attention queries -> tokens."""

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
    """Per-token log-radius head: 256 queries cross-attend to the brain tokens.

    The diagnostic on real bigG tokens shows the per-token radius carries an ~8%
    *content-dependent* spread (only ~7% positional), and the unCLIP decoder
    consumes raw tokens mu + r_i z_i -- so radius accuracy feeds PixCorr directly.
    A per-token cross-attention head (vs. a pooled MLP) lets each radius depend on
    the brain signal relevant to that token.
    """

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
        return self.out(x).squeeze(-1)          # (B, token_len) log-radius


class LowLevelHead(nn.Module):
    """Brain tokens -> blurry RGB image (the spatial layout for img2img init).

    The Baggag tokens say *what* is in the image (semantics); this head predicts
    the coarse *where / what-colour* as a low-res blurry image. The decoder then
    uses that blurry image as an img2img starting point instead of pure noise,
    which is the mechanism behind high PixCorr/SSIM (MindEye2-style). It is fully
    additive: it does not touch the token model.
    """

    def __init__(self, cfg: "TokenStep1Config", voxels_per_subject: dict[int, int]):
        super().__init__()
        self.base = cfg.ll_base
        self.grid = 8
        # Read RAW voxels through the head's OWN per-subject encoder -- NOT the
        # frozen bigG-semantic tokens. The gate experiment (blur_mse ~0.070 for
        # both a mean-pool and a spatial head) proved the semantic backbone
        # discarded the low-level signal; the retinotopic layout lives in the
        # voxels (V1), so the low pathway must tap them directly (MindEye2-style).
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
        return torch.sigmoid(self.out(self.ups(h)))     # (B, 3, ll_size, ll_size) in [0,1]


class LatentLowLevelHead(nn.Module):
    """Brain voxels -> the decoder's img2img LATENT (4 x 96 x 96), not RGB.

    Same voxel-encoder front end as :class:`LowLevelHead`, but it regresses
    ``E(blur(GT))`` -- the thing the sampler actually starts from -- instead of a
    blurry image that then has to survive a sigmoid and a VAE re-encode.

    Two consequences that matter for the failure mode we measured:

    * **no sigmoid.** Latents are roughly unit-Gaussian per channel (that is what
      the model's ``scale_factor`` is for), so the output is linear. A sigmoid on
      an unbounded target is exactly the kind of squashing that makes a head
      collapse to the mean.
    * **the loss lives in latent space**, where the VAE has already thrown away
      the high-frequency detail fMRI cannot predict. In pixel space that
      unpredictable detail dominates the L1 and the cheapest way to reduce it is
      to predict the mean -- which is what runs 344971/344991 did.

    The target is standardised per channel (train-split statistics), so the head
    predicts a unit-scale field and :meth:`unstandardize` puts it back.
    """

    def __init__(self, cfg: "TokenStep1Config", voxels_per_subject: dict[int, int]):
        super().__init__()
        self.base = cfg.ll_base
        self.grid = 12                      # 12 -> 24 -> 48 -> 96
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
        for _ in range(3):                  # 12 -> 96
            nc = max(c // 2, 32)
            blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(c, nc, 3, padding=1), nn.GroupNorm(8, nc), nn.SiLU(),
                nn.Conv2d(nc, nc, 3, padding=1), nn.SiLU()))
            c = nc
        self.ups = nn.Sequential(*blocks)
        self.out = nn.Conv2d(c, 4, 3, padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        # Standardisation of the latent target; overwritten from the cache stats.
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
        """Standardised latent prediction (B, 4, 96, 96)."""
        B = fmri.shape[0]
        x = self.stem(self.input_proj[str(int(subject))](fmri))
        h = self.to_grid(x).view(B, self.base, self.grid, self.grid)
        return self.out(self.ups(self.mix(h)))


# --------------------------------------------------------------------------
#  Patch prior (DiT) -- velocity field on the oblique manifold
# --------------------------------------------------------------------------
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
    """Velocity field v_theta(t, Z | c, c_cls) over the 256 token directions."""

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

        # --- structured conditioning on the global direction c_cls --------
        if cfg.two_head:
            self.cls_kv = nn.Linear(cfg.cls_dim, cfg.brain_dim)    # extra K/V token
            self.cls_ada = nn.Linear(cfg.cls_dim, w)              # AdaLN modulation
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


# --------------------------------------------------------------------------
#  Global CLS prior (Perceiver-style RCFM on S^{dcls-1})
# --------------------------------------------------------------------------
class ClsBlock(nn.Module):
    """AdaLN cross-attention + MLP for a single state query (no self-attention)."""

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
    """v_cls(t, c_t | c) on S^{dcls-1}; a single query attends to brain tokens."""

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


# --------------------------------------------------------------------------
#  Assembled model
# --------------------------------------------------------------------------
class TokenStep1Model(nn.Module):
    def __init__(self, cfg: TokenStep1Config, voxels_per_subject: dict[int, int]):
        super().__init__()
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
        self.low_head = None
        if cfg.low_level:
            head_cls = (LatentLowLevelHead if getattr(cfg, "ll_target", "rgb") == "latent"
                        else LowLevelHead)
            self.low_head = head_cls(cfg, voxels_per_subject)
        self.register_buffer("tgt_mean", torch.zeros(cfg.token_dim))

    def set_target_mean(self, mean):
        self.tgt_mean.copy_(mean.to(self.tgt_mean.device, self.tgt_mean.dtype))

    # ---- time sampling (logit-normal + endpoint mass) -------------------
    def _sample_t(self, B, device):
        z = torch.randn(B, device=device) * self.cfg.logit_normal_s + self.cfg.logit_normal_m
        t = torch.sigmoid(z)
        if self.cfg.uniform_t_prob > 0:
            mix = torch.rand(B, device=device) < self.cfg.uniform_t_prob
            t = torch.where(mix, torch.rand(B, device=device), t)
        return t

    # ====================================================================
    #  Training
    # ====================================================================
    def training_step(self, fmri, subject, target_std, target_raw=None, target_cls=None,
                      target_img=None, low_only=False, use_mixco=False, target_lat=None):
        """One optimisation step.

        ``use_mixco`` mixes the voxel batch and swaps the contrastive target for
        the two-hot MixCo one. The flow / regression / radius targets stay
        *unmixed* (as in MindEye2): a slerp between two different images' token
        sequences is not a valid point on the target manifold, so mixing them
        would corrupt the flow objective rather than regularise it. The mixed
        brain vector paired with an unmixed target acts as input noise for those
        heads, which is the intended effect.
        """
        cfg = self.cfg
        B = fmri.shape[0]; device = fmri.device
        if low_only:
            # Frozen-token warm-start: train ONLY the low-level head. It reads raw
            # voxels through its own encoder, so skip the frozen backbone AND the
            # whole DiT prior / CLS / contrastive forward+backward entirely.
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
            cls_dir = F.normalize(target_cls.float(), dim=-1)        # (B, cls_dim)

        out = self._cls_flow_loss(brain, cls_dir, B, device)         # RCFM term
        if cfg.geometry == "sphere":
            out |= self._patch_step_sphere(brain, target_raw, cls_dir, B, device)
        else:
            out |= self._patch_step_euclidean(brain, target_std, cls_dir, B, device)
        out |= self._contrastive_loss(brain, target_std, cls_dir, device, mix=mix)
        # Unmixed voxels: the low head reads raw fMRI directly and regresses the
        # image's spatial layout, so a blended input against an unblended target
        # is straight label noise for it (it has no contrastive term to absorb it).
        out |= self._low_level_loss(fmri_clean, subject, target_img, target_lat)

        total = (cfg.lambda_flow * out["flow"] + cfg.lambda_reg * out["reg"]
                 + cfg.lambda_cos * out["cos"] + cfg.lambda_clip * out["clip"]
                 + cfg.lambda_rcfm * out["rcfm"])
        if "radius" in out:
            total = total + cfg.lambda_radius * out["radius"]
        if "low" in out:
            total = total + cfg.lambda_low * out["low"]
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
            # Both sides standardised: the head predicts a unit-scale field.
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
        else:                                    # "l1" (default): least mean-seeking
            low = F.l1_loss(pred, target)
        return {"low": low}

    @torch.no_grad()
    def predict_lowlevel(self, fmri, subject):
        """Blurry RGB layout for the decoder's img2img init.

        None when the low pathway is off *or* when it predicts a latent -- use
        :meth:`predict_low_latent` for that. The two are mutually exclusive, so a
        caller can pass both to ``decoder.decode`` and let the None fall through.
        """
        if self.low_head is None or self.low_is_latent:
            return None
        return self.low_head(fmri, subject)

    @torch.no_grad()
    def predict_low_latent(self, fmri, subject):
        """Predicted img2img latent (B,4,96,96) in the decoder's space, or None."""
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
        """Teacher-forced patch conditioning with per-example CFG dropout."""
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

        reg_dir = F.normalize(self.reg_head(brain).float(), dim=-1)
        loss_reg = F.mse_loss(reg_dir, z1)
        loss_cos = 1.0 - F.cosine_similarity(reg_dir, z1, dim=-1).mean()

        logr = self.radius_head(brain).float()
        loss_radius = F.mse_loss(logr, r1.log())

        t = self._sample_t(B, device)
        tb = t[:, None, None]
        z0 = random_sphere(z1.shape, device, z1.dtype)
        zt = slerp(z0, z1, tb)
        ut = slerp_velocity(z0, z1, tb)
        brain_in, cls_in = self._patch_cond(brain, cls_dir, B, device)
        v = project_tangent(self.prior(zt, t, brain_in, cls_in).float(), zt)
        loss_flow = F.mse_loss(v, ut)
        return {"flow": loss_flow, "reg": loss_reg, "cos": loss_cos, "radius": loss_radius}

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
            return {"clip": mixco_nce(b, z, cfg.clip_temp, perm, betas, select)}
        return {"clip": soft_clip_loss(b, z, cfg.clip_temp)}

    # ====================================================================
    #  Sampling
    # ====================================================================
    def _tq(self, t):
        return min(max(t, self.cfg.t_min), self.cfg.t_max)

    @torch.no_grad()
    def _integrate_sphere(self, z, vel_fn, n_steps, solver):
        """Endpoint-safe geodesic integrator (Exp-map Euler / Heun) on a sphere."""
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
    def _sample_prior_sphere(self, brain, cls_hat, n_steps, cfg_scale, solver):
        cfg = self.cfg
        B = brain.shape[0]; device = brain.device
        null_b = self.prior.null_brain.to(brain.dtype).expand(B, -1, -1)
        null_c = (self.prior.null_cls.expand(B, -1) if cfg.two_head else None)

        def vel(z, tq):
            t = torch.full((B,), tq, device=device)
            vc = self.prior(z, t, brain, cls_hat)
            if cfg_scale != 1.0:
                vu = self.prior(z, t, null_b, null_c)
                return vu + cfg_scale * (vc - vu)
            return vc

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

    @torch.no_grad()
    def predict_tokens(self, fmri, subject, stats, *, n_steps=None, cfg_scale=None,
                       solver=None, cond_source=None):
        """Return (raw_tokens[B,N,d], cls_hat[B,dcls] or None)."""
        cfg = self.cfg
        n_steps = n_steps or cfg.n_steps
        cfg_scale = cfg.cfg_scale if cfg_scale is None else cfg_scale
        solver = solver or cfg.solver
        cond_source = cond_source or cfg.cond_source
        brain = self.backbone(fmri, subject)

        cls_hat = None
        if cfg.two_head:
            cls_hat = self._sample_cls(brain, n_steps, cfg.cls_cfg_scale, solver)

        if cfg.geometry == "sphere":
            if cond_source == "regression":
                zdir = F.normalize(self.reg_head(brain).float(), dim=-1)
            elif cond_source == "prior":
                zdir = self._sample_prior_sphere(brain, cls_hat, n_steps, cfg_scale, solver)
            elif cond_source == "blend":
                s = self._sample_prior_sphere(brain, cls_hat, n_steps, cfg_scale, solver)
                r = F.normalize(self.reg_head(brain).float(), dim=-1)
                zdir = F.normalize(cfg.blend_w * s + (1 - cfg.blend_w) * r, dim=-1)
            else:
                raise ValueError(f"Unknown cond_source: {cond_source!r}")
            radius = self.radius_head(brain).float().exp()
            return polar_decode(zdir, radius, self.tgt_mean.float()), cls_hat

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
