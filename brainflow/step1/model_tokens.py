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
from .sphere import (random_sphere, project_tangent, exp_map, log_map,
                     slerp, slerp_velocity, polar_encode, polar_decode,
                     tangent_noise_rad)


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

    # --- R-XFM: what the flow transports FROM ---------------------------
    # "noise"      -- legacy: z0 ~ Uniform(S^{d-1}). On S^1663 a uniform sample has
    #                 cos ~ 0 +/- 1/sqrt(d) with the target, i.e. ~90 deg away and
    #                 equidistant from every target: z0 carries no information and
    #                 the whole burden falls on cross-attention.
    # "anchor_det" -- z0 = the regression anchor mu (a point).
    # "anchor_var" -- z0 ~ q(.|fMRI), a wrapped Gaussian on the sphere centred on
    #                 mu. This is the flow-matching source NeuroFlow's XFM uses,
    #                 specialised to the oblique manifold. The KL keeps sigma from
    #                 collapsing to 0, which would degenerate the flow into a
    #                 residual regressor.
    # Under any anchor source, v == 0 makes the ODE the identity, so the flow
    # recovers the regression baseline exactly -- it can only refine it.
    flow_source: str = "anchor_var"
    # --- R-XFM: WHAT the velocity field parameterises ---------------------
    # For geodesic paths the velocity and the endpoint are exactly equivalent,
    #     u_t = slerp_velocity(z0, z1, t) = Log_{z_t}(z1) / (1 - t),
    # verified to 4e-8 .. 2e-6 max abs error over t in [0.1, 0.9] at d=1664. The
    # choice is therefore purely about the LOSS GEOMETRY, and the two differ a lot:
    #
    #   "velocity" -- mse(v, u_t). Substituting the identity above,
    #       ||v - u_t||^2 = ||Log_{z_t}(zhat1) - Log_{z_t}(z1)||^2 / (1 - t)^2,
    #     i.e. endpoint error implicitly weighted by 1/(1-t)^2: 1.2x at t=0.1,
    #     100x at t=0.9, 2500x at t=0.98. Capacity is spent where the problem is
    #     trivial (near the target) and starved at t->0, which is exactly where
    #     inference STARTS. Additionally ||u_t|| = theta exactly, so as the anchor
    #     improves the whole term shrinks quadratically: measured 738x smaller than
    #     `cos` at theta=68.3deg and 827x at theta=15deg.
    #
    #   "endpoint" -- the net predicts zhat1 = normalize(z_t + raw) and the loss is
    #     1 - cos(zhat1, z1), uniform in t. Same ODE, same solver; removes the
    #     1/(1-t)^2 misweighting, puts the flow term on the same numeric scale as
    #     `cos`, and -- because `out` is zero-initialised -- gives zhat1 = z_t at
    #     init, hence v == 0, hence the identity flow. The prior therefore starts
    #     EXACTLY at the regression anchor by construction rather than by hope.
    # "auto" resolves to "endpoint" on the sphere and "velocity" under
    # geometry="euclidean", where the geodesic identity does not apply. Setting
    # "endpoint" explicitly alongside a non-spherical geometry is an error rather
    # than a silent downgrade.
    flow_param: str = "auto"          # auto | endpoint | velocity
    # Inference-time ablation on the flow's starting point. "shuffle" hands each
    # sample its NEIGHBOUR's anchor: if flow_cos is unchanged the transport is
    # not using z0 at all, which is the difference between R-XFM being wrong and
    # R-XFM not being implemented. Eval-only; training never sets it.
    z0_mode: str = "default"          # default | noise | shuffle

    def resolved_flow_param(self) -> str:
        if self.flow_param != "auto":
            return self.flow_param
        return "endpoint" if self.geometry == "sphere" else "velocity"
    # Posterior width, given as the EXPECTED GEODESIC DISPLACEMENT in radians of a
    # draw from its base point mu. Dimension-independent, unlike a per-coordinate
    # sigma: the two are related by sigma = jitter / sqrt(d - 1).
    #
    # This must stay a SMALL FRACTION of the anchor error, not match it. A draw is
    # displaced in a *random* tangent direction while the target sits in one
    # specific direction, so in high dimensions
    #     cos(z0, z1) ~ cos(jitter) * cos(theta).
    # At the measured anchor error (cos 0.37, theta = 68.3 deg), "calibrating"
    # jitter to theta would drop cos(z0, z1) to 0.126 -- strictly worse than the
    # anchor. The default 0.15 rad (8.6 deg) costs 0.370 -> 0.366, i.e. nothing,
    # while still giving the velocity field a neighbourhood to be defined on.
    anchor_jitter_rad: float = 0.15
    # How the patch prior sees the global direction during TRAINING. It is
    # teacher-forced on ground-truth c_cls today but consumes a *sampled* estimate
    # at inference, so it learns to lean on a signal that is far better at train
    # time than at test time.
    #   "teacher" -- legacy (ground truth)
    #   "jitter"  -- geodesic rotation of the truth by cls_jitter_kappa (cheap)
    #   "sampled" -- scheduled sampling: run the CLS prior, no grad (accurate, slow)
    #   "none"    -- do not condition the patch prior on c_cls
    cls_cond_mode: str = "jitter"
    # Geodesic displacement in RADIANS applied to the true c_cls at train time.
    # Calibrate from the logged val/cls_cos_hat_true: rad = arccos(cls_cos_hat_true).
    # (The previous `cls_jitter_kappa=0.02` was a per-coordinate sigma, which on
    # S^1279 realises 0.02*sqrt(1279) = 0.715 rad = 41 deg -- 36x its nominal value
    # and comparable to the anchor's own 68 deg error, i.e. the conditioning signal
    # was being destroyed rather than de-teacher-forced.)
    cls_jitter_rad: float = 0.30
    cls_ss_prob: float = 0.5           # P(use the sampled estimate) in "sampled" mode
    cls_ss_steps: int = 8              # cheap solver budget for scheduled sampling

    # --- loss weights ---------------------------------------------------
    lambda_flow: float = 1.0      # patch flow
    lambda_rcfm: float = 1.0      # global CLS flow
    # NOTE: for unit vectors mse(a,b) = (2/d) * (1 - cos(a,b)), so `reg` and `cos`
    # are the SAME objective and lambda_reg is worth lambda_cos/832 at d=1664.
    # Default 0: `reg` is still computed and logged, but excluded from the total.
    lambda_reg: float = 0.0       # (deprecated -- see lambda_cos)
    lambda_cos: float = 0.5       # per-token angular fidelity
    lambda_radius: float = 1.0    # radius side-model
    lambda_clip: float = 0.033    # SoftCLIP contrastive (blueprint-tuned)
    # Token-level SoftCLIP on the FLATTENED 256x1664 grid -- the only discriminative
    # signal that touches what the decoder actually consumes. `lambda_clip` above
    # acts on clip_proj(mean(brain)), a 2.4M head that is NOT on the decoder path,
    # so without this every one of the 425,984 numbers the decoder eats is
    # supervised by mean-seeking losses alone. NeuroFlow applies SoftCLIP directly
    # to its flow endpoint over the flattened grid (neurovae.py:57, preds.flatten(1))
    # and ablates it at 86.4% -> 0.3% raw retrieval.
    lambda_clip_tok: float = 0.0
    # Wrapped-Gaussian KL, in nats per tangent dimension. Its only job is to stop
    # sigma collapsing to 0 (which would degenerate the flow into a deterministic
    # residual regressor); it is a floor, not a strong prior.
    lambda_kl: float = 0.1
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
    # An L1/MSE regression has no term that punishes emitting the SAME field for
    # every input, so the head converges on the conditional mean of the latent --
    # which the VAE renders as a near-white frame (run ll_fix_4subj: PixCorr
    # 0.0198, every 2-way metric at chance). This is the failure the token branch
    # had before lambda_clip_tok; in-batch negatives are what fixed it there.
    # Six low-level attempts have failed and none of them could see the backbone:
    # the head runs its own Linear(V_s, ll_hidden) on RAW voxels while a 306M
    # trunk that generalises to 0.80 retrieval sits beside it. Shrinking the head
    # 69.4M -> 32.3M moved val blur_mse only 1.150 -> 1.113, still above the 1.0
    # that predicting the channel mean scores, so the bottleneck is what the head
    # can see, not how many parameters it has.
    ll_use_backbone: bool = False    # build the grid from brain tokens, not voxels
    lambda_low_clip: float = 0.0     # SoftCLIP on the flattened low-level field
    lambda_low_std: float = 0.0      # per-channel second-moment match to target
    ll_strength: float = 0.7      # img2img strength at decode (calibrate via smoke)

    # --- sampling -------------------------------------------------------
    n_steps: int = 50
    cfg_scale: float = 3.0
    solver: str = "heun"          # geodesic Heun (sphere) / Heun (euclidean)
    cond_source: str = "prior"    # prior | regression | blend
    blend_w: float = 0.5
    stochastic_source: bool = False   # draw z0 ~ q at inference (multi-sample arm)
                                      # instead of using the posterior mode mu

    # --- geometry / centering -------------------------------------------
    geometry: str = "sphere"      # sphere (paper) | euclidean (ablation A)
    center_tokens: bool = True
    # How the token mean subtracted before the polar split is pooled.
    #   "global"       -- ONE R^d vector averaged over images AND all N positions
    #                     (targets_bigg._accum_stats reshapes (-1, d) first). Every
    #                     centred direction z_i then retains the offset common to
    #                     position i across all images, which is content-free but
    #                     still counted by every cosine in the repo -- so part of
    #                     the reported anchor cos is a positional prior the flow
    #                     cannot improve and the ranking metrics discount.
    #   "per_position" -- one R^d vector PER position (N, d). z_i is then the
    #                     image-specific deviation at that position, which is the
    #                     quantity the contrastive/2-way metrics actually reward.
    # Run scripts/check_centering.py to measure the split before choosing.
    center_mode: str = "global"   # global | per_position
    subjects: list[int] = field(default_factory=lambda: [1])

    # --- cross-subject encoder -------------------------------------------
    # False -- one full nn.Linear(V_s, enc_hidden) per subject (the original
    #          ModuleDict). Nothing at the input stage is shared, so a second
    #          subject adds ~V*enc_hidden fresh parameters and contributes to the
    #          trunk only. This is what every multi-subject run before this flag
    #          actually trained.
    # True  -- ONE nn.Linear(max_vox, enc_hidden) seen by every subject, plus a
    #          rank-r residual LoRA + bias per subject (SubjectResidualAdapter,
    #          ported from models.py:BrainEncoder). Per-subject cost drops from
    #          V*h (~30M at V=15k, h=2048) to 2*h*r + h (~67k at r=16), and the
    #          shared map receives gradient from every subject.
    #
    # Caveat worth stating: NSD voxel index j is a DIFFERENT cortical location in
    # each subject, so the shared projection is not anatomically aligned -- it has
    # to learn a map that works under an arbitrary per-subject permutation, and
    # the rank-r adapter is what absorbs the mismatch. If shared loses to
    # per-subject on the same data, insufficient adapter capacity is the first
    # thing to test (raise subject_rank), functional alignment the second.
    shared_encoder: bool = True
    subject_rank: int = 16        # LoRA rank of the per-subject residual


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
class SubjectResidualAdapter(nn.Module):
    """Per-subject rank-r residual on the shared hidden vector (LoRA + bias).

    Ported from :class:`brainflow.models.SubjectResidualAdapter`. Zero-initialised
    on all three tensors, so at step 0 the adapter is exactly the identity and the
    model is the pure shared encoder -- adding a subject cannot perturb the others
    until gradients say it should.
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
            # One projection for everyone; subject identity enters as a low-rank
            # residual in hidden space rather than as its own voxel matrix.
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
        """Adapter row index per sample. Accepts a scalar (the loader's current
        per-subject batches) or a per-sample tensor (mixed batches)."""
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
            # Right-pad to max_vox so every subject enters the same matrix. The
            # padded columns are zero, so they contribute nothing to the product
            # and receive no gradient for that subject.
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
        # Backbone route: map each brain token to `base` channels, then map the
        # token axis onto the 12x12 grid. Keeps per-token structure (which a mean
        # pool would destroy, and layout is exactly what this head predicts) for
        # ~74k parameters against the 4M of the raw-voxel projection it replaces.
        self.use_backbone = bool(getattr(cfg, "ll_use_backbone", False))
        if self.use_backbone:
            self.brain_tok = nn.Linear(cfg.brain_dim, self.base)
            self.brain_pos = nn.Linear(cfg.n_brain_tokens, self.grid * self.grid)
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

    def forward(self, fmri, subject, brain=None):
        """Standardised latent prediction (B, 4, 96, 96)."""
        B = fmri.shape[0]
        if self.use_backbone:
            if brain is None:
                raise ValueError("ll_use_backbone needs the backbone tokens; "
                                 "pass brain=backbone(fmri, subject)")
            t = self.brain_tok(brain.float())                  # (B, T, base)
            h = self.brain_pos(t.transpose(1, 2))              # (B, base, grid*grid)
            h = h.view(B, self.base, self.grid, self.grid)
        else:
            x = self.stem(self.input_proj[str(int(subject))](fmri))
            h = self.to_grid(x).view(B, self.base, self.grid, self.grid)
        return self.out(self.ups(self.mix(h)))


def sigma_from_jitter(jitter_rad: float, k: int) -> float:
    """Per-coordinate tangent sigma giving an expected geodesic step of ``jitter_rad``.

    An isotropic tangent Gaussian N(0, sigma^2 I_k) has expected norm ~ sigma*sqrt(k),
    and on the sphere that norm *is* the geodesic distance travelled by Exp_mu. So
    the interpretable knob is the displacement in radians, and sigma follows.
    """
    return jitter_rad / math.sqrt(max(1, k))


class LogSigmaHead(nn.Module):
    """Per-token log-sigma of the anchor posterior q(z0 | fMRI).

    Same cross-attention shape as :class:`RadiusHead` (256 queries over the brain
    tokens, one scalar out) so each token gets its own uncertainty: the anchor is
    not uniformly wrong across the sequence, and a single global sigma would force
    the flow to over-correct confident tokens and under-correct unreliable ones.

    Deliberately a *separate* module rather than an extra output channel on
    ``radius_head`` so that ``--init-from`` can warm-start every existing
    ``step1c`` checkpoint with ``strict=False``: the pretrained ``reg_head`` and
    ``radius_head`` load unchanged and only this head starts fresh.
    """

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
        # Start at the prior width: a fresh head must not inject a huge or a
        # degenerate sigma into the very first flow batches.
        nn.init.zeros_(self.out[-1].weight)
        sigma_p = sigma_from_jitter(cfg.anchor_jitter_rad, cfg.token_dim - 1)
        nn.init.constant_(self.out[-1].bias, math.log(sigma_p))

    def forward(self, brain):
        B = brain.shape[0]
        x = self.queries.expand(B, -1, -1)
        for cr, mlp in zip(self.cross, self.mlps):
            x = cr(x, brain)
            x = x + mlp(x)
        return self.out(x).squeeze(-1)              # (B, token_len) log-sigma


def wrapped_gaussian_kl(logs: torch.Tensor, sigma_p: float) -> torch.Tensor:
    """KL( N(0, sigma^2 I_k) || N(0, sigma_p^2 I_k) ) / k, in nats per tangent dim.

    The posterior is a wrapped Gaussian on S^{d-1}: isotropic in the tangent space
    at the base point mu, pushed onto the manifold by Exp_mu. Its KL against an
    isotropic tangent prior has the usual diagonal-Gaussian closed form,

        KL = (k/2) * [ sigma^2/sigma_p^2 - 1 + 2*log(sigma_p/sigma) ]

    We divide by the tangent dimension k = d-1 so the term is O(1) and
    ``lambda_kl`` is scale-free; multiply back by ``k * n_tokens`` to report nats.
    """
    var_ratio = (2 * logs).exp() / (sigma_p ** 2)
    return 0.5 * (var_ratio - 1.0 - 2 * logs + 2 * math.log(sigma_p))


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
        # Only instantiated for the variational source, so "noise"/"anchor_det"
        # checkpoints stay byte-compatible with step1c.
        self.logs_head = LogSigmaHead(cfg) if cfg.flow_source == "anchor_var" else None
        self.low_head = None
        if cfg.low_level:
            head_cls = (LatentLowLevelHead if getattr(cfg, "ll_target", "rgb") == "latent"
                        else LowLevelHead)
            self.low_head = head_cls(cfg, voxels_per_subject)
        # (d,) for center_mode="global", (N, d) for "per_position". Both broadcast
        # against (B, N, d) in polar_encode/polar_decode.
        self.register_buffer(
            "tgt_mean",
            torch.zeros(cfg.token_len, cfg.token_dim)
            if cfg.center_mode == "per_position" else torch.zeros(cfg.token_dim))

    def set_target_mean(self, mean):
        mean = mean.to(self.tgt_mean.device, self.tgt_mean.dtype)
        if mean.shape != self.tgt_mean.shape:
            if self.cfg.center_mode == "per_position" and mean.dim() == 1:
                # A global mean broadcast to every position: legal, just uninformative.
                mean = mean.unsqueeze(0).expand_as(self.tgt_mean).contiguous()
            else:
                raise ValueError(
                    f"tgt_mean shape {tuple(mean.shape)} does not match "
                    f"center_mode={self.cfg.center_mode!r} "
                    f"(expected {tuple(self.tgt_mean.shape)})")
        self.tgt_mean.copy_(mean)


    # ---- anchor posterior q(z0 | fMRI) ---------------------------------
    def anchor(self, brain):
        """(mu, logs, logr): base direction, per-token log-sigma, log-radius.

        ``mu`` is the regression anchor -- the same tensor ``cond_source
        ="regression"`` decodes today, so the flow's starting point *is* the
        current best-performing configuration.
        """
        mu = F.normalize(self.reg_head(brain).float(), dim=-1)
        logr = self.radius_head(brain).float()
        logs = None
        if self.logs_head is not None:
            k = self.cfg.token_dim - 1
            # Cap sigma so the expected geodesic step sigma*sqrt(k) stays under pi
            # (beyond that the wrapped Gaussian folds around the sphere and the
            # "small perturbation" reading of the posterior stops holding).
            # Cap the learned sigma at a geodesic step of pi/2: beyond that the
            # draw is orthogonal to mu in expectation and the anchor's information
            # is gone (cos(z0, z1) ~ cos(jitter)*cos(theta) -> 0).
            hi = math.log(sigma_from_jitter(math.pi / 2, k))
            logs = self.logs_head(brain).float().clamp(-12.0, hi)
        return mu, logs, logr

    def sample_anchor(self, mu, logs, detach_base: bool = False):
        """Draw z0 ~ q on the manifold; falls back to mu when deterministic.

        ``detach_base`` detaches the BASE POINT mu while leaving sigma's graph
        intact. That distinction is the whole variational apparatus:

        * detaching everything (the previous ``z0 = z0.detach()``) leaves the KL as
          the ONLY gradient into ``logs_head``. Its unique minimiser is
          sigma = sigma_p -- exactly where :class:`LogSigmaHead` is initialised --
          so the head's gradient is identically zero at step 0 and stays there.
          Measured: after 200 AdamW steps log-sigma moved from -3.968687 to
          [-3.969174, -3.968141], a per-token spread of 4.2e-06. The head was a
          constant, and ``anchor_var`` was ``anchor_det`` plus fixed jitter.
        * detaching only mu keeps the intended asymmetry: the flow loss may not
          shorten its own path by degrading the anchor (mu is supervised by
          ``loss_cos``), but it CAN choose how wide a neighbourhood of mu it is
          willing to be defined on. The flow pushes sigma down, the KL floors it,
          and the equilibrium is the usual variational trade-off.
        """
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

    # ---- velocity field: endpoint or velocity parameterisation ----------
    def _prior_velocity(self, zt, t, brain, cls_emb):
        """v(z_t, t) on the oblique manifold, tangent at ``zt``.

        Under ``flow_param="endpoint"`` the net emits a residual toward the
        predicted endpoint and the velocity follows from the exact geodesic
        identity  u_t = Log_{z_t}(z1) / (1 - t).  ``self.prior.out`` is
        zero-initialised, so at init zhat1 == z_t, v == 0, and the ODE is the
        identity map -- the sampler returns the anchor unchanged.
        """
        raw = self.prior(zt, t, brain, cls_emb).float()
        if self.flow_param != "endpoint":
            return raw
        z1_hat = F.normalize(zt.float() + raw, dim=-1)
        # 1 - t is bounded below by 1 - t_max so the division cannot blow up at the
        # final step; _tq applies the same clamp to the integrator's time grid.
        one_minus_t = (1.0 - t).clamp_min(1.0 - self.cfg.t_max)
        while one_minus_t.dim() < zt.dim():
            one_minus_t = one_minus_t.unsqueeze(-1)
        return log_map(zt.float(), z1_hat) / one_minus_t

    def _prior_endpoint(self, zt, t, brain, cls_emb):
        """The predicted endpoint zhat1 itself (endpoint parameterisation only)."""
        raw = self.prior(zt, t, brain, cls_emb).float()
        return F.normalize(zt.float() + raw, dim=-1)

    def _cls_cond(self, brain, cls_dir):
        """Global-direction conditioning for the patch prior at TRAIN time.

        Teacher-forcing on ground-truth ``c_cls`` is an exposure-bias trap: c_cls
        is a near-complete semantic summary of the image, so a prior handed the
        true one for free learns to route around the brain tokens, then meets a
        sampled estimate at inference that it has never seen.
        """
        mode = self.cfg.cls_cond_mode
        if cls_dir is None or mode == "none":
            return None
        if mode == "teacher":
            return cls_dir
        if mode == "jitter":
            # RADIANS, not a per-coordinate sigma -- see cls_jitter_rad.
            return tangent_noise_rad(cls_dir, self.cfg.cls_jitter_rad)
        if mode == "sampled":
            if not self.training or torch.rand(()) >= self.cfg.cls_ss_prob:
                return cls_dir
            with torch.no_grad():
                return self._sample_cls(brain, self.cfg.cls_ss_steps,
                                        self.cfg.cls_cfg_scale, "euler").detach()
        raise ValueError(f"Unknown cls_cond_mode: {mode!r}")

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
                      target_img=None, low_only=False, use_mixco=False, target_lat=None,
                      keep_term_graph=False):
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
            lb = None
            if getattr(cfg, "ll_use_backbone", False):
                with torch.no_grad():
                    lb = self.backbone(fmri, subject)
            out = self._low_level_loss(fmri, subject, target_img, target_lat, lb)
            if "low" not in out:
                raise ValueError(
                    "low_only training needs low_head enabled + the matching target "
                    "(target_img for ll_target=rgb, target_lat for ll_target=latent)")
            out = {k: v.detach() for k, v in out.items()} | {
                "loss": (cfg.lambda_low * out["low"]
                         + cfg.lambda_low_clip * out.get("low_clip", 0.0)
                         + cfg.lambda_low_std * out.get("low_std", 0.0))}
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
        # The RCFM head is trained on the TRUE direction; only the *patch prior's*
        # conditioning is de-teacher-forced (see _cls_cond).
        cls_cond = self._cls_cond(brain, cls_dir)
        if cfg.geometry == "sphere":
            out |= self._patch_step_sphere(brain, target_raw, cls_cond, B, device)
        else:
            out |= self._patch_step_euclidean(brain, target_std, cls_cond, B, device)
        out |= self._contrastive_loss(brain, target_std, cls_dir, device, mix=mix)
        # Unmixed voxels: the low head reads raw fMRI directly and regresses the
        # image's spatial layout, so a blended input against an unblended target
        # is straight label noise for it (it has no contrastive term to absorb it).
        low_brain = None
        if getattr(cfg, "ll_use_backbone", False):
            low_brain = brain if mix is None else self.backbone(fmri_clean, subject)
        out |= self._low_level_loss(fmri_clean, subject, target_img, target_lat,
                                    low_brain)

        total = (cfg.lambda_flow * out["flow"]
                 + cfg.lambda_cos * out["cos"] + cfg.lambda_clip * out["clip"]
                 + cfg.lambda_rcfm * out["rcfm"])
        if cfg.lambda_reg > 0 and "reg" in out:
            # Kept for backward compatibility only; `reg` == `cos` * 2/d.
            total = total + cfg.lambda_reg * out["reg"]
        if "radius" in out:
            total = total + cfg.lambda_radius * out["radius"]
        if "kl" in out:
            total = total + cfg.lambda_kl * out["kl"]
        if "clip_tok" in out:
            total = total + cfg.lambda_clip_tok * out["clip_tok"]
        if "low" in out:
            total = total + cfg.lambda_low * out["low"]
        if "low_clip" in out:
            total = total + cfg.lambda_low_clip * out["low_clip"]
        if "low_std" in out:
            total = total + cfg.lambda_low_std * out["low_std"]
        # Terms are detached for logging by default. `keep_term_graph` retains
        # them so instrument.grad_norms can measure each term's own gradient --
        # without it the probe silently reports nothing.
        if not keep_term_graph:
            out = {k: v.detach() for k, v in out.items()}
        out["loss"] = total
        return out

    @property
    def low_is_latent(self) -> bool:
        return isinstance(self.low_head, LatentLowLevelHead)

    def _low_level_loss(self, fmri, subject, target_img, target_lat=None, brain=None):
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
        pred = self.low_head(fmri, subject, brain)
        mode = getattr(self.cfg, "ll_loss", "l1")
        if mode == "mse":
            low = F.mse_loss(pred, target)
        elif mode == "huber":
            low = F.huber_loss(pred, target, delta=0.1)
        else:                                    # "l1" (default): least mean-seeking
            low = F.l1_loss(pred, target)
        out = {"low": low}
        cfg = self.cfg
        if getattr(cfg, "lambda_low_clip", 0.0) > 0:
            out["low_clip"] = soft_clip_loss(
                F.normalize(pred.flatten(1).float(), dim=-1),
                F.normalize(target.flatten(1).float(), dim=-1), cfg.clip_temp)
        if getattr(cfg, "lambda_low_std", 0.0) > 0:
            # Shrinkage shows up as a per-channel std well below the 1.0 the target
            # was standardised to. Match the second moment directly rather than
            # rescaling after the fact, which cannot add information.
            dims = (0,) + tuple(range(2, pred.dim()))
            out["low_std"] = (pred.float().std(dim=dims)
                              - target.float().std(dim=dims)).abs().mean()
        return out

    @torch.no_grad()
    def predict_lowlevel(self, fmri, subject):
        """Blurry RGB layout for the decoder's img2img init.

        None when the low pathway is off *or* when it predicts a latent -- use
        :meth:`predict_low_latent` for that. The two are mutually exclusive, so a
        caller can pass both to ``decoder.decode`` and let the None fall through.
        """
        if self.low_head is None or self.low_is_latent:
            return None
        b = self.backbone(fmri, subject) if self.cfg.ll_use_backbone else None
        return self.low_head(fmri, subject, b)

    @torch.no_grad()
    def predict_low_latent(self, fmri, subject, moment_match: bool = False):
        """Predicted img2img latent (B,4,96,96) in the decoder's space, or None.

        ``moment_match`` rescales each channel of the standardised prediction to
        unit variance before unstandardising. An L1-regressed head under
        uncertainty shrinks toward the conditional mean, so its per-channel std
        is well below the 1.0 the target was standardised to; multiplying that
        shrunken field by ``lat_std`` yields a latent with too little variance,
        which the VAE decodes as a pale, low-contrast image. Rescaling restores
        the second moment without touching the predicted structure. It cannot
        add information -- if the head is at the mean, this amplifies noise -- so
        it is off by default and belongs in an eval sweep, not in training."""
        if not self.low_is_latent:
            return None
        b = self.backbone(fmri, subject) if self.cfg.ll_use_backbone else None
        z = self.low_head(fmri, subject, b).float()
        if moment_match:
            std = z.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
            z = z / std
        return self.low_head.unstandardize(z)

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
        """R-XFM: transport the anchor posterior onto the bigG token manifold.

        The source is ``q(z0 | fMRI)`` rather than ``Uniform(S^{d-1})``. Two
        consequences that matter here:

        * every point on the geodesic ``slerp(z0, z1, t)`` carries brain
          information, so the DiT learns a short residual transport instead of a
          full generative map from an uninformative start;
        * ``v == 0`` is the identity flow, so the prior is bounded below by the
          regression anchor. Previously the prior *competed* with the anchor and
          lost (blend PixCorr 0.131 < regression 0.164); now it can only refine.
        """
        cfg = self.cfg
        z1, r1 = polar_encode(target_raw.float(), self.tgt_mean.float())

        mu, logs, logr = self.anchor(brain)
        loss_cos = 1.0 - F.cosine_similarity(mu, z1, dim=-1).mean()
        loss_radius = F.mse_loss(logr, r1.log())
        # Same objective as loss_cos up to the factor 2/d (see lambda_reg); kept
        # for log continuity with step1c runs, weighted 0 by default.
        loss_reg = F.mse_loss(mu, z1)

        out = {"reg": loss_reg, "cos": loss_cos, "radius": loss_radius}
        if logs is not None:
            sigma_p = sigma_from_jitter(cfg.anchor_jitter_rad, cfg.token_dim - 1)
            out["kl"] = wrapped_gaussian_kl(logs, sigma_p).mean()

        # The anchor is supervised by loss_cos; letting the flow's loss also push on
        # mu would let the model shorten its own path by degrading the anchor. So
        # the BASE POINT is detached -- but sigma is not, or logs_head receives no
        # gradient at all and freezes at its initialisation (see sample_anchor).
        z0 = self._flow_source(mu, logs, z1.shape, device, z1.dtype, detach_base=True)

        t = self._sample_t(B, device)
        tb = t[:, None, None]
        zt = slerp(z0, z1, tb)
        brain_in, cls_in = self._patch_cond(brain, cls_dir, B, device)

        if self.flow_param == "endpoint":
            # Predict the endpoint and score it directly: uniform in t, same scale
            # as `cos`, and zero-init => zhat1 == zt => v == 0 => identity flow.
            z1_hat = self._prior_endpoint(zt, t, brain_in, cls_in)
            out["flow"] = 1.0 - F.cosine_similarity(z1_hat, z1, dim=-1).mean()
        else:
            ut = slerp_velocity(z0, z1, tb)
            v = project_tangent(self.prior(zt, t, brain_in, cls_in).float(), zt)
            out["flow"] = F.mse_loss(v, ut)

        # Discriminative supervision on the decoder path. `clip` above aligns a
        # pooled 2.4M projection that the decoder never sees; this aligns the
        # flattened 256x1664 grid it actually consumes, preserving per-position
        # structure (NeuroFlow flattens before normalising, neurovae.py:57).
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

        # NOTE: the token-level SoftCLIP lives in _patch_step_sphere, where mu and
        # the centred target z1 already exist -- computing it here would re-run
        # reg_head and would have to mean-pool the sequence, which destroys exactly
        # the per-position structure the term is meant to supervise.
        return out

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
    def _sample_prior_sphere(self, brain, cls_hat, n_steps, cfg_scale, solver,
                             z0=None):
        """Integrate the velocity field from ``z0`` (defaults to the flow source).

        Note on CFG: with an anchor source, guidance is incoherent -- the
        "unconditional" branch drops the brain tokens while ``z0`` is itself
        brain-derived, so the two branches do not differ in the way CFG assumes.
        Keep ``cfg_scale=1.0`` for anchor sources (NeuroFlow's XFM removes
        classifier guidance for the same reason).
        """
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
        """Flow start at INFERENCE time.

        Defaults to the posterior mode ``mu`` rather than a draw: at inference we
        want the lowest-variance estimate. ``stochastic_source=True`` draws
        instead, which is what the multi-sample / latent-averaging arm needs.
        """
        mode = getattr(self.cfg, "z0_mode", "default")
        if mode == "noise":
            return random_sphere(mu.shape, mu.device, mu.dtype)
        if mode == "shuffle":
            return torch.roll(mu, 1, dims=0)
        if self.cfg.flow_source == "noise":
            return random_sphere(mu.shape, mu.device, mu.dtype)
        if self.cfg.stochastic_source:
            return self.sample_anchor(mu, logs)
        return mu

    @torch.no_grad()
    def diagnose(self, fmri, subject, target_raw, *, target_cls=None,
                 n_steps=20, solver="heun"):
        """Latent-space diagnostics for :mod:`brainflow.step1.instrument`.

        Returns per-sample tensors so the caller can aggregate across batches.
        The key output is the pair (anchor_cos, flow_cos): their difference is
        exactly what the flow contributes over its own starting point.
        """
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
        """Return (raw_tokens[B,N,d], cls_hat[B,dcls] or None)."""
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
                sp = self._sample_prior_sphere(brain, cls_hat, n_steps, cfg_scale,
                                               solver, z0=self._infer_z0(mu, logs))
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
