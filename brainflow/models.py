"""BrainFlow v5 model: per-subject input projection + shared encoder + UNet."""
from __future__ import annotations
import math, random
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


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


class BrainEncoder(nn.Module):
    """fMRI → (B, N_TOKENS, BRAIN_DIM) tokens + (B, BRAIN_DIM) cls.
    Per-subject input projection keyed by integer subject id."""
    def __init__(self, cfg: Config, voxels_per_subject: Dict[int, int]):
        super().__init__()
        self.cfg = cfg
        self.subjects = sorted(voxels_per_subject.keys())
        # Per-subject input projection
        self.input_proj = nn.ModuleDict({
            str(s): nn.Linear(voxels_per_subject[s], cfg.enc_hidden)
            for s in self.subjects
        })
        self.stem_norm = nn.LayerNorm(cfg.enc_hidden)
        self.stem_act = nn.GELU()
        self.stem_drop = nn.Dropout(0.3)

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

    def forward(self, fmri: torch.Tensor, subject: torch.Tensor):
        sid = int(subject[0].item())
        proj = self.input_proj[str(sid)]
        h = self.stem_drop(self.stem_act(self.stem_norm(proj(fmri))))
        for b in self.blocks:
            h = b(h)
        tokens = self.to_tokens(h)
        cls = F.normalize(self.cls_head(h), dim=-1)
        return tokens, cls



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
        # HRF-convolve along token axis (treat tokens as a 1D temporal signal)
        x = ctx.transpose(1, 2)                                  # (B, D, N)
        kernel = self.hrf_kernel.expand(D, 1, -1)                # (D, 1, L)
        pad = kernel.shape[-1] // 2
        x = F.conv1d(x, kernel, padding=pad, groups=D)[:, :, :N] # (B, D, N)
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



class BrainFlowV5(nn.Module):
    def __init__(self, cfg: Config, voxels_per_subject: Dict[int, int]):
        super().__init__()
        self.cfg = cfg
        self.brain_enc = BrainEncoder(cfg, voxels_per_subject)
        self.flow_unet = FlowUNet(cfg)
        # Brain-manifold source projector (cfg.method == "hrf"):
        # maps the CLS embedding into the VAE latent space to serve as a
        # structured starting point for the flow (zero-init = pure-noise start).
        if getattr(cfg, "method", "baseline") == "hrf":
            self.cls_to_latent = nn.Linear(
                cfg.brain_dim, cfg.latent_ch * cfg.latent_res * cfg.latent_res
            )
            nn.init.zeros_(self.cls_to_latent.weight)
            nn.init.zeros_(self.cls_to_latent.bias)
        else:
            self.cls_to_latent = None

    def encode_fmri(self, fmri, subject):
        return self.brain_enc(fmri, subject)

    def _source_from_cls(self, cls: torch.Tensor) -> torch.Tensor:
        """Brain-manifold starting point: projected CLS + 0.1 * noise."""
        cfg = self.cfg
        B = cls.shape[0]
        proj = self.cls_to_latent(cls).view(B, cfg.latent_ch, cfg.latent_res, cfg.latent_res)
        return proj + 0.1 * torch.randn_like(proj)

    @torch.no_grad()
    def sample(self, tokens, n_steps=None, cfg_scale=None, solver: str = "midpoint",
               cls: torch.Tensor | None = None):
        """ODE integration from source distribution → VAE latent.

        For cfg.method == "hrf", `cls` MUST be provided so the source distribution
        can be initialized as projected_CLS + 0.1*noise (brain-manifold flow).
        For baseline, source = pure Gaussian noise.

        solver: "euler" | "midpoint" | "heun"
        """
        cfg = self.cfg
        if n_steps is None: n_steps = cfg.ode_steps
        if cfg_scale is None: cfg_scale = cfg.cfg_scale
        B = tokens.shape[0]
        if getattr(cfg, "method", "baseline") == "hrf":
            assert cls is not None, "HRF method requires cls embedding for sampling"
            x = self._source_from_cls(cls)
        else:
            x = torch.randn(B, cfg.latent_ch, cfg.latent_res, cfg.latent_res,
                            device=tokens.device)
        dt = 1.0 / n_steps
        null = self.flow_unet.null_tokens.expand(B, -1, -1)

        def _vel(xt, t_val):
            t = torch.full((B,), t_val, device=xt.device)
            if cfg_scale > 1.0:
                v_cond = self.flow_unet(xt, t, tokens)
                v_uncond = self.flow_unet(xt, t, null)
                return v_uncond + cfg_scale * (v_cond - v_uncond)
            return self.flow_unet(xt, t, tokens)

        for i in range(n_steps):
            if solver == "euler":
                t_cur = i * dt
                v = _vel(x, t_cur)
                x = x + v * dt
            elif solver == "midpoint":
                # Midpoint Euler: evaluate velocity at midpoint of interval
                t_mid = (i + 0.5) * dt
                v = _vel(x, t_mid)
                x = x + v * dt
            elif solver == "heun":
                # Heun's method (2-stage RK): doubles NFE but dramatically
                # more accurate — use for final evaluation runs
                t_cur = i * dt
                t_next = (i + 1) * dt
                v1 = _vel(x, t_cur)
                x_euler = x + v1 * dt
                v2 = _vel(x_euler, t_next)
                x = x + (v1 + v2) * (dt / 2)
            else:
                raise ValueError(f"Unknown solver: {solver!r}. Use 'euler', 'midpoint', or 'heun'.")

            # SB stochastic sampler: add σ(t)·√dt·ε at each step (Euler-Maruyama).
            # σ(t) = σ_max·√(t(1-t)) peaks at t=0.5, vanishes at endpoints so the
            # final sample is deterministic given the velocity field.
            if getattr(cfg, "method", "baseline") == "sb":
                t_next = (i + 1) * dt
                sigma_t = cfg.sb_sigma_max * math.sqrt(max(t_next * (1 - t_next), 0.0))
                if sigma_t > 0:
                    x = x + sigma_t * math.sqrt(dt) * torch.randn_like(x)
        return x

    def training_step(self, batch, device, vae=None, percep_loss_fn=None):
        cfg = self.cfg
        fmri = batch["fmri"].to(device, non_blocking=True)
        latent = batch["latent"].to(device, non_blocking=True)
        clip_emb = batch["clip_emb"].to(device, non_blocking=True)
        subject = batch["subject"].to(device, non_blocking=True)
        B = fmri.shape[0]

        # Mixup: mix only the fMRI input (not the latent target).
        # VAE latent space is not linear — mixed latents are off-manifold.
        # Mixing fMRI still acts as input augmentation without corrupting targets.
        if self.training and cfg.mixup_alpha > 0 and random.random() < 0.3:
            lam = float(np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha))
            lam = max(lam, 1 - lam)
            perm = torch.randperm(B, device=device)
            fmri_mixed = lam * fmri + (1 - lam) * fmri[perm]
        else:
            fmri_mixed = fmri
            lam = 1.0

        tokens, cls_emb = self.encode_fmri(fmri_mixed, subject)

        # Source distribution: brain-manifold (HRF) or pure noise (baseline/sb)
        if getattr(cfg, "method", "baseline") == "hrf":
            x0 = self._source_from_cls(cls_emb)
        else:
            x0 = torch.randn_like(latent)
        t = torch.rand(B, device=device)
        t_expand = t[:, None, None, None]
        xt = (1 - t_expand) * x0 + t_expand * latent
        ut = latent - x0

        # SB noisy-bridge interpolant: add σ(t)·ε to xt (velocity target unchanged)
        if getattr(cfg, "method", "baseline") == "sb":
            sigma_t = cfg.sb_sigma_max * torch.sqrt((t_expand * (1 - t_expand)).clamp(min=0))
            xt = xt + sigma_t * torch.randn_like(xt)

        drop_mask = torch.rand(B, device=device) < cfg.cfg_drop_prob
        null_tokens = self.flow_unet.null_tokens.expand(B, -1, -1)
        ctx = torch.where(drop_mask[:, None, None], null_tokens, tokens)

        # B5: Token dropout — replace dropped tokens with null_tokens so the
        # UNet always sees a valid distribution (not zeros).
        # This is applied AFTER the CFG drop so we don't double-apply to the
        # already-nulled unconditional path.
        if self.training and cfg.token_drop_prob > 0:
            drop = torch.rand(B, ctx.shape[1], 1, device=device) < cfg.token_drop_prob
            null_expand = self.flow_unet.null_tokens.expand(B, -1, -1)
            ctx = torch.where(drop, null_expand, ctx)

        vt = self.flow_unet(xt, t, ctx)
        loss_cfm = F.mse_loss(vt, ut)

        if lam < 1.0:
            _, cls_clean = self.encode_fmri(fmri, subject)
        else:
            cls_clean = cls_emb
        logits = (cls_clean @ clip_emb.T) / cfg.infonce_temp
        labels = torch.arange(B, device=device)
        loss_align = (F.cross_entropy(logits, labels) +
                      F.cross_entropy(logits.T, labels)) / 2

        # Compute perceptual loss if enabled
        loss_percep = torch.tensor(0.0, device=device)
        if percep_loss_fn is not None and vae is not None and cfg.lambda_percep > 0:
            # Decode predicted and target latents to pixel space
            with torch.no_grad():
                gt_imgs = batch["image"].to(device, non_blocking=True)
                # Normalize to [-1, 1] for VAE
                gt_imgs = gt_imgs * 2.0 - 1.0
            
            # Decode predicted latent (use the predicted velocity to get final latent)
            # For training, we sample at a random time step, so we approximate
            # the final latent using: latent ≈ xt + vt * (1 - t)
            pred_latent = xt + vt * (1 - t_expand)
            pred_imgs = vae.decode(pred_latent)
            
            # Compute perceptual loss in pixel space
            loss_percep = percep_loss_fn(pred_imgs, gt_imgs)

        total = cfg.lambda_cfm * loss_cfm + cfg.lambda_align * loss_align + cfg.lambda_percep * loss_percep
        return {"loss": total, "cfm": loss_cfm, "align": loss_align, "percep": loss_percep}
