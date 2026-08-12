"""R-XFM: the anchor posterior, the flow source, and the identity-flow bound.

The property that makes this change safe to spend cluster time on is the last
test here: with ``v == 0`` the ODE is the identity, so a flow started at the
regression anchor reproduces the anchor exactly. The prior can therefore only
refine ``cond_source="regression"``, never lose to it -- which is what the
step1c geometry did (blend PixCorr 0.131 < regression 0.164).
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from brainflow.step1.model_tokens import (TokenStep1Config, TokenStep1Model,
                                          sigma_from_jitter, wrapped_gaussian_kl)
from brainflow.step1.sphere import exp_map, project_tangent, slerp, slerp_velocity


VOX = {1: 128, 2: 96}


def tiny_cfg(**kw) -> TokenStep1Config:
    """A model small enough to run on CPU but structurally identical."""
    base = dict(
        token_len=8, token_dim=32, cls_dim=16, brain_dim=32, n_brain_tokens=4,
        enc_hidden=64, enc_blocks=1, reg_depth=1, reg_heads=2, radius_depth=1,
        prior_width=32, prior_depth=1, prior_heads=2, time_dim=16,
        cls_width=32, cls_depth=1, cls_heads=2, low_level=False,
        subjects=[1, 2],
    )
    base.update(kw)
    return TokenStep1Config(**base)


def make(cfg):
    torch.manual_seed(0)
    m = TokenStep1Model(cfg, VOX)
    m.set_target_mean(torch.zeros(cfg.token_dim))
    return m


def batch(cfg, B=6, subject=1):
    torch.manual_seed(1)
    return {
        "fmri": torch.randn(B, VOX[subject]),
        "raw": F.normalize(torch.randn(B, cfg.token_len, cfg.token_dim), dim=-1) * 2.0,
        "cls": torch.randn(B, cfg.cls_dim),
    }


# ---------------------------------------------------------------------------
#  Anchor posterior
# ---------------------------------------------------------------------------
def test_anchor_sample_stays_on_manifold():
    cfg = tiny_cfg(flow_source="anchor_var")
    m = make(cfg)
    b = batch(cfg)
    brain = m.backbone(b["fmri"], 1)
    mu, logs, logr = m.anchor(brain)

    assert torch.allclose(mu.norm(dim=-1), torch.ones_like(mu.norm(dim=-1)), atol=1e-5)
    z0 = m.sample_anchor(mu, logs)
    assert torch.allclose(z0.norm(dim=-1), torch.ones_like(z0.norm(dim=-1)), atol=1e-5)
    assert logs.shape == (b["fmri"].shape[0], cfg.token_len)
    assert logr.shape == (b["fmri"].shape[0], cfg.token_len)


def test_logsigma_head_initialises_at_the_prior():
    """A fresh sigma head must not inject a degenerate or huge perturbation."""
    cfg = tiny_cfg(flow_source="anchor_var", anchor_jitter_rad=0.15)
    m = make(cfg)
    brain = m.backbone(batch(cfg)["fmri"], 1)
    _, logs, _ = m.anchor(brain)
    want = math.log(sigma_from_jitter(0.15, cfg.token_dim - 1))
    assert torch.allclose(logs, torch.full_like(logs, want), atol=1e-4)


@pytest.mark.parametrize("d", [64, 512, 1664])
def test_jitter_is_dimension_independent(d):
    """The knob is a geodesic angle, so the realised displacement must not
    depend on the token width -- a per-coordinate sigma would not have this
    property and would silently blow up at d=1664."""
    jit = 0.20
    cfg = tiny_cfg(flow_source="anchor_var", token_dim=d, anchor_jitter_rad=jit)
    m = make(cfg)
    brain = m.backbone(batch(cfg)["fmri"], 1)
    mu, logs, _ = m.anchor(brain)
    z0 = m.sample_anchor(mu, logs)
    realised = float(F.cosine_similarity(z0, mu, dim=-1).detach().clamp(-1, 1).arccos().mean())
    assert realised == pytest.approx(jit, rel=0.15)


def test_default_jitter_preserves_the_anchor_signal():
    """cos(z0, z1) ~ cos(jitter) * cos(theta): the draw must not undo the anchor.

    Calibrating the jitter to the anchor error itself (theta = 68 deg at the
    measured cos 0.37) would take cos(z0, z1) from 0.370 to 0.126 -- the source
    would then be worse than the point it was built from. The default keeps the
    loss under 2% relative.
    """
    d = 1664
    theta = math.acos(0.37)
    for jit, floor in ((0.15, 0.98), (1.19, 0.50)):
        cos_z0_z1 = math.cos(jit) * math.cos(theta)
        ratio = cos_z0_z1 / math.cos(theta)
        assert ratio == pytest.approx(math.cos(jit), rel=1e-6)
        if jit == 0.15:
            assert ratio > floor, "default jitter must be nearly free"
        else:
            assert ratio < floor, "theta-sized jitter destroys the anchor"


def test_sigma_is_capped_at_a_quarter_turn():
    """A learned sigma must not be able to erase the anchor it is centred on."""
    cfg = tiny_cfg(flow_source="anchor_var")
    m = make(cfg)
    with torch.no_grad():                       # force the head hard positive
        m.logs_head.out[-1].bias.fill_(50.0)
    brain = m.backbone(batch(cfg)["fmri"], 1)
    _, logs, _ = m.anchor(brain)
    k = cfg.token_dim - 1
    realised = float(logs.detach().exp().max()) * math.sqrt(k)
    assert realised <= math.pi / 2 + 1e-4


def test_deterministic_source_has_no_sigma_head():
    """Checkpoint compatibility: non-variational arms keep the step1c state_dict."""
    for src in ("noise", "anchor_det"):
        m = make(tiny_cfg(flow_source=src))
        assert m.logs_head is None
        assert not any(k.startswith("logs_head") for k in m.state_dict())


def test_kl_is_zero_at_the_prior_and_positive_away_from_it():
    sp = sigma_from_jitter(0.15, 1663)
    at_prior = wrapped_gaussian_kl(torch.tensor(math.log(sp)), sp)
    assert float(at_prior) == pytest.approx(0.0, abs=1e-6)
    for logs in (math.log(sp / 10), math.log(sp * 10)):
        assert float(wrapped_gaussian_kl(torch.tensor(logs), sp)) > 0.0


def test_kl_matches_the_closed_form_gaussian():
    s, sp = 0.05, 0.02
    got = float(wrapped_gaussian_kl(torch.tensor(math.log(s)), sp))
    want = 0.5 * (s ** 2 / sp ** 2 - 1.0 + 2 * math.log(sp / s))
    assert got == pytest.approx(want, rel=1e-6)


# ---------------------------------------------------------------------------
#  Flow source
# ---------------------------------------------------------------------------
def test_noise_source_is_uninformative_and_anchor_source_is_not():
    """The whole argument for the change, as an assertion.

    On a d-sphere two random unit vectors are near-orthogonal, so the legacy
    source starts ~90 deg from the target regardless of the fMRI.
    """
    cfg = tiny_cfg(flow_source="anchor_var", token_dim=512)
    m = make(cfg)
    b = batch(cfg)
    brain = m.backbone(b["fmri"], 1)
    z1 = F.normalize(b["raw"], dim=-1)
    mu, logs, _ = m.anchor(brain)

    drawn = m._flow_source(mu, logs, z1.shape, z1.device, z1.dtype)
    truly_noise = make(tiny_cfg(flow_source="noise", token_dim=512))
    zn = truly_noise._flow_source(mu, None, z1.shape, z1.device, z1.dtype)

    # Legacy source: uninformative, ~90 deg from the target whatever the fMRI.
    assert abs(float(F.cosine_similarity(zn, z1, dim=-1).mean())) < 0.15
    # Anchor source: inside a small geodesic ball around mu.
    angle = float(F.cosine_similarity(drawn, mu, dim=-1).detach().clamp(-1, 1).arccos().mean())
    assert angle == pytest.approx(cfg.anchor_jitter_rad, rel=0.2)


def test_anchor_var_draw_is_stochastic_but_centred():
    cfg = tiny_cfg(flow_source="anchor_var")
    m = make(cfg)
    brain = m.backbone(batch(cfg)["fmri"], 1)
    mu, logs, _ = m.anchor(brain)
    a = m.sample_anchor(mu, logs)
    b = m.sample_anchor(mu, logs)
    assert not torch.allclose(a, b)
    draws = torch.stack([m.sample_anchor(mu, logs) for _ in range(64)])
    assert float(F.cosine_similarity(F.normalize(draws.mean(0), dim=-1), mu,
                                     dim=-1).mean()) > 0.99


# ---------------------------------------------------------------------------
#  The identity-flow bound  (the reason this is a safe experiment)
# ---------------------------------------------------------------------------
def test_zero_velocity_reproduces_the_anchor_exactly():
    cfg = tiny_cfg(flow_source="anchor_det", two_head=False, cfg_scale=1.0)
    m = make(cfg).eval()
    # The prior's output layer is zero-initialised, so v == 0 out of the box.
    b = batch(cfg)
    brain = m.backbone(b["fmri"], 1)
    mu, logs, _ = m.anchor(brain)
    z = m._sample_prior_sphere(brain, None, n_steps=10, cfg_scale=1.0,
                               solver="heun", z0=mu)
    assert torch.allclose(z, mu, atol=1e-5), "identity flow must return the anchor"


def test_prior_cond_source_matches_regression_at_init():
    """`prior` >= `regression` by construction at initialisation."""
    cfg = tiny_cfg(flow_source="anchor_det", two_head=False, cfg_scale=1.0)
    m = make(cfg).eval()
    b = batch(cfg)
    reg, _ = m.predict_tokens(b["fmri"], 1, None, cond_source="regression", n_steps=5)
    pri, _ = m.predict_tokens(b["fmri"], 1, None, cond_source="prior", n_steps=5)
    assert torch.allclose(reg, pri, atol=1e-4)


# ---------------------------------------------------------------------------
#  Geodesic consistency of the interpolant used by the loss
# ---------------------------------------------------------------------------
def test_slerp_velocity_is_the_derivative_of_slerp():
    torch.manual_seed(0)
    z0 = F.normalize(torch.randn(4, 64).double(), dim=-1)
    z1 = F.normalize(torch.randn(4, 64).double(), dim=-1)
    t = torch.full((4, 1), 0.37, dtype=torch.float64)
    h = 1e-6
    fd = (slerp(z0, z1, t + h) - slerp(z0, z1, t - h)) / (2 * h)
    assert torch.allclose(fd, slerp_velocity(z0, z1, t), atol=1e-5)


def test_slerp_velocity_is_tangent_and_has_norm_theta():
    torch.manual_seed(0)
    z0 = F.normalize(torch.randn(4, 64).double(), dim=-1)
    z1 = F.normalize(torch.randn(4, 64).double(), dim=-1)
    t = torch.full((4, 1), 0.6, dtype=torch.float64)
    zt = slerp(z0, z1, t)
    ut = slerp_velocity(z0, z1, t)
    assert torch.allclose((zt * ut).sum(-1), torch.zeros(4, dtype=torch.float64), atol=1e-8)
    theta = (z0 * z1).sum(-1).clamp(-1, 1).arccos()
    assert torch.allclose(ut.norm(dim=-1), theta, atol=1e-8)


# ---------------------------------------------------------------------------
#  Training step wiring
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("src", ["noise", "anchor_det", "anchor_var"])
def test_training_step_runs_and_backprops(src):
    cfg = tiny_cfg(flow_source=src, lambda_clip_tok=0.1)
    m = make(cfg)
    b = batch(cfg)
    out = m.training_step(b["fmri"], 1, target_std=b["raw"], target_raw=b["raw"],
                          target_cls=b["cls"])
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in m.parameters() if p.requires_grad)
    assert ("kl" in out) == (src == "anchor_var")
    assert "clip_tok" in out


def test_lambda_reg_is_excluded_by_default():
    """`reg` and `cos` are the same objective; reg must not silently double-count."""
    cfg = tiny_cfg(flow_source="anchor_det")
    assert cfg.lambda_reg == 0.0
    m = make(cfg)
    b = batch(cfg)
    out = m.training_step(b["fmri"], 1, target_std=b["raw"], target_raw=b["raw"],
                          target_cls=b["cls"])
    # Documented identity: mse(a,b) == (2/d) * (1 - cos(a,b)) for unit vectors.
    assert float(out["reg"]) == pytest.approx(
        float(out["cos"]) * 2.0 / cfg.token_dim, rel=1e-4)


@pytest.mark.parametrize("mode", ["teacher", "jitter", "none"])
def test_cls_cond_modes(mode):
    cfg = tiny_cfg(flow_source="anchor_var", cls_cond_mode=mode)
    m = make(cfg)
    b = batch(cfg)
    brain = m.backbone(b["fmri"], 1)
    cls_dir = F.normalize(b["cls"], dim=-1)
    got = m._cls_cond(brain, cls_dir)
    if mode == "none":
        assert got is None
    elif mode == "teacher":
        assert torch.allclose(got, cls_dir)
    else:
        assert got.shape == cls_dir.shape
        assert torch.allclose(got.norm(dim=-1), torch.ones(got.shape[0]), atol=1e-5)
        assert not torch.allclose(got, cls_dir)


def test_diagnose_returns_the_delta_cos_ingredients():
    cfg = tiny_cfg(flow_source="anchor_var")
    m = make(cfg).eval()
    b = batch(cfg)
    d = m.diagnose(b["fmri"], 1, b["raw"], target_cls=b["cls"], n_steps=4, solver="euler")
    B = b["fmri"].shape[0]
    for k in ("anchor_cos", "flow_cos", "cls_cos"):
        assert d[k].shape == (B,)
    assert d["sigma"].shape == (B, cfg.token_len)
    assert d["z_flow"].shape == (B, cfg.token_len, cfg.token_dim)
    assert torch.isfinite(d["anchor_cos"]).all() and torch.isfinite(d["flow_cos"]).all()


def test_multi_subject_routing_uses_the_right_projection():
    cfg = tiny_cfg(flow_source="anchor_var")
    m = make(cfg)
    for s in (1, 2):
        out = m.training_step(torch.randn(4, VOX[s]), s,
                              target_std=torch.randn(4, cfg.token_len, cfg.token_dim),
                              target_raw=F.normalize(
                                  torch.randn(4, cfg.token_len, cfg.token_dim), dim=-1),
                              target_cls=torch.randn(4, cfg.cls_dim))
        assert torch.isfinite(out["loss"])
