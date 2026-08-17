from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from rxfm.model import TokenStep1Config, TokenStep1Model
from rxfm.sphere import log_map, slerp, slerp_velocity


VOX = {1: 128}


def tiny_cfg(**kw) -> TokenStep1Config:
    base = dict(
        token_len=8, token_dim=32, cls_dim=16, brain_dim=32, n_brain_tokens=4,
        enc_hidden=64, enc_blocks=1, reg_depth=1, reg_heads=2, radius_depth=1,
        prior_width=32, prior_depth=1, prior_heads=2, time_dim=16,
        cls_width=32, cls_depth=1, cls_heads=2, low_level=False, subjects=[1],
    )
    base.update(kw)
    return TokenStep1Config(**base)


def make(cfg):
    torch.manual_seed(0)
    m = TokenStep1Model(cfg, VOX)
    m.set_target_mean(torch.zeros(cfg.token_len, cfg.token_dim)
                      if cfg.center_mode == "per_position"
                      else torch.zeros(cfg.token_dim))
    return m


def batch(cfg, B=6):
    torch.manual_seed(1)
    return {
        "fmri": torch.randn(B, VOX[1]),
        "raw": F.normalize(torch.randn(B, cfg.token_len, cfg.token_dim), dim=-1) * 2.0,
        "cls": torch.randn(B, cfg.cls_dim),
    }


@pytest.mark.parametrize("t", [0.1, 0.3, 0.5, 0.7, 0.9])
@pytest.mark.parametrize("deg", [68.3, 15.0])
def test_velocity_equals_log_over_one_minus_t(t, deg):
    d, N, B = 1664, 2, 16
    th = math.radians(deg)
    torch.manual_seed(0)
    z0 = F.normalize(torch.randn(B, N, d), dim=-1)
    g = F.normalize(torch.randn(B, N, d), dim=-1)
    g = F.normalize(g - (g * z0).sum(-1, keepdim=True) * z0, dim=-1)
    z1 = math.cos(th) * z0 + math.sin(th) * g

    tb = torch.full((B, 1, 1), t)
    zt = slerp(z0, z1, tb)
    u = slerp_velocity(z0, z1, tb)
    v = log_map(zt, z1) / (1 - t)
    assert float((u - v).abs().max()) < 1e-5


def test_velocity_target_norm_is_theta_and_shrinks_with_the_anchor():
    d, N, B = 1664, 2, 16
    prev = None
    for deg in (90.0, 68.3, 30.0, 15.0):
        th = math.radians(deg)
        torch.manual_seed(0)
        z0 = F.normalize(torch.randn(B, N, d), dim=-1)
        g = F.normalize(torch.randn(B, N, d), dim=-1)
        g = F.normalize(g - (g * z0).sum(-1, keepdim=True) * z0, dim=-1)
        z1 = math.cos(th) * z0 + math.sin(th) * g
        tb = torch.rand(B, 1, 1)
        u = slerp_velocity(z0, z1, tb)
        assert float(u.norm(dim=-1).mean()) == pytest.approx(th, rel=1e-3)

        ratio = float((1 - F.cosine_similarity(z0, z1, dim=-1).mean())
                      / F.mse_loss(torch.zeros_like(u), u))
        assert ratio > 500, "cos should dominate the velocity MSE by >500x"
        if prev is not None:
            assert ratio > prev, "the imbalance must worsen as the anchor improves"
        prev = ratio


@pytest.mark.parametrize("param", ["endpoint", "velocity"])
def test_zero_init_prior_is_the_identity_flow(param):
    cfg = tiny_cfg(flow_source="anchor_det", flow_param=param)
    m = make(cfg).eval()
    b = batch(cfg)
    brain = m.backbone(b["fmri"], 1)
    mu, logs, _ = m.anchor(brain)
    z = m._sample_prior_sphere(brain, None, n_steps=10, cfg_scale=1.0,
                               solver="heun", z0=mu)
    assert float(F.cosine_similarity(z, mu.detach(), dim=-1).min()) > 1 - 1e-5


def test_endpoint_head_predicts_zt_at_init():
    cfg = tiny_cfg(flow_param="endpoint")
    m = make(cfg).eval()
    b = batch(cfg)
    brain = m.backbone(b["fmri"], 1)
    zt = F.normalize(torch.randn(b["fmri"].shape[0], cfg.token_len, cfg.token_dim),
                     dim=-1)
    t = torch.full((b["fmri"].shape[0],), 0.4)
    z1_hat = m._prior_endpoint(zt, t, brain, None)
    assert torch.allclose(z1_hat, zt, atol=1e-5)
    v = m._prior_velocity(zt, t, brain, None)
    assert float(v.abs().max()) < 1e-5


def test_endpoint_and_velocity_agree_when_the_residual_is_matched():
    cfg = tiny_cfg(flow_param="endpoint")
    m = make(cfg).eval()
    b = batch(cfg)
    B = b["fmri"].shape[0]
    brain = m.backbone(b["fmri"], 1)
    torch.manual_seed(3)
    with torch.no_grad():
        m.prior.out.bias.normal_(0, 0.1)
    zt = F.normalize(torch.randn(B, cfg.token_len, cfg.token_dim), dim=-1)
    for tv in (0.1, 0.5, 0.9):
        t = torch.full((B,), tv)
        z1_hat = m._prior_endpoint(zt, t, brain, None)
        want = log_map(zt, z1_hat) / (1 - tv)
        got = m._prior_velocity(zt, t, brain, None)
        assert float((want - got).abs().max()) < 1e-5


def test_endpoint_flow_loss_is_on_the_same_scale_as_cos():
    cfg = tiny_cfg(token_dim=1664, token_len=4, flow_param="endpoint",
                   flow_source="anchor_det")
    m = make(cfg)
    b = batch(cfg)
    out = m.training_step(b["fmri"], 1, b["raw"], target_raw=b["raw"],
                          target_cls=b["cls"])
    ratio = float(out["cos"]) / max(float(out["flow"]), 1e-12)
    assert 0.05 < ratio < 20.0, f"flow/cos scale ratio {1/ratio:.3g} is not O(1)"


def test_per_position_centering_changes_the_target_directions():
    cfg_g = tiny_cfg(center_mode="global")
    cfg_p = tiny_cfg(center_mode="per_position")
    assert make(cfg_g).tgt_mean.shape == (cfg_g.token_dim,)
    m = make(cfg_p)
    assert m.tgt_mean.shape == (cfg_p.token_len, cfg_p.token_dim)

    torch.manual_seed(0)
    per_pos = torch.randn(cfg_p.token_len, cfg_p.token_dim)
    m.set_target_mean(per_pos)
    from rxfm.sphere import polar_encode
    x = torch.randn(4, cfg_p.token_len, cfg_p.token_dim)
    z_p, _ = polar_encode(x, m.tgt_mean)
    z_g, _ = polar_encode(x, per_pos.mean(0))
    assert float((z_p - z_g).abs().max()) > 1e-3


def test_set_target_mean_rejects_a_mismatched_shape():
    m = make(tiny_cfg(center_mode="global"))
    with pytest.raises(ValueError, match="center_mode"):
        m.set_target_mean(torch.zeros(m.cfg.token_len, m.cfg.token_dim))


def test_endpoint_param_rejects_euclidean_geometry():
    with pytest.raises(ValueError, match="sphere-only"):
        TokenStep1Model(tiny_cfg(flow_param="endpoint", geometry="euclidean"), VOX)
