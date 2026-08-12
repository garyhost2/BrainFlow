"""Diagnostics must be correct *and* cheap, since they run at every eval."""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from brainflow.step1.instrument import (dispersion, effective_rank, grad_norms,
                                        latent_diagnostics, param_report,
                                        train_val_gap, write_manifest)
from brainflow.step1.model_tokens import TokenStep1Config, TokenStep1Model

VOX = {1: 128}


def tiny_cfg(**kw):
    base = dict(token_len=8, token_dim=32, cls_dim=16, brain_dim=32, n_brain_tokens=4,
                enc_hidden=64, enc_blocks=1, reg_depth=1, reg_heads=2, radius_depth=1,
                prior_width=32, prior_depth=1, prior_heads=2, time_dim=16,
                cls_width=32, cls_depth=1, cls_heads=2, low_level=False, subjects=[1])
    base.update(kw)
    return TokenStep1Config(**base)


class _DS(Dataset):
    def __init__(self, cfg, n=24):
        torch.manual_seed(0)
        self.f = torch.randn(n, VOX[1])
        self.e = F.normalize(torch.randn(n, cfg.token_len, cfg.token_dim), dim=-1) * 2
        self.c = torch.randn(n, cfg.cls_dim)

    def __len__(self):
        return len(self.f)

    def __getitem__(self, i):
        return {"fmri": self.f[i], "emb": self.e[i], "cls": self.c[i], "subject": 1}


def _collate(b):
    return {"fmri": torch.stack([x["fmri"] for x in b]),
            "emb": torch.stack([x["emb"] for x in b]),
            "cls": torch.stack([x["cls"] for x in b]),
            "subject": 1}


def loader(cfg, bs=8):
    return DataLoader(_DS(cfg), batch_size=bs, collate_fn=_collate)


def make(cfg):
    torch.manual_seed(0)
    m = TokenStep1Model(cfg, VOX)
    m.set_target_mean(torch.zeros(cfg.token_dim))
    return m


# ---------------------------------------------------------------------------
def test_latent_diagnostics_reports_the_full_key_set():
    cfg = tiny_cfg(flow_source="anchor_var")
    d = latent_diagnostics(make(cfg).eval(), loader(cfg), torch.device("cpu"),
                           n_steps=4, solver="euler", prefix="val/")
    for k in ("val/anchor_cos", "val/flow_cos", "val/delta_cos",
              "val/anchor_theta_deg", "val/cls_cos_hat_true", "val/radius_corr",
              "val/sigma_mean", "val/retr_fwd_300", "val/eff_rank"):
        assert k in d, f"missing {k}"
        assert isinstance(d[k], float)


def test_delta_cos_is_zero_for_an_untrained_identity_flow():
    """The prior's out-layer is zero-init, so the flow is the identity and it
    adds exactly nothing -- delta_cos must report 0, not a flattering number."""
    cfg = tiny_cfg(flow_source="anchor_det", cfg_scale=1.0)
    d = latent_diagnostics(make(cfg).eval(), loader(cfg), torch.device("cpu"),
                           n_steps=4, solver="euler", prefix="val/")
    assert d["val/delta_cos"] == pytest.approx(0.0, abs=1e-5)
    assert d["val/anchor_cos"] == pytest.approx(d["val/flow_cos"], abs=1e-5)


def test_delta_cos_is_signed_and_detects_a_harmful_flow():
    """A flow that moves away from the target must produce delta_cos < 0.

    This is the whole point of the metric: step1c's blend config was measurably
    worse than its own regression anchor and nothing in the logs said so.

    The anchor is made *good* first (targets set to the anchor's own output, so
    anchor_cos == 1) -- otherwise a random flow has nothing to destroy and
    delta_cos sits at 0 for the wrong reason.
    """
    cfg = tiny_cfg(flow_source="anchor_det", cfg_scale=1.0)
    m = make(cfg).eval()
    ds = _DS(cfg)
    with torch.no_grad():
        mu, _, _ = m.anchor(m.backbone(ds.f, 1))
        ds.e = mu.clone() * 2.0                 # perfect anchor: cos(mu, z1) == 1
    dl = DataLoader(ds, batch_size=8, collate_fn=_collate)

    base = latent_diagnostics(m, dl, torch.device("cpu"), n_steps=8,
                              solver="euler", prefix="val/")
    assert base["val/anchor_cos"] == pytest.approx(1.0, abs=1e-4)
    assert base["val/delta_cos"] == pytest.approx(0.0, abs=1e-5)

    with torch.no_grad():                       # random, non-zero velocity field
        for p_ in m.prior.out.parameters():
            p_.copy_(torch.randn_like(p_) * 0.5)
    d = latent_diagnostics(m, dl, torch.device("cpu"), n_steps=8,
                           solver="euler", prefix="val/")
    assert d["val/delta_cos"] < -1e-3, "a harmful flow must register as negative"
    assert 0.0 <= d["val/delta_cos_win_rate"] <= 1.0


def test_noise_source_anchor_cos_is_near_zero():
    cfg = tiny_cfg(flow_source="noise", token_dim=256, cfg_scale=1.0)
    d = latent_diagnostics(make(cfg).eval(), loader(cfg), torch.device("cpu"),
                           n_steps=4, solver="euler", prefix="val/")
    assert abs(d["val/anchor_cos"]) < 0.15
    assert d["val/anchor_theta_deg"] == pytest.approx(90.0, abs=10.0)


def test_diagnostics_restore_training_mode():
    cfg = tiny_cfg(flow_source="anchor_var")
    m = make(cfg)
    m.train()
    latent_diagnostics(m, loader(cfg), torch.device("cpu"), n_steps=2, solver="euler")
    assert m.training, "diagnostics must not silently leave the model in eval mode"


def test_effective_rank_detects_dimensional_collapse():
    n, d = 64, 32
    assert effective_rank(torch.randn(n, d)) > d // 2
    # Genuinely low-rank: all rows live in a 2-D subspace.
    basis = torch.randn(2, d)
    low_rank = torch.randn(n, 2) @ basis
    assert effective_rank(low_rank) <= 2


def test_dispersion_detects_mean_collapse():
    """The failure mode four low-level heads actually hit.

    Centring makes constant-plus-noise look full rank, so effective_rank is blind
    to it -- which is exactly why both are logged.
    """
    n, d = 64, 32
    spread = torch.randn(n, d)
    collapsed = torch.randn(1, d).repeat(n, 1) * 10.0 + torch.randn(n, d) * 1e-3

    assert dispersion(spread) == pytest.approx(1.0, abs=0.15)
    assert dispersion(collapsed) < 0.01
    assert effective_rank(collapsed) > d // 2      # documents the blind spot


def test_train_val_gap_pairs_matching_keys():
    tr = {"train/flow_cos": 0.8, "train/retr_fwd_300": 0.9, "train/only": 1.0}
    va = {"val/flow_cos": 0.6, "val/retr_fwd_300": 0.5}
    g = train_val_gap(tr, va)
    assert g["gap/flow_cos"] == pytest.approx(0.2)
    assert g["gap/retr_fwd_300"] == pytest.approx(0.4)
    assert "gap/only" not in g


# ---------------------------------------------------------------------------
def test_grad_norms_expose_the_reg_cos_scale_identity():
    """`reg` and `cos` are the same objective; their gradient norms must differ
    by the documented factor 2/d, which is what makes lambda_reg meaningless."""
    cfg = tiny_cfg(flow_source="anchor_det")
    m = make(cfg)
    torch.manual_seed(0)
    fmri = torch.randn(6, VOX[1])
    raw = F.normalize(torch.randn(6, cfg.token_len, cfg.token_dim), dim=-1) * 2
    out = m.training_step(fmri, 1, target_std=raw, target_raw=raw,
                          target_cls=torch.randn(6, cfg.cls_dim),
                          keep_term_graph=True)
    gn = grad_norms({"reg": out["reg"], "cos": out["cos"]}, m.parameters())
    assert gn["gradnorm/reg"] > 0 and gn["gradnorm/cos"] > 0
    assert gn["gradnorm/reg"] / gn["gradnorm/cos"] == pytest.approx(
        2.0 / cfg.token_dim, rel=1e-3)


def test_grad_norms_leave_no_stale_graph():
    cfg = tiny_cfg(flow_source="anchor_var")
    m = make(cfg)
    fmri = torch.randn(4, VOX[1])
    raw = F.normalize(torch.randn(4, cfg.token_len, cfg.token_dim), dim=-1)
    out = m.training_step(fmri, 1, target_std=raw, target_raw=raw,
                          target_cls=torch.randn(4, cfg.cls_dim),
                          keep_term_graph=True)
    grad_norms({k: v for k, v in out.items() if k != "loss"}, m.parameters())
    out["loss"].backward()          # must still work after the probes


def test_terms_are_detached_by_default():
    """The probe is opt-in; the hot path must not retain the term graphs."""
    cfg = tiny_cfg(flow_source="anchor_var")
    m = make(cfg)
    raw = F.normalize(torch.randn(4, cfg.token_len, cfg.token_dim), dim=-1)
    out = m.training_step(torch.randn(4, VOX[1]), 1, target_std=raw, target_raw=raw,
                          target_cls=torch.randn(4, cfg.cls_dim))
    assert not out["flow"].requires_grad
    assert out["loss"].requires_grad


# ---------------------------------------------------------------------------
def test_manifest_is_written_and_reloadable(tmp_path):
    cfg = tiny_cfg(flow_source="anchor_var")
    m = make(cfg)

    class A:
        pass
    a = A(); a.lr = 3e-4; a.subjects = [1]

    p = write_manifest(tmp_path, args=a, cfg=cfg, model=m, voxels=VOX)
    blob = json.loads(p.read_text())
    assert blob["config"]["flow_source"] == "anchor_var"
    assert blob["config_hash"]
    assert blob["params"]["params_M/total"] > 0
    assert blob["voxels_per_subject"]["1"] == 128


def test_param_report_covers_every_submodule():
    cfg = tiny_cfg(flow_source="anchor_var")
    r = param_report(make(cfg))
    for k in ("params_M/backbone", "params_M/prior", "params_M/reg_head",
              "params_M/logs_head", "params_M/total"):
        assert k in r
    assert r["params_M/total"] >= r["params_M/prior"]
