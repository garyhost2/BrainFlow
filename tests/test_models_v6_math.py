import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brainflow.config import Config
from brainflow.models import CLIPPrior, _sample_t, _sinkhorn_ot


def test_sample_t_schedules_are_bounded():
    cfg = Config()
    for sched in ("linear", "cosine", "logit_normal"):
        cfg.t_schedule = sched
        t = _sample_t(128, torch.device("cpu"), cfg)
        assert (t >= 0).all() and (t <= 1).all()


def test_sinkhorn_ot_returns_valid_indices():
    B = 6
    x0 = torch.randn(B, 4)
    x1 = torch.randn(B, 4)
    C = torch.cdist(x1, x0).pow(2)
    idx = _sinkhorn_ot(C, reg=0.05, n_iter=10)
    assert idx.shape == (B,)
    assert ((idx >= 0) & (idx < B)).all()


def test_clip_prior_vfm_sigma_and_sphere_sampling():
    cfg = Config(
        clip_dim=16,
        brain_dim=16,
        prior_dim=32,
        prior_blocks=1,
        clip_prior_geometry="sphere",
        flow_objective="vfm",
    )
    prior = CLIPPrior(cfg).eval()
    B = 4
    cls = F.normalize(torch.randn(B, cfg.brain_dim), dim=-1)
    clip_gt = F.normalize(torch.randn(B, cfg.clip_dim), dim=-1)
    t = torch.rand(B)

    mu, sigma, log_sigma = prior.forward_with_sigma(clip_gt, t, cls)
    assert mu.shape == clip_gt.shape
    assert sigma.shape == clip_gt.shape
    assert log_sigma.shape == clip_gt.shape
    assert (sigma >= 1e-4).all() and (sigma <= 10.0).all()

    loss = prior.flow_loss(clip_gt, cls, vfm_kl_weight=0.1)
    assert torch.isfinite(loss)

    out = prior.sample(cls, n_steps=5, solver="dpm2m")
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
