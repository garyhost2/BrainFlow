"""Tests for spherical geometry helpers in brainflow/clip_prior.py.

All tests run on CPU without network access.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brainflow.clip_prior import (
    random_sphere,
    slerp,
    slerp_tangent,
    exp_map,
    log_map,
    project_tangent,
    ClipPrior,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

B, D = 8, 16  # small batch and dimension for fast tests


def unit_pair() -> tuple[torch.Tensor, torch.Tensor]:
    """Two random unit-vector batches."""
    x0 = F.normalize(torch.randn(B, D), dim=-1)
    x1 = F.normalize(torch.randn(B, D), dim=-1)
    return x0, x1


# ── random_sphere ─────────────────────────────────────────────────────────────

class TestRandomSphere:
    def test_unit_norm(self):
        x = random_sphere((B, D), device="cpu")
        norms = x.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(B), atol=1e-5)

    def test_shape(self):
        x = random_sphere((B, D), device="cpu")
        assert x.shape == (B, D)

    def test_distinct_samples(self):
        x0 = random_sphere((B, D), device="cpu")
        x1 = random_sphere((B, D), device="cpu")
        assert not torch.allclose(x0, x1)


# ── slerp ─────────────────────────────────────────────────────────────────────

class TestSlerp:
    def test_t0_returns_x0(self):
        x0, x1 = unit_pair()
        t = torch.zeros(B)
        out = slerp(x0, x1, t)
        assert torch.allclose(out, x0, atol=1e-5)

    def test_t1_returns_x1(self):
        x0, x1 = unit_pair()
        t = torch.ones(B)
        out = slerp(x0, x1, t)
        assert torch.allclose(out, x1, atol=1e-5)

    def test_output_is_unit_norm(self):
        x0, x1 = unit_pair()
        t = torch.rand(B)
        out = slerp(x0, x1, t)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(B), atol=1e-5)

    def test_midpoint_is_on_sphere(self):
        x0, x1 = unit_pair()
        t = torch.full((B,), 0.5)
        out = slerp(x0, x1, t)
        assert torch.allclose(out.norm(dim=-1), torch.ones(B), atol=1e-5)

    def test_collinear_fallback(self):
        """Near-collinear pairs (x0 ≈ x1) should not produce NaN."""
        x0 = F.normalize(torch.randn(B, D), dim=-1)
        x1 = x0 + 1e-8 * torch.randn(B, D)  # almost identical
        x1 = F.normalize(x1, dim=-1)
        t = torch.rand(B)
        out = slerp(x0, x1, t)
        assert not torch.isnan(out).any()
        assert torch.allclose(out.norm(dim=-1), torch.ones(B), atol=1e-4)

    def test_antipodal_fallback(self):
        """Antipodal pairs (x1 = -x0) should not produce NaN."""
        x0 = F.normalize(torch.randn(B, D), dim=-1)
        x1 = -x0
        t = torch.rand(B)
        out = slerp(x0, x1, t)
        assert not torch.isnan(out).any()


# ── slerp_tangent ─────────────────────────────────────────────────────────────

class TestSlerpTangent:
    def test_shape(self):
        x0, x1 = unit_pair()
        t = torch.rand(B)
        ut = slerp_tangent(x0, x1, t)
        assert ut.shape == (B, D)

    def test_tangent_orthogonal_to_xt(self):
        """u_t must be orthogonal to x_t (i.e., tangent to the sphere)."""
        x0, x1 = unit_pair()
        t = torch.rand(B)
        xt = slerp(x0, x1, t)
        ut = slerp_tangent(x0, x1, t)
        dot = (ut * xt).sum(dim=-1)
        assert torch.allclose(dot, torch.zeros(B), atol=1e-4)

    def test_no_nan(self):
        x0, x1 = unit_pair()
        t = torch.rand(B)
        ut = slerp_tangent(x0, x1, t)
        assert not torch.isnan(ut).any()


# ── exp_map ───────────────────────────────────────────────────────────────────

class TestExpMap:
    def test_output_is_unit_norm(self):
        x0, _ = unit_pair()
        v = torch.randn(B, D) * 0.3
        out = exp_map(x0, v)
        assert torch.allclose(out.norm(dim=-1), torch.ones(B), atol=1e-5)

    def test_zero_tangent_returns_x(self):
        """exp_x(0) should return x."""
        x0, _ = unit_pair()
        v = torch.zeros(B, D)
        out = exp_map(x0, v)
        # normalize(x + 0) = x for unit x
        assert torch.allclose(out, x0, atol=1e-5)

    def test_round_trip_with_log_map(self):
        """exp_x(log_x(y)) ≈ y for generic pairs."""
        x0, x1 = unit_pair()
        v = log_map(x0, x1)
        y = exp_map(x0, v)
        assert torch.allclose(y, x1, atol=1e-4)


# ── log_map ───────────────────────────────────────────────────────────────────

class TestLogMap:
    def test_shape(self):
        x0, x1 = unit_pair()
        v = log_map(x0, x1)
        assert v.shape == (B, D)

    def test_tangent(self):
        """log_x(y) must be orthogonal to x."""
        x0, x1 = unit_pair()
        v = log_map(x0, x1)
        dot = (v * x0).sum(dim=-1)
        assert torch.allclose(dot, torch.zeros(B), atol=1e-4)

    def test_no_nan(self):
        x0, x1 = unit_pair()
        v = log_map(x0, x1)
        assert not torch.isnan(v).any()


# ── project_tangent ───────────────────────────────────────────────────────────

class TestProjectTangent:
    def test_output_orthogonal_to_x(self):
        x0, _ = unit_pair()
        v = torch.randn(B, D)
        pv = project_tangent(x0, v)
        dot = (pv * x0).sum(dim=-1)
        assert torch.allclose(dot, torch.zeros(B), atol=1e-5)

    def test_already_tangent_unchanged(self):
        """Projecting an already-tangent vector should be a no-op."""
        x0, _ = unit_pair()
        v = torch.randn(B, D)
        pv = project_tangent(x0, v)   # project once
        ppv = project_tangent(x0, pv)  # project again → same
        assert torch.allclose(pv, ppv, atol=1e-5)


# ── ClipPrior geometry flag ───────────────────────────────────────────────────

class TestClipPriorGeometryFlag:
    def _make_prior(self, geometry: str) -> ClipPrior:
        return ClipPrior(clip_dim=32, ctx_dim=64, dim=64, depth=2, heads=4,
                         geometry=geometry)

    def test_euclidean_flow_loss_runs(self):
        prior = self._make_prior("euclidean")
        prior._stats_fitted = False
        clip_emb = torch.randn(4, 32)
        ctx = torch.randn(4, 8, 64)
        loss = prior.flow_loss(clip_emb, ctx)
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_sphere_flow_loss_runs(self):
        prior = self._make_prior("sphere")
        clip_emb = F.normalize(torch.randn(4, 32), dim=-1)
        ctx = torch.randn(4, 8, 64)
        loss = prior.flow_loss(clip_emb, ctx)
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_sphere_sample_returns_unit_vectors(self):
        prior = self._make_prior("sphere").eval()
        ctx = torch.randn(2, 8, 64)
        out = prior.sample(ctx, n_steps=5, cfg_scale=1.0)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-4)

    def test_euclidean_sample_shape(self):
        prior = self._make_prior("euclidean").eval()
        ctx = torch.randn(2, 8, 64)
        out = prior.sample(ctx, n_steps=5, cfg_scale=1.0)
        assert out.shape == (2, 32)

    def test_invalid_geometry_raises(self):
        with pytest.raises(ValueError, match="geometry"):
            ClipPrior(geometry="hyperbolic")

    def test_euclidean_default(self):
        prior = ClipPrior(clip_dim=32, ctx_dim=64, dim=64, depth=1, heads=4)
        assert prior.geometry == "euclidean"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
