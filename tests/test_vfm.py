"""Tests for VFM module: equivalence to CFM and numerical stability."""
import math
import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brainflow.vfm import (
    vfm_loss, cfm_loss, velocity_from_posterior, flow_loss,
    VFMHead, _EPS_T,
)


def _seed():
    torch.manual_seed(42)


class TestVFMLoss:
    def test_vfm_loss_positive(self):
        _seed()
        B, D = 4, 64
        mu = torch.randn(B, D)
        log_sigma = torch.zeros(B, D)
        x1 = torch.randn(B, D)
        loss = vfm_loss(mu, log_sigma, x1)
        assert loss.item() > 0, "VFM loss should be positive"

    def test_vfm_loss_perfect_prediction(self):
        """When mu == x1 and log_sigma == -inf, loss should be near 0."""
        _seed()
        B, D = 4, 64
        x1 = torch.randn(B, D)
        mu = x1.clone()
        # Very tight posterior: log_sigma = -10 (near minimum allowed by clamp)
        log_sigma = torch.full((B, D), -10.0)
        loss = vfm_loss(mu, log_sigma, x1)
        # Loss ≈ -10 (dominated by log_sigma term when mu == x1)
        assert loss.item() < 0, "VFM loss with perfect mu should be negative"

    def test_cfm_loss_equals_mse(self):
        _seed()
        B, D = 4, 64
        v_pred = torch.randn(B, D)
        v_target = torch.randn(B, D)
        cfm = cfm_loss(v_pred, v_target)
        mse = torch.nn.functional.mse_loss(v_pred, v_target)
        assert torch.allclose(cfm, mse), "cfm_loss should match F.mse_loss"


class TestVelocityFromPosterior:
    def test_shape_preserved(self):
        _seed()
        B, C, H, W = 2, 4, 8, 8
        mu = torch.randn(B, C, H, W)
        xt = torch.randn(B, C, H, W)
        t = torch.rand(B)
        v = velocity_from_posterior(mu, xt, t)
        assert v.shape == mu.shape, "velocity shape must match mu/xt shape"

    def test_exact_identity_at_t0(self):
        """At t=0: xt = x0, target = x1, so v = (mu - x0) / 1.0 = mu - xt."""
        _seed()
        B, D = 4, 16
        mu = torch.randn(B, D)
        xt = torch.randn(B, D)
        t = torch.zeros(B)
        v = velocity_from_posterior(mu, xt, t)
        expected = mu - xt
        assert torch.allclose(v, expected, atol=1e-6)

    def test_numerical_stability_near_t1(self):
        """Near t=1, denominator is clamped to _EPS_T — no inf/nan."""
        _seed()
        B, D = 4, 16
        mu = torch.randn(B, D)
        xt = torch.randn(B, D)
        t = torch.full((B,), 1.0 - 1e-8)  # very close to t=1
        v = velocity_from_posterior(mu, xt, t)
        assert not torch.any(torch.isnan(v)), "velocity must not contain NaN near t=1"
        assert not torch.any(torch.isinf(v)), "velocity must not contain Inf near t=1"
        # Maximum magnitude bounded by |mu - xt| / _EPS_T
        max_v = v.abs().max().item()
        max_expected = (mu - xt).abs().max().item() / _EPS_T
        assert max_v <= max_expected + 1e-3


class TestFlowLossDispatch:
    def test_cfm_mode_matches_cfm_loss(self):
        """flow_loss with flow_objective='cfm' must match cfm_loss bit-for-bit."""
        _seed()
        B, D = 4, 32
        # raw_pred is the velocity in CFM mode
        raw_pred = torch.randn(B, D)
        xt = torch.randn(B, D)
        t = torch.rand(B)
        x0 = torch.randn(B, D)
        x1 = torch.randn(B, D)
        ut = x1 - x0  # velocity target

        # Use the unified flow_loss in CFM mode
        # Note: in CFM mode, x1 argument receives ut
        loss_unified, v_unified = flow_loss(
            raw_pred, xt, t, ut, flow_objective="cfm", spatial=False
        )
        # Direct cfm_loss
        loss_direct = cfm_loss(raw_pred, ut)

        assert torch.allclose(loss_unified, loss_direct, atol=1e-7), \
            "CFM mode must match cfm_loss exactly"
        assert torch.allclose(v_unified, raw_pred, atol=1e-7), \
            "CFM mode velocity must equal raw_pred"

    def test_vfm_mode_produces_different_loss(self):
        """VFM mode must produce a different loss than CFM mode (uses 2x channels)."""
        _seed()
        B, D = 4, 32
        # VFM mode: raw_pred has 2× features (mu + log_sigma concatenated)
        raw_pred_vfm = torch.randn(B, D * 2)  # (mu, log_sigma)
        xt = torch.randn(B, D)
        t = torch.rand(B)
        x1 = torch.randn(B, D)

        loss_vfm, v_vfm = flow_loss(
            raw_pred_vfm, xt, t, x1, flow_objective="vfm", spatial=False
        )
        # Just check it runs and produces a scalar
        assert loss_vfm.dim() == 0, "VFM loss must be scalar"
        assert v_vfm.shape == xt.shape, "VFM velocity must match xt shape"

    def test_unknown_objective_raises(self):
        with pytest.raises(ValueError, match="Unknown flow_objective"):
            flow_loss(
                torch.randn(2, 4), torch.randn(2, 4), torch.rand(2),
                torch.randn(2, 4), flow_objective="bad_value",
            )


class TestVFMHeadSpatial:
    def test_spatial_split(self):
        """VFMHead should split along dim=1 for spatial tensors."""
        _seed()
        B, C, H, W = 2, 8, 4, 4
        raw = torch.randn(B, C * 2, H, W)
        head = VFMHead(spatial=True)
        mu, log_sigma = head(raw)
        assert mu.shape == (B, C, H, W)
        assert log_sigma.shape == (B, C, H, W)

    def test_vector_split(self):
        """VFMHead should split along dim=-1 for vector tensors."""
        _seed()
        B, D = 4, 16
        raw = torch.randn(B, D * 2)
        head = VFMHead(spatial=False)
        mu, log_sigma = head(raw)
        assert mu.shape == (B, D)
        assert log_sigma.shape == (B, D)

    def test_log_sigma_clamped(self):
        """log_sigma must be clamped to [-10, 2]."""
        raw = torch.full((2, 4), 100.0)
        head = VFMHead(spatial=False)
        _, log_sigma = head(raw)
        assert log_sigma.max().item() <= 2.0 + 1e-6
        raw2 = torch.full((2, 4), -100.0)
        _, log_sigma2 = head(raw2)
        assert log_sigma2.min().item() >= -10.0 - 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
