"""Tests for minibatch OT coupling in brainflow/ot_coupling.py.

All tests run on CPU without network access.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brainflow.ot_coupling import ot_minibatch_coupling


B, D = 8, 16   # small batch and feature dim for fast CPU tests


class TestOTMinibatchCoupling:
    def test_output_shape(self):
        x0 = torch.randn(B, D)
        x1 = torch.randn(B, D)
        out = ot_minibatch_coupling(x0, x1)
        assert out.shape == x0.shape

    def test_output_dtype(self):
        x0 = torch.randn(B, D)
        x1 = torch.randn(B, D)
        out = ot_minibatch_coupling(x0, x1)
        assert out.dtype == x0.dtype

    def test_valid_permutation_of_x0_rows(self):
        """Each row of the output must be one of the rows of x0."""
        x0 = torch.randn(B, D)
        x1 = torch.randn(B, D)
        out = ot_minibatch_coupling(x0, x1)
        for row in out:
            match = (x0 - row.unsqueeze(0)).abs().sum(dim=-1).min().item()
            assert match < 1e-5, f"Row {row} not found in x0"

    def test_reduces_transport_cost_vs_identity(self):
        """OT reordering should not increase total squared transport cost."""
        torch.manual_seed(42)
        # Create x0, x1 with a known structure: x0[i] is close to x1[B-1-i]
        # (reversed order), so OT should reverse x0 to match x1
        x1 = torch.randn(B, D)
        x0_reversed = x1.flip(0) + 0.01 * torch.randn(B, D)

        # Identity pairing cost
        cost_identity = ((x0_reversed - x1) ** 2).sum(dim=-1).sum().item()

        # OT-coupled cost
        x0_coupled = ot_minibatch_coupling(x0_reversed, x1, n_iters=100)
        cost_coupled = ((x0_coupled - x1) ** 2).sum(dim=-1).sum().item()

        assert cost_coupled <= cost_identity + 1e-3, (
            f"OT cost {cost_coupled:.4f} > identity cost {cost_identity:.4f}"
        )

    def test_4d_tensor_input(self):
        """Should work with latent tensors of shape (B, C, H, W)."""
        x0 = torch.randn(B, 4, 8, 8)
        x1 = torch.randn(B, 4, 8, 8)
        out = ot_minibatch_coupling(x0, x1)
        assert out.shape == (B, 4, 8, 8)

    def test_no_grad_output(self):
        """Output should be detached (no gradient)."""
        x0 = torch.randn(B, D, requires_grad=True)
        x1 = torch.randn(B, D, requires_grad=True)
        out = ot_minibatch_coupling(x0, x1)
        assert not out.requires_grad

    def test_unsupported_p_raises(self):
        x0 = torch.randn(B, D)
        x1 = torch.randn(B, D)
        with pytest.raises(ValueError, match="p=2"):
            ot_minibatch_coupling(x0, x1, p=1)

    def test_deterministic_with_same_input(self):
        """Same inputs should yield the same assignment."""
        torch.manual_seed(0)
        x0 = torch.randn(B, D)
        x1 = torch.randn(B, D)
        out1 = ot_minibatch_coupling(x0.clone(), x1.clone())
        out2 = ot_minibatch_coupling(x0.clone(), x1.clone())
        assert torch.allclose(out1, out2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
