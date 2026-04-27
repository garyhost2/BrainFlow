"""Tests for FlowCLIPDiT per-channel CLIP standardization round-trip."""
import sys
from pathlib import Path
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brainflow.config import Config
from brainflow.flow_clip_dit import FlowCLIPDiT, CLIP_TOKEN_DIM, CLIP_GRID_H, CLIP_GRID_W


def _make_cfg() -> Config:
    cfg = Config()
    cfg.dit_dim = 64
    cfg.dit_depth = 1
    cfg.dit_heads = 4
    cfg.time_emb_dim = 64
    cfg.brain_dim = 64
    cfg.n_tokens = 4
    cfg.flow_objective = "cfm"
    cfg.lambda_cos = 0.1
    return cfg


class TestCLIPStandardization:
    def test_initial_buffers_are_identity(self):
        """Before fitting, standardize/destandardize should be near-identity."""
        cfg = _make_cfg()
        model = FlowCLIPDiT(cfg)
        x = torch.randn(2, CLIP_GRID_H, CLIP_GRID_W, CLIP_TOKEN_DIM)
        x_std = model.standardize(x)
        x_back = model.destandardize(x_std)
        max_err = (x - x_back).abs().max().item()
        assert max_err < 1e-5, f"Round-trip error {max_err:.2e} > 1e-5 before fit"

    def test_fit_standardization(self):
        """After fitting, clip_token_mean and clip_token_std should reflect data stats."""
        torch.manual_seed(42)
        cfg = _make_cfg()
        model = FlowCLIPDiT(cfg)

        # Create synthetic data with known mean/std
        N = 200
        true_mean = torch.linspace(-1, 1, CLIP_TOKEN_DIM)
        true_std  = torch.linspace(0.5, 2.0, CLIP_TOKEN_DIM)
        # (N, 16, 16, 1024) = (N, H, W, C)
        data = true_mean + true_std * torch.randn(N, CLIP_GRID_H, CLIP_GRID_W, CLIP_TOKEN_DIM)

        model.fit_standardization(data)

        # Check stored mean/std are close to true values
        mean_err = (model.clip_token_mean - true_mean).abs().max().item()
        std_err  = (model.clip_token_std  - true_std ).abs().max().item()
        # With N=200 and ~256 tokens per sample we have 200*256=51200 samples per channel
        assert mean_err < 0.1, f"Mean error {mean_err:.4f} too large"
        assert std_err  < 0.1, f"Std error  {std_err:.4f} too large"

    def test_standardize_destandardize_roundtrip_after_fit(self):
        """standardize → destandardize must have max abs error < 1e-5."""
        torch.manual_seed(0)
        cfg = _make_cfg()
        model = FlowCLIPDiT(cfg)

        # Fit with random data
        data = torch.randn(100, CLIP_GRID_H, CLIP_GRID_W, CLIP_TOKEN_DIM)
        model.fit_standardization(data)

        # Test round-trip on a fresh batch
        x = torch.randn(4, CLIP_GRID_H, CLIP_GRID_W, CLIP_TOKEN_DIM)
        x_std = model.standardize(x)
        x_back = model.destandardize(x_std)

        max_err = (x - x_back).abs().max().item()
        assert max_err < 1e-5, f"Round-trip max abs error {max_err:.2e} exceeds 1e-5"

    def test_standardize_output_shape(self):
        cfg = _make_cfg()
        model = FlowCLIPDiT(cfg)
        x = torch.randn(3, CLIP_GRID_H, CLIP_GRID_W, CLIP_TOKEN_DIM)
        assert model.standardize(x).shape == x.shape

    def test_destandardize_output_shape(self):
        cfg = _make_cfg()
        model = FlowCLIPDiT(cfg)
        x = torch.randn(3, CLIP_GRID_H, CLIP_GRID_W, CLIP_TOKEN_DIM)
        assert model.destandardize(x).shape == x.shape

    def test_standardized_data_has_zero_mean_unit_std(self):
        """After fitting and standardizing training data, it should be ~N(0,1)."""
        torch.manual_seed(7)
        cfg = _make_cfg()
        model = FlowCLIPDiT(cfg)

        N = 500
        data = torch.randn(N, CLIP_GRID_H, CLIP_GRID_W, CLIP_TOKEN_DIM)
        model.fit_standardization(data)

        # Standardize the same data that was used for fitting
        data_std = model.standardize(data)
        flat = data_std.reshape(-1, CLIP_TOKEN_DIM)
        channel_mean = flat.mean(0).abs().max().item()
        channel_std  = (flat.std(0) - 1.0).abs().max().item()

        assert channel_mean < 0.05, f"Standardized mean not near 0: {channel_mean:.4f}"
        assert channel_std  < 0.1,  f"Standardized std not near 1: {channel_std:.4f}"

    def test_buffers_move_with_model(self):
        """clip_token_mean and clip_token_std should be proper buffers (registered)."""
        cfg = _make_cfg()
        model = FlowCLIPDiT(cfg)
        buffer_names = {name for name, _ in model.named_buffers()}
        assert "clip_token_mean" in buffer_names
        assert "clip_token_std" in buffer_names

    def test_fit_accepts_3d_input(self):
        """fit_standardization should work with (N, 256, 1024) flat token arrays."""
        cfg = _make_cfg()
        model = FlowCLIPDiT(cfg)
        data = torch.randn(100, CLIP_GRID_H * CLIP_GRID_W, CLIP_TOKEN_DIM)
        # Should not raise
        model.fit_standardization(data)
        assert model._standardization_fitted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
