"""Smoke tests for Flow_CLIP DiT: forward/backward on dummy input."""
import sys
from pathlib import Path
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brainflow.config import Config
from brainflow.flow_clip_dit import FlowCLIPDiT, CLIP_GRID_H, CLIP_GRID_W, CLIP_TOKEN_DIM


def _make_cfg(flow_objective: str = "vfm") -> Config:
    cfg = Config()
    cfg.dit_dim = 64       # tiny for tests
    cfg.dit_depth = 2
    cfg.dit_heads = 4
    cfg.time_emb_dim = 64
    cfg.brain_dim = 64
    cfg.n_tokens = 8
    cfg.flow_objective = flow_objective
    cfg.lambda_cos = 0.1
    return cfg


def _make_batch(B: int = 2, device: str = "cpu"):
    """Create dummy (B, 1024, 16, 16) CLIP grid + (B, n_tokens, brain_dim) brain ctx."""
    torch.manual_seed(0)
    clip_grid = torch.randn(B, CLIP_TOKEN_DIM, CLIP_GRID_H, CLIP_GRID_W, device=device)
    brain_ctx = torch.randn(B, 8, 64, device=device)   # n_tokens=8, brain_dim=64
    return clip_grid, brain_ctx


class TestFlowCLIPDiTForward:
    def test_forward_vfm_output_shape(self):
        """Forward pass in VFM mode should produce 2× output channels."""
        cfg = _make_cfg("vfm")
        model = FlowCLIPDiT(cfg)
        model.eval()
        B = 2
        clip_grid, brain_ctx = _make_batch(B)
        t = torch.rand(B)
        out = model(clip_grid, t, brain_ctx)
        # VFM: 2× channels (mu + log_sigma)
        assert out.shape == (B, CLIP_TOKEN_DIM * 2, CLIP_GRID_H, CLIP_GRID_W), \
            f"Expected {(B, CLIP_TOKEN_DIM * 2, CLIP_GRID_H, CLIP_GRID_W)}, got {out.shape}"

    def test_forward_cfm_output_shape(self):
        """Forward pass in CFM mode should produce 1× output channels."""
        cfg = _make_cfg("cfm")
        model = FlowCLIPDiT(cfg)
        model.eval()
        B = 2
        clip_grid, brain_ctx = _make_batch(B)
        t = torch.rand(B)
        out = model(clip_grid, t, brain_ctx)
        assert out.shape == (B, CLIP_TOKEN_DIM, CLIP_GRID_H, CLIP_GRID_W), \
            f"Expected {(B, CLIP_TOKEN_DIM, CLIP_GRID_H, CLIP_GRID_W)}, got {out.shape}"

    def test_forward_backward_vfm(self):
        """Backward pass should not raise in VFM mode."""
        cfg = _make_cfg("vfm")
        model = FlowCLIPDiT(cfg)
        model.train()
        B = 2
        clip_grid, brain_ctx = _make_batch(B)
        t = torch.rand(B)
        out = model(clip_grid, t, brain_ctx)
        loss = out.mean()
        loss.backward()
        # Check that gradients exist on key parameters
        assert model.patch_embed.weight.grad is not None

    def test_forward_backward_cfm(self):
        """Backward pass should not raise in CFM mode."""
        cfg = _make_cfg("cfm")
        model = FlowCLIPDiT(cfg)
        model.train()
        B = 2
        clip_grid, brain_ctx = _make_batch(B)
        t = torch.rand(B)
        out = model(clip_grid, t, brain_ctx)
        loss = out.mean()
        loss.backward()
        assert model.patch_embed.weight.grad is not None


class TestFlowCLIPDiTFlowLoss:
    def test_flow_loss_vfm(self):
        """flow_loss() in VFM mode should return a dict with 'loss' key."""
        cfg = _make_cfg("vfm")
        model = FlowCLIPDiT(cfg)
        model.train()
        B = 2
        clip_grid, brain_ctx = _make_batch(B)
        result = model.flow_loss(clip_grid, brain_ctx)
        assert "loss" in result
        assert result["loss"].dim() == 0
        assert not torch.isnan(result["loss"]), "VFM loss must not be NaN"

    def test_flow_loss_cfm(self):
        """flow_loss() in CFM mode should return a dict with 'loss' key."""
        cfg = _make_cfg("cfm")
        model = FlowCLIPDiT(cfg)
        model.train()
        B = 2
        clip_grid, brain_ctx = _make_batch(B)
        result = model.flow_loss(clip_grid, brain_ctx)
        assert "loss" in result
        assert result["loss"].dim() == 0

    def test_flow_loss_backward(self):
        """flow_loss() backward should not raise."""
        cfg = _make_cfg("vfm")
        model = FlowCLIPDiT(cfg)
        model.train()
        B = 2
        clip_grid, brain_ctx = _make_batch(B)
        result = model.flow_loss(clip_grid, brain_ctx)
        result["loss"].backward()
        assert model.patch_embed.weight.grad is not None


class TestFlowCLIPDiTStandardization:
    def test_standardization_buffers_initialized(self):
        cfg = _make_cfg()
        model = FlowCLIPDiT(cfg)
        assert model.clip_token_mean.shape == (CLIP_TOKEN_DIM,)
        assert model.clip_token_std.shape == (CLIP_TOKEN_DIM,)
        # Before fitting: std should be ones, mean zeros
        assert torch.allclose(model.clip_token_std, torch.ones(CLIP_TOKEN_DIM))
        assert torch.allclose(model.clip_token_mean, torch.zeros(CLIP_TOKEN_DIM))


class TestFlowCLIPDiTSampling:
    def test_sample_shape(self):
        cfg = _make_cfg("cfm")
        model = FlowCLIPDiT(cfg)
        model.eval()
        B = 2
        _, brain_ctx = _make_batch(B)
        with torch.no_grad():
            out = model.sample(brain_ctx, n_steps=2)
        assert out.shape == (B, CLIP_TOKEN_DIM, CLIP_GRID_H, CLIP_GRID_W)

    def test_sample_no_nan(self):
        cfg = _make_cfg("cfm")
        model = FlowCLIPDiT(cfg)
        model.eval()
        B = 2
        _, brain_ctx = _make_batch(B)
        with torch.no_grad():
            out = model.sample(brain_ctx, n_steps=5)
        assert not torch.any(torch.isnan(out)), "Sampled tokens must not contain NaN"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
