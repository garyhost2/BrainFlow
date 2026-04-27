"""Tests for FlowUNet with and without clip_ctx (back-compat)."""
import sys
from pathlib import Path
import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brainflow.config import Config
from brainflow.flow_unet import FlowUNet


def _make_cfg() -> Config:
    cfg = Config()
    cfg.latent_ch = 4
    cfg.latent_res = 8      # tiny for tests (normally 32)
    cfg.unet_base_ch = 16   # small for tests (normally 128)
    cfg.brain_dim = 32      # tiny
    cfg.n_tokens = 4
    cfg.attn_heads = 2
    cfg.time_emb_dim = 32
    cfg.method = "baseline"
    return cfg


def _make_batch(cfg: Config, B: int = 2, device: str = "cpu"):
    x = torch.randn(B, cfg.latent_ch, cfg.latent_res, cfg.latent_res, device=device)
    t = torch.rand(B, device=device)
    brain_ctx = torch.randn(B, cfg.n_tokens, cfg.brain_dim, device=device)
    # CLIP context: (B, 256, 1024) — ViT-L/14 patch tokens
    clip_ctx = torch.randn(B, 16, 1024, device=device)  # using 16 tokens for speed
    return x, t, brain_ctx, clip_ctx


class TestFlowUNetBackwardCompat:
    """FlowUNet must produce valid output when clip_ctx=None."""

    def test_forward_no_clip_ctx(self):
        cfg = _make_cfg()
        model = FlowUNet(cfg)
        model.eval()
        x, t, brain_ctx, _ = _make_batch(cfg)
        with torch.no_grad():
            out = model(x, t, brain_ctx, clip_ctx=None)
        assert out.shape == x.shape, "Output shape must match input latent shape"

    def test_forward_no_clip_ctx_no_nan(self):
        cfg = _make_cfg()
        model = FlowUNet(cfg)
        model.eval()
        x, t, brain_ctx, _ = _make_batch(cfg)
        with torch.no_grad():
            out = model(x, t, brain_ctx, clip_ctx=None)
        assert not torch.any(torch.isnan(out)), "Output must not contain NaN (no clip_ctx)"

    def test_backward_no_clip_ctx(self):
        cfg = _make_cfg()
        model = FlowUNet(cfg)
        model.train()
        x, t, brain_ctx, _ = _make_batch(cfg)
        out = model(x, t, brain_ctx, clip_ctx=None)
        loss = out.mean()
        loss.backward()
        # Check at least one param has gradient
        assert model.op.bias.grad is not None

    def test_null_tokens_shape(self):
        cfg = _make_cfg()
        model = FlowUNet(cfg)
        # null_tokens should be (1, n_tokens, brain_dim)
        assert model.null_tokens.shape == (1, cfg.n_tokens, cfg.brain_dim)


class TestFlowUNetWithClipCtx:
    """FlowUNet with clip_ctx must produce valid output and different activations."""

    def test_forward_with_clip_ctx_shape(self):
        cfg = _make_cfg()
        model = FlowUNet(cfg)
        model.eval()
        x, t, brain_ctx, clip_ctx = _make_batch(cfg)
        with torch.no_grad():
            out = model(x, t, brain_ctx, clip_ctx=clip_ctx)
        assert out.shape == x.shape

    def test_forward_with_clip_ctx_no_nan(self):
        cfg = _make_cfg()
        model = FlowUNet(cfg)
        model.eval()
        x, t, brain_ctx, clip_ctx = _make_batch(cfg)
        with torch.no_grad():
            out = model(x, t, brain_ctx, clip_ctx=clip_ctx)
        assert not torch.any(torch.isnan(out))

    def test_backward_with_clip_ctx(self):
        cfg = _make_cfg()
        model = FlowUNet(cfg)
        model.train()
        x, t, brain_ctx, clip_ctx = _make_batch(cfg)
        out = model(x, t, brain_ctx, clip_ctx=clip_ctx)
        loss = out.mean()
        loss.backward()
        assert model.clip_context_proj.weight.grad is not None, \
            "clip_context_proj must receive gradient"

    def test_clip_ctx_changes_output(self):
        """With clip_ctx, output should differ from no-clip_ctx (not identical)."""
        torch.manual_seed(99)
        cfg = _make_cfg()
        model = FlowUNet(cfg)
        model.eval()
        x, t, brain_ctx, clip_ctx = _make_batch(cfg)
        with torch.no_grad():
            out_no_clip = model(x, t, brain_ctx, clip_ctx=None)
            out_with_clip = model(x, t, brain_ctx, clip_ctx=clip_ctx)
        # Outputs should differ (CLIP context affects the activations)
        assert not torch.allclose(out_no_clip, out_with_clip), \
            "clip_ctx must change the output (different activations)"

    def test_clip_context_proj_shape(self):
        cfg = _make_cfg()
        model = FlowUNet(cfg)
        # clip_context_proj maps 1024 → brain_dim
        assert model.clip_context_proj.in_features == 1024
        assert model.clip_context_proj.out_features == cfg.brain_dim


class TestFlowUNetHRF:
    """HRF method still works without clip_ctx."""

    def test_forward_hrf_no_clip(self):
        cfg = _make_cfg()
        cfg.method = "hrf"
        cfg.hrf_len = cfg.n_tokens
        model = FlowUNet(cfg)
        model.eval()
        x, t, brain_ctx, _ = _make_batch(cfg)
        with torch.no_grad():
            out = model(x, t, brain_ctx, clip_ctx=None)
        assert out.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
