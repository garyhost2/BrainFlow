"""Tests for the eval_full harness.

Runs evaluate_full on tiny random CPU tensors — no GPU, dataset, or checkpoint
required.  Only PixCorr and SSIM are exercised (all network-dependent metrics
are skipped).
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brainflow.metrics_full import evaluate_full

_SKIP = ["AlexNet(2)", "AlexNet(5)", "Inception", "CLIP", "EffNet-B", "SwAV"]
_EXPECTED_KEYS = {"PixCorr", "SSIM"}


class TestEvaluateFullOffline:
    def _make_batch(self, B: int = 4, H: int = 64, W: int = 64):
        pred = torch.rand(B, 3, H, W)
        target = torch.rand(B, 3, H, W)
        return pred, target

    def test_returns_expected_keys(self):
        pred, target = self._make_batch()
        result = evaluate_full(pred, target, device="cpu", skip=_SKIP)
        assert _EXPECTED_KEYS <= result.keys(), (
            f"Missing keys: {_EXPECTED_KEYS - result.keys()}"
        )

    def test_all_values_finite(self):
        pred, target = self._make_batch()
        result = evaluate_full(pred, target, device="cpu", skip=_SKIP)
        for name, val in result.items():
            assert isinstance(val, float), f"{name} is not a float"
            assert val == val, f"{name} is NaN"  # NaN != NaN
            assert abs(val) < 1e9, f"{name}={val} suspiciously large"

    def test_pixcorr_identical_images(self):
        """PixCorr should be ~1.0 when pred == target."""
        x = torch.rand(4, 3, 64, 64)
        result = evaluate_full(x, x.clone(), device="cpu", skip=_SKIP)
        assert result["PixCorr"] > 0.99

    def test_ssim_identical_images(self):
        """SSIM should be ~1.0 when pred == target."""
        x = torch.rand(4, 3, 64, 64)
        result = evaluate_full(x, x.clone(), device="cpu", skip=_SKIP)
        assert result["SSIM"] > 0.99

    def test_pixcorr_random_images_less_than_identical(self):
        pred = torch.rand(4, 3, 64, 64)
        target = torch.rand(4, 3, 64, 64)
        r_random = evaluate_full(pred, target, device="cpu", skip=_SKIP)
        r_same = evaluate_full(pred, pred.clone(), device="cpu", skip=_SKIP)
        assert r_same["PixCorr"] > r_random["PixCorr"]

    def test_skip_works(self):
        pred, target = self._make_batch()
        result = evaluate_full(pred, target, device="cpu", skip=_SKIP)
        for skipped in _SKIP:
            assert skipped not in result, f"Skipped metric {skipped!r} appeared in results"


class TestSelfTestScript:
    """End-to-end test of the --self-test CLI path."""

    def test_self_test_exits_zero(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "scripts.eval_full", "--self-test"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert result.returncode == 0, (
            f"--self-test failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_self_test_prints_pixcorr(self):
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "scripts.eval_full", "--self-test"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        assert "PixCorr" in result.stdout
        assert "SSIM" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
