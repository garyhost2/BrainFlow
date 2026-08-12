"""Tests for the eval_full harness.

Runs evaluate_full on tiny random CPU tensors — no GPU, dataset, or checkpoint
required.  Only PixCorr and SSIM are exercised (all network-dependent metrics
are skipped).
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brainflow.metrics_full import (evaluate_full, two_way_identification,
                                    correlation_distance, retrieval_metrics,
                                    ssim_grayscale, format_comparison)

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
            # The script prints non-ASCII; a cp1252 console (Windows) would raise
            # UnicodeEncodeError on the print, not on anything being measured.
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
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
            # The script prints non-ASCII; a cp1252 console (Windows) would raise
            # UnicodeEncodeError on the print, not on anything being measured.
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        assert "PixCorr" in result.stdout
        assert "SSIM" in result.stdout


class TestTwoWayIdentification:
    def test_perfect_alignment_is_one(self):
        feats = torch.eye(8)
        assert two_way_identification(feats, feats.clone()) == 1.0

    def test_random_alignment_is_near_half(self):
        torch.manual_seed(0)
        p = torch.randn(128, 64)
        t = torch.randn(128, 64)
        score = two_way_identification(p, t)
        assert 0.35 <= score <= 0.65


class TestEvaluateFullTwoWayKeys:
    def test_returns_2way_keys(self, monkeypatch):
        def _pair(pred, target, device):
            B = pred.shape[0]
            x = torch.eye(B, dtype=pred.dtype, device=pred.device)
            return x, x.clone()

        monkeypatch.setattr("brainflow.metrics_full._alexnet_feature_pair",
                            lambda pred, target, layer, device: _pair(pred, target, device))
        monkeypatch.setattr("brainflow.metrics_full._inception_feature_pair", _pair)
        monkeypatch.setattr("brainflow.metrics_full._clip_feature_pair", _pair)
        monkeypatch.setattr("brainflow.metrics_full._effnet_feature_pair", _pair)
        monkeypatch.setattr("brainflow.metrics_full._swav_feature_pair", _pair)

        pred = torch.rand(4, 3, 64, 64)
        target = torch.rand(4, 3, 64, 64)
        result = evaluate_full(pred, target, device="cpu")
        expected = {
            "AlexNet(2)_2way", "AlexNet(5)_2way", "Inception_2way",
            "CLIP_2way", "EffNet-B_2way", "SwAV_2way",
        }
        assert expected <= result.keys()


class TestCorrelationDistance:
    """EffNet-B / SwAV are DISTANCES in the NSD tables: lower = better."""

    def test_identical_features_are_zero(self):
        x = torch.randn(16, 64)
        assert correlation_distance(x, x.clone()) == pytest.approx(0.0, abs=1e-5)

    def test_random_features_near_one(self):
        torch.manual_seed(0)
        d = correlation_distance(torch.randn(256, 512), torch.randn(256, 512))
        assert 0.9 <= d <= 1.1

    def test_better_prediction_has_smaller_distance(self):
        torch.manual_seed(0)
        target = torch.randn(64, 128)
        close = target + 0.1 * torch.randn_like(target)
        far = target + 2.0 * torch.randn_like(target)
        assert correlation_distance(close, target) < correlation_distance(far, target)

    def test_evaluate_full_reports_effnet_as_distance(self, monkeypatch):
        """A perfect reconstruction must give EffNet-B ~0, not ~1."""
        def _pair(pred, target, device):
            x = torch.randn(pred.shape[0], 32)
            return x, x.clone()
        monkeypatch.setattr("brainflow.metrics_full._effnet_feature_pair", _pair)
        monkeypatch.setattr("brainflow.metrics_full._swav_feature_pair", _pair)
        x = torch.rand(4, 3, 64, 64)
        result = evaluate_full(x, x.clone(), device="cpu",
                               skip=["AlexNet(2)", "AlexNet(5)", "Inception", "CLIP"])
        assert result["EffNet-B"] < 0.01
        assert result["SwAV"] < 0.01


class TestSSIMGrayscale:
    def test_identical_is_one(self):
        x = torch.rand(4, 3, 64, 64)
        assert ssim_grayscale(x, x.clone()) == pytest.approx(1.0, abs=1e-4)

    def test_noise_is_low(self):
        torch.manual_seed(0)
        assert ssim_grayscale(torch.rand(4, 3, 64, 64), torch.rand(4, 3, 64, 64)) < 0.2

    def test_valid_window_not_zero_padded(self):
        """A uniform image pair scores 1.0; zero-padded borders would drag it down."""
        x = torch.full((2, 3, 32, 32), 0.7)
        assert ssim_grayscale(x, x.clone()) == pytest.approx(1.0, abs=1e-4)


class TestRetrieval:
    def test_perfect_prediction_is_one(self):
        torch.manual_seed(0)
        emb = torch.randn(64, 8, 16)
        r = retrieval_metrics(emb, emb.clone(), batch_size=32)
        assert r["retrieval_fwd"] == 1.0 and r["retrieval_bwd"] == 1.0

    def test_random_prediction_is_chance(self):
        torch.manual_seed(0)
        r = retrieval_metrics(torch.randn(300, 128), torch.randn(300, 128),
                              batch_size=300)
        assert r["retrieval_fwd"] < 0.05          # chance = 1/300
        assert r["retrieval_bwd"] < 0.05

    def test_drops_short_trailing_pool(self):
        """982 images at pool=300 must score 3 pools of 300, not 3 + a soft 82."""
        emb = torch.randn(982, 64)
        r = retrieval_metrics(emb, emb.clone(), batch_size=300)
        assert r["retrieval_pools"] == 3 and r["retrieval_pool"] == 300

    def test_pool_smaller_than_n(self):
        emb = torch.randn(10, 64)
        r = retrieval_metrics(emb, emb.clone(), batch_size=300)
        assert r["retrieval_pool"] == 10 and r["retrieval_pools"] == 1


class TestComparisonTable:
    def test_renders_our_row_and_references(self):
        metrics = {"PixCorr": 0.164, "SSIM": 0.348, "CLIP_2way": 0.703,
                   "EffNet-B": 0.85, "SwAV": 0.60}
        table = format_comparison(metrics)
        assert "**ours**" in table and "MindEye2" in table
        assert "0.164" in table
        assert "--" in table          # AlexNet columns are absent -> placeholder


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
