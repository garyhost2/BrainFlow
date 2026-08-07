"""Blurry-latent target cache -> loader -> training step, offline.

No GPU, no VAE, no dataset: the cache blob is fabricated in the exact format
:func:`build_or_load_latent_targets` writes, so a silent break in the format or
the loader wiring is caught here rather than on the cluster an hour into a job.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brainflow.step1.targets_latent import (build_or_load_latent_targets,
                                            _subj_file, _accum_stats)
from brainflow.step1.data import build_step1_loaders
from brainflow.step1.model_tokens import TokenStep1Config, TokenStep1Model

N_TR, N_TE, V = 20, 8, 30


def _write_cache(dirpath, subj=1, ll_size=64, mean=1.0, std=3.0):
    lat_tr = (torch.randn(N_TR, 4, 96, 96) * std + mean).half()
    lat_te = (torch.randn(N_TE, 4, 96, 96) * std + mean).half()
    cnt, s, sq = _accum_stats(lat_tr)
    torch.save({"lat_train": lat_tr, "lat_test": lat_te, "ll_size": ll_size,
                "accum": {"count": cnt, "sum": s, "sqsum": sq}},
               _subj_file(dirpath, subj, ll_size))


def _fake_data():
    tensors = {
        "fmri_train_1": torch.randn(N_TR, V), "fmri_test_1": torch.randn(N_TE, V),
        "imgs_train_1": torch.rand(N_TR, 3, 32, 32),
        "imgs_test_1": torch.rand(N_TE, 3, 32, 32),
        "voxels": {1: V},
        "fmri_stats": {1: {"mu": torch.zeros(V), "std": torch.ones(V)}},
    }
    targets = {"emb_train_1": torch.randn(N_TR, 4, 8),
               "emb_test_1": torch.randn(N_TE, 4, 8),
               "cls_train_1": torch.randn(N_TR, 10),
               "cls_test_1": torch.randn(N_TE, 10)}
    return tensors, targets


def _tiny_latent_model(stats):
    cfg = TokenStep1Config(
        token_len=4, token_dim=8, cls_dim=10, brain_dim=6, n_brain_tokens=3,
        enc_hidden=16, enc_blocks=1, reg_depth=1, reg_heads=2,
        prior_width=16, prior_depth=1, prior_heads=2, time_dim=8,
        cls_width=16, cls_depth=1, cls_heads=2, ll_target="latent",
        ll_hidden=32, ll_base=32, n_steps=4, subjects=[1])
    m = TokenStep1Model(cfg, {1: V})
    m.set_target_mean(torch.zeros(8))
    m.low_head.set_latent_stats(stats["mean"], stats["std"])
    return m


@pytest.fixture
def cache_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


class TestLatentCache:
    def test_stats_recover_channel_mean_and_std(self, cache_dir):
        _write_cache(cache_dir, mean=1.0, std=3.0)
        tensors, _ = _fake_data()
        out = build_or_load_latent_targets(tensors, [1], cache_dir, decoder=None,
                                           ll_size=64)
        st = out["_lat_stats"]
        assert torch.allclose(st["mean"], torch.full((4,), 1.0), atol=0.15)
        assert torch.allclose(st["std"], torch.full((4,), 3.0), atol=0.15)

    def test_missing_cache_without_decoder_is_a_clear_error(self, cache_dir):
        tensors, _ = _fake_data()
        with pytest.raises(FileNotFoundError, match="build_latent_targets"):
            build_or_load_latent_targets(tensors, [1], cache_dir, decoder=None)

    def test_blur_levels_do_not_collide(self, cache_dir):
        """A cache blurred at 64 must not be picked up for a 128 run.

        Each blur level gets its own filename, so the 128 request simply finds
        nothing -- it must not silently reuse the 64 blob.
        """
        _write_cache(cache_dir, ll_size=64)
        tensors, _ = _fake_data()
        with pytest.raises(FileNotFoundError):
            build_or_load_latent_targets(tensors, [1], cache_dir, decoder=None,
                                         ll_size=128)
        assert build_or_load_latent_targets(tensors, [1], cache_dir, decoder=None,
                                            ll_size=64)["_lat_stats"]["ll_size"] == 64

    def test_contradictory_inner_ll_size_is_refused(self, cache_dir):
        """Defends against a hand-copied/renamed cache whose contents disagree."""
        _write_cache(cache_dir, ll_size=64)
        fp = _subj_file(cache_dir, 1, 64)
        blob = torch.load(str(fp), map_location="cpu")
        blob["ll_size"] = 128                      # filename says 64, contents say 128
        torch.save(blob, fp)
        tensors, _ = _fake_data()
        with pytest.raises(ValueError, match="ll_size"):
            build_or_load_latent_targets(tensors, [1], cache_dir, decoder=None,
                                         ll_size=64)


class TestLoaderWiring:
    def test_lat_reaches_the_batch(self, cache_dir):
        _write_cache(cache_dir)
        tensors, targets = _fake_data()
        targets |= build_or_load_latent_targets(tensors, [1], cache_dir, decoder=None)
        bundle = build_step1_loaders(tensors, targets, [1], batch_size=4,
                                     num_workers=0, val_frac=0.25)
        for loader in (bundle.train, bundle.val, bundle.eval):
            batch = next(iter(loader))
            assert "lat" in batch
            assert batch["lat"].shape[1:] == (4, 96, 96)

    def test_no_lat_key_when_cache_absent(self):
        """Every existing rgb/prior-only run must keep working untouched."""
        tensors, targets = _fake_data()
        bundle = build_step1_loaders(tensors, targets, [1], batch_size=4, num_workers=0)
        assert "lat" not in next(iter(bundle.train))


class TestTrainingStep:
    def test_standardized_target_is_unit_scale(self, cache_dir):
        _write_cache(cache_dir, mean=1.0, std=3.0)
        tensors, targets = _fake_data()
        targets |= build_or_load_latent_targets(tensors, [1], cache_dir, decoder=None)
        bundle = build_step1_loaders(tensors, targets, [1], batch_size=8, num_workers=0)
        m = _tiny_latent_model(targets["_lat_stats"])
        z = m.low_head.standardize(next(iter(bundle.eval))["lat"])
        assert abs(z.std().item() - 1.0) < 0.2
        assert abs(z.mean().item()) < 0.2

    def test_untrained_l1_sits_at_the_predict_the_mean_line(self, cache_dir):
        """The head is zero-init, so `low` starts at E|N(0,1)| = 0.798.

        That makes the logged number self-calibrating: below ~0.798 the head has
        learned something, at 0.798 it is predicting the channel mean. The RGB
        runs needed an external reference (blur_mse ~0.07) to see this.
        """
        _write_cache(cache_dir)
        tensors, targets = _fake_data()
        targets |= build_or_load_latent_targets(tensors, [1], cache_dir, decoder=None)
        bundle = build_step1_loaders(tensors, targets, [1], batch_size=8, num_workers=0)
        m = _tiny_latent_model(targets["_lat_stats"])
        batch = next(iter(bundle.eval))
        low = m._low_level_loss(batch["fmri"], 1, None, batch["lat"])["low"]
        assert abs(low.item() - 0.798) < 0.08

    def test_full_and_low_only_steps_both_train(self, cache_dir):
        _write_cache(cache_dir)
        tensors, targets = _fake_data()
        targets |= build_or_load_latent_targets(tensors, [1], cache_dir, decoder=None)
        bundle = build_step1_loaders(tensors, targets, [1], batch_size=4, num_workers=0)
        m = _tiny_latent_model(targets["_lat_stats"])
        batch = next(iter(bundle.train))

        out = m.training_step(batch["fmri"], batch["subject"], None,
                              target_raw=batch["emb"], target_cls=batch["cls"],
                              target_img=batch["image"], target_lat=batch["lat"])
        assert torch.isfinite(out["loss"])
        out["loss"].backward()
        assert m.low_head.out.weight.grad is not None

        only = m.training_step(batch["fmri"], batch["subject"], None,
                               target_lat=batch["lat"], low_only=True)
        assert set(only) == {"low", "loss"}

    def test_low_only_without_a_latent_target_fails_loudly(self, cache_dir):
        _write_cache(cache_dir)
        tensors, targets = _fake_data()
        targets |= build_or_load_latent_targets(tensors, [1], cache_dir, decoder=None)
        m = _tiny_latent_model(targets["_lat_stats"])
        with pytest.raises(ValueError, match="target_lat"):
            m.training_step(torch.randn(2, V), 1, None,
                            target_img=torch.rand(2, 3, 32, 32), low_only=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
