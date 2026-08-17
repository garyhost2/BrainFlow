from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from rxfm.frontier import (FrontierPoint, format_surface, latent_frontier,
                                      pareto_front, parse_ks, predicted_frontier,
                                      predicted_resultant)
from rxfm.model import TokenStep1Config, TokenStep1Model
from rxfm.target_stats import TargetStats


VOX = {1: 96}


def tiny_cfg(**kw) -> TokenStep1Config:
    base = dict(
        token_len=6, token_dim=32, cls_dim=16, brain_dim=32, n_brain_tokens=4,
        enc_hidden=64, enc_blocks=1, reg_depth=1, reg_heads=2, radius_depth=1,
        prior_width=32, prior_depth=1, prior_heads=2, time_dim=16,
        cls_width=32, cls_depth=1, cls_heads=2, low_level=False, subjects=[1],
        flow_source="anchor_var", n_steps=3,
    )
    base.update(kw)
    return TokenStep1Config(**base)


def make(cfg):
    torch.manual_seed(0)
    m = TokenStep1Model(cfg, VOX)
    m.set_target_mean(torch.zeros(cfg.token_dim))
    return m.eval()


class _DS(Dataset):
    def __init__(self, cfg, n=24):
        torch.manual_seed(0)
        self.f = torch.randn(n, VOX[1])
        self.e = F.normalize(torch.randn(n, cfg.token_len, cfg.token_dim), dim=-1) * 2
        self.c = torch.randn(n, cfg.cls_dim)

    def __len__(self):
        return len(self.f)

    def __getitem__(self, i):
        return {"fmri": self.f[i], "emb": self.e[i], "cls": self.c[i], "subject": 1}


def _coll(b):
    return {"fmri": torch.stack([x["fmri"] for x in b]),
            "emb": torch.stack([x["emb"] for x in b]),
            "cls": torch.stack([x["cls"] for x in b]), "subject": 1}


def loader(cfg, bs=8):
    return DataLoader(_DS(cfg), batch_size=bs, collate_fn=_coll)


def stats_for(cfg):
    return TargetStats(mean=torch.zeros(cfg.token_dim), std=torch.ones(cfg.token_dim))


def test_parse_ks_accepts_anchor_and_dedupes():
    assert parse_ks(["0", "1", "4", "4", "anchor"]) == [0, 1, 4]
    assert parse_ks("1,2,8") == [1, 2, 8]


def test_predicted_frontier_matches_the_proposition():
    m = 0.370
    pred = predicted_frontier(m, [1, 2, 4, 1024])
    assert pred[1] == pytest.approx(m * m, rel=1e-9)
    assert pred[1024] == pytest.approx(m, rel=1e-2)
    assert pred[1] < pred[2] < pred[4] < pred[1024]


def test_predicted_frontier_reproduces_the_reported_table():
    expected = {1: 0.137, 2: 0.181, 4: 0.231, 8: 0.277, 16: 0.314, 32: 0.338}
    pred = predicted_frontier(0.370, expected)
    for k, want in expected.items():
        assert pred[k] == pytest.approx(want, abs=1e-3)


def test_predicted_resultant_interpolates_one_to_anchor():
    m = 0.370
    assert predicted_resultant(m, 1) == pytest.approx(1.0, abs=1e-9)
    assert predicted_resultant(m, 10 ** 6) == pytest.approx(m, rel=1e-3)


def test_predicted_frontier_is_monotone_and_bounded():
    for m in (0.05, 0.37, 0.9):
        pred = predicted_frontier(m, [1, 2, 4, 8, 16, 4096])
        vals = [pred[k] for k in sorted(pred)]
        assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))
        assert vals[0] >= m * m - 1e-9
        assert vals[-1] <= m + 1e-6


@pytest.mark.parametrize("ks", [[1], [0, 1, 2], [0, 1, 2, 4]])
def test_frontier_returns_every_requested_k(ks):
    cfg = tiny_cfg()
    m = make(cfg)
    fmri = torch.randn(5, VOX[1])
    got = m.predict_tokens_frontier(fmri, 1, stats_for(cfg),
                                    ks=[k for k in ks if k >= 1])
    for k in ks:
        key = "anchor" if k == 0 else k
        assert key in got
        assert got[key]["tokens"].shape == (5, cfg.token_len, cfg.token_dim)


def test_resultant_length_increases_with_k():
    cfg = tiny_cfg()
    m = make(cfg)
    fmri = torch.randn(6, VOX[1])
    got = m.predict_tokens_frontier(fmri, 1, stats_for(cfg), ks=[1, 2, 8, 32])
    r = [float(got[k]["resultant"].mean()) for k in (1, 2, 8, 32)]
    assert r[0] == pytest.approx(1.0, abs=1e-4)
    assert all(b <= a + 1e-4 for a, b in zip(r, r[1:]))


def test_k_averaging_reduces_dispersion():
    cfg = tiny_cfg(anchor_jitter_rad=0.6)
    m = make(cfg)
    fmri = torch.randn(8, VOX[1])
    got = m.predict_tokens_frontier(fmri, 1, stats_for(cfg), ks=[1, 16])
    spread_1 = float(got[1]["tokens"].std(dim=0).mean())
    spread_16 = float(got[16]["tokens"].std(dim=0).mean())
    assert spread_16 <= spread_1


def test_deterministic_source_collapses_the_frontier():
    cfg = tiny_cfg(flow_source="anchor_det")
    m = make(cfg)
    fmri = torch.randn(4, VOX[1])
    got = m.predict_tokens_frontier(fmri, 1, stats_for(cfg), ks=[1, 8])
    assert torch.allclose(got[1]["tokens"], got[8]["tokens"], atol=1e-4)


def test_predict_tokens_n_samples_matches_the_frontier_shape():
    cfg = tiny_cfg()
    m = make(cfg)
    fmri = torch.randn(4, VOX[1])
    tok1, cls1 = m.predict_tokens(fmri, 1, stats_for(cfg), cond_source="prior",
                                  n_samples=1)
    tok8, cls8 = m.predict_tokens(fmri, 1, stats_for(cfg), cond_source="prior",
                                  n_samples=8)
    assert tok1.shape == tok8.shape
    assert cls1.shape == cls8.shape
    assert torch.isfinite(tok8).all()


def test_predict_tokens_n_samples_is_a_noop_for_regression():
    cfg = tiny_cfg()
    m = make(cfg)
    fmri = torch.randn(4, VOX[1])
    a, _ = m.predict_tokens(fmri, 1, stats_for(cfg), cond_source="regression",
                            n_samples=1)
    b, _ = m.predict_tokens(fmri, 1, stats_for(cfg), cond_source="regression",
                            n_samples=16)
    assert torch.allclose(a, b, atol=1e-6)


def test_latent_frontier_reports_every_k_and_key():
    cfg = tiny_cfg()
    m = make(cfg)
    pts = latent_frontier(m, loader(cfg), stats_for(cfg), torch.device("cpu"),
                          ks=[0, 1, 4], n_steps=2, max_batches=2, retrieval_pool=8)
    assert {p.k for p in pts} == {0, 1, 4}
    for p in pts:
        for key in ("token_cos", "two_way", "retr_fwd", "resultant", "dispersion"):
            assert key in p.latent
            assert not math.isnan(p.latent[key])


def test_latent_frontier_restores_training_mode():
    cfg = tiny_cfg()
    m = make(cfg).train()
    latent_frontier(m, loader(cfg), stats_for(cfg), torch.device("cpu"),
                    ks=[1], n_steps=2, max_batches=1, retrieval_pool=8)
    assert m.training


def test_pareto_front_drops_dominated_points():
    pts = [
        FrontierPoint(k=1, strength=1.0, subject=1, n=10,
                      image={"PixCorr": 0.10, "CLIP_2way": 0.90}),
        FrontierPoint(k=2, strength=1.0, subject=1, n=10,
                      image={"PixCorr": 0.20, "CLIP_2way": 0.80}),
        FrontierPoint(k=4, strength=1.0, subject=1, n=10,
                      image={"PixCorr": 0.05, "CLIP_2way": 0.70}),
    ]
    front = pareto_front(pts)
    assert len(front) == 2
    assert {round(r["image/PixCorr"], 2) for r in front} == {0.10, 0.20}


def test_format_surface_renders_a_grid():
    pts = [FrontierPoint(k=k, strength=s, subject=1, n=4,
                         image={"PixCorr": 0.1 * k, "CLIP_2way": 0.5 + 0.1 * s})
           for k in (1, 2) for s in (0.5, 1.0)]
    text = format_surface(pts)
    assert "K\\s" in text
    assert "0.50" in text and "1.00" in text


def test_frontier_point_flattens_with_namespaced_keys():
    p = FrontierPoint(k=2, strength=0.7, subject=1, n=8,
                      latent={"token_cos": 0.3}, image={"PixCorr": 0.2})
    flat = p.flat()
    assert flat["latent/token_cos"] == 0.3
    assert flat["image/PixCorr"] == 0.2
    assert flat["k"] == 2 and flat["strength"] == 0.7
