"""The aggregator that produces the paper's tables, exercised on fixtures.

This is the script whose output would go into a submission, so the things worth
pinning are the ones a reader could be misled by: metric direction, which row
is called best, and per-subject vs pooled aggregation.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent


def _metrics(pix, ssim, clip, eff, swav, **kw):
    d = {"PixCorr": pix, "SSIM": ssim, "AlexNet(2)_2way": 0.77, "AlexNet(5)_2way": 0.86,
         "Inception_2way": 0.77, "CLIP_2way": clip, "EffNet-B": eff, "SwAV": swav,
         "retrieval_fwd": 0.14, "retrieval_bwd": 0.13, "n": 982}
    d.update(kw)
    return d


@pytest.fixture
def tree():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "outputs"
        for arm, m in [("paper_sphere_s0", _metrics(.132, .240, .761, .865, .503)),
                       ("paper_euclid_s0", _metrics(.110, .215, .712, .900, .560))]:
            p = root / arm / "eval_full"
            p.mkdir(parents=True)
            (p / "metrics.json").write_text(json.dumps(
                {"config": {"cond_source": "blend"}, "subjects": [1],
                 "per_subject": {"1": m}, "mean": m}))
        sw = root / "paper_sphere_s0" / "sweep"
        sw.mkdir(parents=True)
        (sw / "sweep.json").write_text(json.dumps({"rows": [
            dict(cond_source="regression", cfg_scale=1.0, blend_w=None, strength=1.0,
                 **_metrics(.141, .276, .756, .860, .500)),
            dict(cond_source="blend", cfg_scale=3.0, blend_w=0.5, strength=1.0,
                 **_metrics(.132, .240, .761, .865, .503))]}))
        yield root


def _run(root, *extra):
    out = root / "paper"
    r = subprocess.run(
        [sys.executable, "-m", "scripts.paper_report", "--root", str(root),
         "--out", str(out), *extra],
        capture_output=True, text=True, cwd=str(REPO),
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    assert r.returncode == 0, r.stderr
    return (out / "results.md").read_text(encoding="utf-8"), json.loads(
        (out / "results.json").read_text(encoding="utf-8"))


class TestPaperReport:
    def test_finds_both_arms_and_the_references(self, tree):
        md, js = _run(tree)
        assert "paper_sphere_s0" in js and "paper_euclid_s0" in js
        assert "MindEye2" in md and "Takagi" in md

    def test_best_run_is_ranked_first(self, tree):
        md, _ = _run(tree)
        lead = md.split("## 1. Leaderboard")[1].split("## ")[0]
        assert lead.index("paper_sphere_s0") < lead.index("paper_euclid_s0")

    def test_rank_by_is_configurable(self, tree):
        md, _ = _run(tree, "--rank-by", "PixCorr")
        assert "Best run by PixCorr" in md

    def test_gap_is_negative_when_worse_in_both_directions(self, tree):
        """EffNet-B is lower-better; being worse there must still read negative."""
        md, _ = _run(tree)
        gap = md.split("## 2. Gap to MindEye2")[1].split("## ")[0]
        for line in gap.splitlines():
            if line.startswith("| PixCorr") or line.startswith("| EffNet-B"):
                assert "-0." in line, f"expected a negative gap in: {line}"

    def test_lower_is_better_is_stated(self, tree):
        md, _ = _run(tree)
        assert "LOWER better" in md

    def test_operating_points_table_lists_every_sweep_row(self, tree):
        md, _ = _run(tree)
        ops = md.split("## 4. Operating points")[1]
        assert "regression cfg=1.0" in ops and "blend cfg=3.0 bw=0.5" in ops

    def test_empty_tree_fails_loudly(self):
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run(
                [sys.executable, "-m", "scripts.paper_report", "--root", d,
                 "--out", str(Path(d) / "paper")],
                capture_output=True, text=True, cwd=str(REPO))
            assert r.returncode != 0
            assert "no metrics.json" in (r.stderr + r.stdout)


class TestPerSubject:
    def test_mean_row_is_the_mean_not_the_first_subject(self, tree):
        p = tree / "paper_sphere_s0" / "eval_full" / "metrics.json"
        ps = {"1": _metrics(.100, .200, .700, .900, .500),
              "2": _metrics(.200, .300, .800, .700, .400)}
        mean = {k: (ps["1"][k] + ps["2"][k]) / 2 for k in ps["1"]}
        p.write_text(json.dumps({"config": {}, "subjects": [1, 2],
                                 "per_subject": ps, "mean": mean}))
        md, js = _run(tree)
        assert js["paper_sphere_s0"]["headline"]["PixCorr"] == pytest.approx(0.15)
        sec = md.split("## 5. Per-subject")[1]
        assert "subj1" in sec and "subj2" in sec and "**mean**" in sec


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
