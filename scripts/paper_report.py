"""Collect every scored run under outputs/ into the paper's tables.

Walks ``outputs/**/metrics.json`` (from ``eval_step1b``) and ``outputs/**/sweep.json``
(from ``sweep_step1b``) and emits:

  1. **the leaderboard table** -- our best row per run next to the published NSD
     numbers, in the units the tables actually use (AlexNet/Inception/CLIP are
     2-way accuracies, EffNet-B/SwAV are correlation distances, lower better);
  2. **the ablation table** -- runs paired by whatever differs in their name, so
     sphere-vs-euclidean and mixco-vs-baseline read off directly;
  3. **the operating-point curve** -- every (cond_source, cfg_scale, blend_w)
     config of a sweep, which is the PixCorr<->CLIP trade the paper has to be
     honest about, since no single point is good at both ends.

Runs entirely on files already on disk -- no GPU, no model, no decoder.

    python -m scripts.paper_report --root outputs --out outputs/paper
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from brainflow.metrics_full import NSD_REFERENCE, LOWER_IS_BETTER

# The report is UTF-8 (arrows mark metric direction). Don't let a cp1252 console
# turn a finished report into a traceback -- the file is written either way.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

# The columns the NSD tables report, in their order.
COLS = ["PixCorr", "SSIM", "AlexNet(2)_2way", "AlexNet(5)_2way",
        "Inception_2way", "CLIP_2way", "EffNet-B", "SwAV"]
EXTRA = ["retrieval_fwd", "retrieval_bwd"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="outputs",
                    help="directory tree to scan for metrics.json / sweep.json")
    ap.add_argument("--out", type=str, default="outputs/paper")
    ap.add_argument("--rank-by", type=str, default="CLIP_2way",
                    help="metric used to pick each run's headline row")
    return ap.parse_args()


def _fmt(d, k):
    v = d.get(k)
    return f"{v:.3f}" if isinstance(v, (int, float)) and v == v else "--"


def _table(rows, cols, first_col="run"):
    head = [first_col] + [c.replace("_2way", "").replace("retrieval_", "ret-")
                          for c in cols]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, d in rows:
        out.append("| " + " | ".join([name] + [_fmt(d, c) for c in cols]) + " |")
    return "\n".join(out)


def load_runs(root: Path):
    """Return ``{run_name: {"headline": metrics, "source": path, "rows": [...]}}``."""
    runs = {}
    for mf in sorted(root.rglob("metrics.json")):
        try:
            blob = json.loads(mf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        # eval_step1b writes {"mean": ..., "per_subject": ...}; older files were flat.
        m = blob.get("mean", blob)
        if "PixCorr" not in m:
            continue
        name = mf.parent.parent.name if mf.parent.name.startswith("eval") else mf.parent.name
        runs.setdefault(name, {"rows": []})
        runs[name]["headline"] = m
        runs[name]["source"] = str(mf)
        runs[name]["per_subject"] = blob.get("per_subject", {})
        runs[name]["config"] = blob.get("config", {})
    for sf in sorted(root.rglob("sweep.json")):
        try:
            blob = json.loads(sf.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        rows = [r for r in blob.get("rows", []) if "PixCorr" in r]
        if not rows:
            continue
        name = sf.parent.parent.name
        runs.setdefault(name, {"rows": []})
        runs[name]["rows"] = rows
        runs[name].setdefault("source", str(sf))
    return runs


def main():
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(root)
    if not runs:
        raise SystemExit(f"no metrics.json / sweep.json with a PixCorr under {root}/")

    # Headline per run: the eval's mean row, else the sweep's best row by --rank-by.
    best = {}
    for name, r in runs.items():
        cand = r.get("headline")
        if cand is None and r["rows"]:
            cand = max(r["rows"], key=lambda x: x.get(args.rank_by, float("-inf")))
        if cand:
            best[name] = cand

    lines = ["# BrainFlow results", "",
             "All numbers are full-test-set evaluations. AlexNet/Inception/CLIP are "
             "2-way identification accuracies (higher better); **EffNet-B and SwAV are "
             "correlation distances (LOWER better)**.", ""]

    lines += ["## 1. Leaderboard", ""]
    ranked = sorted(best.items(), key=lambda kv: kv[1].get(args.rank_by, 0), reverse=True)
    rows = [(f"**{n}**", d) for n, d in ranked]
    rows += [(k, v) for k, v in NSD_REFERENCE.items()]
    lines += [_table(rows, COLS), ""]

    lines += ["### retrieval (300-way, on the predicted embedding)", "",
              _table([(f"**{n}**", d) for n, d in ranked], EXTRA), ""]

    # Gap to the strongest published baseline, per metric, for the best run.
    if ranked:
        top_name, top = ranked[0]
        me2 = NSD_REFERENCE["MindEye2 (1 subj, 40h)"]
        lines += ["## 2. Gap to MindEye2", "",
                  f"Best run by {args.rank_by}: `{top_name}`", "",
                  "`gap` is signed so **negative always means we are worse**, for "
                  "higher-is-better and lower-is-better metrics alike.", "",
                  "| metric | ours | MindEye2 | gap |", "|---|---|---|---|"]
        for c in COLS:
            if c not in top or c not in me2:
                continue
            lower = c in LOWER_IS_BETTER
            gap = (me2[c] - top[c]) if lower else (top[c] - me2[c])
            arrow = "↓" if lower else "↑"
            lines.append(f"| {c.replace('_2way','')} {arrow} | {top[c]:.3f} | "
                         f"{me2[c]:.3f} | {gap:+.3f} |")
        lines.append("")

    # Ablation: runs sharing a prefix differ by exactly one factor most of the time,
    # so print them adjacent rather than guessing the pairing.
    lines += ["## 3. All runs (ablation view)", "",
              _table([(n, d) for n, d in sorted(best.items())], COLS), ""]

    lines += ["## 4. Operating points (the PixCorr <-> CLIP trade)", ""]
    for name, r in sorted(runs.items()):
        if not r["rows"]:
            continue
        lines += [f"### {name}", ""]
        tbl = []
        for x in sorted(r["rows"], key=lambda x: -x.get("CLIP_2way", 0)):
            bw = x.get("blend_w")
            label = (f"{x.get('cond_source')} cfg={x.get('cfg_scale')}"
                     + (f" bw={bw}" if bw is not None else "")
                     + (f" str={x.get('strength')}" if x.get("strength", 1.0) != 1.0 else ""))
            tbl.append((label, x))
        lines += [_table(tbl, COLS + EXTRA, first_col="config"), ""]

    # Per-subject spread, when a run scored more than one.
    multi = {n: r for n, r in runs.items() if len(r.get("per_subject", {})) > 1}
    if multi:
        lines += ["## 5. Per-subject", ""]
        for name, r in sorted(multi.items()):
            lines += [f"### {name}", ""]
            tbl = [(f"subj{s}", m) for s, m in sorted(r["per_subject"].items())]
            tbl.append(("**mean**", r["headline"]))
            lines += [_table(tbl, COLS, first_col="subject"), ""]

    report = "\n".join(lines)
    (out_dir / "results.md").write_text(report, encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps(
        {n: {"headline": best.get(n), "rows": r["rows"],
             "per_subject": r.get("per_subject", {}), "source": r.get("source")}
         for n, r in runs.items()}, indent=2))
    print(report)
    print(f"\n✓ {out_dir/'results.md'}")
    print(f"✓ {out_dir/'results.json'}")


if __name__ == "__main__":
    main()
