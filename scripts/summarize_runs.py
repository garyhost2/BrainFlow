"""Every training run in outputs/, one row each, with the config that produced it.

Reads outputs/*/diagnostics.jsonl for the curves and logs/*.out for the banner
that records what was actually run (which is not always what was requested --
gate1 arm 3 asks for --flow-param endpoint and TRAIN_EXTRA has overridden it to
velocity in at least two runs).

    python -m scripts.summarize_runs
    python -m scripts.summarize_runs --sort delta --full
"""
import argparse
import json
import pathlib
import re

BANNER = re.compile(r"R-XFM source=(\S+) param=(\S+) jitter=(\S+) rad \| cls_cond=(\S+)")
ARM = re.compile(r"── arm \d+: (\S+) → (\S+)")


def _log_configs(root):
    """map output-dir -> (source, param, jitter, cls_cond) by reading the logs."""
    out = {}
    for p in sorted((root / "logs").glob("*.out")):
        try:
            txt = p.read_text(errors="ignore")
        except OSError:
            continue
        arm = ARM.search(txt)
        ban = BANNER.search(txt)
        if not arm or not ban:
            continue
        out[pathlib.Path(arm.group(2)).name] = {
            "source": ban.group(1), "param": ban.group(2),
            "jitter": ban.group(3), "cls_cond": ban.group(4),
            "log": p.name,
        }
    return out


def _rows(path):
    rows = []
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "val/delta_cos" in r:
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".")
    ap.add_argument("--sort", type=str, default="delta",
                    choices=["delta", "name", "flow", "retr"])
    ap.add_argument("--full", action="store_true", help="also print every epoch")
    args = ap.parse_args()

    root = pathlib.Path(args.root).resolve()
    cfgs = _log_configs(root)

    recs = []
    for d in sorted((root / "outputs").glob("*/diagnostics.jsonl")):
        name = d.parent.name
        rows = _rows(d)
        if not rows:
            continue
        best = max(rows, key=lambda r: r["val/delta_cos"])
        last = rows[-1]
        c = cfgs.get(name, {})
        recs.append({
            "name": name, "n": len(rows),
            "param": c.get("param", "?"), "cls": c.get("cls_cond", "?"),
            "src": c.get("source", "?"),
            "best_delta": best["val/delta_cos"], "best_ep": best.get("epoch"),
            "best_win": best.get("val/delta_cos_win_rate", float("nan")),
            "last_ep": last.get("epoch"),
            "flow": last.get("val/flow_cos", float("nan")),
            "anchor": last.get("val/anchor_cos", float("nan")),
            "cls_cos": last.get("val/cls_cos_hat_true", float("nan")),
            "two_way": last.get("val/two_way", float("nan")),
            "retr": last.get("val/retr_fwd_300", float("nan")),
            "rows": rows,
        })

    key = {"delta": lambda r: -r["best_delta"], "name": lambda r: r["name"],
           "flow": lambda r: -r["flow"], "retr": lambda r: -r["retr"]}[args.sort]
    recs.sort(key=key)

    hdr = (f"{'run':<32}{'param':<10}{'cls_cond':<14}{'ep':>4}"
           f"{'bestD':>9}{'@ep':>5}{'win':>7}"
           f"{'flow':>8}{'anch':>8}{'ratio':>7}{'theta':>7}"
           f"{'clsC':>7}{'2way':>7}{'retr':>7}")
    print(hdr)
    print("-" * len(hdr))
    import math
    for r in recs:
        ratio = r["flow"] / r["anchor"] if r["anchor"] else float("nan")
        theta = math.degrees(math.acos(max(-1.0, min(1.0, ratio)))) if ratio == ratio else float("nan")
        print(f"{r['name']:<32}{r['param']:<10}{r['cls']:<14}{r['n']:>4}"
              f"{r['best_delta']:>+9.4f}{str(r['best_ep']):>5}{r['best_win']:>7.3f}"
              f"{r['flow']:>8.4f}{r['anchor']:>8.4f}{ratio:>7.3f}{theta:>6.1f}°"
              f"{r['cls_cos']:>7.3f}{r['two_way']:>7.3f}{r['retr']:>7.3f}")

    if args.full:
        for r in recs:
            print(f"\n== {r['name']}  ({r['param']}, cls={r['cls']})")
            for row in r["rows"]:
                print(f"   ep{str(row.get('epoch')):>5}  "
                      f"delta={row['val/delta_cos']:+.4f}  "
                      f"win={row.get('val/delta_cos_win_rate', float('nan')):.4f}  "
                      f"flow={row.get('val/flow_cos', float('nan')):.4f}  "
                      f"anchor={row.get('val/anchor_cos', float('nan')):.4f}  "
                      f"cls={row.get('val/cls_cos_hat_true', float('nan')):.3f}  "
                      f"2way={row.get('val/two_way', float('nan')):.3f}  "
                      f"retr={row.get('val/retr_fwd_300', float('nan')):.3f}")


if __name__ == "__main__":
    main()
