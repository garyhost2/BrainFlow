"""What is actually inside a step1b checkpoint.

Written because the checkpoints are ~5.4 GB and the login node OOM-kills any
attempt to open one there, so every question about a trained model needed a
throwaway heredoc pasted over a flaky link. Run it on a compute node:

    srun -p gpu-all -c 2 --mem=64G -t 00:10:00 \
        python -m scripts.inspect_ckpt outputs/ll_fix_4subj/last.pt
"""
import argparse
import pathlib

import torch


def _fmt_bytes(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=str)
    ap.add_argument("--grep", type=str, default="low,lat,ll_",
                    help="comma-separated substrings to match against cfg fields")
    args = ap.parse_args()

    p = pathlib.Path(args.ckpt)
    print(f"== {p}  ({_fmt_bytes(p.stat().st_size)})", flush=True)
    ck = torch.load(p, map_location="cpu")
    print("top-level keys:", list(ck.keys()), flush=True)

    cfg = ck.get("cfg")
    if cfg is not None:
        d = vars(cfg) if hasattr(cfg, "__dict__") else dict(cfg)
        terms = [t for t in args.grep.split(",") if t]
        hits = {k: v for k, v in d.items() if any(t in k for t in terms)}
        print(f"cfg fields matching {terms}:", flush=True)
        for k in sorted(hits):
            print(f"   {k:28s} {hits[k]!r}", flush=True)

    sd = ck.get("model") or ck.get("state_dict") or {}

    # The question this was built for: did the low-level latent head ever get
    # its standardisation stats? lat_std == all ones means set_lat_stats never
    # ran, so the head regressed against an unstandardised target.
    for k, v in sd.items():
        if "lat_mean" in k or "lat_std" in k:
            print(f"{k} -> {[round(x, 6) for x in v.flatten().tolist()]}", flush=True)

    groups = {}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        head = k.split(".")[0]
        g = groups.setdefault(head, [0, 0])
        g[0] += 1
        g[1] += v.numel()
    print("model state_dict by top-level module:", flush=True)
    for head in sorted(groups, key=lambda h: -groups[h][1]):
        n, params = groups[head]
        print(f"   {head:24s} {n:4d} tensors  {params/1e6:9.2f}M params", flush=True)
    print(f"   {'TOTAL':24s} {sum(g[0] for g in groups.values()):4d} tensors  "
          f"{sum(g[1] for g in groups.values())/1e6:9.2f}M params", flush=True)

    # Where the 5.4 GB goes. AdamW keeps two fp32 moments per parameter, so an
    # optimizer blob roughly 2x the model is expected -- and is dead weight in
    # every checkpoint that is only ever loaded for eval.
    print("checkpoint sections:", flush=True)
    for k, v in ck.items():
        if isinstance(v, dict):
            tot = 0
            stack = [v]
            while stack:
                cur = stack.pop()
                for x in cur.values():
                    if torch.is_tensor(x):
                        tot += x.numel() * x.element_size()
                    elif isinstance(x, dict):
                        stack.append(x)
            print(f"   {k:24s} {len(v):5d} entries  {_fmt_bytes(tot)}", flush=True)


if __name__ == "__main__":
    main()
