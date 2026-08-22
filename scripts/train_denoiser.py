"""Train the brain-space denoiser and measure it against doing nothing.

    python -m scripts.train_denoiser --subjects 1 2 5 7 --epochs 60 \
        --out outputs/denoise_4subj

The number that decides this is `dcos` in the per-epoch line: mean cosine to the
leave-one-out target for the model, minus the same cosine for the raw trial, on
held-out images. Positive means the denoiser moved a trial closer to the
response its own image actually produces. Zero or negative means trial noise is
irreducible from a single trial and the idea is dead -- which is a one-run
answer, not a research programme.

Everything stays resident on the GPU and is indexed directly; a DataLoader here
would spend more time collating 15k-dim vectors than the model spends on them.
"""
import argparse
import json
import pathlib
import time

import torch

from brainflow.denoise import (BrainDenoiser, RepeatIndex, denoise_loss,
                               loo_target)
from brainflow.tensor_cache import assert_tensor_cache_alignment


def _load(args, device):
    p = pathlib.Path(args.tensor_cache)
    blob = assert_tensor_cache_alignment(
        str(p), torch.load(p, map_location="cpu", mmap=True))
    subj = {}
    for s in args.subjects:
        fmri = blob[f"fmri_train_{s}"].float()
        stats = blob["fmri_stats"][s]
        mu = stats["mu"].float()
        sd = stats["std"].float().clamp_min(1e-6)
        idx = RepeatIndex.cached(blob[f"imgs_train_{s}"],
                                 p.parent / f"repeat_index_s{s}.pt")
        tr, va = idx.split(args.val_frac, seed=args.seed)
        x = ((fmri - mu) / sd).to(device)
        sums, sizes = idx.loo_machinery(x)
        subj[s] = {"x": x, "gid": idx.gid.to(device), "sums": sums,
                   "sizes": sizes, "train": tr.to(device), "val": va.to(device),
                   "n_groups": len(idx.groups), "n_vox": x.shape[1]}
        print(f"  subj{s:02d}: {x.shape[0]} trials, {len(idx.groups)} images "
              f"with >=2 repeats, {len(tr)} train / {len(va)} val trials, "
              f"{x.shape[1]} voxels", flush=True)
    return subj


@torch.no_grad()
def evaluate(model, subj, chunk=1024):
    model.eval()
    out = {}
    for s, d in subj.items():
        idx, x = d["val"], d["x"]
        cos_raw, cos_hat, mse_hat, n = 0.0, 0.0, 0.0, 0
        for i0 in range(0, len(idx), chunk):
            sel = idx[i0:i0 + chunk]
            xb = x[sel]
            tgt = loo_target(x, sel, d["gid"], d["sums"], d["sizes"])
            pred = model(xb, s)
            cos_raw += torch.cosine_similarity(xb, tgt, dim=1).sum().item()
            cos_hat += torch.cosine_similarity(pred, tgt, dim=1).sum().item()
            mse_hat += torch.nn.functional.mse_loss(
                pred, tgt, reduction="sum").item() / xb.shape[1]
            n += len(sel)
        out[s] = {"cos_raw": cos_raw / n, "cos_hat": cos_hat / n,
                  "mse": mse_hat / n, "n": n}
        out[s]["dcos"] = out[s]["cos_hat"] - out[s]["cos_raw"]
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor-cache", type=str,
                    default="./mindeyev2_cache/all_subjects_tensors.pt")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--width", type=int, default=2048)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--lambda-cos", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="outputs/denoise")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    print(f"▶ brain-space denoiser | subjects {args.subjects} | device {device}")
    subj = _load(args, device)

    model = BrainDenoiser({s: d["n_vox"] for s, d in subj.items()},
                          width=args.width, depth=args.depth,
                          dropout=args.dropout).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"✓ {n_par/1e6:.1f}M parameters, output zero-initialised "
          f"(step 0 is exactly the identity)", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    # One batch = one subject, because voxel counts differ. Sample subjects in
    # proportion to their trial count so no subject is over-weighted.
    counts = torch.tensor([len(subj[s]["train"]) for s in subj], dtype=torch.float)
    order = list(subj)
    steps = int(counts.sum().item()) // args.batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(1, args.epochs * steps), pct_start=0.1)

    base = evaluate(model, subj)
    for s, m in base.items():
        print(f"  baseline subj{s:02d}: cos(raw, loo) = {m['cos_raw']:.4f} "
              f"(n={m['n']})", flush=True)

    best, hist = -1e9, []
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        agg = {"mse": 0.0, "cos": 0.0, "n": 0}
        for _ in range(steps):
            s = order[int(torch.multinomial(counts, 1).item())]
            d = subj[s]
            sel = d["train"][torch.randint(len(d["train"]), (args.batch_size,),
                                           device=device)]
            xb = d["x"][sel]
            tgt = loo_target(d["x"], sel, d["gid"], d["sums"], d["sizes"])
            ld = denoise_loss(model(xb, s), tgt, args.lambda_cos)
            opt.zero_grad(set_to_none=True)
            ld["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            agg["mse"] += ld["mse"].item(); agg["cos"] += ld["cos"].item()
            agg["n"] += 1

        ev = evaluate(model, subj)
        dcos = sum(m["dcos"] for m in ev.values()) / len(ev)
        row = {"epoch": ep, "train_mse": agg["mse"] / agg["n"],
               "train_cos": agg["cos"] / agg["n"], "val_dcos": dcos,
               "per_subject": {str(s): m for s, m in ev.items()},
               "secs": round(time.time() - t0, 1)}
        hist.append(row)
        with open(out_dir / "denoise_log.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")

        mark = ""
        if dcos > best:
            best = dcos
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "voxels": {s: d["n_vox"] for s, d in subj.items()},
                        "epoch": ep, "val_dcos": dcos}, out_dir / "best.pt")
            mark = "  ✓best.pt"
        detail = " ".join(f"s{s}:{m['dcos']:+.4f}" for s, m in sorted(ev.items()))
        print(f"Ep {ep:3d} | train mse={row['train_mse']:.4f} "
              f"cos={row['train_cos']:.4f} | val dcos={dcos:+.4f} | {detail} "
              f"| {row['secs']}s{mark}", flush=True)

    print(f"\nDone. best mean val dcos = {best:+.4f}")
    print("Positive means a denoised trial sits closer to its own image's "
          "response than the raw trial did. Zero means trial noise is "
          "irreducible from one trial and this line of work stops here.")


if __name__ == "__main__":
    main()
