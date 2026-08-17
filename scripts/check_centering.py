from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-dir", type=str, default="./mindeyev2_cache")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1])
    ap.add_argument("--max-images", type=int, default=4000,
                    help="cap on images streamed (the full block is ~4 GB/subject)")
    ap.add_argument("--chunk", type=int, default=256)
    return ap.parse_args()


@torch.no_grad()
def analyse(emb: torch.Tensor, chunk: int) -> dict:
    n, N, d = emb.shape

    total = torch.zeros(N, d, dtype=torch.float64)
    for i in range(0, n, chunk):
        total += emb[i:i + chunk].float().sum(dim=0).double()
    m_pos = (total / n).float()
    m_bar = m_pos.mean(dim=0)

    gam_sq_num = torch.zeros((), dtype=torch.float64)
    gam_sq_den = torch.zeros((), dtype=torch.float64)
    cos_free_sum = torch.zeros((), dtype=torch.float64)
    norm_glob = torch.zeros((), dtype=torch.float64)
    norm_pos = torch.zeros((), dtype=torch.float64)
    seen = 0

    free = m_pos - m_bar
    free_dir = F.normalize(free, dim=-1)

    for i in range(0, n, chunk):
        x = emb[i:i + chunk].float()
        b = x.shape[0]
        cg = x - m_bar
        cp = x - m_pos
        gam_sq_num += free.pow(2).sum().double() * b
        gam_sq_den += cg.pow(2).sum().double()
        cos_free_sum += F.cosine_similarity(
            free_dir.unsqueeze(0).expand_as(cg), cg, dim=-1).sum().double()
        norm_glob += cg.norm(dim=-1).sum().double()
        norm_pos += cp.norm(dim=-1).sum().double()
        seen += b

    return {
        "n_images": seen,
        "gamma": float((gam_sq_num / gam_sq_den).sqrt()),
        "cos_free": float(cos_free_sum / (seen * N)),
        "norm_ratio_pos_over_glob": float(norm_pos / norm_glob),
        "m_pos": m_pos,
        "m_bar": m_bar,
    }


@torch.no_grad()
def anchor_floor(emb: torch.Tensor, m_pos: torch.Tensor, m_bar: torch.Tensor,
                 chunk: int, n_pairs: int = 512) -> dict:
    n = emb.shape[0]
    k = min(n_pairs, n // 2)
    a = emb[:k].float()
    b = emb[k:2 * k].float()
    return {
        "cross_image_cos_global": float(
            F.cosine_similarity(a - m_bar, b - m_bar, dim=-1).mean()),
        "cross_image_cos_per_position": float(
            F.cosine_similarity(a - m_pos, b - m_pos, dim=-1).mean()),
    }


def main():
    args = parse_args()
    for s in args.subjects:
        fp = Path(args.target_dir) / f"step1b_bigg_s{s}.pt"
        if not fp.exists():
            raise SystemExit(f"missing {fp}; build the bigG targets first")
        blob = torch.load(str(fp), map_location="cpu", mmap=True)
        emb = blob["emb_train"]
        if args.max_images and len(emb) > args.max_images:
            emb = emb[:args.max_images]
        n, N, d = emb.shape
        print(f"\n=== subject {s}: {n} images x {N} positions x {d} dims ===")

        r = analyse(emb, args.chunk)
        x = anchor_floor(emb, r["m_pos"], r["m_bar"], args.chunk)

        print(f"  gamma  (positional share of the centred norm) = {r['gamma']:.4f}")
        print(f"  cos_free  (cosine of a CONSTANT per-position predictor,")
        print(f"             i.e. the floor under every anchor number) = {r['cos_free']:.4f}")
        print(f"  cross-image cos, global centring        = {x['cross_image_cos_global']:.4f}")
        print(f"  cross-image cos, per-position centring  = {x['cross_image_cos_per_position']:.4f}")
        print(f"  ||x - m_pos|| / ||x - m_bar||           = {r['norm_ratio_pos_over_glob']:.4f}")

        free = r["cos_free"]
        print()
        if free < 0.05:
            print("  => the positional prior is negligible. --center-mode global is fine;")
            print("     the measured anchor cos is genuinely image-specific.")
        elif free < 0.15:
            print(f"  => {free:.3f} of the anchor cosine is free. Worth switching to")
            print("     --center-mode per_position, but it does not restate the results.")
        else:
            print(f"  => {free:.3f} of the anchor cosine is a CONSTANT predictor's score.")
            print("     Against the measured anchor cos 0.37 that is "
                  f"{100 * free / 0.37:.0f}% of the number.")
            print("     Re-run with --center-mode per_position before comparing any")
            print("     anchor/delta_cos figure to the ones recorded in PROGRESS.md.")


if __name__ == "__main__":
    main()
