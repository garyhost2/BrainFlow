from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

TOKEN_DIM = 1664


def _load(target_dir: Path, subject: int):
    fp = Path(target_dir) / f"step1b_bigg_s{subject}.pt"
    if not fp.exists():
        raise FileNotFoundError(f"{fp} not found. Build targets first (train/eval/smoke).")
    blob = torch.load(str(fp), map_location="cpu", mmap=True)
    a = blob["accum"]
    mu = (a["sum"].double() / int(a["count"])).float()
    return blob, mu


def _sample(emb, n_images, seed=0):
    g = torch.Generator().manual_seed(seed)
    N = emb.shape[0]
    idx = torch.randperm(N, generator=g)[:min(n_images, N)]
    return emb[idx].float()


def _radius_stats(x, mu):
    n, T, d = x.shape
    cen = x - mu
    r = cen.norm(dim=-1)
    r_un = x.norm(dim=-1)

    flat = r.reshape(-1)
    rel = (flat.std() / flat.mean()).item()
    rel_un = (r_un.std() / r_un.mean()).item()

    per_pos_mean = r.mean(dim=0)
    between = ((per_pos_mean - flat.mean()) ** 2).mean().item()
    total = flat.var(unbiased=False).item()
    frac_pos = between / max(total, 1e-12)

    resid = (r - per_pos_mean[None, :])
    rel_resid = (resid.std() / flat.mean()).item()
    return {
        "rel_spread_centered": rel, "rel_spread_uncentered": rel_un,
        "frac_var_by_position": frac_pos, "rel_spread_residual": rel_resid,
        "r_mean": flat.mean().item(), "r_min": flat.min().item(), "r_max": flat.max().item(),
    }


def _angle_stats(x, mu, n_pairs=200_000, seed=0):
    g = torch.Generator().manual_seed(seed)
    n, T, d = x.shape
    cen = torch.nn.functional.normalize(x - mu, dim=-1)
    un = torch.nn.functional.normalize(x, dim=-1)
    img = torch.randint(0, n, (n_pairs,), generator=g)
    i = torch.randint(0, T, (n_pairs,), generator=g)
    j = torch.randint(0, T, (n_pairs,), generator=g)
    cos_c = (cen[img, i] * cen[img, j]).sum(-1)
    cos_u = (un[img, i] * un[img, j]).sum(-1)
    return {
        "cos_centered_mean": cos_c.mean().item(), "cos_centered_absmean": cos_c.abs().mean().item(),
        "cos_centered_std": cos_c.std().item(),
        "cos_uncentered_mean": cos_u.mean().item(), "cos_uncentered_absmean": cos_u.abs().mean().item(),
        "theoretical_std": 1.0 / math.sqrt(d),
    }


def _verdict(rad, ang):
    paper_rel = 1.0 / math.sqrt(2 * TOKEN_DIM)
    lines = ["", "--- VERDICT ---------------------------------------------"]
    rel, fpos, resid = rad["rel_spread_centered"], rad["frac_var_by_position"], rad["rel_spread_residual"]
    lines.append(f"Radius rel-spread (centered): {rel*100:.1f}%   (paper isotropic ref: {paper_rel*100:.2f}%)")
    lines.append(f"  ... of which {fpos*100:.0f}% of variance is explained by token POSITION,")
    lines.append(f"      leaving a residual per-token rel-spread of {resid*100:.1f}% for the side-head.")
    if resid < 0.05:
        lines.append("  -> RADIUS IS EASY: residual spread <5%. Peeling it off is well justified;")
        lines.append("     the sphere bet's premise (1) HOLDS on real tokens. Sphere is sound.")
    elif resid < 0.15:
        lines.append("  -> RADIUS IS MODERATE: a per-token head should track it, but verify the")
        lines.append("     radius-loss converges; the sphere bet is plausible, not free.")
    else:
        lines.append("  -> RADIUS IS HARD: residual spread >15%. Peeling it ADDS real error vs ECFM.")
        lines.append("     Baggag's preference for Euclidean CFM is well-founded here; do NOT assume")
        lines.append("     the sphere wins on low-level metrics without the head-to-head ablation.")
    lines.append("")
    ac, au = ang["cos_centered_absmean"], ang["cos_uncentered_absmean"]
    lines.append(f"Direction overlap |cos| -- centered: {ac:.3f}  uncentered: {au:.3f}  "
                 f"(near-orthogonal ref: {ang['theoretical_std']:.3f})")
    if ac < au * 0.9:
        lines.append("  -> Centering measurably de-clusters the directions (good for transport).")
    else:
        lines.append("  -> Centering barely changes spread; its benefit is small on this data.")
    if ang["cos_centered_std"] < 3 * ang["theoretical_std"]:
        lines.append("  -> Directions are well-spread; slerp/geodesics are well-conditioned (premise 2 holds).")
    else:
        lines.append("  -> Directions are clustered (high cos spread); geodesics transport mass a long way.")
    lines.append("---------------------------------------------------------")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-dir", type=str, default="./mindeyev2_cache")
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--split", type=str, default="test", choices=["train", "test"])
    ap.add_argument("--n-images", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    blob, mu = _load(Path(args.target_dir), args.subject)
    emb = blob[f"emb_{args.split}"]
    print(f"> subject {args.subject}  split={args.split}  tokens={tuple(emb.shape)}  "
          f"sampling {args.n_images} images")
    x = _sample(emb, args.n_images, args.seed)

    rad = _radius_stats(x, mu)
    ang = _angle_stats(x, mu, seed=args.seed)

    print("\n--- RADIUS (||x_i - mu||) -------------------------------")
    for k, v in rad.items():
        print(f"  {k:24s} {v:.5f}" + ("  (= {:.2f}%)".format(v * 100) if "spread" in k or "frac" in k else ""))
    print("\n--- DIRECTION (cosine between token directions) ---------")
    for k, v in ang.items():
        print(f"  {k:24s} {v:.5f}")
    print(_verdict(rad, ang))


if __name__ == "__main__":
    main()
