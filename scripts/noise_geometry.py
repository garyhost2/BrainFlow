"""Geometry of what varies when the same image is shown again.

Zero training. Runs on the cached betas. Every quantity has a null it is
reported against, because in 15,724 dimensions from 27,000 trials the leading
directions of anything are partly Marchenko-Pastur artifacts.

    srun -p gpu-A100 --account=a100_intern --qos=a100_intern \
        --gres=gpu:A100_80GB:1 -c 8 --mem=200G -t 02:00:00 \
        python -m scripts.noise_geometry --subjects 1 2 5 7 --out outputs/noise_geom

What it measures, per subject, on the train split:

1.  Sigma_noise  pooled within-image residual covariance, dof N - G.
2.  Sigma_signal covariance of the per-image means, MINUS Sigma_noise/nbar --
    the means still carry 1/nbar of the trial noise, and not subtracting it
    inflates every alignment number that follows.
3.  MP edge      the bulk edge of a white covariance at matched (d, n) and
    variance. Only eigenvalues above it count as structure. Participation
    ratio is reported beside it as a shape statistic that needs no threshold.
4.  Split-half   top-k noise directions estimated on disjoint halves of the
    IMAGES, then subspace overlap. Directions that do not replicate across
    halves are sampling noise; this is the first control a reviewer asks for.
5.  Alignment    tr(P_sig^T Sigma_noise P_sig) / tr(Sigma_noise) against its
    chance value k/d. This is the number that forecasts the expensive
    experiments: noise orthogonal to the signal subspace is nearly harmless to
    a downstream encoder, so alignment at chance predicts that repeat
    consistency and denoising buy little.
6.  Session ablation  everything again after removing per-session means. If
    the anisotropy collapses, it was scanner drift, not trial noise.
7.  DIR / NORM   direction and norm repeatability, and the norm regressed on
    state (session index, position within session) rather than stimulus.

Caveat that belongs in the caption, not the discussion: NSD ships
betas_fithrf_GLMdenoise_RR, so this is the variability that SURVIVED GLMsingle
(per-voxel HRF fitting, cross-validated noise regressors, ridge). The framework
being scaled here is noise correlations and multiplicative gain, measured at
pattern level on NSD rather than on single units.
"""
import argparse
import json
import pathlib

import torch

from brainflow.denoise import RepeatIndex
from brainflow.tensor_cache import assert_tensor_cache_alignment

NSD_TRIALS_PER_SESSION = 750


def _covariances(x, gid, valid, n_groups):
    """Pooled within-image residual covariance and the bias-corrected signal one."""
    d = x.shape[1]
    g = gid[valid]
    xv = x[valid]

    sums = torch.zeros(n_groups, d, device=x.device, dtype=x.dtype)
    sums.index_add_(0, g, xv)
    counts = torch.zeros(n_groups, device=x.device, dtype=x.dtype)
    counts.index_add_(0, g, torch.ones(len(g), device=x.device, dtype=x.dtype))
    means = sums / counts.unsqueeze(1)

    resid = xv - means[g]
    n_eff = len(g) - n_groups                      # one dof lost per group mean
    sigma_noise = (resid.T @ resid) / n_eff

    mc = means - means.mean(0, keepdim=True)
    sigma_means = (mc.T @ mc) / (n_groups - 1)
    nbar = counts.mean().item()
    sigma_signal = sigma_means - sigma_noise / nbar

    return sigma_noise, sigma_signal, resid, means, n_eff, nbar


def _spectrum_stats(sigma, n_eff):
    d = sigma.shape[0]
    ev = torch.linalg.eigvalsh(sigma.double()).flip(0)          # descending
    tr = ev.sum()
    # Marchenko-Pastur bulk edge for a white covariance of the same total
    # variance at this (d, n). Anything below it is consistent with no structure.
    var = (tr / d).item()
    ratio = d / max(n_eff, 1)
    edge = var * (1.0 + ratio ** 0.5) ** 2
    pr = (tr ** 2 / (ev ** 2).sum()).item()                     # participation ratio
    return {
        "trace": tr.item(),
        "mean_eigenvalue": var,
        "mp_edge": edge,
        "n_above_mp": int((ev > edge).sum().item()),
        "participation_ratio": pr,
        "pr_over_d": pr / d,
        "top10_frac": (ev[:10].sum() / tr).item(),
        "top_eigenvalue": ev[0].item(),
    }, ev


def _topk_dirs(resid, k):
    """Top-k eigenvectors of resid^T resid, without forming resid^T resid."""
    _, _, v = torch.svd_lowrank(resid, q=min(k + 16, min(resid.shape) - 1), niter=4)
    return v[:, :k]


def _split_half_overlap(x, gid, valid, n_groups, k, seed):
    """Estimate the top-k noise directions on disjoint IMAGE halves and compare.

    Overlap is ||U1^T U2||_F^2 / k, which is 1 for identical subspaces and
    k/d in expectation for two random ones.
    """
    gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(n_groups, generator=gen)
    half = n_groups // 2
    a = torch.zeros(n_groups, dtype=torch.bool); a[order[:half]] = True
    a = a.to(x.device)

    subs = []
    for mask in (a, ~a):
        sel = valid[mask[gid[valid]]]
        g = gid[sel]
        uniq, inv = torch.unique(g, return_inverse=True)
        sums = torch.zeros(len(uniq), x.shape[1], device=x.device, dtype=x.dtype)
        sums.index_add_(0, inv, x[sel])
        cnt = torch.zeros(len(uniq), device=x.device, dtype=x.dtype)
        cnt.index_add_(0, inv, torch.ones(len(sel), device=x.device, dtype=x.dtype))
        subs.append(x[sel] - (sums / cnt.unsqueeze(1))[inv])

    u1, u2 = _topk_dirs(subs[0], k), _topk_dirs(subs[1], k)
    overlap = ((u1.T @ u2) ** 2).sum().item() / k
    return {"k": k, "overlap": overlap, "chance": k / x.shape[1]}


def _alignment(sigma_noise, sigma_signal, k):
    """Fraction of noise variance living inside the top-k signal subspace."""
    ev, evec = torch.linalg.eigh(sigma_signal.double())
    p = evec[:, -k:].to(sigma_noise.dtype)
    inside = torch.einsum("ij,jk,ki->", p.T, sigma_noise, p).item()
    total = torch.diagonal(sigma_noise).sum().item()
    d = sigma_noise.shape[0]
    return {"k": k, "frac_noise_in_signal_subspace": inside / total,
            "chance": k / d, "ratio_to_chance": (inside / total) / (k / d)}


def _dir_norm(x, gid, valid, n_groups, sessions, seed=0):
    g = gid[valid]
    xv = x[valid]
    norms = xv.norm(dim=1)
    dirs = xv / norms.unsqueeze(1).clamp_min(1e-8)

    sums = torch.zeros(n_groups, x.shape[1], device=x.device, dtype=x.dtype)
    sums.index_add_(0, g, dirs)
    cnt = torch.zeros(n_groups, device=x.device, dtype=x.dtype)
    cnt.index_add_(0, g, torch.ones(len(g), device=x.device, dtype=x.dtype))
    # mean pairwise within-group cosine, from the group resultant:
    #   (|sum|^2 - n) / (n(n-1))
    res2 = (sums ** 2).sum(1)
    ok = cnt >= 2
    cos_within = ((res2[ok] - cnt[ok]) / (cnt[ok] * (cnt[ok] - 1))).mean().item()

    gen = torch.Generator(device=x.device).manual_seed(seed)
    perm = torch.randperm(len(dirs), generator=gen, device=x.device)
    cos_between = (dirs * dirs[perm]).sum(1).mean().item()

    nsum = torch.zeros(n_groups, device=x.device, dtype=x.dtype)
    nsum.index_add_(0, g, norms)
    nmean = nsum / cnt
    var_within = ((norms - nmean[g]) ** 2).sum().item() / max(len(g) - n_groups, 1)
    var_total = norms.var(unbiased=True).item()

    out = {
        "cos_within": cos_within, "cos_between": cos_between,
        "dir_repeat": (cos_within - cos_between) / max(1e-8, 1 - cos_between),
        "norm_repeat": 1.0 - var_within / max(var_total, 1e-8),
        "norm_cv": (norms.std() / norms.mean()).item(),
    }
    out["dir_over_norm"] = out["dir_repeat"] / max(out["norm_repeat"], 1e-8)

    # Norm against STATE rather than stimulus: session index and position within
    # session. H1 predicts the norm tracks these and the direction does not.
    sv = sessions[valid].to(x.dtype)
    pos = (torch.arange(len(x), device=x.device)[valid] % NSD_TRIALS_PER_SESSION
           ).to(x.dtype)
    for name, cov in (("session", sv), ("trial_in_session", pos)):
        c = torch.corrcoef(torch.stack([norms, cov]))[0, 1].item()
        out[f"norm_r_{name}"] = c
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor-cache", type=str,
                    default="./mindeyev2_cache/all_subjects_tensors.pt")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1])
    ap.add_argument("--k", type=int, default=64, help="subspace size for 4 and 5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="outputs/noise_geom")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = pathlib.Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    p = pathlib.Path(args.tensor_cache)
    blob = assert_tensor_cache_alignment(
        str(p), torch.load(p, map_location="cpu", mmap=True))

    report = {}
    for s in args.subjects:
        print(f"\n═══ subject {s:02d}", flush=True)
        fmri = blob[f"fmri_train_{s}"].float()
        st = blob["fmri_stats"][s]
        x = ((fmri - st["mu"].float()) / st["std"].float().clamp_min(1e-6)).to(device)
        idx = RepeatIndex.cached(blob[f"imgs_train_{s}"],
                                 p.parent / f"repeat_index_s{s}.pt")
        gid = idx.gid.to(device); valid = idx.valid.to(device)
        n_groups = len(idx.groups); d = x.shape[1]
        sessions = torch.arange(x.shape[0], device=device) // NSD_TRIALS_PER_SESSION
        print(f"  {x.shape[0]} trials, {d} voxels, {n_groups} images with >=2 repeats, "
              f"{int(sessions.max().item()) + 1} sessions", flush=True)

        r = {"n_trials": int(x.shape[0]), "n_voxels": d, "n_groups": n_groups}

        for tag, xx in (("raw", x), ("session_demeaned", None)):
            if tag == "session_demeaned":
                xx = x.clone()
                for ses in sessions.unique():
                    m = sessions == ses
                    xx[m] -= xx[m].mean(0, keepdim=True)

            sn, sg, resid, means, n_eff, nbar = _covariances(xx, gid, valid, n_groups)
            stats, _ = _spectrum_stats(sn, n_eff)
            block = {"noise_spectrum": stats,
                     "n_eff": n_eff, "nbar": nbar,
                     "signal_trace": torch.diagonal(sg).sum().item(),
                     "snr_trace": (torch.diagonal(sg).sum()
                                   / torch.diagonal(sn).sum()).item()}
            block["alignment"] = _alignment(sn, sg, args.k)
            del sn, sg, resid, means
            torch.cuda.empty_cache()
            block["split_half"] = _split_half_overlap(xx, gid, valid, n_groups,
                                                      args.k, args.seed)
            block["dir_norm"] = _dir_norm(xx, gid, valid, n_groups, sessions, args.seed)
            r[tag] = block

            sp = block["noise_spectrum"]
            print(f"  [{tag:16s}] eig>MP {sp['n_above_mp']:5d}/{d}  "
                  f"PR {sp['participation_ratio']:8.1f} ({sp['pr_over_d']:.3f}d)  "
                  f"top10 {sp['top10_frac']:.3f}  "
                  f"align {block['alignment']['frac_noise_in_signal_subspace']:.4f} "
                  f"({block['alignment']['ratio_to_chance']:.2f}x chance)  "
                  f"splithalf {block['split_half']['overlap']:.3f} "
                  f"(chance {block['split_half']['chance']:.4f})  "
                  f"DIR/NORM {block['dir_norm']['dir_over_norm']:.2f}", flush=True)
            del xx
            torch.cuda.empty_cache()

        report[str(s)] = r
        del x
        torch.cuda.empty_cache()

    (out_dir / "noise_geometry.json").write_text(json.dumps(report, indent=2))
    print(f"\n✓ {out_dir / 'noise_geometry.json'}")
    print("\nFalsifiers, as pre-registered:")
    print("  spectrum at the MP edge          -> no structure to model")
    print("  alignment at chance              -> denoising cannot help the encoder")
    print("  anisotropy gone after demeaning  -> it was drift, not trial noise")


if __name__ == "__main__":
    main()
