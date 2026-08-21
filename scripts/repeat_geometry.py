"""Is stimulus content carried by the DIRECTION of a voxel pattern or by its NORM?

NSD presents each image ~3 times. build_or_load_tensors averages the repeats on
the TEST split (data.py:173) but keeps every TRAIN trial, so the train half is a
ready-made repeat experiment: ~27k trials over ~9k unique images.

For each image with >= 2 presentations, split every trial's standardised voxel
pattern into a norm and a unit direction, then ask which component is
REPEATABLE -- similar across repeats of one image relative to its spread across
different images. Reported as an ICC-style ratio in [0, 1]:

    direction:  (cos_within - cos_between) / (1 - cos_between)
    norm:       1 - var_within / var_total

Both are 0 when the component carries nothing image-specific and 1 when it is
perfectly determined by the image. They are on the same scale, which the raw
val/radius_corr and val/anchor_cos columns are NOT -- a Pearson r on a scalar and
a cosine in 1664 dimensions have different chance levels (0 and ~1/sqrt(1664)).

    srun -p gpu-all -c 4 --mem=128G -t 01:00:00 \
        python -m scripts.repeat_geometry --subject 1

Direction repeatability >> norm repeatability supports the claim that the code is
directional and the norm is trial state (gain, arousal, drift). Comparable values
refute it.
"""
import argparse
import hashlib
import pathlib
from collections import defaultdict

import torch

from brainflow.tensor_cache import assert_tensor_cache_alignment


def _group_by_image(imgs, chunk=512):
    """image hash -> list of trial indices. Repeats are byte-identical."""
    groups = defaultdict(list)
    for i0 in range(0, imgs.shape[0], chunk):
        block = imgs[i0:i0 + chunk].contiguous()
        for j in range(block.shape[0]):
            h = hashlib.blake2b(block[j].numpy().tobytes(), digest_size=16).digest()
            groups[h].append(i0 + j)
    return groups


def _stats(fmri, groups, min_repeats, seed=0):
    keep = [v for v in groups.values() if len(v) >= min_repeats]
    if not keep:
        raise SystemExit(f"no image had >= {min_repeats} presentations")

    mu = fmri.mean(0, keepdim=True)
    sd = fmri.std(0, keepdim=True).clamp_min(1e-6)
    z = (fmri - mu) / sd

    norms = z.norm(dim=1)
    dirs = z / norms.unsqueeze(1).clamp_min(1e-8)

    # ---- direction: within-image vs between-image cosine ---------------------
    within = []
    for idx in keep:
        d = dirs[idx]
        c = d @ d.T
        iu = torch.triu_indices(len(idx), len(idx), offset=1)
        within.append(c[iu[0], iu[1]])
    cos_within = torch.cat(within).mean().item()

    g = torch.Generator().manual_seed(seed)
    firsts = torch.tensor([v[0] for v in keep])
    perm = firsts[torch.randperm(len(firsts), generator=g)]
    pairs = min(len(firsts) - 1, 20000)
    cos_between = (dirs[firsts[:pairs]] * dirs[perm[:pairs]]).sum(1).mean().item()

    # ---- norm: within-image variance vs total variance -----------------------
    var_within, n_w = 0.0, 0
    for idx in keep:
        v = norms[idx]
        var_within += ((v - v.mean()) ** 2).sum().item()
        n_w += len(idx) - 1
    var_within /= max(n_w, 1)
    flat = torch.cat([norms[torch.tensor(v)] for v in keep])
    var_total = flat.var(unbiased=True).item()

    return {
        "n_images": len(keep),
        "n_trials": sum(len(v) for v in keep),
        "cos_within": cos_within,
        "cos_between": cos_between,
        "dir_repeat": (cos_within - cos_between) / max(1e-8, 1.0 - cos_between),
        "var_within": var_within,
        "var_total": var_total,
        "norm_repeat": 1.0 - var_within / max(var_total, 1e-8),
        "norm_mean": flat.mean().item(),
        "norm_cv": (flat.std().item() / max(flat.mean().item(), 1e-8)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor-cache", type=str,
                    default="./mindeyev2_cache/all_subjects_tensors.pt")
    ap.add_argument("--subject", type=int, nargs="+", default=[1])
    ap.add_argument("--min-repeats", type=int, default=2)
    args = ap.parse_args()

    p = pathlib.Path(args.tensor_cache)
    blob = assert_tensor_cache_alignment(str(p), torch.load(p, map_location="cpu",
                                                            mmap=True))
    print(f"{'subj':>5} {'images':>7} {'trials':>7} {'cosW':>8} {'cosB':>8} "
          f"{'DIR':>8} {'NORM':>8} {'normCV':>8}", flush=True)
    for s in args.subject:
        fmri = blob[f"fmri_train_{s}"].float()
        imgs = blob[f"imgs_train_{s}"]
        groups = _group_by_image(imgs)
        r = _stats(fmri, groups, args.min_repeats)
        print(f"{s:5d} {r['n_images']:7d} {r['n_trials']:7d} "
              f"{r['cos_within']:8.4f} {r['cos_between']:8.4f} "
              f"{r['dir_repeat']:8.4f} {r['norm_repeat']:8.4f} "
              f"{r['norm_cv']:8.4f}", flush=True)


if __name__ == "__main__":
    main()
