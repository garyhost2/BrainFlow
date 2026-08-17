"""Gate 0: decide the behav->betas trial offset from the data itself.

``rxfm/data.py`` indexes the betas HDF5 with ``all_trial - 1`` while indexing
the COCO image HDF5 with ``all_coco`` unshifted. MindEye2's ``dataset_creation``
notebook defines behav column 5 as::

    "global_trial": (int(behav.iloc[jj]['SESSION']) - 1) * 750 + jj      # 5

``jj`` is a positional ``.iloc`` index and ``SESSION - 1`` zeroes the session
offset, so column 5 is 0-based and MindEye2 consumes it unshifted
(``voxels[...][behav[:, 0, 5].long()]``). That implies our ``-1`` pairs the fMRI
of trial *t-1* with the image of trial *t*.

Rather than argue from source, this script measures it. For each candidate offset
we fit a ridge from voxels to the frozen bigG CLS embedding on a train split and
score top-1 retrieval on a held-out split. The correct pairing must win, and by a
wide margin -- a one-trial shift destroys the tightly stimulus-locked component of
the signal while leaving slow global state partly intact, so the *gap* is the
evidence, not the absolute number.

Runs on CPU in a few minutes and needs no training. It is wired into
``tests/test_trial_offset.py`` so a regression cannot land silently.

Usage::

    python -m scripts.check_offset --subject 1
    python -m scripts.check_offset --subject 1 --offsets 0 -1 --alpha 1e4
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import h5py
import numpy as np
import torch
import webdataset as wds

# Chance top-1 within a pool of `n_eval` candidates; anything near this is noise.
_MIN_SHARDS_TRAIN = 64
_MIN_SHARDS_TEST = 8


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="./mindeyev2_cache")
    ap.add_argument("--target-dir", type=str, default="./mindeyev2_cache")
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--offsets", type=int, nargs="+", default=[0, -1],
                    help="candidate offsets applied to behav column 5")
    ap.add_argument("--alpha", type=float, default=1e4, help="ridge regularisation")
    ap.add_argument("--n-eval", type=int, default=1000,
                    help="held-out trials used as the retrieval pool")
    ap.add_argument("--max-train", type=int, default=12000,
                    help="cap on ridge training rows (keeps this a CPU job)")
    ap.add_argument("--min-margin", type=float, default=0.02,
                    help="top-1 margin the winner must clear to count as conclusive")
    return ap.parse_args()


def load_behav(data_dir: Path, subject: int) -> np.ndarray:
    """Concatenated behav rows for the TRAIN shards of one subject."""
    s = f"0{subject}"
    rows = []
    for i in range(_MIN_SHARDS_TRAIN):
        p = data_dir / f"wds/subj{s}/train/{i}.tar"
        if not p.exists():
            continue
        for sample in wds.WebDataset(str(p), handler=wds.warn_and_continue):
            if "behav.npy" in sample:
                rows.append(np.load(io.BytesIO(sample["behav.npy"])))
    if not rows:
        raise SystemExit(f"no train shards under {data_dir}/wds/subj{s}/train/")
    return np.concatenate(rows)


def ridge_fit(X: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge in whichever space is smaller (dual when n < d)."""
    n, d = X.shape
    if n >= d:
        A = X.T @ X + alpha * np.eye(d, dtype=X.dtype)
        return np.linalg.solve(A, X.T @ Y)
    K = X @ X.T + alpha * np.eye(n, dtype=X.dtype)
    return X.T @ np.linalg.solve(K, Y)


def score_offset(betas_path: Path, coco_cls: torch.Tensor, behav: np.ndarray,
                 target_idx: np.ndarray, offset: int, args) -> dict:
    """Fit voxels -> bigG CLS under one offset, report held-out cos and top-1."""
    trial = behav[:, 5].astype(int)

    with h5py.File(betas_path, "r", rdcc_nbytes=256 * 1024 * 1024) as f:
        key = list(f.keys())[0]
        n_tot, _ = f[key].shape
        idx = np.clip(trial + offset, 0, n_tot - 1)
        # Read the unique rows once, then expand -- h5py cannot fancy-index dups.
        u, inv = np.unique(idx, return_inverse=True)
        X = f[key][u][inv].astype(np.float32)

    # ``coco_cls`` is cls_train, which targets_bigg builds from imgs_train -- one
    # row PER TRIAL, already image-aligned. It must therefore be indexed by ROW.
    # Indexing it with behav[:, 0] (the 73k COCO id, range 0..72999) against an
    # array of length n ~ 25k clipped roughly two thirds of all trials onto the
    # same final row, giving both offsets a near-chance score and making the test
    # structurally unable to decide anything.
    Y = coco_cls[target_idx].float().numpy()

    n = len(X)
    n_eval = min(args.n_eval, n // 4)
    n_train = min(args.max_train, n - n_eval)
    Xtr, Ytr = X[:n_train], Y[:n_train]
    Xte, Yte = X[-n_eval:], Y[-n_eval:]

    # Standardise voxels on TRAIN statistics only.
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True).clip(1e-6)
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    W = ridge_fit(Xtr, Ytr, args.alpha)
    P = Xte @ W

    P /= np.linalg.norm(P, axis=1, keepdims=True).clip(1e-8)
    T = Yte / np.linalg.norm(Yte, axis=1, keepdims=True).clip(1e-8)
    S = P @ T.T
    top1 = float((S.argmax(1) == np.arange(len(S))).mean())
    return {"offset": offset, "cos": float(np.diag(S).mean()), "top1": top1,
            "pool": len(S), "chance": 1.0 / len(S)}


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    s = f"0{args.subject}"

    betas = data_dir / f"betas_all_subj{s}_fp32_renorm.hdf5"
    if not betas.exists():
        raise SystemExit(f"missing {betas}; run the data prep first")

    # bigG CLS per 73k COCO id. Reuse the step1b target cache when present so this
    # stays a CPU job; otherwise fall back to the raw image hdf5 is NOT attempted
    # (that would need the embedder and a GPU).
    cache = Path(args.target_dir) / f"step1b_bigg_s{args.subject}.pt"
    if not cache.exists():
        raise SystemExit(
            f"missing {cache}. Build the bigG targets first "
            f"(train_step1b does it automatically on a GPU node).")
    blob = torch.load(str(cache), map_location="cpu", mmap=True)

    behav = load_behav(data_dir, args.subject)
    # The cache is ordered like the train split, so align lengths defensively.
    cls_train = blob["cls_train"]
    n = min(len(behav), len(cls_train))
    behav = behav[:n]

    # Index CLS by row (already image-aligned in the cache) rather than by COCO id.
    coco_cls = cls_train[:n]
    behav_idx = np.arange(n)

    results = []
    for off in args.offsets:
        r = score_offset(betas, coco_cls, behav, behav_idx,
                         offset=off, args=argparse.Namespace(**vars(args)))
        results.append(r)
        print(f"offset {off:+d}:  cos={r['cos']:.4f}   "
              f"top-1/{r['pool']}={r['top1']:.4f}   (chance={r['chance']:.4f})")

    best = max(results, key=lambda r: r["top1"])
    runner = sorted(results, key=lambda r: -r["top1"])[1] if len(results) > 1 else None
    print(f"\n=> best offset: {best['offset']:+d}")
    conclusive = True
    if runner is not None:
        margin = best["top1"] - runner["top1"]
        print(f"   margin over next best: {margin:+.4f}")
        if margin < args.min_margin:
            conclusive = False
            print(f"   [warn] offsets are within noise of each other (margin "
                  f"{margin:+.4f} < {args.min_margin}) -- inconclusive. "
                  "Raise --n-eval or --max-train before trusting this.")
    if best["top1"] < 4 * best["chance"]:
        conclusive = False
        print(f"   [warn] the winning offset scores {best['top1']:.4f} against a "
              f"chance of {best['chance']:.4f}: the ridge is barely above chance for "
              "EVERY candidate, so this cannot discriminate. Check the target cache "
              "and the voxel/behav alignment before trusting any arm downstream.")
    if best["offset"] != -1:
        print("\n   rxfm/data.py:132 currently uses `all_trial - 1`. "
              f"This says it should be `all_trial + {best['offset']}`.")
    # Machine-readable verdict: train_arms.sbatch greps for this line, so an
    # inconclusive run can no longer unblock the ablation just by existing.
    print(f"\nGATE0: verdict={'CONCLUSIVE' if conclusive else 'INCONCLUSIVE'} "
          f"offset={best['offset']:+d} top1={best['top1']:.4f} "
          f"chance={best['chance']:.4f}")
    return 0 if conclusive else 1


if __name__ == "__main__":
    raise SystemExit(main())
