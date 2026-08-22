"""Brain-space denoiser: one noisy trial -> the response that trial is a sample of.

NSD presents each image ~3 times. build_or_load_tensors averages the repeats on
the TEST split (data.py:173) and keeps every TRAIN trial, so the train half is a
paired denoising dataset nobody uses: ~27k trials over ~9k images.

Measured on the cached betas (scripts/repeat_geometry.py), two presentations of
one image correlate at cos 0.15 against 0.006 for different images. That 0.15 is
what survives GLMsingle -- NSD ships betas_fithrf_GLMdenoise_RR, i.e. per-voxel
HRF fitting, cross-validated noise regressors and ridge already applied. So the
baseline here is the field's standard single-trial denoiser, and the opening is
that GLMsingle is per-voxel and linear while the residual variability is
anisotropic: direction repeats 1.75-3.55x better than norm in all four subjects.

Two invariants this module exists to enforce, because getting either wrong
produces a number that looks good and means nothing:

* **Leave-one-out targets.** The target for trial i is the mean of the OTHER
  repeats of its image. Regressing onto a mean that contains i lets the model
  read the answer off its own input.
* **Grouping by image, not by trial.** A held-out split that puts two repeats of
  one image on opposite sides is measuring memorisation.
"""
from __future__ import annotations

import hashlib
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F


class RepeatIndex:
    """Groups trials by stimulus and serves leave-one-out targets.

    Repeats share a byte-identical image tensor, so the grouping needs no
    metadata beyond what the cache already holds. Hashing ~27k x 150 KB costs a
    few seconds, so the result is cached to disk and keyed by the tensor cache's
    own stamp.
    """

    def __init__(self, groups: list[list[int]], n_trials: int):
        keep = [g for g in groups if len(g) >= 2]
        if not keep:
            raise ValueError("no stimulus had >= 2 presentations; LOO is undefined")
        self.groups = keep
        self.n_trials = n_trials
        # trial -> group id, and group id -> size. -1 marks a singleton trial,
        # which has no LOO target and is excluded from every split.
        gid = torch.full((n_trials,), -1, dtype=torch.long)
        for k, g in enumerate(keep):
            gid[torch.tensor(g, dtype=torch.long)] = k
        self.gid = gid
        self.sizes = torch.tensor([len(g) for g in keep], dtype=torch.float32)
        self.valid = (gid >= 0).nonzero(as_tuple=True)[0]

    @staticmethod
    def build(imgs: torch.Tensor, chunk: int = 256) -> "RepeatIndex":
        buckets: dict[bytes, list[int]] = {}
        n = imgs.shape[0]
        for i0 in range(0, n, chunk):
            block = imgs[i0:i0 + chunk].contiguous()
            b = block.shape[0]
            # A 1-D memoryview slices without copying; a 2-D one raises
            # "multi-dimensional sub-views are not implemented", and .tobytes()
            # per row would copy 150 KB 27,000 times.
            mv = memoryview(block.numpy().reshape(-1))
            stride = len(mv) // b
            for j in range(b):
                h = hashlib.blake2b(mv[j * stride:(j + 1) * stride],
                                    digest_size=16).digest()
                buckets.setdefault(h, []).append(i0 + j)
        return RepeatIndex(list(buckets.values()), n)

    @staticmethod
    def cached(imgs: torch.Tensor, path: pathlib.Path) -> "RepeatIndex":
        if path.is_file():
            blob = torch.load(path, map_location="cpu")
            if blob.get("n_trials") == imgs.shape[0]:
                return RepeatIndex(blob["groups"], blob["n_trials"])
        idx = RepeatIndex.build(imgs)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"groups": idx.groups, "n_trials": idx.n_trials}, path)
        return idx

    def split(self, val_frac: float, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Split by GROUP, so no image contributes trials to both sides."""
        g = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(self.groups), generator=g)
        n_val = max(1, int(round(val_frac * len(self.groups))))
        val_groups = set(order[:n_val].tolist())
        tr, va = [], []
        for k, grp in enumerate(self.groups):
            (va if k in val_groups else tr).extend(grp)
        return torch.tensor(sorted(tr)), torch.tensor(sorted(va))

    def loo_machinery(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-group sums and sizes, on x's device. Targets are then O(1):

            target_i = (group_sum[gid_i] - x_i) / (size[gid_i] - 1)
        """
        sums = torch.zeros(len(self.groups), x.shape[1], device=x.device, dtype=x.dtype)
        gid = self.gid.to(x.device)
        valid = self.valid.to(x.device)
        sums.index_add_(0, gid[valid], x[valid])
        return sums, self.sizes.to(x.device)


def loo_target(x: torch.Tensor, idx: torch.Tensor, gid: torch.Tensor,
               sums: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    g = gid[idx]
    return (sums[g] - x[idx]) / (sizes[g] - 1.0).unsqueeze(1)


class BrainDenoiser(nn.Module):
    """Per-subject projections around a shared residual trunk.

    The output layer is zero-initialised, so at step 0 the model is exactly the
    identity and can only improve on "do nothing" -- the same discipline that
    made the flow's v == 0 the safe starting point, and the reason a run that
    fails to learn shows up as a flat curve rather than a corrupted one.
    """

    def __init__(self, voxels: dict[int, int], width: int = 2048,
                 depth: int = 4, dropout: float = 0.15):
        super().__init__()
        self.subjects = sorted(voxels)
        self.inp = nn.ModuleDict({str(s): nn.Linear(v, width) for s, v in voxels.items()})
        self.out = nn.ModuleDict({str(s): nn.Linear(width, v) for s, v in voxels.items()})
        blocks = []
        for _ in range(depth):
            blocks.append(nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, width * 2), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(width * 2, width)))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(width)
        for s in voxels:
            nn.init.zeros_(self.out[str(s)].weight)
            nn.init.zeros_(self.out[str(s)].bias)

    def forward(self, x: torch.Tensor, subject: int) -> torch.Tensor:
        h = self.inp[str(subject)](x)
        for b in self.blocks:
            h = h + b(h)
        return x + self.out[str(subject)](self.norm(h))


def denoise_loss(pred: torch.Tensor, target: torch.Tensor,
                 lambda_cos: float = 1.0) -> dict[str, torch.Tensor]:
    """MSE plus a direction term.

    repeat_geometry says direction is the repeatable part (1.75-3.55x the norm),
    so supervising it explicitly is not a hedge -- it is where the recoverable
    signal was measured to be.
    """
    mse = F.mse_loss(pred, target)
    cos = 1.0 - F.cosine_similarity(pred.float(), target.float(), dim=1).mean()
    return {"mse": mse, "cos": cos, "loss": mse + lambda_cos * cos}
