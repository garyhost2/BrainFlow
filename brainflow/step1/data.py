"""Step-1 data: reuse the existing tensor cache, attach decoder-target embeddings.

We deliberately do NOT couple to the full multi-phase `build_dataloaders`.  We
load the already-built `all_subjects_tensors.pt` (fMRI + uint8 images + train
stats) and the Step-1 target-embedding cache, then build simple per-subject
datasets.  Single-subject batches (homogeneous voxel shape) via SubjectSampler.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler


class Step1Dataset(Dataset):
    """fMRI (z-scored with TRAIN stats) + raw target embedding + image (for eval)."""
    def __init__(self, fmri, target_emb, images, subject_id, fmri_mu, fmri_std,
                 augment=False, noise_std=0.0):
        assert len(fmri) == len(target_emb) == len(images)
        self.fmri = ((fmri.float() - fmri_mu) / fmri_std)
        self.emb = target_emb.float()
        self.images = images  # uint8
        self.subject_id = int(subject_id)
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self):
        return len(self.fmri)

    def __getitem__(self, i):
        f = self.fmri[i]
        if self.augment and self.noise_std > 0:
            f = f + torch.randn_like(f) * self.noise_std
        img = self.images[i]
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        return {"fmri": f, "emb": self.emb[i], "image": img, "subject": self.subject_id}


def _collate(batch):
    return {
        "fmri": torch.stack([b["fmri"] for b in batch]),
        "emb": torch.stack([b["emb"] for b in batch]),
        "image": torch.stack([b["image"] for b in batch]),
        "subject": int(batch[0]["subject"]),  # homogeneous batch
    }


class SubjectSampler(Sampler):
    """Single-subject batches across a ConcatDataset of per-subject datasets."""
    def __init__(self, lengths, batch_size, shuffle=True, drop_last=True, seed=0):
        self.lengths = list(lengths)
        self.bs = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.ranges = []
        off = 0
        for L in self.lengths:
            self.ranges.append((off, off + L)); off += L

    def set_epoch(self, e):
        self.epoch = e

    def __iter__(self):
        g = torch.Generator(); g.manual_seed(self.seed + self.epoch)
        batches = []
        for (start, end) in self.ranges:
            n = end - start
            idx = torch.arange(start, end)
            if self.shuffle:
                idx = idx[torch.randperm(n, generator=g)]
            full = (n // self.bs) * self.bs
            if full == 0:
                if not self.drop_last:
                    batches.append(idx.tolist())
                continue
            idx = idx[:full]
            for s in range(0, full, self.bs):
                batches.append(idx[s:s + self.bs].tolist())
        if self.shuffle:
            order = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in order]
        yield from batches

    def __len__(self):
        return sum(L // self.bs for L in self.lengths)


@dataclass
class LoaderBundle:
    train: DataLoader
    eval: DataLoader
    voxels: dict
    train_sampler: SubjectSampler


def build_step1_loaders(tensors: dict, targets: dict, subjects: list[int],
                        batch_size: int, num_workers: int = 8,
                        fmri_noise_std: float = 0.0) -> LoaderBundle:
    fmri_stats = tensors.get("fmri_stats", {})
    voxels = tensors.get("voxels", {})
    train_sets, eval_sets = [], []
    for s in subjects:
        if s not in voxels:
            voxels[s] = tensors[f"fmri_train_{s}"].shape[1]
        mu = fmri_stats[s]["mu"]; std = fmri_stats[s]["std"].clamp_min(1e-6)
        train_sets.append(Step1Dataset(
            tensors[f"fmri_train_{s}"], targets[f"emb_train_{s}"], tensors[f"imgs_train_{s}"],
            s, mu, std, augment=True, noise_std=fmri_noise_std))
        eval_sets.append(Step1Dataset(
            tensors[f"fmri_test_{s}"], targets[f"emb_test_{s}"], tensors[f"imgs_test_{s}"],
            s, mu, std, augment=False))

    train_set = ConcatDataset(train_sets)
    eval_set = ConcatDataset(eval_sets)
    train_sampler = SubjectSampler([len(d) for d in train_sets], batch_size, shuffle=True, drop_last=True)
    eval_sampler = SubjectSampler([len(d) for d in eval_sets], batch_size, shuffle=False, drop_last=False)

    train = DataLoader(train_set, batch_sampler=train_sampler, num_workers=num_workers,
                       pin_memory=True, persistent_workers=num_workers > 0,
                       prefetch_factor=4 if num_workers > 0 else None, collate_fn=_collate)
    ev = DataLoader(eval_set, batch_sampler=eval_sampler, num_workers=num_workers,
                    pin_memory=True, persistent_workers=num_workers > 0,
                    prefetch_factor=4 if num_workers > 0 else None, collate_fn=_collate)
    return LoaderBundle(train=train, eval=ev, voxels=voxels, train_sampler=train_sampler)
