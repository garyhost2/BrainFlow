from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, Sampler

class Step1Dataset(Dataset):
    def __init__(self, fmri, target_emb, images, subject_id, fmri_mu, fmri_std,
                 cls_emb=None, augment=False, noise_std=0.0, lat=None):
        assert len(fmri) == len(target_emb) == len(images)
        self.fmri = ((fmri.float() - fmri_mu) / fmri_std)

        self.emb = target_emb
        self.cls = cls_emb
        self.lat = lat
        self.images = images
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
        item = {"fmri": f, "emb": self.emb[i].float(), "image": img,
                "subject": self.subject_id}
        if self.cls is not None:
            item["cls"] = self.cls[i].float()
        if self.lat is not None:
            item["lat"] = self.lat[i].float()
        return item

def _collate(batch):
    out = {
        "fmri": torch.stack([b["fmri"] for b in batch]),
        "emb": torch.stack([b["emb"] for b in batch]),
        "image": torch.stack([b["image"] for b in batch]),
        "subject": int(batch[0]["subject"]),
    }
    if "cls" in batch[0]:
        out["cls"] = torch.stack([b["cls"] for b in batch])
    if "lat" in batch[0]:
        out["lat"] = torch.stack([b["lat"] for b in batch])
    return out

class SubjectSampler(Sampler):
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
            for s in range(0, full, self.bs):
                batches.append(idx[s:s + self.bs].tolist())
            if not self.drop_last and full < n:
                batches.append(idx[full:].tolist())
        if self.shuffle:
            order = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in order]
        yield from batches

    def __len__(self):
        if self.drop_last:
            return sum(L // self.bs for L in self.lengths)
        return sum(-(-L // self.bs) for L in self.lengths)

@dataclass
class LoaderBundle:
    train: DataLoader
    eval: DataLoader
    voxels: dict
    train_sampler: SubjectSampler
    val: Optional[DataLoader] = None

def _loader(dataset, sampler, num_workers):
    return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers,
                      pin_memory=True, persistent_workers=num_workers > 0,
                      prefetch_factor=4 if num_workers > 0 else None, collate_fn=_collate)

def build_step1_loaders(tensors: dict, targets: dict, subjects: list[int],
                        batch_size: int, num_workers: int = 8,
                        fmri_noise_std: float = 0.0, val_frac: float = 0.0,
                        seed: int = 0) -> LoaderBundle:
    fmri_stats = tensors.get("fmri_stats", {})
    voxels = tensors.get("voxels", {})
    train_sets, val_sets, eval_sets = [], [], []
    for s in subjects:
        if s not in voxels:
            voxels[s] = tensors[f"fmri_train_{s}"].shape[1]
        mu = fmri_stats[s]["mu"]; std = fmri_stats[s]["std"].clamp_min(1e-6)
        cls_tr = targets.get(f"cls_train_{s}")
        cls_te = targets.get(f"cls_test_{s}")
        lat_tr = targets.get(f"lat_train_{s}")
        lat_te = targets.get(f"lat_test_{s}")
        tr_aug = Step1Dataset(
            tensors[f"fmri_train_{s}"], targets[f"emb_train_{s}"], tensors[f"imgs_train_{s}"],
            s, mu, std, cls_emb=cls_tr, augment=True, noise_std=fmri_noise_std, lat=lat_tr)
        if val_frac > 0:
            tr_plain = Step1Dataset(
                tensors[f"fmri_train_{s}"], targets[f"emb_train_{s}"], tensors[f"imgs_train_{s}"],
                s, mu, std, cls_emb=cls_tr, augment=False, lat=lat_tr)
            n = len(tr_aug)
            g = torch.Generator().manual_seed(seed + int(s))
            perm = torch.randperm(n, generator=g).tolist()
            n_val = max(1, int(n * val_frac))
            train_sets.append(Subset(tr_aug, perm[n_val:]))
            val_sets.append(Subset(tr_plain, perm[:n_val]))
        else:
            train_sets.append(tr_aug)
        eval_sets.append(Step1Dataset(
            tensors[f"fmri_test_{s}"], targets[f"emb_test_{s}"], tensors[f"imgs_test_{s}"],
            s, mu, std, cls_emb=cls_te, augment=False, lat=lat_te))

    train_sampler = SubjectSampler([len(d) for d in train_sets], batch_size, shuffle=True, drop_last=True)
    eval_sampler = SubjectSampler([len(d) for d in eval_sets], batch_size, shuffle=False, drop_last=False)
    train = _loader(ConcatDataset(train_sets), train_sampler, num_workers)
    ev = _loader(ConcatDataset(eval_sets), eval_sampler, num_workers)

    val = None
    if val_sets:
        val_sampler = SubjectSampler([len(d) for d in val_sets], batch_size, shuffle=False, drop_last=False)
        val = _loader(ConcatDataset(val_sets), val_sampler, num_workers)
    return LoaderBundle(train=train, eval=ev, voxels=voxels, train_sampler=train_sampler, val=val)
