"""
BrainFlow — MindEyeV2 fMRI dataloader with DDP support.
All tunable parameters live in config.yaml — edit that file, not this one.

Usage:
    # Multi-GPU (num_gpus must match config.yaml → distribution.num_gpus)
    python launch.py --nproc_per_node=2 dataloader.py

    # Single-GPU / CPU
    python dataloader.py
"""

import os, io, h5py, torch, numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.nn.functional as F
import webdataset as wds
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import yaml

# ── Patch TCPStore (libuv not available in Windows CPU builds) ─────────────────
_OrigTCPStore = dist.TCPStore
class _PatchedTCPStore(_OrigTCPStore):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_libuv", False)
        super().__init__(*args, **kwargs)
dist.TCPStore = _PatchedTCPStore
torch.distributed.TCPStore = _PatchedTCPStore
import importlib
_rdzv_mod = importlib.import_module("torch.distributed.rendezvous")
_rdzv_mod.TCPStore = _PatchedTCPStore

load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# Config loading
# ══════════════════════════════════════════════════════════════════════════════
_CONFIG_PATH = Path(__file__).parent / "config.yaml"

def _load_cfg() -> dict:
    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {_CONFIG_PATH}. "
            "Please place config.yaml next to dataloader.py."
        )
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)

_CFG = _load_cfg()

# Convenience accessors
_DIST_CFG  = _CFG["distribution"]
_DATA_CFG  = _CFG["data"]
_DL_CFG    = _CFG["dataloader"]
_TRAIN_CFG = _CFG["training"]

# ── Resolved config values ─────────────────────────────────────────────────────
HF_REPO          = _DATA_CFG["hf_repo"]
DATA_DIR         = Path(_DATA_CFG["data_dir"])
TENSOR_CACHE     = DATA_DIR / _DATA_CFG["tensor_cache"]
HF_TOKEN         = os.environ.get("HF_TOKEN")
SUBJECTS         = list(_DATA_CFG["subjects"])
IMG_SIZE         = int(_DATA_CFG["img_size"])
MAX_TRAIN        = int(_DATA_CFG["max_train"])
MAX_TEST         = int(_DATA_CFG["max_test"])
DOWNLOAD_THREADS = int(_DATA_CFG["download_threads"])
FORCE_REBUILD    = bool(_DATA_CFG["force_rebuild"])

BATCH_SIZE       = int(_DL_CFG["batch_size_per_gpu"])
NUM_WORKERS      = int(_DL_CFG["num_workers"])
PREFETCH_FACTOR  = int(_DL_CFG["prefetch_factor"])
PIN_MEMORY       = bool(_DL_CFG["pin_memory"])
PERSISTENT_WKRS  = bool(_DL_CFG["persistent_workers"])
DROP_LAST_TRAIN  = bool(_DL_CFG["drop_last_train"])
DROP_LAST_TEST   = bool(_DL_CFG["drop_last_test"])

NUM_EPOCHS       = int(_TRAIN_CFG["num_epochs"])
LOG_EVERY        = int(_TRAIN_CFG["log_every_n_steps"])

_BACKEND_CFG     = _DIST_CFG["backend"]   # "auto" | "nccl" | "gloo"
INIT_METHOD      = _DIST_CFG["init_method"]


# ══════════════════════════════════════════════════════════════════════════════
# DDP helpers
# ══════════════════════════════════════════════════════════════════════════════
def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()

def rank() -> int:
    return dist.get_rank() if is_dist() else 0

def world_size() -> int:
    return dist.get_world_size() if is_dist() else 1

def is_main() -> bool:
    return rank() == 0


def setup_ddp():
    """Initialize the process group. Backend and init_method come from config."""
    if "LOCAL_RANK" not in os.environ:
        return  # single-GPU / CPU — no DDP needed

    local_rank = int(os.environ["LOCAL_RANK"])

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl" if _BACKEND_CFG == "auto" else _BACKEND_CFG
    else:
        backend = "gloo" if _BACKEND_CFG == "auto" else _BACKEND_CFG

    dist.init_process_group(backend=backend, init_method=INIT_METHOD)


def cleanup_ddp():
    if is_dist():
        dist.destroy_process_group()


# ══════════════════════════════════════════════════════════════════════════════
# Download helpers
# ══════════════════════════════════════════════════════════════════════════════
def dl_hf(fname: str) -> Path:
    """Download a single file from HuggingFace (skips if already present)."""
    p = DATA_DIR / fname
    if p.exists():
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(
        repo_id=HF_REPO, filename=fname,
        repo_type="dataset", local_dir=str(DATA_DIR), token=HF_TOKEN
    ))


def prefetch_subject_files(subject: int) -> dict:
    """Download all files for one subject using DOWNLOAD_THREADS parallel threads."""
    s = f"0{subject}"
    tasks = [f"betas_all_subj{s}_fp32_renorm.hdf5"]
    for split in ["train", "test"]:
        n = 8 if split == "train" else 1
        for i in range(n):
            tasks.append(f"wds/subj{s}/{split}/{i}.tar")

    with ThreadPoolExecutor(max_workers=DOWNLOAD_THREADS) as pool:
        futures  = {pool.submit(dl_hf, t): t for t in tasks}
        results  = {}
        for f in as_completed(futures):
            fname = futures[f]
            try:
                results[fname] = f.result()
            except Exception:
                pass  # shard may not exist for this subject
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Per-subject loader
# ══════════════════════════════════════════════════════════════════════════════
def load_subject(subject: int, coco_h5_path: Path) -> dict:
    s = f"0{subject}"

    file_map  = prefetch_subject_files(subject)
    beta_key  = f"betas_all_subj{s}_fp32_renorm.hdf5"
    beta_path = file_map.get(beta_key) or dl_hf(beta_key)

    # Collect available shard paths
    shards = []
    for split in ["train", "test"]:
        n = 8 if split == "train" else 1
        for i in range(n):
            k = f"wds/subj{s}/{split}/{i}.tar"
            if k in file_map and file_map[k].exists():
                shards.append((split, file_map[k]))

    # Read behavioural metadata from WebDataset shards
    behav = {"train": [], "test": []}
    for split, path in shards:
        ds = wds.WebDataset(str(path), handler=wds.warn_and_continue)
        for sample in ds:
            if "behav.npy" in sample:
                behav[split].append(np.load(io.BytesIO(sample["behav.npy"])))

    tr = np.concatenate(behav["train"])[:MAX_TRAIN]
    te = (np.concatenate(behav["test"])[:MAX_TEST]
          if behav["test"] else tr[-MAX_TEST:])

    all_trial = np.concatenate([tr[:, 5].astype(int), te[:, 5].astype(int)])
    all_coco  = np.concatenate([tr[:, 0].astype(int), te[:, 0].astype(int)])
    n_tr      = len(tr)

    # Load fMRI betas — vectorised unique-index reads, 256 MB chunk cache
    with h5py.File(beta_path, "r", rdcc_nbytes=256 * 1024 * 1024) as f:
        key         = list(f.keys())[0]
        n_tot, n_vox = f[key].shape
        idx0        = np.clip(all_trial - 1, 0, n_tot - 1)
        u, inv      = np.unique(idx0, return_inverse=True)
        fmri        = f[key][u][inv]

    # Load COCO images — same pattern
    with h5py.File(coco_h5_path, "r", rdcc_nbytes=256 * 1024 * 1024) as f:
        key     = list(f.keys())[0]
        cl      = np.clip(all_coco, 0, f[key].shape[0] - 1)
        uc, ic  = np.unique(cl, return_inverse=True)
        imgs    = f[key][uc][ic]

    # Image preprocessing (float32, CPU; GPU transfer happens in training loop)
    imgs = torch.from_numpy(imgs.astype(np.float32))
    if imgs.max() > 1.5:
        imgs = imgs / 255.0
    imgs = imgs.clamp(0, 1)
    if imgs.shape[-1] != IMG_SIZE:
        imgs = F.interpolate(imgs, IMG_SIZE,
                             mode="bilinear", align_corners=False).clamp(0, 1)

    return {
        "subject":      subject,
        "n_voxels":     n_vox,
        "fmri_train":   torch.tensor(fmri[:n_tr],  dtype=torch.float32),
        "fmri_test":    torch.tensor(fmri[n_tr:],  dtype=torch.float32),
        "images_train": imgs[:n_tr],
        "images_test":  imgs[n_tr:],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Dataset & collate
# ══════════════════════════════════════════════════════════════════════════════
class MindEyeDataset(Dataset):
    def __init__(self, fmri: torch.Tensor, images: torch.Tensor, subject_id: int):
        assert len(fmri) == len(images)
        cuda_up = torch.cuda.is_available() and PIN_MEMORY
        self.fmri       = fmri.pin_memory()   if cuda_up else fmri
        self.images     = images.pin_memory() if cuda_up else images
        self.subject_id = subject_id

    def __len__(self):
        return len(self.fmri)

    def __getitem__(self, idx):
        return {
            "fmri":    self.fmri[idx],
            "image":   self.images[idx],
            "subject": self.subject_id,
        }


def fast_collate(batch):
    """Stack tensors directly, skipping the default collate overhead."""
    return {
        "fmri":    torch.stack([b["fmri"]    for b in batch]),
        "image":   torch.stack([b["image"]   for b in batch]),
        "subject": torch.tensor([b["subject"] for b in batch], dtype=torch.long),
    }


# ══════════════════════════════════════════════════════════════════════════════
# DataLoader builder
# ══════════════════════════════════════════════════════════════════════════════
def build_dataloaders():
    """
    Returns (train_loader, test_loader, train_sampler).
    All parameters come from config.yaml — nothing is hard-coded here.
    """
    # ── Fast path: tensor cache already on disk ────────────────────────────────
    if TENSOR_CACHE.exists() and not FORCE_REBUILD:
        if is_main():
            print(f"✓ Loading tensor cache from {TENSOR_CACHE}")
        saved = torch.load(TENSOR_CACHE, map_location="cpu")

        train_datasets, test_datasets = [], []
        for subj in SUBJECTS:
            train_datasets.append(MindEyeDataset(
                saved[f"fmri_train_{subj}"], saved[f"imgs_train_{subj}"], subj))
            test_datasets.append(MindEyeDataset(
                saved[f"fmri_test_{subj}"],  saved[f"imgs_test_{subj}"],  subj))

        return _wrap_loaders(train_datasets, test_datasets)

    # ── Slow path: download → process → cache (rank 0 only) ───────────────────
    if is_main():
        coco_h5 = dl_hf("COCO_73k_subj_indices.hdf5")
        train_datasets, test_datasets = [], []
        save_dict = {}

        for subj in tqdm(SUBJECTS, desc="Loading subjects"):
            data = load_subject(subj, coco_h5)
            print(f"  subj{subj:02d} → "
                  f"fmri_train: {tuple(data['fmri_train'].shape)} | "
                  f"imgs_train: {tuple(data['images_train'].shape)} | "
                  f"voxels: {data['n_voxels']}")

            save_dict[f"fmri_train_{subj}"] = data["fmri_train"]
            save_dict[f"fmri_test_{subj}"]  = data["fmri_test"]
            save_dict[f"imgs_train_{subj}"] = data["images_train"]
            save_dict[f"imgs_test_{subj}"]  = data["images_test"]

            train_datasets.append(MindEyeDataset(
                data["fmri_train"], data["images_train"], subj))
            test_datasets.append(MindEyeDataset(
                data["fmri_test"],  data["images_test"],  subj))

        print(f"\nSaving tensor cache → {TENSOR_CACHE}  (~12-15 GB, one-time)")
        torch.save(save_dict, TENSOR_CACHE)
        print("✓ Tensor cache saved.\n")

    # All ranks sync before building loaders
    if is_dist():
        dist.barrier()

        if not is_main():
            saved = torch.load(TENSOR_CACHE, map_location="cpu")
            train_datasets, test_datasets = [], []
            for subj in SUBJECTS:
                train_datasets.append(MindEyeDataset(
                    saved[f"fmri_train_{subj}"], saved[f"imgs_train_{subj}"], subj))
                test_datasets.append(MindEyeDataset(
                    saved[f"fmri_test_{subj}"],  saved[f"imgs_test_{subj}"],  subj))

    return _wrap_loaders(train_datasets, test_datasets)


def _wrap_loaders(train_datasets, test_datasets):
    """Wrap datasets into DDP-aware DataLoaders using config.yaml settings."""
    train_set = ConcatDataset(train_datasets)
    test_set  = ConcatDataset(test_datasets)

    # DistributedSampler shards across however many ranks are active
    train_sampler = DistributedSampler(
        train_set,
        num_replicas=world_size(),
        rank=rank(),
        shuffle=True,
        drop_last=DROP_LAST_TRAIN,
    ) if is_dist() else None

    test_sampler = DistributedSampler(
        test_set,
        num_replicas=world_size(),
        rank=rank(),
        shuffle=False,
        drop_last=DROP_LAST_TEST,
    ) if is_dist() else None

    persistent = PERSISTENT_WKRS and NUM_WORKERS > 0
    pf         = PREFETCH_FACTOR  if NUM_WORKERS > 0 else None

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=persistent,
        prefetch_factor=pf,
        collate_fn=fast_collate,
        drop_last=DROP_LAST_TRAIN,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        sampler=test_sampler,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=persistent,
        prefetch_factor=pf,
        collate_fn=fast_collate,
        drop_last=DROP_LAST_TEST,
    )

    return train_loader, test_loader, train_sampler


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    setup_ddp()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device     = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )

    if is_main():
        n_gpus = world_size()
        print(f"Running on {n_gpus} process(es)  "
              f"[backend: {_BACKEND_CFG}  |  "
              f"effective batch: {BATCH_SIZE} × {n_gpus} = {BATCH_SIZE * n_gpus}]\n")

    train_loader, test_loader, train_sampler = build_dataloaders()

    if is_main():
        print(f"Train batches per GPU : {len(train_loader)}")
        print(f"Test  batches per GPU : {len(test_loader)}")
        print(f"Effective batch size  : {BATCH_SIZE * world_size()}\n")

    # ── Training loop skeleton ────────────────────────────────────────────────
    for epoch in range(NUM_EPOCHS):

        # CRITICAL: reseed sampler each epoch so each GPU gets a fresh permutation
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for step, batch in enumerate(train_loader):
            fmri    = batch["fmri"].to(device, non_blocking=True)
            images  = batch["image"].to(device, non_blocking=True)
            subject = batch["subject"].to(device, non_blocking=True)

            # ── your model forward / backward pass here ──
            # loss = model(fmri, images)
            # loss.backward()
            # optimizer.step()

            if is_main() and step % LOG_EVERY == 0:
                print(f"epoch {epoch} | step {step:04d} | "
                      f"fmri {tuple(fmri.shape)} | img {tuple(images.shape)}")

            if step == 2:
                break   # remove in real training

    cleanup_ddp()