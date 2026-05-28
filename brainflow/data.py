"""Multi-subject MindEyeV2 data pipeline with DDP support."""
from __future__ import annotations
import os, io, gc, random, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
import h5py
import webdataset as wds
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

from .config import Config

HF_TOKEN = os.environ.get("HF_TOKEN")
_CACHE_FORMAT_V2 = "v2_imagenet_norm"
_LATENT_CACHE_FORMAT_V2 = "v2_sdvae_scaled"
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])



def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()

def rank() -> int:
    return dist.get_rank() if is_dist() else 0

def world_size() -> int:
    return dist.get_world_size() if is_dist() else 1

def is_main() -> bool:
    return rank() == 0


def _git_sha() -> str:
    try:
        root = Path(__file__).resolve().parent.parent
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except Exception:
        return "unknown"


def _v2_cache_meta() -> dict[str, str]:
    return {
        "format": _CACHE_FORMAT_V2,
        "clip_model": "ViT-L-14",
        "pretrained": "openai",
        "git_sha": _git_sha(),
    }


def _assert_v2_cache(cache: Path, payload: dict) -> None:
    meta = payload.get("_meta")
    if not isinstance(meta, dict) or meta.get("format") != _CACHE_FORMAT_V2:
        raise RuntimeError(
            f"Cache format mismatch for {cache}. Expected _meta.format={_CACHE_FORMAT_V2!r}. "
            f"Set force_rebuild: true to rebuild this cache."
        )


def _latent_cache_meta() -> dict[str, str]:
    return {
        "format": _LATENT_CACHE_FORMAT_V2,
        "git_sha": _git_sha(),
    }


def _assert_latent_cache(cache: Path, payload: dict) -> None:
    meta = payload.get("_meta")
    if not isinstance(meta, dict) or meta.get("format") != _LATENT_CACHE_FORMAT_V2:
        raise RuntimeError(
            f"Cache format mismatch for {cache}. Expected _meta.format={_LATENT_CACHE_FORMAT_V2!r}. "
            f"Set force_rebuild: true to rebuild this cache."
        )


# ─── HF download ──────────────────────────────────────────────────────────────
def dl_hf(fname: str, data_dir: Path, repo: str) -> Path:
    p = data_dir / fname
    if p.exists():
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(
        repo_id=repo, filename=fname, repo_type="dataset",
        local_dir=str(data_dir), token=HF_TOKEN,
    ))


def prefetch_subject_files(subject: int, cfg: Config) -> dict:
    s = f"0{subject}"
    tasks = [f"betas_all_subj{s}_fp32_renorm.hdf5"]
    for split, n in [("train", 64), ("test", 8)]:
        for i in range(n):
            tasks.append(f"wds/subj{s}/{split}/{i}.tar")
    results = {}
    with ThreadPoolExecutor(max_workers=cfg.download_threads) as pool:
        futures = {pool.submit(dl_hf, t, cfg.data_dir, cfg.hf_repo): t for t in tasks}
        for f in as_completed(futures):
            try:
                results[futures[f]] = f.result()
            except Exception:
                pass
    return results


# ─── Per-subject loader (raw fMRI + images) ───────────────────────────────────
def load_subject(subject: int, coco_h5: Path, cfg: Config) -> dict:
    s = f"0{subject}"
    file_map = prefetch_subject_files(subject, cfg)
    beta_key = f"betas_all_subj{s}_fp32_renorm.hdf5"
    beta_path = file_map.get(beta_key) or dl_hf(beta_key, cfg.data_dir, cfg.hf_repo)

    shards = []
    for split, n in [("train", 64), ("test", 8)]:
        for i in range(n):
            k = f"wds/subj{s}/{split}/{i}.tar"
            if k in file_map and file_map[k].exists():
                shards.append((split, file_map[k]))

    behav = {"train": [], "test": []}
    for split, path in shards:
        for sample in wds.WebDataset(str(path), handler=wds.warn_and_continue):
            if "behav.npy" in sample:
                behav[split].append(np.load(io.BytesIO(sample["behav.npy"])))

    tr = np.concatenate(behav["train"])[:cfg.max_train]
    te = (np.concatenate(behav["test"])[:cfg.max_test]
          if behav["test"] else tr[-cfg.max_test:])
    all_trial = np.concatenate([tr[:, 5].astype(int), te[:, 5].astype(int)])
    all_coco = np.concatenate([tr[:, 0].astype(int), te[:, 0].astype(int)])
    n_tr = len(tr)
    te_coco = te[:, 0].astype(int)

    with h5py.File(beta_path, "r", rdcc_nbytes=256 * 1024 * 1024) as f:
        key = list(f.keys())[0]
        n_tot, n_vox = f[key].shape
        idx0 = np.clip(all_trial - 1, 0, n_tot - 1)
        u, inv = np.unique(idx0, return_inverse=True)
        fmri = f[key][u][inv]

    with h5py.File(coco_h5, "r", rdcc_nbytes=256 * 1024 * 1024) as f:
        key = list(f.keys())[0]
        cl = np.clip(all_coco, 0, f[key].shape[0] - 1)
        uc, ic = np.unique(cl, return_inverse=True)
        imgs = f[key][uc][ic]

    imgs = torch.from_numpy(imgs.astype(np.float32))
    if imgs.max() > 1.5:
        imgs = imgs / 255.0
    imgs = imgs.clamp(0, 1)
    if imgs.shape[-1] != cfg.img_size:
        imgs = F.interpolate(imgs, cfg.img_size, mode="bilinear",
                             align_corners=False).clamp(0, 1)
    # Store images as uint8 to keep cache small (~4× smaller than float32).
    # Decoded back to float in NSDDataset / CLIP / VAE encoders.
    imgs = (imgs * 255.0).round().clamp(0, 255).to(torch.uint8)

    fmri_train = torch.tensor(fmri[:n_tr], dtype=torch.float32)
    fmri_test_raw = torch.tensor(fmri[n_tr:], dtype=torch.float32)
    imgs_train = imgs[:n_tr]
    imgs_test_raw = imgs[n_tr:]

    # B4: Average test betas across repetitions of the same coco_id
    # NSD shows each test image 3× per subject; average for cleaner signal
    n_before = len(te_coco)
    unique_coco = np.unique(te_coco)
    # Build averaged fmri and deduplicated images per unique coco_id (te_coco is already a numpy array)
    avg_fmri_rows = []
    avg_img_rows = []
    for uid in unique_coco:
        mask = te_coco == uid
        avg_fmri_rows.append(fmri_test_raw[mask].mean(0))
        avg_img_rows.append(imgs_test_raw[mask][0])
    fmri_test = torch.stack(avg_fmri_rows)
    imgs_test = torch.stack(avg_img_rows)
    n_after = len(fmri_test)
    print(f"  subj{subject:02d}: test trials averaged {n_before} → {n_after} unique images")

    # Compute train fMRI statistics (used to normalise test set in NSDDataset)
    fmri_mu = fmri_train.mean(0)
    fmri_std = fmri_train.std(0).clamp(1e-6)

    return {
        "subject": subject,
        "n_voxels": n_vox,
        "fmri_train": fmri_train,
        "fmri_test": fmri_test,
        "images_train": imgs_train,
        "images_test": imgs_test,
        "fmri_mu": fmri_mu,
        "fmri_std": fmri_std,
    }



def build_or_load_tensors(cfg: Config) -> dict:
    cache = cfg.data_dir / cfg.tensor_cache
    if cache.exists() and not cfg.force_rebuild:
        if is_main():
            print(f"✓ Loading tensor cache: {cache}")
        out = torch.load(cache, map_location="cpu")
        if is_dist(): dist.barrier()
        return out

    if not is_main():
        if is_dist(): dist.barrier()
        return torch.load(cache, map_location="cpu")

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    coco = dl_hf("coco_images_224_float16.hdf5", cfg.data_dir, cfg.hf_repo)
    out = {"voxels": {}, "fmri_stats": {}}
    for subj in tqdm(cfg.subjects, desc="Loading subjects"):
        d = load_subject(subj, coco, cfg)
        out[f"fmri_train_{subj}"] = d["fmri_train"]
        out[f"fmri_test_{subj}"] = d["fmri_test"]
        out[f"imgs_train_{subj}"] = d["images_train"]
        out[f"imgs_test_{subj}"] = d["images_test"]
        out["voxels"][subj] = d["n_voxels"]
        out["fmri_stats"][subj] = {"mu": d["fmri_mu"], "std": d["fmri_std"]}
        print(f"  subj{subj:02d}: train={tuple(d['fmri_train'].shape)} "
              f"test={tuple(d['fmri_test'].shape)} voxels={d['n_voxels']}")
    print(f"\nSaving tensor cache → {cache}")
    torch.save(out, cache)
    if is_dist(): dist.barrier()
    return out


# ─── CLIP encoding (rank-0 only, cached) ──────────────────────────────────────
def compute_or_load_clip(tensors: dict, cfg: Config) -> dict:
    cache = cfg.data_dir / cfg.clip_cache
    if cache.exists() and not cfg.force_rebuild:
        if is_main():
            print(f"✓ Loading CLIP cache: {cache}")
        out = torch.load(cache, map_location="cpu")
        _assert_v2_cache(cache, out)
        if is_dist(): dist.barrier()
        return out
    if not is_main():
        if is_dist(): dist.barrier()
        out = torch.load(cache, map_location="cpu")
        _assert_v2_cache(cache, out)
        return out

    import open_clip
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading CLIP ViT-L/14 …")
    m, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    m = m.eval().to(device)
    for p in m.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def _enc(imgs, bs=32):
        out = []
        for i in tqdm(range(0, len(imgs), bs), desc="CLIP", leave=False):
            chunk = imgs[i:i+bs]
            if chunk.dtype == torch.uint8:
                chunk = chunk.float() / 255.0
            b = F.interpolate(chunk, 224, mode="bilinear",
                              align_corners=False).to(device)
            b = (b - _CLIP_MEAN.to(device).view(1, 3, 1, 1)) / _CLIP_STD.to(device).view(1, 3, 1, 1)
            e = F.normalize(m.encode_image(b).float(), dim=-1).cpu()
            out.append(e)
        return torch.cat(out)

    res = {}
    for subj in cfg.subjects:
        print(f"CLIP encoding subj{subj:02d} …")
        res[f"clip_train_{subj}"] = _enc(tensors[f"imgs_train_{subj}"])
        res[f"clip_test_{subj}"] = _enc(tensors[f"imgs_test_{subj}"])
    res["_meta"] = _v2_cache_meta()
    del m; gc.collect(); torch.cuda.empty_cache()
    torch.save(res, cache)
    print(f"✓ CLIP cache saved: {cache}")
    if is_dist(): dist.barrier()
    return res


# ─── VAE latent encoding (rank-0 only, cached) ────────────────────────────────
def compute_or_load_latents(tensors: dict, cfg: Config) -> dict:
    from .vae import FrozenVAE
    cache = cfg.data_dir / cfg.latent_cache
    if cache.exists() and not cfg.force_rebuild:
        if is_main():
            print(f"✓ Loading VAE latent cache: {cache}")
        out = torch.load(cache, map_location="cpu")
        _assert_latent_cache(cache, out)
        if is_dist(): dist.barrier()
        return out
    if not is_main():
        if is_dist(): dist.barrier()
        out = torch.load(cache, map_location="cpu")
        _assert_latent_cache(cache, out)
        return out

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = FrozenVAE(cache_dir=cfg.data_dir / "hf_cache").to(device)

    @torch.no_grad()
    def _enc(imgs, bs=16):
        out = []
        for i in tqdm(range(0, len(imgs), bs), desc="VAE", leave=False):
            chunk = imgs[i:i+bs]
            if chunk.dtype == torch.uint8:
                chunk = chunk.float() / 255.0
            out.append(vae.encode(chunk.to(device)).cpu())
        return torch.cat(out)

    res = {}
    for subj in cfg.subjects:
        print(f"VAE encoding subj{subj:02d} …")
        res[f"lat_train_{subj}"] = _enc(tensors[f"imgs_train_{subj}"])
        res[f"lat_test_{subj}"] = _enc(tensors[f"imgs_test_{subj}"])
    res["_meta"] = _latent_cache_meta()
    del vae; gc.collect(); torch.cuda.empty_cache()
    torch.save(res, cache)
    print(f"✓ VAE latent cache saved: {cache}")
    if is_dist(): dist.barrier()
    return res


# ─── CLIP patch token encoding (rank-0 only, cached) ──────────────────────────
def compute_or_load_clip_patches(tensors: dict, cfg: Config) -> dict:
    """Extract and cache ViT-L/14 patch tokens (196 → 256 zero-padded, fp16).

    Returns dict with keys "clip_patch_train_{subj}" and "clip_patch_test_{subj}",
    each a (N, 256, 1024) fp16 tensor stored on CPU.
    """
    cache = cfg.data_dir / cfg.clip_patch_cache
    if cache.exists() and not cfg.force_rebuild:
        if is_main():
            print(f"✓ Loading CLIP patch cache: {cache}")
        out = torch.load(cache, map_location="cpu")
        _assert_v2_cache(cache, out)
        if is_dist(): dist.barrier()
        return out
    if not is_main():
        if is_dist(): dist.barrier()
        out = torch.load(cache, map_location="cpu")
        _assert_v2_cache(cache, out)
        return out

    import open_clip
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading CLIP ViT-L/14 for patch token extraction …")
    m, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    m = m.eval().to(device)
    for p in m.parameters(): p.requires_grad_(False)

    @torch.no_grad()
    def _vit_patch_tokens(visual, x: torch.Tensor) -> torch.Tensor:
        """Unrolled ViT-L/14 forward → patch tokens (B, 196, 1024).

        Compatible with all open_clip versions (avoids output_tokens kwarg
        which was added in open_clip >= 2.20.0).
        Applies ln_post to patch tokens only (no projection).
        """
        # Patch embedding: (B, C, H, W) → (B, 196, 1024)
        x = visual.conv1(x)                                          # (B, 1024, 14, 14)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1) # (B, 196, 1024)
        B = x.shape[0]
        # Prepend CLS token
        cls = visual.class_embedding.to(x.dtype).reshape(1, 1, -1).expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)                               # (B, 197, 1024)
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)
        # Transformer (open_clip uses LND layout internally)
        x = x.permute(1, 0, 2)   # NLD → LND
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)   # LND → NLD
        # Apply ln_post to patch tokens only (drop CLS at index 0)
        patch_tok = visual.ln_post(x[:, 1:, :])                      # (B, 196, 1024)
        return patch_tok

    @torch.no_grad()
    def _extract_patches(imgs, bs=16):
        """imgs: (N, 3, H, W) uint8. Returns (N, 256, 1024) fp16."""
        out = []
        for i in tqdm(range(0, len(imgs), bs), desc="CLIP patches", leave=False):
            chunk = imgs[i:i + bs]
            if chunk.dtype == torch.uint8:
                chunk = chunk.float() / 255.0
            b = F.interpolate(chunk, 224, mode="bilinear",
                              align_corners=False).to(device)
            b = (b - _CLIP_MEAN.to(device).view(1, 3, 1, 1)) / _CLIP_STD.to(device).view(1, 3, 1, 1)
            patch_tok = _vit_patch_tokens(m.visual, b)               # (B, 196, 1024)
            # Zero-pad 196 → 256 so the grid is exactly 16×16
            padded = F.pad(patch_tok.float(), (0, 0, 0, 256 - patch_tok.shape[1]))
            out.append(padded.cpu().half())
        return torch.cat(out)  # (N, 256, 1024) fp16

    res = {}
    for subj in cfg.subjects:
        print(f"CLIP patch extraction subj{subj:02d} …")
        res[f"clip_patch_train_{subj}"] = _extract_patches(
            tensors[f"imgs_train_{subj}"])
        res[f"clip_patch_test_{subj}"] = _extract_patches(
            tensors[f"imgs_test_{subj}"])
    res["_meta"] = _v2_cache_meta()
    del m; gc.collect(); torch.cuda.empty_cache()
    torch.save(res, cache)
    print(f"✓ CLIP patch cache saved: {cache}")
    if is_dist(): dist.barrier()
    return res


class NSDDataset(Dataset):
    def __init__(self, fmri, latents, images, clip_embs, subject_id,
                 augment=False, cfg: Config | None = None,
                 fmri_mu=None, fmri_std=None,
                 clip_patches=None):
        assert len(fmri) == len(latents) == len(images) == len(clip_embs)
        self.fmri = fmri.float()
        self.latents = latents.float()
        # Keep images as uint8 in RAM (decode to float on __getitem__)
        self.images = images if images.dtype == torch.uint8 else images.float()
        self.clip_embs = clip_embs.float()
        self.subject_id = int(subject_id)
        self.augment = augment
        self.cfg = cfg
        # B1: Z-score normalise fMRI per-subject using train statistics.
        # If fmri_mu/fmri_std are provided (test set), use them to avoid
        # data leak from normalising with test-set statistics.
        if fmri_mu is not None and fmri_std is not None:
            self.fmri_mu = fmri_mu
            self.fmri_std = fmri_std
        else:
            self.fmri_mu = self.fmri.mean(0)
            self.fmri_std = self.fmri.std(0).clamp(1e-6)
        self.fmri = (self.fmri - self.fmri_mu) / self.fmri_std
        # Phase 2: optional patch tokens (N, 256, 1024) fp16
        self.clip_patches = clip_patches

    def __len__(self): return len(self.fmri)

    def __getitem__(self, i):
        f = self.fmri[i]
        if self.augment and self.cfg is not None:
            f = f + torch.randn_like(f) * self.cfg.fmri_noise_std
            if random.random() < 0.5:
                mask = torch.rand_like(f) > self.cfg.fmri_mask_prob
                f = f * mask
        img = self.images[i]
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        result = {
            "fmri": f,
            "latent": self.latents[i],
            "image": img,
            "clip_emb": self.clip_embs[i],
            "subject": self.subject_id,
        }
        if self.clip_patches is not None:
            result["clip_patches"] = self.clip_patches[i].float()  # (256, 1024)
        return result


def variable_collate(batch):
    """Stack same-subject samples (subject grouping enforced by SubjectBatchSampler)."""
    result = {
        "fmri": torch.stack([b["fmri"] for b in batch]),
        "latent": torch.stack([b["latent"] for b in batch]),
        "image": torch.stack([b["image"] for b in batch]),
        "clip_emb": torch.stack([b["clip_emb"] for b in batch]),
        "subject": torch.tensor([b["subject"] for b in batch], dtype=torch.long),
    }
    if "clip_patches" in batch[0]:
        result["clip_patches"] = torch.stack([b["clip_patches"] for b in batch])
    return result


from torch.utils.data import Sampler

class SubjectBatchSampler(Sampler):
    """Yields same-subject batches across `ConcatDataset` of per-subject datasets, DDP-aware."""
    def __init__(self, dataset_lengths, batch_size, shuffle=True, drop_last=True,
                 num_replicas=1, rank=0, seed=0):
        self.lengths = list(dataset_lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.ranges = []
        offset = 0
        for L in self.lengths:
            self.ranges.append((offset, offset + L))
            offset += L

    def set_epoch(self, e: int):
        self.epoch = e

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        all_batches = []
        for (start, end) in self.ranges:
            n = end - start
            idx = torch.arange(start, end)
            if self.shuffle:
                idx = idx[torch.randperm(n, generator=g)]
            full = (n // self.batch_size) * self.batch_size
            if full == 0 and not self.drop_last:
                all_batches.append(idx.tolist())
            else:
                idx = idx[:full]
                for s in range(0, full, self.batch_size):
                    all_batches.append(idx[s:s + self.batch_size].tolist())
        if self.shuffle:
            order = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in order]
        all_batches = all_batches[self.rank::self.num_replicas]
        for b in all_batches:
            yield b

    def __len__(self):
        total = 0
        for L in self.lengths:
            n_full = L // self.batch_size
            total += n_full
        return total // self.num_replicas


def build_dataloaders(cfg: Config):
    """Returns (train_loader, test_loader, eval_loader, train_sampler, voxels_per_subject).

    eval_loader is identical to test_loader but NOT sharded across DDP ranks
    (num_replicas=1, rank=0) so rank-0 evaluation sees the full test set.
    """
    tensors = build_or_load_tensors(cfg)
    clips = compute_or_load_clip(tensors, cfg)
    lats = compute_or_load_latents(tensors, cfg)

    # Phase 2: load CLIP patch token cache if needed
    clip_patches_cache = None
    if getattr(cfg, "training_stage", "").startswith("2"):
        clip_patches_cache = compute_or_load_clip_patches(tensors, cfg)

    fmri_stats = tensors.get("fmri_stats", {})
    train_sets, test_sets = [], []
    voxels = tensors.get("voxels", {})
    for subj in cfg.subjects:
        if subj not in voxels:
            voxels[subj] = tensors[f"fmri_train_{subj}"].shape[1]
        tr_cp = clip_patches_cache.get(f"clip_patch_train_{subj}") if clip_patches_cache else None
        te_cp = clip_patches_cache.get(f"clip_patch_test_{subj}") if clip_patches_cache else None
        # Build train dataset first so we can read its normalisation stats
        tr_ds = NSDDataset(
            tensors[f"fmri_train_{subj}"], lats[f"lat_train_{subj}"],
            tensors[f"imgs_train_{subj}"], clips[f"clip_train_{subj}"],
            subject_id=subj, augment=True, cfg=cfg,
            clip_patches=tr_cp,
        )
        train_sets.append(tr_ds)
        # B1: Pass train fmri stats to test dataset to avoid data leak
        if subj in fmri_stats:
            te_mu = fmri_stats[subj]["mu"]
            te_std = fmri_stats[subj]["std"]
        else:
            te_mu = tr_ds.fmri_mu
            te_std = tr_ds.fmri_std
        test_sets.append(NSDDataset(
            tensors[f"fmri_test_{subj}"], lats[f"lat_test_{subj}"],
            tensors[f"imgs_test_{subj}"], clips[f"clip_test_{subj}"],
            subject_id=subj, augment=False, cfg=cfg,
            fmri_mu=te_mu, fmri_std=te_std,
            clip_patches=te_cp,
        ))

    train_set = ConcatDataset(train_sets)
    test_set = ConcatDataset(test_sets)

    train_sampler = SubjectBatchSampler(
        [len(d) for d in train_sets], cfg.batch_size_per_gpu,
        shuffle=True, drop_last=cfg.drop_last_train,
        num_replicas=world_size(), rank=rank(),
    )
    test_sampler = SubjectBatchSampler(
        [len(d) for d in test_sets], cfg.batch_size_per_gpu,
        shuffle=False, drop_last=cfg.drop_last_test,
        num_replicas=world_size(), rank=rank(),
    )
    # Eval sampler: always rank=0, num_replicas=1 → full test set on rank-0
    eval_sampler = SubjectBatchSampler(
        [len(d) for d in test_sets], cfg.batch_size_per_gpu,
        shuffle=False, drop_last=False,
        num_replicas=1, rank=0,
    )

    pf = cfg.prefetch_factor if cfg.num_workers > 0 else None
    persistent = cfg.persistent_workers and cfg.num_workers > 0

    train_loader = DataLoader(
        train_set, batch_sampler=train_sampler,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        persistent_workers=persistent, prefetch_factor=pf,
        collate_fn=variable_collate,
    )
    test_loader = DataLoader(
        test_set, batch_sampler=test_sampler,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        persistent_workers=persistent, prefetch_factor=pf,
        collate_fn=variable_collate,
    )
    eval_loader = DataLoader(
        test_set, batch_sampler=eval_sampler,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        persistent_workers=persistent, prefetch_factor=pf,
        collate_fn=variable_collate,
    )
    return train_loader, test_loader, eval_loader, train_sampler, voxels
