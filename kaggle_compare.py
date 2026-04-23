#!/usr/bin/env python3
"""
BrainFlow — Kaggle Comparison Script
=====================================
Compares THREE methods on a SHARED backbone (v5 BrainEncoder + FlowUNet):

  M0  Baseline           — Conditional Flow Matching (deterministic ODE)
  MA  SB + OT            — Schrödinger Bridge (drift+score, SDE) + cross-subject
                            Sinkhorn alignment in CLS-token space.
                            Unique eval axes: (1) per-pixel uncertainty from K samples
                                              (2) zero-shot subject transfer (subj 5).
  MB  HRF Brain-Manifold — CFM whose source distribution is the projected brain
                            CLS embedding (NOT Gaussian noise) + a fixed double-gamma
                            HRF velocity bias gated by α(t) = sin(πt).

Compute budget: ~11h on Kaggle T4×2 (or P100). 60 epochs per method, fixed.
Subject 1 is the head-to-head set. Subject 2 is loaded only for OT alignment
(token-space, no latents). Subject 5 is loaded only for the zero-shot transfer
test (~5 min of fMRI; alignment-only fine-tune of its stem).

NOTE: This is a single-file Kaggle notebook script. Drop into a Kaggle cell.
"""
# %% [code]
import os, sys, io, math, time, random, warnings, gc, json, subprocess
from pathlib import Path
from collections import defaultdict
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from scipy.stats import pearsonr

# ── Kaggle-only installs ────────────────────────────────────────────────────────
subprocess.run([
    "pip", "install", "-q",
    "open-clip-torch>=2.20.0", "torchdiffeq", "webdataset",
    "h5py", "wandb", "diffusers", "geomloss",
], check=True)

from huggingface_hub import hf_hub_download, login
from diffusers import AutoencoderKL
import open_clip, h5py, webdataset as wds
import wandb

# ── Try geomloss; fall back to inline Sinkhorn if it fails ─────────────────────
try:
    from geomloss import SamplesLoss
    HAS_GEOMLOSS = True
except Exception as e:
    print(f"[warn] geomloss import failed ({e}); using inline Sinkhorn")
    HAS_GEOMLOSS = False

# ── Kaggle secrets ─────────────────────────────────────────────────────────────
try:
    from kaggle_secrets import UserSecretsClient
    _sec = UserSecretsClient()
    HF_TOKEN = _sec.get_secret("HF_TOKEN")
    WANDB_KEY = _sec.get_secret("wandb_api")
    login(token=HF_TOKEN, add_to_git_credential=False)
    wandb.login(key=WANDB_KEY)
except Exception as e:
    print(f"[warn] Kaggle secrets unavailable ({e}); using env vars")
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    if os.environ.get("WANDB_API_KEY"):
        wandb.login(key=os.environ["WANDB_API_KEY"])

SEED = 42
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
set_seed()
torch.backends.cudnn.benchmark = True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %% [code]
# ════════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════════
class CFG:
    # ── Smoke test toggle ──
    SMOKE_TEST = False           # if True: 2 epochs, 256 train, 64 test
    RUN_BASELINE = True
    RUN_SB_OT = True
    RUN_HRF = True

    # ── Dataset ──
    HF_REPO = "pscotti/mindeyev2"
    PRIMARY_SUBJECT = 1          # head-to-head
    AUX_SUBJECT = 2              # OT alignment partner (token-space only)
    ZEROSHOT_SUBJECT = 5         # zero-shot transfer test
    IMG_SIZE = 256; LATENT_RES = 32; LATENT_CH = 4
    MAX_TRAIN = 8859; MAX_TEST = 982

    CLIP_DIM = 768; BRAIN_DIM = 768

    # ── Encoder ──
    ENC_HIDDEN = 1024; ENC_BLOCKS = 4
    ENC_DROP = 0.25; N_TOKENS = 16
    TOKEN_DROP_PROB = 0.1; STOCHASTIC_DEPTH = 0.1

    # ── UNet ──
    UNET_BASE_CH = 128; ATTN_HEADS = 8; TIME_EMB_DIM = 512

    # ── Training ──
    BATCH_SIZE = 16; GRAD_ACCUM = 4
    TOTAL_EPOCHS = 60            # FIXED across methods
    LR = 2e-4; WARMUP_EPOCHS = 5
    MIN_LR = 1e-5; GRAD_CLIP = 0.5
    WEIGHT_DECAY = 0.05

    # ── Losses ──
    INFONCE_TEMP = 0.07
    LAMBDA_ALIGN = 0.1
    LAMBDA_CFM = 1.0
    LAMBDA_SCORE = 1.0           # SB score-matching weight
    LAMBDA_OT = 0.05             # Sinkhorn weight (Method A)

    # ── CFG ──
    CFG_DROP_PROB = 0.20
    CFG_SCALE = 2.0

    # ── SB ──
    SIGMA_MAX = 0.5              # I²SB-style σ(t) = σ_max·sqrt(t(1-t))
    K_SAMPLES_SB = 4             # samples/trial for uncertainty maps

    # ── HRF ──
    HRF_LEN = 16                 # match N_TOKENS

    # ── Augmentation ──
    FMRI_NOISE_STD = 0.05; FMRI_MASK_PROB = 0.05
    MIXUP_ALPHA = 0.2

    # ── EMA ──
    EMA_DECAY = 0.9995; EMA_START = 200

    # ── Eval ──
    EVAL_FREQ = 10
    ODE_STEPS = 20

    # ── Zero-shot subj transfer ──
    ZEROSHOT_STEPS = 100         # gradient steps to fit subj-5 stem only
    ZEROSHOT_LR = 5e-4

    # ── Paths ──
    CACHE_DIR = Path("/kaggle/working/cache")
    DATA_DIR = Path("/kaggle/working/data")
    FIG_DIR = Path("/kaggle/working/figures")
    TMP_DIR = Path("/kaggle/tmp")
    OUT_DIR = Path("/kaggle/working")

if CFG.SMOKE_TEST:
    CFG.TOTAL_EPOCHS = 2; CFG.MAX_TRAIN = 256; CFG.MAX_TEST = 64
    CFG.EVAL_FREQ = 1; CFG.WARMUP_EPOCHS = 1; CFG.EMA_START = 10
    CFG.ZEROSHOT_STEPS = 10

for d in [CFG.CACHE_DIR, CFG.DATA_DIR, CFG.FIG_DIR, CFG.TMP_DIR, CFG.OUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# %% [code]
# ════════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════════════════════════
def dl_hf(fname, local_dir=CFG.DATA_DIR):
    p = local_dir / fname
    if p.exists(): return p
    p.parent.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(
        repo_id=CFG.HF_REPO, filename=fname,
        repo_type="dataset", local_dir=str(local_dir), token=HF_TOKEN,
    ))

def _load_behav(subject):
    """Return (train_behav, test_behav) numpy arrays from webdataset shards."""
    s = f"0{subject}"
    behav = {"train": [], "test": []}
    for split in ["train", "test"]:
        n_shards = 8 if split == "train" else 1
        for i in range(n_shards):
            try:
                p = dl_hf(f"wds/subj{s}/{split}/{i}.tar")
            except Exception:
                continue
            for sample in wds.WebDataset(str(p), handler=wds.warn_and_continue):
                if "behav.npy" in sample:
                    behav[split].append(np.load(io.BytesIO(sample["behav.npy"])))
    tr = np.concatenate(behav["train"])[:CFG.MAX_TRAIN] if behav["train"] else np.zeros((0, 17))
    te = np.concatenate(behav["test"])[:CFG.MAX_TEST] if behav["test"] else tr[-CFG.MAX_TEST:]
    return tr, te

def load_subject_full(subject, coco_h5):
    """Full load: fmri + images + (optional) latents will be encoded later."""
    s = f"0{subject}"
    bp = dl_hf(f"betas_all_subj{s}_fp32_renorm.hdf5")
    tr, te = _load_behav(subject)
    all_trial = np.concatenate([tr[:, 5].astype(int), te[:, 5].astype(int)])
    with h5py.File(bp, "r") as f:
        key = list(f.keys())[0]
        n_tot, n_vox = f[key].shape
        idx0 = np.clip(all_trial - 1, 0, n_tot - 1)
        u, inv = np.unique(idx0, return_inverse=True)
        fmri = f[key][u][inv]
    n_tr = len(tr)
    all_coco = np.concatenate([tr[:, 0].astype(int), te[:, 0].astype(int)])
    with h5py.File(coco_h5, "r") as f:
        key = list(f.keys())[0]
        cl = np.clip(all_coco, 0, f[key].shape[0] - 1)
        uc, ic = np.unique(cl, return_inverse=True)
        imgs = f[key][uc][ic]
    imgs = torch.from_numpy(imgs.astype(np.float32))
    if imgs.max() > 1.5: imgs = imgs / 255.0
    imgs = imgs.clamp(0, 1)
    if imgs.shape[-1] != CFG.IMG_SIZE:
        imgs = F.interpolate(imgs, CFG.IMG_SIZE, mode="bilinear",
                             align_corners=False).clamp(0, 1)
    return {
        "fmri_train": torch.tensor(fmri[:n_tr], dtype=torch.float32),
        "fmri_test": torch.tensor(fmri[n_tr:], dtype=torch.float32),
        "images_train": imgs[:n_tr], "images_test": imgs[n_tr:],
        "n_voxels": n_vox,
    }

def load_subject_fmri_only(subject):
    """Lightweight load for OT alignment partner: fmri only, paired with COCO ids."""
    s = f"0{subject}"
    bp = dl_hf(f"betas_all_subj{s}_fp32_renorm.hdf5")
    tr, _ = _load_behav(subject)
    if len(tr) == 0:
        raise RuntimeError(f"No training behav for subj {subject}")
    with h5py.File(bp, "r") as f:
        key = list(f.keys())[0]
        n_tot, n_vox = f[key].shape
        idx0 = np.clip(tr[:, 5].astype(int) - 1, 0, n_tot - 1)
        u, inv = np.unique(idx0, return_inverse=True)
        fmri = f[key][u][inv]
    coco_ids = tr[:, 0].astype(int)
    return {
        "fmri_train": torch.tensor(fmri, dtype=torch.float32),
        "coco_ids": torch.tensor(coco_ids, dtype=torch.long),
        "n_voxels": n_vox,
    }


# ── COCO HDF5 (one download serves all subjects) ─────────────────────────────
def get_coco_h5():
    p = CFG.TMP_DIR / "coco_images_224_float16.hdf5"
    if p.exists(): return p
    alt = CFG.DATA_DIR / "coco_images_224_float16.hdf5"
    if alt.exists(): return alt
    return Path(hf_hub_download(
        repo_id=CFG.HF_REPO, filename="coco_images_224_float16.hdf5",
        repo_type="dataset", local_dir=str(CFG.TMP_DIR), token=HF_TOKEN,
    ))


# ── CLIP encoder (used once at preprocessing) ────────────────────────────────
def compute_clip(images, bs=32):
    m, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    m = m.eval().to(DEVICE)
    for p in m.parameters(): p.requires_grad_(False)
    out = []
    for i in tqdm(range(0, len(images), bs), desc="CLIP encode", leave=False):
        b = F.interpolate(images[i:i+bs], 224, mode="bilinear",
                          align_corners=False).to(DEVICE)
        with torch.no_grad(): e = m.encode_image(b)
        out.append(F.normalize(e.float(), dim=-1).cpu())
        del b, e
    del m; gc.collect(); torch.cuda.empty_cache()
    return torch.cat(out)


# ── Frozen SD-VAE ────────────────────────────────────────────────────────────
class FrozenVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            cache_dir=str(CFG.CACHE_DIR), torch_dtype=torch.float32,
        )
        self.vae.eval()
        for p in self.vae.parameters(): p.requires_grad_(False)
        self.scale = 0.18215

    @torch.no_grad()
    def encode(self, x):
        return self.vae.encode(x * 2 - 1).latent_dist.sample() * self.scale

    @torch.no_grad()
    def decode(self, z):
        return (self.vae.decode(z / self.scale).sample.clamp(-1, 1) + 1) / 2


# %% [code]
# ── Run preprocessing ────────────────────────────────────────────────────────
print("=" * 70); print("Phase 0: Data preprocessing"); print("=" * 70)

cache_subj1 = CFG.CACHE_DIR / f"raw_subj{CFG.PRIMARY_SUBJECT}.pt"
cache_subj_aux = CFG.CACHE_DIR / f"raw_subj{CFG.AUX_SUBJECT}_fmri.pt"
cache_subj_zs = CFG.CACHE_DIR / f"raw_subj{CFG.ZEROSHOT_SUBJECT}.pt"

coco_h5 = get_coco_h5()

if not cache_subj1.exists():
    print(f"Loading subj {CFG.PRIMARY_SUBJECT} (full)…")
    raw1 = load_subject_full(CFG.PRIMARY_SUBJECT, coco_h5)
    raw1["clip_train"] = compute_clip(raw1["images_train"])
    raw1["clip_test"] = compute_clip(raw1["images_test"])
    torch.save(raw1, cache_subj1)
else:
    raw1 = torch.load(cache_subj1, map_location="cpu")
print(f"  subj{CFG.PRIMARY_SUBJECT}: {raw1['fmri_train'].shape[0]} train, "
      f"{raw1['fmri_test'].shape[0]} test, {raw1['n_voxels']} voxels")

if CFG.RUN_SB_OT:
    if not cache_subj_aux.exists():
        print(f"Loading subj {CFG.AUX_SUBJECT} (fmri only)…")
        raw_aux = load_subject_fmri_only(CFG.AUX_SUBJECT)
        # CLIP embeddings keyed by coco id — we will index by coco id to pair with subj1
        # Easier: precompute aux subject's CLIP embeddings using its own COCO ids
        # Pull images for those coco ids
        with h5py.File(coco_h5, "r") as f:
            key = list(f.keys())[0]
            cl = np.clip(raw_aux["coco_ids"].numpy(), 0, f[key].shape[0] - 1)
            uc, ic = np.unique(cl, return_inverse=True)
            aux_imgs = f[key][uc][ic]
        aux_imgs = torch.from_numpy(aux_imgs.astype(np.float32))
        if aux_imgs.max() > 1.5: aux_imgs = aux_imgs / 255.0
        aux_imgs = aux_imgs.clamp(0, 1)
        if aux_imgs.shape[-1] != CFG.IMG_SIZE:
            aux_imgs = F.interpolate(aux_imgs, CFG.IMG_SIZE, mode="bilinear",
                                     align_corners=False).clamp(0, 1)
        raw_aux["clip_train"] = compute_clip(aux_imgs)
        del aux_imgs
        torch.save(raw_aux, cache_subj_aux)
    else:
        raw_aux = torch.load(cache_subj_aux, map_location="cpu")
    print(f"  subj{CFG.AUX_SUBJECT}: {raw_aux['fmri_train'].shape[0]} train (alignment), "
          f"{raw_aux['n_voxels']} voxels")

    if not cache_subj_zs.exists():
        try:
            print(f"Loading subj {CFG.ZEROSHOT_SUBJECT} (small set for zero-shot)…")
            raw_zs = load_subject_full(CFG.ZEROSHOT_SUBJECT, coco_h5)
            # Keep only ~5min ≈ 250 trials
            for k in ["fmri_train", "images_train"]:
                raw_zs[k] = raw_zs[k][:250]
            raw_zs["clip_train"] = compute_clip(raw_zs["images_train"])
            torch.save(raw_zs, cache_subj_zs)
        except Exception as e:
            print(f"[warn] subj{CFG.ZEROSHOT_SUBJECT} unavailable: {e}; zero-shot will be skipped")
            raw_zs = None
    else:
        raw_zs = torch.load(cache_subj_zs, map_location="cpu")
    if raw_zs is not None:
        print(f"  subj{CFG.ZEROSHOT_SUBJECT}: {raw_zs['fmri_train'].shape[0]} train, "
              f"{raw_zs['fmri_test'].shape[0]} test")
else:
    raw_aux = None; raw_zs = None


# %% [code]
# ── VAE-encode subj 1 latents ────────────────────────────────────────────────
vae = FrozenVAE().to(DEVICE)
lt = CFG.CACHE_DIR / "lat_tr_subj1.pt"
le = CFG.CACHE_DIR / "lat_te_subj1.pt"
if lt.exists() and le.exists():
    latents_tr = torch.load(lt, map_location="cpu")
    latents_te = torch.load(le, map_location="cpu")
else:
    def _enc(imgs, bs=16):
        o = []
        for i in tqdm(range(0, len(imgs), bs), desc="VAE encode", leave=False):
            o.append(vae.encode(imgs[i:i+bs].to(DEVICE)).cpu())
        return torch.cat(o)
    latents_tr = _enc(raw1["images_train"])
    latents_te = _enc(raw1["images_test"])
    torch.save(latents_tr, lt); torch.save(latents_te, le)
vae.cpu(); gc.collect(); torch.cuda.empty_cache()


# %% [code]
# ════════════════════════════════════════════════════════════════════════════════
#  DATASET
# ════════════════════════════════════════════════════════════════════════════════
class NSDDataset(Dataset):
    def __init__(self, fmri, latents=None, images=None, clip_embs=None, augment=False):
        self.fmri = fmri.float()
        self.latents = latents.float() if latents is not None else None
        self.images = images.float() if images is not None else None
        self.clip_embs = clip_embs.float() if clip_embs is not None else None
        self.augment = augment
        mu = self.fmri.mean(0, keepdim=True)
        std = self.fmri.std(0, keepdim=True).clamp(1e-6)
        self.fmri = (self.fmri - mu) / std

    def __len__(self): return len(self.fmri)

    def __getitem__(self, i):
        f = self.fmri[i]
        if self.augment:
            f = f + torch.randn_like(f) * CFG.FMRI_NOISE_STD
            if random.random() < 0.5:
                mask = torch.rand_like(f) > CFG.FMRI_MASK_PROB
                f = f * mask
        out = {"fmri": f, "index": i}
        if self.latents is not None: out["latent"] = self.latents[i]
        if self.images is not None: out["image"] = self.images[i]
        if self.clip_embs is not None: out["clip_emb"] = self.clip_embs[i]
        return out


train_ds = NSDDataset(raw1["fmri_train"], latents_tr, raw1["images_train"],
                      raw1["clip_train"], augment=True)
test_ds = NSDDataset(raw1["fmri_test"], latents_te, raw1["images_test"],
                     raw1["clip_test"], augment=False)
train_dl = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True,
                      num_workers=2, pin_memory=True, drop_last=True,
                      persistent_workers=True)
test_dl = DataLoader(test_ds, batch_size=CFG.BATCH_SIZE, shuffle=False,
                     num_workers=2, pin_memory=True, persistent_workers=True)

# Aux subject loader (Method A only)
aux_dl = None; aux_ds = None
if raw_aux is not None:
    aux_ds = NSDDataset(raw_aux["fmri_train"], None, None,
                        raw_aux["clip_train"], augment=True)
    aux_dl = DataLoader(aux_ds, batch_size=CFG.BATCH_SIZE, shuffle=True,
                        num_workers=2, pin_memory=True, drop_last=True,
                        persistent_workers=True)


# %% [code]
# ════════════════════════════════════════════════════════════════════════════════
#  SHARED MODEL COMPONENTS (lifted from v5)
# ════════════════════════════════════════════════════════════════════════════════
class ResBlockMLP(nn.Module):
    def __init__(self, dim, drop=0.1, mult=4, stoch_depth=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult), nn.GELU(), nn.Dropout(drop),
            nn.Linear(dim * mult, dim), nn.Dropout(drop),
        )
        self.drop_prob = stoch_depth

    def forward(self, x):
        if self.training and self.drop_prob > 0 and random.random() < self.drop_prob:
            return x
        return x + self.net(self.norm(x))


class BrainEncoder(nn.Module):
    """fMRI → 16 tokens (B,16,768) + CLS (B,768).
    Stem is per-subject (multi-subject support); trunk is shared.
    """
    def __init__(self, voxel_dims_by_subject):
        super().__init__()
        self.stems = nn.ModuleDict({
            str(sid): nn.Sequential(
                nn.Linear(n_vox, CFG.ENC_HIDDEN),
                nn.LayerNorm(CFG.ENC_HIDDEN), nn.GELU(), nn.Dropout(0.3),
            ) for sid, n_vox in voxel_dims_by_subject.items()
        })
        self.blocks = nn.ModuleList([
            ResBlockMLP(CFG.ENC_HIDDEN, CFG.ENC_DROP, mult=4,
                        stoch_depth=CFG.STOCHASTIC_DEPTH * (i + 1) / CFG.ENC_BLOCKS)
            for i in range(CFG.ENC_BLOCKS)
        ])
        self.to_tokens = nn.Sequential(
            nn.Linear(CFG.ENC_HIDDEN, CFG.N_TOKENS * CFG.BRAIN_DIM),
            nn.Unflatten(-1, (CFG.N_TOKENS, CFG.BRAIN_DIM)),
            nn.LayerNorm(CFG.BRAIN_DIM), nn.Dropout(0.1),
        )
        self.cls_head = nn.Linear(CFG.ENC_HIDDEN, CFG.BRAIN_DIM)

    def add_subject(self, sid, n_vox, device):
        """Hot-add a new subject stem (used for zero-shot fine-tune)."""
        sid = str(sid)
        if sid in self.stems: return
        self.stems[sid] = nn.Sequential(
            nn.Linear(n_vox, CFG.ENC_HIDDEN),
            nn.LayerNorm(CFG.ENC_HIDDEN), nn.GELU(), nn.Dropout(0.3),
        ).to(device)

    def forward(self, x, subject_id):
        h = self.stems[str(subject_id)](x)
        for b in self.blocks: h = b(h)
        tokens = self.to_tokens(h)
        cls = F.normalize(self.cls_head(h), dim=-1)
        if self.training and CFG.TOKEN_DROP_PROB > 0:
            mask = (torch.rand(tokens.shape[0], tokens.shape[1], 1,
                               device=tokens.device) > CFG.TOKEN_DROP_PROB).float()
            tokens = tokens * mask
        return tokens, cls


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half).float() / (half - 1))
        self.register_buffer("freqs", freqs)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(),
                                 nn.Linear(dim * 4, dim))

    def forward(self, t):
        t = t.unsqueeze(-1) * self.freqs
        return self.mlp(torch.cat([t.sin(), t.cos()], dim=-1))


class CrossAttention(nn.Module):
    def __init__(self, qd, cd, nh=8, hd=64):
        super().__init__()
        inner = nh * hd
        self.scale = hd ** -0.5; self.nh = nh; self.hd = hd
        self.to_q = nn.Linear(qd, inner, bias=False)
        self.to_k = nn.Linear(cd, inner, bias=False)
        self.to_v = nn.Linear(cd, inner, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, qd), nn.Dropout(0.05))
        self.norm = nn.LayerNorm(qd)

    def forward(self, x, ctx):
        B, L, _ = x.shape
        res = x; x = self.norm(x)
        def rsh(t, s): return t.view(B, s, self.nh, self.hd).transpose(1, 2)
        Q = rsh(self.to_q(x), L)
        K = rsh(self.to_k(ctx), ctx.shape[1])
        V = rsh(self.to_v(ctx), ctx.shape[1])
        out = torch.softmax(Q @ K.transpose(-2, -1) * self.scale, dim=-1) @ V
        return res + self.to_out(out.transpose(1, 2).reshape(B, L, -1))


class UNetResBlock(nn.Module):
    def __init__(self, ic, oc, td):
        super().__init__()
        self.n1 = nn.GroupNorm(min(32, ic), ic)
        self.c1 = nn.Conv2d(ic, oc, 3, padding=1)
        self.n2 = nn.GroupNorm(min(32, oc), oc)
        self.c2 = nn.Conv2d(oc, oc, 3, padding=1)
        self.act = nn.SiLU()
        self.tp = nn.Linear(td, oc * 2)
        self.skip = nn.Conv2d(ic, oc, 1) if ic != oc else nn.Identity()
        self.dropout = nn.Dropout2d(0.05)

    def forward(self, x, te):
        h = self.act(self.n1(x)); h = self.c1(h)
        sc, sh = self.act(self.tp(te))[:, :, None, None].chunk(2, dim=1)
        h = self.n2(h) * (1 + sc) + sh
        h = self.act(h); h = self.dropout(h); h = self.c2(h)
        return h + self.skip(x)


class FlowUNet(nn.Module):
    """
    UNet velocity field. Optional second head for SB score.
    Optional HRF velocity bias (Method B) injected as additive correction.
    """
    def __init__(self, has_score_head=False, has_hrf_bias=False):
        super().__init__()
        ic, bc = CFG.LATENT_CH, CFG.UNET_BASE_CH
        bd, td, nh = CFG.BRAIN_DIM, CFG.TIME_EMB_DIM, CFG.ATTN_HEADS
        chs = [bc, bc * 2, bc * 4, bc * 4]
        self.te = SinusoidalTimeEmbedding(td)
        self.ip = nn.Conv2d(ic, bc, 3, padding=1)
        self.null_tokens = nn.Parameter(torch.randn(1, CFG.N_TOKENS, bd) * 0.01)

        self.eb, self.ea, self.ed = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        ch = bc
        for i, co in enumerate(chs):
            self.eb.append(UNetResBlock(ch, co, td))
            self.ea.append(CrossAttention(co, bd, nh))
            self.ed.append(nn.Conv2d(co, co, 4, stride=2, padding=1)
                           if i < len(chs) - 1 else nn.Identity())
            ch = co
        self.m1 = UNetResBlock(chs[-1], chs[-1], td)
        self.ma = CrossAttention(chs[-1], bd, nh)
        self.m2 = UNetResBlock(chs[-1], chs[-1], td)
        self.du, self.db, self.da = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i, co in enumerate(reversed(chs)):
            self.du.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(ch, ch, 3, padding=1),
            ) if i > 0 else nn.Identity())
            self.db.append(UNetResBlock(ch + chs[len(chs)-1-i], co, td))
            self.da.append(CrossAttention(co, bd, nh))
            ch = co
        self.on = nn.GroupNorm(min(32, bc), bc)
        self.op = nn.Conv2d(bc, ic, 3, padding=1)
        nn.init.zeros_(self.op.weight); nn.init.zeros_(self.op.bias)

        # SB second head
        self.has_score_head = has_score_head
        if has_score_head:
            self.score_head = nn.Conv2d(bc, ic, 3, padding=1)
            nn.init.zeros_(self.score_head.weight)
            nn.init.zeros_(self.score_head.bias)

        # HRF bias projector (Method B): tokens → spatial map
        self.has_hrf_bias = has_hrf_bias
        if has_hrf_bias:
            self.hrf_proj = nn.Linear(bd, ic * CFG.LATENT_RES * CFG.LATENT_RES)
            nn.init.zeros_(self.hrf_proj.weight); nn.init.zeros_(self.hrf_proj.bias)
            self.register_buffer("hrf_kernel", _double_gamma_hrf(CFG.HRF_LEN))

    def _trunk(self, x, t, ctx):
        te = self.te(t); h = self.ip(x); sk = []
        for res, attn, dn in zip(self.eb, self.ea, self.ed):
            h = res(h, te)
            B, C, H, W = h.shape
            h = attn(h.permute(0, 2, 3, 1).reshape(B, H*W, C), ctx)\
                .reshape(B, H, W, C).permute(0, 3, 1, 2)
            sk.append(h); h = dn(h)
        h = self.m1(h, te)
        B, C, H, W = h.shape
        h = self.ma(h.permute(0, 2, 3, 1).reshape(B, H*W, C), ctx)\
            .reshape(B, H, W, C).permute(0, 3, 1, 2)
        h = self.m2(h, te)
        for i, (up, res, attn) in enumerate(zip(self.du, self.db, self.da)):
            h = up(h); h = torch.cat([h, sk[-(i+1)]], dim=1)
            h = res(h, te)
            B, C, H, W = h.shape
            h = attn(h.permute(0, 2, 3, 1).reshape(B, H*W, C), ctx)\
                .reshape(B, H, W, C).permute(0, 3, 1, 2)
        return F.silu(self.on(h)), te

    def forward(self, x, t, ctx, return_score=False):
        h, te = self._trunk(x, t, ctx)
        v = self.op(h)
        if self.has_hrf_bias:
            # Convolve token sequence with fixed HRF kernel along token axis
            B, L, D = ctx.shape
            ctx_t = ctx.transpose(1, 2)                         # (B, D, L)
            kern = self.hrf_kernel.view(1, 1, -1).expand(D, 1, -1)
            ctx_conv = F.conv1d(ctx_t, kern, padding=CFG.HRF_LEN // 2,
                                groups=D)[..., :L]              # (B, D, L)
            ctx_summary = ctx_conv.mean(-1)                     # (B, D)
            bias = self.hrf_proj(ctx_summary).view(
                B, CFG.LATENT_CH, CFG.LATENT_RES, CFG.LATENT_RES)
            alpha = torch.sin(math.pi * t).view(B, 1, 1, 1)
            v = v + alpha * bias
        if return_score and self.has_score_head:
            s = self.score_head(h)
            return v, s
        return v


def _double_gamma_hrf(length, peak=6.0, undershoot=16.0,
                      ratio=6.0, dt=1.0):
    """Standard SPM-style double-gamma HRF, length samples, normalized."""
    from math import lgamma, exp
    t = np.arange(length, dtype=np.float64) * dt + dt / 2
    def gamma_pdf(t, k):
        return np.exp((k - 1) * np.log(t + 1e-12) - t - lgamma(k))
    h = gamma_pdf(t, peak) - gamma_pdf(t, undershoot) / ratio
    h = h / (np.abs(h).sum() + 1e-12)
    return torch.tensor(h, dtype=torch.float32)


# %% [code]
# ════════════════════════════════════════════════════════════════════════════════
#  SINKHORN (fallback) + OT loss
# ════════════════════════════════════════════════════════════════════════════════
def sinkhorn_div(a, b, blur=0.05, n_iter=50):
    """Inline Sinkhorn divergence on uniform-weight samples (fallback)."""
    def _cost(x, y):
        return torch.cdist(x, y, p=2).pow(2)
    def _ot(x, y):
        n, m = x.shape[0], y.shape[0]
        C = _cost(x, y) / (2 * blur ** 2)
        log_a = torch.full((n,), -math.log(n), device=x.device)
        log_b = torch.full((m,), -math.log(m), device=x.device)
        f = torch.zeros(n, device=x.device); g = torch.zeros(m, device=x.device)
        for _ in range(n_iter):
            f = -torch.logsumexp(g[None, :] - C, dim=1) + log_b.logsumexp(0) - log_a
            g = -torch.logsumexp(f[:, None] - C, dim=0) + log_a.logsumexp(0) - log_b
        # Approximate transport cost
        P = torch.exp(f[:, None] + g[None, :] - C)
        return (P * C).sum() * (2 * blur ** 2)
    return _ot(a, b) - 0.5 * _ot(a, a) - 0.5 * _ot(b, b)


if HAS_GEOMLOSS:
    _geom_loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.7)
    def ot_loss(a, b):
        return _geom_loss(a, b)
else:
    def ot_loss(a, b):
        return sinkhorn_div(a, b, blur=0.05, n_iter=30)


# %% [code]
# ════════════════════════════════════════════════════════════════════════════════
#  EMA
# ════════════════════════════════════════════════════════════════════════════════
class EMA:
    def __init__(self, model, decay=CFG.EMA_DECAY):
        self.decay = decay
        self.shadow = {k: v.clone().detach().float()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow and self.shadow[k].shape == v.shape:
                self.shadow[k].mul_(self.decay).add_(v.float(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.clone().detach().float()

    def apply(self, model):
        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device
        compatible = {}
        sd = model.state_dict()
        for k, v in sd.items():
            if k in self.shadow and self.shadow[k].shape == v.shape:
                compatible[k] = self.shadow[k].to(dtype).to(device)
            else:
                compatible[k] = v
        model.load_state_dict(compatible)

    def restore(self, sd, model):
        model.load_state_dict(sd)


# %% [code]
# ════════════════════════════════════════════════════════════════════════════════
#  METRICS + EVAL
# ════════════════════════════════════════════════════════════════════════════════
def pixel_correlation(pred, target):
    p = pred.flatten(1).cpu().numpy()
    t = target.flatten(1).cpu().numpy()
    return float(np.nanmean([pearsonr(pi, ti)[0] for pi, ti in zip(p, t)]))


def ssim_pytorch(pred, target, ws=11, sigma=1.5):
    C1, C2 = 0.01**2, 0.03**2
    B, C, H, W = pred.shape
    g = torch.arange(ws, dtype=torch.float32) - ws // 2
    g = torch.exp(-g**2 / (2 * sigma**2)); g = g / g.sum()
    k = g.outer(g).unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1).to(pred.device)
    pad = ws // 2
    def mu(x): return F.conv2d(x, k, padding=pad, groups=C)
    mx = mu(pred); my = mu(target)
    sx = mu(pred * pred) - mx**2
    sy = mu(target * target) - my**2
    sxy = mu(pred * target) - mx * my
    return float(((2 * mx * my + C1) * (2 * sxy + C2) /
                  ((mx**2 + my**2 + C1) * (sx + sy + C2))).mean())


_clip_eval_model = None
def _get_clip_eval():
    global _clip_eval_model
    if _clip_eval_model is None:
        m, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai")
        m = m.eval().to(DEVICE)
        for p in m.parameters(): p.requires_grad_(False)
        _clip_eval_model = m
    return _clip_eval_model


@torch.no_grad()
def clip_similarity(pi, gt):
    m = _get_clip_eval()
    pi_r = F.interpolate(pi, 224, mode="bilinear", align_corners=False)
    gt_r = F.interpolate(gt, 224, mode="bilinear", align_corners=False)
    ep = F.normalize(m.encode_image(pi_r).float(), dim=-1)
    et = F.normalize(m.encode_image(gt_r).float(), dim=-1)
    return float((ep * et).sum(-1).mean())


@torch.no_grad()
def evaluate(model, loader, sample_fn, subject_id=1, n_batches=8,
             collect_uncertainty=False, k_samples=1):
    """Generic eval. sample_fn(model, tokens, k_samples) -> (latent_pred,
    optional std_pixel)."""
    model.eval(); vae.to(DEVICE)
    pcs, sss, cls = [], [], []
    unc_means = []
    n_batches = min(n_batches, len(loader))
    for i, batch in enumerate(loader):
        if i >= n_batches: break
        fmri = batch["fmri"].to(DEVICE)
        images = batch["image"].to(DEVICE)
        tokens, _ = model.brain_enc(fmri, subject_id)
        if collect_uncertainty and k_samples > 1:
            # Sample K times, return mean + per-pixel std in pixel space
            preds = []
            for _ in range(k_samples):
                pl = sample_fn(model, tokens)
                preds.append(vae.decode(pl))
            preds = torch.stack(preds, 0)        # (K,B,C,H,W)
            pi = preds.mean(0)
            unc_means.append(float(preds.std(0).mean()))
        else:
            pl = sample_fn(model, tokens)
            pi = vae.decode(pl)
        pcs.append(pixel_correlation(pi, images))
        sss.append(ssim_pytorch(pi.cpu(), images.cpu()))
        cls.append(clip_similarity(pi, images))
    vae.cpu(); gc.collect(); torch.cuda.empty_cache()
    model.train()
    res = {"PixCorr": float(np.mean(pcs)),
           "SSIM": float(np.mean(sss)),
           "CLIP_Sim": float(np.mean(cls))}
    if collect_uncertainty and unc_means:
        res["Uncertainty"] = float(np.mean(unc_means))
    return res


# %% [code]
# ════════════════════════════════════════════════════════════════════════════════
#  WRAPPER MODELS (one per method)
# ════════════════════════════════════════════════════════════════════════════════
class BaselineModel(nn.Module):
    """M0: standard CFM."""
    def __init__(self, voxel_dims):
        super().__init__()
        self.brain_enc = BrainEncoder(voxel_dims)
        self.flow_unet = FlowUNet(has_score_head=False, has_hrf_bias=False)

    def training_step(self, batch, subject_id=1):
        fmri = batch["fmri"].to(DEVICE)
        latent = batch["latent"].to(DEVICE)
        clip_emb = batch["clip_emb"].to(DEVICE)
        B = fmri.shape[0]

        if self.training and CFG.MIXUP_ALPHA > 0 and random.random() < 0.3:
            lam = max(np.random.beta(CFG.MIXUP_ALPHA, CFG.MIXUP_ALPHA),
                      1 - np.random.beta(CFG.MIXUP_ALPHA, CFG.MIXUP_ALPHA))
            perm = torch.randperm(B, device=DEVICE)
            latent = lam * latent + (1 - lam) * latent[perm]
            fmri_in = lam * fmri + (1 - lam) * fmri[perm]
        else:
            fmri_in = fmri; lam = 1.0

        tokens, cls_emb = self.brain_enc(fmri_in, subject_id)
        x0 = torch.randn_like(latent)
        t = torch.rand(B, device=DEVICE)
        te = t[:, None, None, None]
        xt = (1 - te) * x0 + te * latent
        ut = latent - x0

        drop = torch.rand(B, device=DEVICE) < CFG.CFG_DROP_PROB
        null = self.flow_unet.null_tokens.expand(B, -1, -1)
        ctx = torch.where(drop[:, None, None], null, tokens)
        vt = self.flow_unet(xt, t, ctx)
        loss_cfm = F.mse_loss(vt, ut)

        if lam < 1.0:
            _, cls_clean = self.brain_enc(fmri, subject_id)
        else:
            cls_clean = cls_emb
        logits = (cls_clean @ clip_emb.T) / CFG.INFONCE_TEMP
        labels = torch.arange(B, device=DEVICE)
        loss_align = (F.cross_entropy(logits, labels)
                      + F.cross_entropy(logits.T, labels)) / 2

        total = CFG.LAMBDA_CFM * loss_cfm + CFG.LAMBDA_ALIGN * loss_align
        return {"loss": total, "cfm": loss_cfm, "align": loss_align}

    @torch.no_grad()
    def sample_ode(self, tokens, n_steps=CFG.ODE_STEPS, cfg_scale=CFG.CFG_SCALE):
        B = tokens.shape[0]
        x = torch.randn(B, CFG.LATENT_CH, CFG.LATENT_RES, CFG.LATENT_RES,
                        device=tokens.device)
        dt = 1.0 / n_steps
        null = self.flow_unet.null_tokens.expand(B, -1, -1)
        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=x.device)
            if cfg_scale > 1.0:
                vc = self.flow_unet(x, t, tokens)
                vu = self.flow_unet(x, t, null)
                v = vu + cfg_scale * (vc - vu)
            else:
                v = self.flow_unet(x, t, tokens)
            x = x + v * dt
        return x


class SBOTModel(nn.Module):
    """MA: Schrödinger Bridge + OT cross-subject alignment."""
    def __init__(self, voxel_dims):
        super().__init__()
        self.brain_enc = BrainEncoder(voxel_dims)
        self.flow_unet = FlowUNet(has_score_head=True, has_hrf_bias=False)

    @staticmethod
    def _sigma(t):
        return CFG.SIGMA_MAX * torch.sqrt(t * (1 - t) + 1e-6)

    def training_step(self, batch, batch_aux=None, subject_id=1, aux_subject_id=2):
        fmri = batch["fmri"].to(DEVICE)
        latent = batch["latent"].to(DEVICE)
        clip_emb = batch["clip_emb"].to(DEVICE)
        B = fmri.shape[0]

        tokens, cls_emb = self.brain_enc(fmri, subject_id)

        # SB endpoints: x0=noise, x1=z_GT
        x0 = torch.randn_like(latent)
        t = torch.rand(B, device=DEVICE).clamp(1e-3, 1 - 1e-3)
        te = t[:, None, None, None]
        sigma_t = self._sigma(t)[:, None, None, None]
        eps = torch.randn_like(latent)
        xt = (1 - te) * x0 + te * latent + sigma_t * eps

        v_target = latent - x0
        s_target = -eps / sigma_t.clamp(1e-3)

        drop = torch.rand(B, device=DEVICE) < CFG.CFG_DROP_PROB
        null = self.flow_unet.null_tokens.expand(B, -1, -1)
        ctx = torch.where(drop[:, None, None], null, tokens)
        v_pred, s_pred = self.flow_unet(xt, t, ctx, return_score=True)

        # Weight score loss by σ² to keep it well-scaled across t
        loss_drift = F.mse_loss(v_pred, v_target)
        loss_score = (((s_pred - s_target) * sigma_t) ** 2).mean()

        # InfoNCE alignment (subj 1 only — clean pairs)
        logits = (cls_emb @ clip_emb.T) / CFG.INFONCE_TEMP
        labels = torch.arange(B, device=DEVICE)
        loss_align = (F.cross_entropy(logits, labels)
                      + F.cross_entropy(logits.T, labels)) / 2

        # OT alignment between subj1 and aux subj CLS embeddings
        loss_ot = torch.tensor(0.0, device=DEVICE)
        if batch_aux is not None:
            fmri_aux = batch_aux["fmri"].to(DEVICE)
            clip_aux = batch_aux["clip_emb"].to(DEVICE)
            _, cls_aux = self.brain_enc(fmri_aux, aux_subject_id)
            loss_ot = ot_loss(cls_emb, cls_aux)
            # Aux subject also gets its own InfoNCE alignment to its CLIP
            logits_aux = (cls_aux @ clip_aux.T) / CFG.INFONCE_TEMP
            labels_aux = torch.arange(cls_aux.shape[0], device=DEVICE)
            loss_align = loss_align + 0.5 * (
                F.cross_entropy(logits_aux, labels_aux)
                + F.cross_entropy(logits_aux.T, labels_aux)) / 2

        total = (CFG.LAMBDA_CFM * loss_drift
                 + CFG.LAMBDA_SCORE * loss_score
                 + CFG.LAMBDA_ALIGN * loss_align
                 + CFG.LAMBDA_OT * loss_ot)
        return {"loss": total, "cfm": loss_drift, "score": loss_score,
                "align": loss_align, "ot": loss_ot}

    @torch.no_grad()
    def sample_sde(self, tokens, n_steps=CFG.ODE_STEPS, cfg_scale=CFG.CFG_SCALE):
        """Euler-Maruyama on the SB SDE with CFG on drift only."""
        B = tokens.shape[0]
        x = torch.randn(B, CFG.LATENT_CH, CFG.LATENT_RES, CFG.LATENT_RES,
                        device=tokens.device)
        dt = 1.0 / n_steps
        null = self.flow_unet.null_tokens.expand(B, -1, -1)
        for i in range(n_steps):
            t = torch.full((B,), i * dt + 1e-3, device=x.device).clamp(0, 1 - 1e-3)
            sigma_t = self._sigma(t)[:, None, None, None]
            if cfg_scale > 1.0:
                vc, sc_ = self.flow_unet(x, t, tokens, return_score=True)
                vu, su = self.flow_unet(x, t, null, return_score=True)
                v = vu + cfg_scale * (vc - vu)
                s = su + cfg_scale * (sc_ - su)
            else:
                v, s = self.flow_unet(x, t, tokens, return_score=True)
            # SB SDE: dx = (v + sigma^2 * s) dt + sigma * dW
            drift = v + (sigma_t ** 2) * s
            x = x + drift * dt + sigma_t * math.sqrt(dt) * torch.randn_like(x)
        return x


class HRFModel(nn.Module):
    """MB: brain-manifold flow with HRF velocity bias."""
    def __init__(self, voxel_dims):
        super().__init__()
        self.brain_enc = BrainEncoder(voxel_dims)
        self.flow_unet = FlowUNet(has_score_head=False, has_hrf_bias=True)
        self.cls_to_latent = nn.Linear(
            CFG.BRAIN_DIM, CFG.LATENT_CH * CFG.LATENT_RES * CFG.LATENT_RES)
        nn.init.zeros_(self.cls_to_latent.weight)
        nn.init.zeros_(self.cls_to_latent.bias)

    def project_source(self, cls_emb):
        return self.cls_to_latent(cls_emb).view(
            -1, CFG.LATENT_CH, CFG.LATENT_RES, CFG.LATENT_RES)

    def training_step(self, batch, subject_id=1):
        fmri = batch["fmri"].to(DEVICE)
        latent = batch["latent"].to(DEVICE)
        clip_emb = batch["clip_emb"].to(DEVICE)
        B = fmri.shape[0]

        tokens, cls_emb = self.brain_enc(fmri, subject_id)
        x0 = self.project_source(cls_emb) + 0.1 * torch.randn_like(latent)

        t = torch.rand(B, device=DEVICE)
        te = t[:, None, None, None]
        xt = (1 - te) * x0 + te * latent
        ut = latent - x0

        drop = torch.rand(B, device=DEVICE) < CFG.CFG_DROP_PROB
        null = self.flow_unet.null_tokens.expand(B, -1, -1)
        ctx = torch.where(drop[:, None, None], null, tokens)
        vt = self.flow_unet(xt, t, ctx)
        loss_cfm = F.mse_loss(vt, ut)

        logits = (cls_emb @ clip_emb.T) / CFG.INFONCE_TEMP
        labels = torch.arange(B, device=DEVICE)
        loss_align = (F.cross_entropy(logits, labels)
                      + F.cross_entropy(logits.T, labels)) / 2

        total = CFG.LAMBDA_CFM * loss_cfm + CFG.LAMBDA_ALIGN * loss_align
        return {"loss": total, "cfm": loss_cfm, "align": loss_align}

    @torch.no_grad()
    def sample_ode(self, tokens, cls_emb, n_steps=CFG.ODE_STEPS,
                   cfg_scale=CFG.CFG_SCALE):
        B = tokens.shape[0]
        x = self.project_source(cls_emb) + 0.1 * torch.randn(
            B, CFG.LATENT_CH, CFG.LATENT_RES, CFG.LATENT_RES, device=tokens.device)
        dt = 1.0 / n_steps
        null = self.flow_unet.null_tokens.expand(B, -1, -1)
        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=x.device)
            if cfg_scale > 1.0:
                vc = self.flow_unet(x, t, tokens)
                vu = self.flow_unet(x, t, null)
                v = vu + cfg_scale * (vc - vu)
            else:
                v = self.flow_unet(x, t, tokens)
            x = x + v * dt
        return x


# %% [code]
# ════════════════════════════════════════════════════════════════════════════════
#  TRAINING LOOP (generic)
# ════════════════════════════════════════════════════════════════════════════════
def make_optimizer_scheduler(model, n_epochs):
    opt = AdamW(model.parameters(), lr=CFG.LR,
                weight_decay=CFG.WEIGHT_DECAY, betas=(0.9, 0.999), eps=1e-8)
    def lr_lambda(epoch):
        if epoch < CFG.WARMUP_EPOCHS:
            return epoch / max(1, CFG.WARMUP_EPOCHS)
        progress = (epoch - CFG.WARMUP_EPOCHS) / max(1, n_epochs - CFG.WARMUP_EPOCHS)
        return max(CFG.MIN_LR / CFG.LR, 0.5 * (1 + math.cos(math.pi * progress)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return opt, sched


def train_one_method(method_name, model, train_dl, eval_sample_fn,
                     ckpt_path, aux_iter=None, n_epochs=None,
                     uses_cls_for_sample=False,
                     k_samples_eval=1, collect_uncertainty=False):
    n_epochs = n_epochs or CFG.TOTAL_EPOCHS
    print(f"\n{'='*70}\nTraining {method_name}\n{'='*70}")
    model.to(DEVICE)
    opt, sched = make_optimizer_scheduler(model, n_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    ema = EMA(model)
    step = 0; best_pc = -1.0; best_metrics = {}
    history = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        losses = defaultdict(list)
        opt.zero_grad()
        pbar = tqdm(train_dl, leave=False, desc=f"{method_name} Ep{epoch}")
        for bi, batch in enumerate(pbar):
            with torch.cuda.amp.autocast():
                if aux_iter is not None:
                    try: batch_aux = next(aux_iter)
                    except StopIteration:
                        aux_iter = iter(aux_dl); batch_aux = next(aux_iter)
                    ld = model.training_step(
                        batch, batch_aux=batch_aux,
                        subject_id=CFG.PRIMARY_SUBJECT,
                        aux_subject_id=CFG.AUX_SUBJECT)
                else:
                    ld = model.training_step(batch, subject_id=CFG.PRIMARY_SUBJECT)
                loss = ld["loss"] / CFG.GRAD_ACCUM
            scaler.scale(loss).backward()
            if (bi + 1) % CFG.GRAD_ACCUM == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)
                scaler.step(opt); scaler.update(); opt.zero_grad()
                if step >= CFG.EMA_START: ema.update(model)
                step += 1
            for k, v in ld.items():
                losses[k].append(v.item() if torch.is_tensor(v) else float(v))
            pbar.set_postfix(c=f"{ld['cfm'].item():.3f}")
        sched.step()
        train_log = {f"{method_name}/train/{k}": float(np.mean(v))
                     for k, v in losses.items()}
        train_log[f"{method_name}/train/lr"] = opt.param_groups[0]["lr"]
        train_log["epoch"] = epoch

        if (epoch % CFG.EVAL_FREQ == 0 or epoch == n_epochs):
            gc.collect(); torch.cuda.empty_cache()
            if step >= CFG.EMA_START:
                orig = {k: v.clone() for k, v in model.state_dict().items()}
                ema.apply(model)
                m = evaluate(model, test_dl, eval_sample_fn,
                             subject_id=CFG.PRIMARY_SUBJECT,
                             collect_uncertainty=collect_uncertainty,
                             k_samples=k_samples_eval)
                ema.restore(orig, model); tag = "EMA"
            else:
                m = evaluate(model, test_dl, eval_sample_fn,
                             subject_id=CFG.PRIMARY_SUBJECT,
                             collect_uncertainty=collect_uncertainty,
                             k_samples=k_samples_eval)
                tag = "raw"
            for k, v in m.items(): train_log[f"{method_name}/test/{k}"] = v
            tqdm.write(f"[{method_name}] Ep{epoch:3d}/{n_epochs} | "
                       f"cfm={train_log[f'{method_name}/train/cfm']:.3f} | "
                       f"PC={m['PixCorr']:.3f} SSIM={m['SSIM']:.3f} "
                       f"CLIP={m['CLIP_Sim']:.3f} ({tag})")
            if m["PixCorr"] > best_pc:
                best_pc = m["PixCorr"]; best_metrics = m
                torch.save(model.state_dict(), ckpt_path)
            history.append({"epoch": epoch, **m})
        wandb.log(train_log, step=epoch + (0 if method_name == "M0" else
                                            1000 if method_name == "MA" else 2000))
    # Final EMA save
    if step >= CFG.EMA_START:
        ema.apply(model)
    torch.save(model.state_dict(), str(ckpt_path).replace(".pt", "_ema.pt"))
    print(f"[{method_name}] best PC={best_pc:.4f} | metrics={best_metrics}")
    return best_metrics, history, ema


# %% [code]
# ════════════════════════════════════════════════════════════════════════════════
#  ZERO-SHOT SUBJECT TRANSFER (Method A only)
# ════════════════════════════════════════════════════════════════════════════════
def zero_shot_transfer(model, raw_zs):
    """Fit only the new subject's stem with alignment loss; eval reconstruction."""
    if raw_zs is None:
        return {"PixCorr": float("nan"), "SSIM": float("nan"),
                "CLIP_Sim": float("nan")}
    print("\n[MA] Zero-shot subject transfer…")
    model.to(DEVICE)
    n_vox = raw_zs["fmri_train"].shape[1]
    sid = CFG.ZEROSHOT_SUBJECT
    model.brain_enc.add_subject(sid, n_vox, DEVICE)

    # Freeze everything except the new stem
    for p in model.parameters(): p.requires_grad_(False)
    new_stem = model.brain_enc.stems[str(sid)]
    for p in new_stem.parameters(): p.requires_grad_(True)

    # Z-score subj 5 fmri
    f = raw_zs["fmri_train"].float()
    mu = f.mean(0, keepdim=True); std = f.std(0, keepdim=True).clamp(1e-6)
    f = (f - mu) / std
    f = f.to(DEVICE)
    clip_zs = raw_zs["clip_train"].float().to(DEVICE)

    opt = AdamW(new_stem.parameters(), lr=CFG.ZEROSHOT_LR)
    bs = min(32, f.shape[0])
    model.train()
    for step in range(CFG.ZEROSHOT_STEPS):
        idx = torch.randperm(f.shape[0], device=DEVICE)[:bs]
        _, cls_emb = model.brain_enc(f[idx], sid)
        logits = (cls_emb @ clip_zs[idx].T) / CFG.INFONCE_TEMP
        labels = torch.arange(bs, device=DEVICE)
        loss = (F.cross_entropy(logits, labels)
                + F.cross_entropy(logits.T, labels)) / 2
        opt.zero_grad(); loss.backward(); opt.step()

    # Eval reconstruction on subj 5 train images (since we have no held-out test for it)
    model.eval(); vae.to(DEVICE)
    pcs, sss, cls_ = [], [], []
    n_eval = min(64, raw_zs["images_train"].shape[0])
    images = raw_zs["images_train"][:n_eval].to(DEVICE)
    with torch.no_grad():
        for i in range(0, n_eval, CFG.BATCH_SIZE):
            fb = f[i:i+CFG.BATCH_SIZE]
            ib = images[i:i+CFG.BATCH_SIZE]
            tokens, _ = model.brain_enc(fb, sid)
            pl = model.sample_sde(tokens)
            pi = vae.decode(pl)
            pcs.append(pixel_correlation(pi, ib))
            sss.append(ssim_pytorch(pi.cpu(), ib.cpu()))
            cls_.append(clip_similarity(pi, ib))
    vae.cpu()
    for p in model.parameters(): p.requires_grad_(True)
    return {"PixCorr": float(np.mean(pcs)),
            "SSIM": float(np.mean(sss)),
            "CLIP_Sim": float(np.mean(cls_))}


# %% [code]
# ════════════════════════════════════════════════════════════════════════════════
#  RUN ALL THREE METHODS
# ════════════════════════════════════════════════════════════════════════════════
voxel_dims = {str(CFG.PRIMARY_SUBJECT): raw1["n_voxels"]}
if raw_aux is not None:
    voxel_dims[str(CFG.AUX_SUBJECT)] = raw_aux["n_voxels"]

run = wandb.init(project="brainflow-compare",
                 name=f"compare_smoke{int(CFG.SMOKE_TEST)}",
                 config={k: v for k, v in vars(CFG).items()
                         if not k.startswith("_") and isinstance(v, (int, float, str, bool))})

results = {}; histories = {}

# %% [code]
# ── M0: Baseline ──
if CFG.RUN_BASELINE:
    m0 = BaselineModel(voxel_dims)
    print(f"M0 params: {sum(p.numel() for p in m0.parameters())/1e6:.1f}M")
    def m0_sample(model, tokens):
        return model.sample_ode(tokens)
    results["M0"], histories["M0"], _ = train_one_method(
        "M0", m0, train_dl, m0_sample,
        ckpt_path=CFG.OUT_DIR / "best_M0.pt")
    m0.cpu(); del m0; gc.collect(); torch.cuda.empty_cache()

# %% [code]
# ── MA: SB + OT ──
if CFG.RUN_SB_OT:
    ma = SBOTModel(voxel_dims)
    print(f"MA params: {sum(p.numel() for p in ma.parameters())/1e6:.1f}M")
    aux_iter = iter(aux_dl) if aux_dl is not None else None
    def ma_sample(model, tokens):
        return model.sample_sde(tokens)
    results["MA"], histories["MA"], ma_ema = train_one_method(
        "MA", ma, train_dl, ma_sample,
        ckpt_path=CFG.OUT_DIR / "best_MA.pt", aux_iter=aux_iter,
        k_samples_eval=CFG.K_SAMPLES_SB, collect_uncertainty=True)
    # Zero-shot transfer
    zs_metrics = zero_shot_transfer(ma, raw_zs)
    results["MA"]["ZeroShot_PixCorr"] = zs_metrics["PixCorr"]
    results["MA"]["ZeroShot_SSIM"] = zs_metrics["SSIM"]
    results["MA"]["ZeroShot_CLIP_Sim"] = zs_metrics["CLIP_Sim"]
    print(f"[MA] zero-shot subj{CFG.ZEROSHOT_SUBJECT}: {zs_metrics}")
    # keep ma alive for the figure later
    ma_for_viz = ma
else:
    ma_for_viz = None

# %% [code]
# ── MB: HRF Brain-Manifold ──
if CFG.RUN_HRF:
    mb = HRFModel(voxel_dims)
    print(f"MB params: {sum(p.numel() for p in mb.parameters())/1e6:.1f}M")
    def mb_sample(model, tokens):
        # We need cls_emb too — refetch via dataset hook isn't trivial; use a closure
        # Recompute cls inside: pass a wrapped fn via evaluate that knows subject
        # Simpler: monkey-patch evaluate to compute cls; we use trick: rerun encoder
        # via batch passed in evaluate. But evaluate only passes tokens.
        # Workaround: store last batch fmri on the model.
        return model.sample_ode(tokens, model._last_cls)

    # Override evaluate-side: monkey-patch HRFModel.brain_enc forward to cache cls
    _orig_forward = mb.brain_enc.forward
    def _cache_forward(x, sid):
        tokens, cls = _orig_forward(x, sid)
        mb._last_cls = cls
        return tokens, cls
    mb.brain_enc.forward = _cache_forward

    results["MB"], histories["MB"], _ = train_one_method(
        "MB", mb, train_dl, mb_sample,
        ckpt_path=CFG.OUT_DIR / "best_MB.pt")
    mb_for_viz = mb
else:
    mb_for_viz = None


# %% [code]
# ════════════════════════════════════════════════════════════════════════════════
#  PHASE 4 — UNIFIED COMPARISON
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70); print("Phase 4: Final comparison"); print("=" * 70)

# WandB table
table_rows = []
for method, m in results.items():
    row = [method,
           m.get("PixCorr", float("nan")),
           m.get("SSIM", float("nan")),
           m.get("CLIP_Sim", float("nan")),
           m.get("Uncertainty", float("nan")),
           m.get("ZeroShot_PixCorr", float("nan")),
           m.get("ZeroShot_CLIP_Sim", float("nan"))]
    table_rows.append(row)
    print(f"  {method}: {m}")

cols = ["Method", "PixCorr", "SSIM", "CLIP_Sim", "Uncertainty",
        "ZS_PixCorr", "ZS_CLIP_Sim"]
wandb.log({"final_comparison": wandb.Table(columns=cols, data=table_rows)})

# Save JSON summary
with open(CFG.OUT_DIR / "compare_summary.json", "w") as f:
    json.dump({"results": results, "histories": histories,
               "config": {k: str(v) for k, v in vars(CFG).items()
                          if not k.startswith("_")}}, f, indent=2, default=str)

# Side-by-side figure: 4 sample trials × {GT, M0, MA mean, MA std, MB}
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
n_viz = 4
ncols = 5  # GT, M0, MA mean, MA uncertainty, MB
fig, axes = plt.subplots(n_viz, ncols, figsize=(ncols * 2.6, n_viz * 2.6))
fig.suptitle("BrainFlow comparison — GT vs methods (subj 1 test)",
             fontsize=12, fontweight="bold")

vae.to(DEVICE)
# Re-load best checkpoints for each method (saved during training)
m0_eval = BaselineModel(voxel_dims).to(DEVICE).eval()
if (CFG.OUT_DIR / "best_M0.pt").exists():
    m0_eval.load_state_dict(torch.load(CFG.OUT_DIR / "best_M0.pt", map_location=DEVICE))

if ma_for_viz is not None:
    ma_for_viz.to(DEVICE).eval()
if mb_for_viz is not None:
    mb_for_viz.to(DEVICE).eval()

with torch.no_grad():
    for i in range(n_viz):
        s = test_ds[i]
        fmri = s["fmri"].unsqueeze(0).to(DEVICE)
        gt = s["image"].permute(1, 2, 0).numpy().clip(0, 1)
        axes[i, 0].imshow(gt); axes[i, 0].axis("off")
        if i == 0: axes[i, 0].set_title("GT", fontsize=9)

        # M0
        if CFG.RUN_BASELINE:
            tk, _ = m0_eval.brain_enc(fmri, CFG.PRIMARY_SUBJECT)
            pl = m0_eval.sample_ode(tk)
            pr = vae.decode(pl).squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)
            axes[i, 1].imshow(pr); axes[i, 1].axis("off")
            if i == 0: axes[i, 1].set_title("M0 Baseline", fontsize=9)

        # MA mean + uncertainty
        if ma_for_viz is not None:
            tk, _ = ma_for_viz.brain_enc(fmri, CFG.PRIMARY_SUBJECT)
            preds = []
            for _ in range(CFG.K_SAMPLES_SB):
                pl = ma_for_viz.sample_sde(tk)
                preds.append(vae.decode(pl))
            preds = torch.stack(preds, 0)
            mean_img = preds.mean(0).squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)
            std_img = preds.std(0).squeeze(0).mean(0).cpu().numpy()
            axes[i, 2].imshow(mean_img); axes[i, 2].axis("off")
            axes[i, 3].imshow(std_img, cmap="hot"); axes[i, 3].axis("off")
            if i == 0:
                axes[i, 2].set_title("MA SB+OT mean", fontsize=9)
                axes[i, 3].set_title("MA uncertainty", fontsize=9)

        # MB
        if mb_for_viz is not None:
            tk, cls = mb_for_viz.brain_enc(fmri, CFG.PRIMARY_SUBJECT)
            pl = mb_for_viz.sample_ode(tk, cls)
            pr = vae.decode(pl).squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)
            axes[i, 4].imshow(pr); axes[i, 4].axis("off")
            if i == 0: axes[i, 4].set_title("MB HRF-Flow", fontsize=9)

plt.tight_layout()
out_png = CFG.OUT_DIR / "compare_grid.png"
plt.savefig(out_png, bbox_inches="tight", dpi=180)
wandb.log({"compare_grid": wandb.Image(str(out_png))})
print(f"Saved {out_png}")

vae.cpu(); gc.collect(); torch.cuda.empty_cache()
wandb.finish()
print("\nAll done.")
