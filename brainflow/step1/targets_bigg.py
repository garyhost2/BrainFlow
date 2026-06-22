from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .targets import TargetStats

TOKEN_LEN = 256
TOKEN_DIM = 1664

def _add_sgm_to_path(mindeye_src: Path):
    mindeye_src = Path(mindeye_src)
    gm = mindeye_src / "generative_models"
    if not gm.exists():
        raise FileNotFoundError(
            f"Vendored sgm not found at {gm}. Run scripts/setup_step1b.sh first.")
    for p in (str(mindeye_src), str(gm)):
        if p not in sys.path:
            sys.path.insert(0, p)

@torch.no_grad()
def _load_embedder(device, mindeye_src: Path):
    _add_sgm_to_path(mindeye_src)
    from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder
    emb = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14", version="laion2b_s39b_b160k",
        output_tokens=True, only_tokens=True,
    )
    emb = emb.eval().to(device)
    for p in emb.parameters():
        p.requires_grad_(False)
    return emb

@torch.no_grad()
def _encode(emb, imgs, device, bs=16):
    outs = []
    for i in tqdm(range(0, len(imgs), bs), desc="bigG tokens", leave=False):
        x = imgs[i:i + bs]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        x = x.clamp(0, 1).to(device)
        outs.append(emb(x).float().cpu().half())
    return torch.cat(outs)

def _accum_stats(e_tr_fp16, chunk=1024):
    cnt = 0
    s = torch.zeros(TOKEN_DIM, dtype=torch.float64)
    sq = torch.zeros(TOKEN_DIM, dtype=torch.float64)
    for i in range(0, len(e_tr_fp16), chunk):
        f = e_tr_fp16[i:i + chunk].float().reshape(-1, TOKEN_DIM)
        cnt += f.shape[0]
        s += f.sum(0).double()
        sq += (f * f).sum(0).double()
    return cnt, s, sq

def _subj_file(cache_dir: Path, subj: int) -> Path:
    return Path(cache_dir) / f"step1b_bigg_s{subj}.pt"

def build_or_load_bigg_targets(tensors, subjects, cache_dir, device, mindeye_src,
                               hf_cache=None, force_rebuild=False):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    embedder = None
    out = {}
    tot_cnt = 0
    tot_s = torch.zeros(TOKEN_DIM, dtype=torch.float64)
    tot_sq = torch.zeros(TOKEN_DIM, dtype=torch.float64)

    for s in subjects:
        fp = _subj_file(cache_dir, s)
        if not fp.exists() or force_rebuild:
            if embedder is None:
                embedder = _load_embedder(device, mindeye_src)
            print(f"  bigG encode subj{s:02d}")
            e_tr = _encode(embedder, tensors[f"imgs_train_{s}"], device)
            e_te = _encode(embedder, tensors[f"imgs_test_{s}"], device)
            cnt, ssum, sqsum = _accum_stats(e_tr)
            torch.save({"emb_train": e_tr, "emb_test": e_te,
                        "accum": {"count": cnt, "sum": ssum, "sqsum": sqsum}}, fp)
            print(f"  ✓ saved {fp} ({fp.stat().st_size/1e9:.1f} GB)")
            del e_tr, e_te
            gc.collect()

        blob = torch.load(str(fp), map_location="cpu", mmap=True)
        out[f"emb_train_{s}"] = blob["emb_train"]
        out[f"emb_test_{s}"] = blob["emb_test"]
        a = blob["accum"]
        tot_cnt += int(a["count"])
        tot_s += a["sum"].double()
        tot_sq += a["sqsum"].double()

    if embedder is not None:
        del embedder
        gc.collect()
        torch.cuda.empty_cache()

    mean = (tot_s / tot_cnt).float()
    var = (tot_sq / tot_cnt - (tot_s / tot_cnt) ** 2).clamp_min(1e-12).float()
    out["_stats"] = TargetStats(mean=mean, std=var.sqrt().clamp_min(1e-6))
    out["_meta"] = {"embedder": "ViT-bigG-14/laion2b_s39b_b160k",
                    "token_len": TOKEN_LEN, "token_dim": TOKEN_DIM, "norm": "raw_tokens"}
    return out
