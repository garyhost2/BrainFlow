"""Offline bigG target extraction: the *structured* (CLS + patch) target.

Each image is passed once through the frozen OpenCLIP ViT-bigG/14 embedder in
token-output mode (``output_tokens=True, only_tokens=False``), yielding both
objects of the structured factorisation (paper Stage 2a):

  * the pooled, projected **class embedding** ``c_cls`` in R^{1280} -- the global
    direction fed to the unCLIP decoder's pooled slot and the SoftCLIP anchor;
  * the **penultimate patch-token sequence** ``X`` in R^{256 x 1664} -- the
    cross-attention context the patch prior transports on the oblique manifold.

The per-channel token mean/std (``TargetStats``) are accumulated on the *training
split only* (no test leakage); they standardise the Euclidean baseline and centre
the polar decomposition (``mu`` = the token mean). The CLS directions need no
statistics -- they are unit-normalised at use.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .target_stats import TargetStats

TOKEN_LEN = 256
TOKEN_DIM = 1664
CLS_DIM = 1280


def _add_sgm_to_path(mindeye_src: Path):
    mindeye_src = Path(mindeye_src)
    gm = mindeye_src / "generative_models"
    if not gm.exists():
        raise FileNotFoundError(
            f"Vendored sgm not found at {gm}. Run scripts/setup_decoder.sh first.")
    for p in (str(mindeye_src), str(gm)):
        if p not in sys.path:
            sys.path.insert(0, p)


@torch.no_grad()
def _load_embedder(device, mindeye_src: Path):
    _add_sgm_to_path(mindeye_src)
    from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder
    emb = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14", version="laion2b_s39b_b160k",
        output_tokens=True, only_tokens=False,   # <- keep BOTH cls and tokens
    )
    emb = emb.eval().to(device)
    for p in emb.parameters():
        p.requires_grad_(False)
    return emb


def _split_outputs(out):
    """Return (cls[B,1280], tokens[B,256,1664]) from the embedder output.

    sgm returns ``(pooled, tokens)`` when ``output_tokens=True``; we identify the
    members by rank rather than position so the order cannot silently flip.
    """
    if not isinstance(out, (tuple, list)) or len(out) < 2:
        raise RuntimeError(
            "bigG embedder must return (pooled, tokens); construct it with "
            "output_tokens=True, only_tokens=False.")
    a, b = out[0], out[1]
    tok, cls = (a, b) if a.ndim == 3 else (b, a)
    if cls.ndim != 2 or tok.ndim != 3:
        raise RuntimeError(f"Unexpected bigG output ranks: {a.shape}, {b.shape}")
    return cls, tok


@torch.no_grad()
def _encode_dual(emb, imgs, device, bs=16):
    cls_out, tok_out = [], []
    for i in tqdm(range(0, len(imgs), bs), desc="bigG cls+tokens", leave=False):
        x = imgs[i:i + bs]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        x = x.clamp(0, 1).to(device)
        cls, tok = _split_outputs(emb(x))
        cls_out.append(cls.float().cpu().half())
        tok_out.append(tok.float().cpu().half())
    return torch.cat(cls_out), torch.cat(tok_out)


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
        have = fp.exists() and not force_rebuild
        if have:
            # A pre-existing token-only cache lacks the CLS; trigger a rebuild.
            keys = torch.load(str(fp), map_location="cpu", mmap=True).keys()
            have = "cls_train" in keys
        if not have:
            if embedder is None:
                embedder = _load_embedder(device, mindeye_src)
            print(f"  bigG encode subj{s:02d} (cls + tokens)")
            cls_tr, e_tr = _encode_dual(embedder, tensors[f"imgs_train_{s}"], device)
            cls_te, e_te = _encode_dual(embedder, tensors[f"imgs_test_{s}"], device)
            cnt, ssum, sqsum = _accum_stats(e_tr)
            torch.save({"emb_train": e_tr, "emb_test": e_te,
                        "cls_train": cls_tr, "cls_test": cls_te,
                        "accum": {"count": cnt, "sum": ssum, "sqsum": sqsum}}, fp)
            print(f"  ✓ saved {fp} ({fp.stat().st_size/1e9:.1f} GB)")
            del e_tr, e_te, cls_tr, cls_te
            gc.collect()

        blob = torch.load(str(fp), map_location="cpu", mmap=True)
        out[f"emb_train_{s}"] = blob["emb_train"]
        out[f"emb_test_{s}"] = blob["emb_test"]
        out[f"cls_train_{s}"] = blob["cls_train"]
        out[f"cls_test_{s}"] = blob["cls_test"]
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
                    "token_len": TOKEN_LEN, "token_dim": TOKEN_DIM,
                    "cls_dim": CLS_DIM, "norm": "raw_tokens", "only_tokens": False}
    return out
