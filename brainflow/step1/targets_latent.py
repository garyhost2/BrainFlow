"""Offline blurry-latent target: the low-level pathway's regression target.

Four attempts at the low-level head regressed *RGB* (a blurry image), and the
decoder then re-encoded that RGB through the VAE to get its img2img init. Every
one failed -- the first three underfit to the per-image mean colour, the fourth
(L1 @ 64^2, run 347830) fit the train set and overfit, with val ``blur_mse``
rising monotonically while PixCorr stayed flat.

This module supplies the alternative target: the **VAE latent of the blurred
image**, ``E(blur(GT))`` in R^{4 x 96 x 96}, which is what the decoder actually
consumes. Predicting it directly

  * removes a lossy round-trip (head -> sigmoid RGB -> VAE encode -> latent);
  * puts the loss in the space the diffusion model is trained on, where the VAE
    has already discarded the high-frequency detail fMRI cannot predict, instead
    of in pixel space where that unpredictable detail dominates the L1.

Latents are cached per subject exactly like the bigG targets -- the VAE never
runs during training. Storage is ~74 KB per image (fp16), i.e. ~2 GB per subject
for 27k train trials, on top of the ~23 GB bigG cache.

Channel statistics are accumulated on the TRAIN split only and used to
standardise the regression target; ``encode_first_stage`` already applies the
model's ``scale_factor``, so the stored latents are in the same space the
sampler expects for ``init_latent``.
"""

from __future__ import annotations

import gc
from pathlib import Path

import torch
from tqdm.auto import tqdm

LATENT_C, LATENT_H, LATENT_W = 4, 96, 96


def _subj_file(cache_dir: Path, subj: int, ll_size: int) -> Path:
    return Path(cache_dir) / f"step1c_latent_s{subj}_b{ll_size}.pt"


@torch.no_grad()
def _encode_split(decoder, imgs, ll_size: int, bs: int = 8) -> torch.Tensor:
    """Blur -> VAE-encode a stack of images, in chunks (768^2 encodes are heavy)."""
    out = []
    for i in tqdm(range(0, len(imgs), bs), desc=f"blur{ll_size}->latent", leave=False):
        z = decoder.encode_blurry(imgs[i:i + bs], ll_size=ll_size)
        out.append(z.float().cpu().half())
    return torch.cat(out)


def _accum_stats(lat_fp16, chunk: int = 512):
    """Per-channel mean/std over the train split (N,4,96,96) in float64."""
    cnt = 0
    s = torch.zeros(LATENT_C, dtype=torch.float64)
    sq = torch.zeros(LATENT_C, dtype=torch.float64)
    for i in range(0, len(lat_fp16), chunk):
        f = lat_fp16[i:i + chunk].float()
        cnt += f.shape[0] * f.shape[2] * f.shape[3]
        s += f.sum(dim=(0, 2, 3)).double()
        sq += (f * f).sum(dim=(0, 2, 3)).double()
    return cnt, s, sq


def build_or_load_latent_targets(tensors, subjects, cache_dir, decoder,
                                 ll_size: int = 64, force_rebuild: bool = False):
    """Return ``{lat_train_{s}, lat_test_{s}, _lat_stats}``; build any missing cache.

    ``decoder`` is an :class:`SDXLUnCLIPDecoder` -- required only when a cache is
    missing. Pass ``None`` to load-or-fail without holding the VAE in memory.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    tot_cnt = 0
    tot_s = torch.zeros(LATENT_C, dtype=torch.float64)
    tot_sq = torch.zeros(LATENT_C, dtype=torch.float64)

    for s in subjects:
        fp = _subj_file(cache_dir, s, ll_size)
        if not fp.exists() or force_rebuild:
            if decoder is None:
                raise FileNotFoundError(
                    f"{fp} missing and no decoder given to build it. Run "
                    f"scripts/build_latent_targets.py (needs the A100 + vendored sgm).")
            print(f"  blurry-latent encode subj{s:02d} (ll_size={ll_size})", flush=True)
            lat_tr = _encode_split(decoder, tensors[f"imgs_train_{s}"], ll_size)
            lat_te = _encode_split(decoder, tensors[f"imgs_test_{s}"], ll_size)
            cnt, ssum, sqsum = _accum_stats(lat_tr)
            torch.save({"lat_train": lat_tr, "lat_test": lat_te, "ll_size": ll_size,
                        "accum": {"count": cnt, "sum": ssum, "sqsum": sqsum}}, fp)
            print(f"  ✓ saved {fp} ({fp.stat().st_size / 1e9:.2f} GB)", flush=True)
            del lat_tr, lat_te
            gc.collect()

        blob = torch.load(str(fp), map_location="cpu", mmap=True)
        if int(blob.get("ll_size", ll_size)) != ll_size:
            raise ValueError(
                f"{fp} was built with ll_size={blob['ll_size']}, not {ll_size}. "
                f"Delete it or pass the matching --ll-size.")
        out[f"lat_train_{s}"] = blob["lat_train"]
        out[f"lat_test_{s}"] = blob["lat_test"]
        a = blob["accum"]
        tot_cnt += int(a["count"])
        tot_s += a["sum"].double()
        tot_sq += a["sqsum"].double()

    mean = (tot_s / tot_cnt).float()
    var = (tot_sq / tot_cnt - (tot_s / tot_cnt) ** 2).clamp_min(1e-12).float()
    out["_lat_stats"] = {"mean": mean, "std": var.sqrt().clamp_min(1e-6),
                         "ll_size": ll_size}
    return out
