"""Calibrate the low-level img2img pathway (find the best --ll-strength).

Decodes GROUND-TRUTH bigG tokens (+ CLS) with a GROUND-TRUTH *blurry* image as the
img2img init, sweeping ``strength``. This both (a) validates the img2img code path
produces sane images on the real engine and (b) shows how much PixCorr/SSIM the
blurry init buys vs. the token-only decode (strength=1.0). Pick the strength that
maximises PixCorr/SSIM without collapsing CLIP, and pass it as --ll-strength when
training the low-level head.

Run: sbatch slurm/smoke_step1b_a100.sbatch  (or adapt) -> python -m scripts.smoke_lowlevel
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from brainflow.tensor_cache import assert_tensor_cache_alignment
from brainflow.step1.targets_bigg import _load_embedder, _encode_dual
from brainflow.step1.decoder_sgm import SDXLUnCLIPDecoder
from brainflow.step1.metrics import pixcorr, ssim, CLIPMetric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="./mindeyev2_cache")
    ap.add_argument("--tensor-cache", type=str, default="all_subjects_tensors.pt")
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--ll-size", type=int, default=128)
    ap.add_argument("--strengths", type=float, nargs="+", default=[1.0, 0.8, 0.7, 0.6, 0.5])
    ap.add_argument("--mindeye-src", type=str, default="third_party/MindEyeV2/src")
    ap.add_argument("--ckpt-path", type=str, default="third_party/unclip6_epoch0_step110000.ckpt")
    ap.add_argument("--out", type=str, default="outputs/step1b/smoke_lowlevel")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir); hf_cache = data_dir / "hf_cache"

    tensors = assert_tensor_cache_alignment(data_dir / args.tensor_cache, torch.load(data_dir / args.tensor_cache, map_location="cpu"))
    imgs = tensors[f"imgs_test_{args.subject}"][:args.n]
    gt = (imgs.float() / 255.0).clamp(0, 1) if imgs.dtype == torch.uint8 else imgs.float().clamp(0, 1)

    print("▶ Encoding GT bigG (cls + tokens)…")
    emb = _load_embedder(device, args.mindeye_src)
    cls, tokens = _encode_dual(emb, imgs, device)
    cls, tokens = cls.float(), tokens.float()
    del emb; torch.cuda.empty_cache()

    # GROUND-TRUTH blurry init = GT downsampled to ll_size (best case the head can predict)
    blurry = F.interpolate(gt, args.ll_size, mode="bilinear", align_corners=False).clamp(0, 1)

    print("▶ Loading SDXL-unCLIP decoder…")
    decoder = SDXLUnCLIPDecoder(device, args.mindeye_src, args.ckpt_path)
    clip_metric = CLIPMetric(device, hf_cache=hf_cache)

    def gt_match(recon):
        g = gt
        if g.shape[-1] != recon.shape[-1]:
            g = F.interpolate(g, recon.shape[-1], mode="bilinear", align_corners=False)
        m = {"PixCorr": pixcorr(recon, g), "SSIM": ssim(recon, g)}
        m.update(clip_metric.score(recon, g))
        return m

    print(f"\n  strength | PixCorr   SSIM   CLIP_2way   (GT tokens + GT-blurry img2img init)")
    rows = []
    for s in args.strengths:
        recon = decoder.decode(tokens, cls, init_image=blurry, strength=s)
        m = gt_match(recon)
        rows.append((s, m))
        tag = " (token-only)" if s >= 1.0 else ""
        print(f"   {s:5.2f}   | {m['PixCorr']:.3f}    {m['SSIM']:.3f}   {m['CLIP_2way']:.3f}{tag}", flush=True)
        try:
            from torchvision.utils import save_image
            k = min(8, recon.shape[0])
            save_image(torch.cat([gt[:k], recon[:k]]), str(out_dir / f"grid_s{s:.2f}.png"), nrow=k)
        except Exception:
            pass

    best = max(rows, key=lambda r: r[1]["PixCorr"] + r[1]["SSIM"])
    print(f"\n  => best low-level strength ≈ {best[0]:.2f} "
          f"(PixCorr={best[1]['PixCorr']:.3f} SSIM={best[1]['SSIM']:.3f}). "
          f"Train with --ll-strength {best[0]:.2f}.")
    print("  (This is the GT-blurry CEILING; the trained head's blurry recon is imperfect,")
    print("   so real gains are smaller — but it confirms the mechanism + the strength knob.)")


if __name__ == "__main__":
    main()
