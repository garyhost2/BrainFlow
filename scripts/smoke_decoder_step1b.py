"""Step-1b smoke test: decode GROUND-TRUTH bigG tokens -> images.

Proves the SDXL-unCLIP decoder + bigG embedder + token space all work, BEFORE
training anything.  If real tokens give clean reconstructions, the decoder path
is correct and the only remaining job is the brain->token prior.

    python -m scripts.smoke_decoder_step1b \
        --data-dir ./mindeyev2_cache --subject 1 --n 8 \
        --mindeye-src third_party/MindEyeV2/src \
        --ckpt-path third_party/unclip6_epoch0_step110000.ckpt \
        --out outputs/step1b/smoke
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from brainflow.step1.targets_bigg import _load_embedder, _encode
from brainflow.step1.decoder_sgm import SDXLUnCLIPDecoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="./mindeyev2_cache")
    ap.add_argument("--tensor-cache", type=str, default="all_subjects_tensors.pt")
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--mindeye-src", type=str, default="third_party/MindEyeV2/src")
    ap.add_argument("--ckpt-path", type=str, default="third_party/unclip6_epoch0_step110000.ckpt")
    ap.add_argument("--out", type=str, default="outputs/step1b/smoke")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    tensors = torch.load(data_dir / args.tensor_cache, map_location="cpu")
    imgs = tensors[f"imgs_test_{args.subject}"][:args.n]   # uint8 (n,3,H,W)

    print("▶ Encoding ground-truth bigG tokens…")
    emb = _load_embedder(device, args.mindeye_src)
    tokens = _encode(emb, imgs, device).float()           # (n,256,1664)
    del emb; torch.cuda.empty_cache()

    print("▶ Loading SDXL-unCLIP decoder…")
    decoder = SDXLUnCLIPDecoder(device, args.mindeye_src, args.ckpt_path)

    print("▶ Decoding…")
    recon = decoder.decode(tokens)                        # (n,3,256,256) in [0,1]

    gt = (imgs.float() / 255.0).clamp(0, 1)
    if gt.shape[-1] != recon.shape[-1]:
        gt = torch.nn.functional.interpolate(gt, recon.shape[-1], mode="bilinear", align_corners=False)
    try:
        from torchvision.utils import save_image
        save_image(torch.cat([gt, recon]), str(out_dir / "smoke_grid.png"), nrow=args.n)
        print(f"✓ Wrote {out_dir/'smoke_grid.png'} — top row GT, bottom row decoded-from-GT-tokens.")
        print("  If the bottom row looks like the top row, the decoder path is CORRECT.")
    except Exception as e:
        print(f"[grid skipped] {e}")


if __name__ == "__main__":
    main()
