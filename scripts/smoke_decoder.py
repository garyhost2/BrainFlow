from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rxfm.tensor_cache import assert_tensor_cache_alignment
from rxfm.targets import _load_embedder, _encode_dual
from rxfm.decoder import SDXLUnCLIPDecoder
from rxfm.image_metrics import pixcorr, ssim

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="./mindeyev2_cache")
    ap.add_argument("--tensor-cache", type=str, default="all_subjects_tensors.pt")
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--mindeye-src", type=str, default="third_party/MindEyeV2/src")
    ap.add_argument("--ckpt-path", type=str, default="third_party/unclip6_epoch0_step110000.ckpt")
    ap.add_argument("--cls-vector-slot", type=int, nargs=2, default=[0, 1280],
                    help="[lo hi] positions in the unCLIP 'vector' slot to fill with c_cls")
    ap.add_argument("--out", type=str, default="outputs/step1b/smoke")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    tensors = assert_tensor_cache_alignment(data_dir / args.tensor_cache, torch.load(data_dir / args.tensor_cache, map_location="cpu"))
    imgs = tensors[f"imgs_test_{args.subject}"][:args.n]

    print("▶ Encoding ground-truth bigG (cls + tokens)…")
    emb = _load_embedder(device, args.mindeye_src)
    cls, tokens = _encode_dual(emb, imgs, device)
    cls, tokens = cls.float(), tokens.float()
    del emb; torch.cuda.empty_cache()

    print("▶ Loading SDXL-unCLIP decoder…")
    decoder = SDXLUnCLIPDecoder(device, args.mindeye_src, args.ckpt_path,
                                cls_vector_slot=tuple(args.cls_vector_slot))

    print("▶ Decoding from GT tokens (unclip6 conditions on the 256x1664 crossattn)…")
    recon = decoder.decode(tokens, cls)

    gt = (imgs.float() / 255.0).clamp(0, 1)
    if gt.shape[-1] != recon.shape[-1]:
        gt = torch.nn.functional.interpolate(gt, recon.shape[-1], mode="bilinear", align_corners=False)

    pc, ss = pixcorr(recon, gt), ssim(recon, gt)
    print(f"▶ Decoder ceiling (GROUND-TRUTH tokens): PixCorr={pc:.4f}  SSIM={ss:.4f}")
    print(f"  Full-run eval was PixCorr=0.077 SSIM=0.207 — compare to this ceiling.")
    print(f"  If this ceiling is also low, the floor is the decoder design (no structural")
    print(f"  conditioning), not the brain->token prior.")
    try:
        from torchvision.utils import save_image
        save_image(torch.cat([gt, recon]), str(out_dir / "smoke_grid.png"), nrow=args.n)
        print(f"✓ Wrote {out_dir/'smoke_grid.png'} — top row GT, bottom row decoded-from-GT-tokens.")
        print("  If the bottom row looks like the top row, the decoder path is CORRECT.")
    except Exception as e:
        print(f"[grid skipped] {e}")

if __name__ == "__main__":
    main()
