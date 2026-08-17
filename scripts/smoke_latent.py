from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from rxfm.tensor_cache import assert_tensor_cache_alignment
from rxfm.targets import _load_embedder, _encode_dual
from rxfm.decoder import SDXLUnCLIPDecoder, quiet_benign_warnings
from rxfm.image_metrics import pixcorr, ssim, CLIPMetric


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="./mindeyev2_cache")
    ap.add_argument("--tensor-cache", type=str, default="all_subjects_tensors.pt")
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--ll-size", type=int, default=64)
    ap.add_argument("--strengths", type=float, nargs="+", default=[1.0, 0.7, 0.6, 0.5])
    ap.add_argument("--arms", type=str, nargs="+", default=["rgb", "latent", "mean"],
                    choices=["rgb", "latent", "mean"])
    ap.add_argument("--mindeye-src", type=str, default="third_party/MindEyeV2/src")
    ap.add_argument("--ckpt-path", type=str,
                    default="third_party/unclip6_epoch0_step110000.ckpt")
    ap.add_argument("--out", type=str, default="outputs/step1c_smoke_latent")
    return ap.parse_args()


def main():
    args = parse_args()
    quiet_benign_warnings()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir); hf_cache = data_dir / "hf_cache"

    tensors = assert_tensor_cache_alignment(data_dir / args.tensor_cache, torch.load(data_dir / args.tensor_cache, map_location="cpu"))
    imgs = tensors[f"imgs_test_{args.subject}"][:args.n]
    gt = (imgs.float() / 255.0).clamp(0, 1) if imgs.dtype == torch.uint8 else imgs.float().clamp(0, 1)

    print("▶ encoding GT bigG (cls + tokens)…", flush=True)
    emb = _load_embedder(device, args.mindeye_src)
    cls, tokens = _encode_dual(emb, imgs, device)
    cls, tokens = cls.float(), tokens.float()
    del emb
    torch.cuda.empty_cache()

    print("▶ loading SDXL-unCLIP decoder…", flush=True)
    decoder = SDXLUnCLIPDecoder(device, args.mindeye_src, args.ckpt_path)
    clip_metric = CLIPMetric(device, hf_cache=hf_cache)

    blurry = F.interpolate(gt, args.ll_size, mode="bilinear", align_corners=False).clamp(0, 1)
    lat = decoder.encode_blurry(gt, ll_size=args.ll_size).float().cpu()
    lat_mean = lat.mean(dim=(0, 2, 3)).view(1, -1, 1, 1).expand_as(lat).contiguous()
    print(f"  latent {tuple(lat.shape)} | per-channel mean={lat.mean(dim=(0,2,3)).tolist()}")
    print(f"  per-channel std={lat.std(dim=(0,2,3)).tolist()}")

    def score(recon):
        g = gt
        if g.shape[-1] != recon.shape[-1]:
            g = F.interpolate(g, recon.shape[-1], mode="bilinear", align_corners=False)
        m = {"PixCorr": pixcorr(recon, g), "SSIM": ssim(recon, g)}
        m.update(clip_metric.score(recon, g))
        return m

    arms = {
        "rgb":    lambda s: decoder.decode(tokens, cls, init_image=blurry, strength=s),
        "latent": lambda s: decoder.decode(tokens, cls, init_latent=lat, strength=s),
        "mean":   lambda s: decoder.decode(tokens, cls, init_latent=lat_mean, strength=s),
    }

    rows = []
    print(f"\n  arm      strength | PixCorr   SSIM   CLIP_2way")
    for arm in args.arms:
        for s in args.strengths:
            recon = arms[arm](s)
            m = score(recon)
            rows.append({"arm": arm, "strength": s, **m})
            tag = "  (token-only; init ignored)" if s >= 1.0 else ""
            print(f"  {arm:8s} {s:5.2f}    | {m['PixCorr']:.3f}    {m['SSIM']:.3f}   "
                  f"{m['CLIP_2way']:.3f}{tag}", flush=True)
            try:
                from torchvision.utils import save_image
                k = min(8, recon.shape[0])
                save_image(torch.cat([gt[:k], recon[:k]]),
                           str(out_dir / f"grid_{arm}_s{s:.2f}.png"), nrow=k)
            except Exception:
                pass

    (out_dir / "smoke_latent.json").write_text(json.dumps(rows, indent=2))

    def best(arm):
        cand = [r for r in rows if r["arm"] == arm and r["strength"] < 1.0]
        return max(cand, key=lambda r: r["PixCorr"]) if cand else None

    b_lat, b_rgb, b_mean = best("latent"), best("rgb"), best("mean")
    print("\n=== GATE ===")
    if b_lat and b_rgb:
        d = abs(b_lat["PixCorr"] - b_rgb["PixCorr"])
        verdict = "PASS" if d < 0.05 else "FAIL"
        print(f"  [{verdict}] latent vs rgb path: PixCorr {b_lat['PixCorr']:.3f} vs "
              f"{b_rgb['PixCorr']:.3f} (|delta|={d:.3f}; the two encode the same blur, "
              f"so they must agree)")
    if b_lat and b_mean:
        gap = b_lat["PixCorr"] - b_mean["PixCorr"]
        print(f"  informative headroom: PixCorr {b_lat['PixCorr']:.3f} (true blur) - "
              f"{b_mean['PixCorr']:.3f} (mean latent) = {gap:.3f}")
        if gap < 0.10:
            print("  [STOP] a collapsed head already captures most of the img2img gain. "
                  "Training this head cannot honestly buy much PixCorr.")
        else:
            print(f"  [GO] {gap:.3f} PixCorr is genuinely attributable to predicting the "
                  f"layout. Train with --ll-target latent --ll-strength "
                  f"{b_lat['strength']:.2f}.")


if __name__ == "__main__":
    main()
