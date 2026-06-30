"""Sampling-calibration sweep: find the best (cond_source, cfg_scale) per metric.

The flow trades pixel-alignment for semantics as guidance rises: high cfg / pure
prior sharpens CLIP but collapses PixCorr, while regression / blend / low cfg
stays near the conditional mean (high PixCorr, softer CLIP). This script loads a
checkpoint + the decoder once and sweeps the decode configs on a fixed subset of
test images, so you can read off the PixCorr-optimal and CLIP-optimal settings
without retraining.

Run via slurm/sweep_step1b.sbatch (needs the A100 / decoder).
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import torch

from brainflow.step1.model_tokens import TokenStep1Config, TokenStep1Model
from brainflow.step1.targets import TargetStats
from brainflow.step1.targets_bigg import build_or_load_bigg_targets
from brainflow.step1.data import build_step1_loaders
from brainflow.step1.decoder_sgm import SDXLUnCLIPDecoder
from brainflow.step1.metrics import pixcorr, ssim, CLIPMetric


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data-dir", type=str, default="./mindeyev2_cache")
    ap.add_argument("--tensor-cache", type=str, default="all_subjects_tensors.pt")
    ap.add_argument("--target-dir", type=str, default="./mindeyev2_cache")
    ap.add_argument("--mindeye-src", type=str, default="third_party/MindEyeV2/src")
    ap.add_argument("--ckpt-path", type=str, default="third_party/unclip6_epoch0_step110000.ckpt")
    ap.add_argument("--n-images", type=int, default=200)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--solver", type=str, default="heun", choices=["euler", "heun"])
    ap.add_argument("--cond-sources", type=str, nargs="+",
                    default=["regression", "prior"])
    ap.add_argument("--cfg-scales", type=float, nargs="+", default=[1.0, 3.0])
    ap.add_argument("--ll-strengths", type=float, nargs="+", default=[1.0, 0.7, 0.6, 0.5],
                    help="img2img strengths (1.0 = token-only, no low-level init)")
    ap.add_argument("--out", type=str, default="outputs/step1c_sphere/sweep")
    return ap.parse_args()


def _load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg: TokenStep1Config = ckpt["cfg"]
    stats = TargetStats.from_dict(ckpt["stats"])
    model = TokenStep1Model(cfg, ckpt["voxels"])
    model.load_state_dict(ckpt["model"])
    if "ema" in ckpt:                                   # prefer EMA weights
        sd = model.state_dict()
        for k, v in ckpt["ema"].items():
            sd[k].copy_(v)
        model.load_state_dict(sd)
    return model.to(device).eval(), cfg, stats, ckpt["subjects"], ckpt["voxels"]


@torch.no_grad()
def _collect(loader, n):
    """A fixed subset of test batches (same images for every config = fair)."""
    out, tot = [], 0
    for b in loader:
        out.append((b["fmri"], int(b["subject"]), b["image"]))
        tot += b["fmri"].shape[0]
        if tot >= n:
            break
    return out


@torch.no_grad()
def _eval_config(model, subset, stats, decoder, clip_metric, device, *,
                 cond_source, cfg_scale, strength, steps, solver, n):
    preds, gts = [], []
    for fmri, subj, imgs in subset:
        f = fmri.to(device)
        tok, cls_hat = model.predict_tokens(
            f, subj, stats, cond_source=cond_source, cfg_scale=cfg_scale,
            n_steps=steps, solver=solver)
        blur = model.predict_lowlevel(f, subj)
        preds.append(decoder.decode(tok, cls_hat, init_image=blur, strength=strength))
        gts.append(imgs)
    pred = torch.cat(preds)[:n]
    gt = torch.cat(gts)[:n]
    m = {"PixCorr": pixcorr(pred, gt), "SSIM": ssim(pred, gt)}
    m.update(clip_metric.score(pred, gt))
    return m


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir); hf_cache = data_dir / "hf_cache"

    model, cfg, stats, subjects, voxels = _load_model(args.ckpt, device)
    tensors = torch.load(data_dir / args.tensor_cache, map_location="cpu")
    targets = build_or_load_bigg_targets(
        tensors, subjects, args.target_dir, device, args.mindeye_src, hf_cache=hf_cache)
    bundle = build_step1_loaders(tensors, targets, subjects, batch_size=32, num_workers=8)
    subset = _collect(bundle.eval, args.n_images)

    decoder = SDXLUnCLIPDecoder(device, args.mindeye_src, args.ckpt_path)
    clip_metric = CLIPMetric(device, hf_cache=hf_cache)

    # Grid: cond_source x cfg_scale x img2img strength (regression ignores cfg).
    configs = []
    for cond in args.cond_sources:
        scales = [1.0] if cond == "regression" else args.cfg_scales
        for cfgs in scales:
            for st in args.ll_strengths:
                configs.append((cond, cfgs, st))
    print(f"  sweeping {len(configs)} configs x {args.n_images} images", flush=True)

    rows = []
    for cond, cfgs, st in configs:
        m = _eval_config(model, subset, stats, decoder, clip_metric, device,
                         cond_source=cond, cfg_scale=cfgs, strength=st,
                         steps=args.steps, solver=args.solver, n=args.n_images)
        row = {"cond_source": cond, "cfg_scale": cfgs, "strength": st, **m}
        rows.append(row)
        print(f"  {cond:11s} cfg={cfgs:<4} str={st:<4} | PixCorr={m['PixCorr']:.3f} "
              f"SSIM={m['SSIM']:.3f} CLIP_2way={m['CLIP_2way']:.3f}", flush=True)

    best_pix = max(rows, key=lambda r: r["PixCorr"])
    best_clip = max(rows, key=lambda r: r["CLIP_2way"])
    summary = {"ckpt": args.ckpt, "n_images": args.n_images, "rows": rows,
               "best_PixCorr": best_pix, "best_CLIP_2way": best_clip}
    (out_dir / "sweep.json").write_text(json.dumps(summary, indent=2))

    print("\n=== BEST CONFIGS ===")
    print(f"  PixCorr : {best_pix['cond_source']} cfg={best_pix['cfg_scale']} "
          f"str={best_pix['strength']} -> PixCorr={best_pix['PixCorr']:.3f} "
          f"SSIM={best_pix['SSIM']:.3f} CLIP_2way={best_pix['CLIP_2way']:.3f}")
    print(f"  CLIP_2way: {best_clip['cond_source']} cfg={best_clip['cfg_scale']} "
          f"str={best_clip['strength']} -> CLIP_2way={best_clip['CLIP_2way']:.3f} "
          f"PixCorr={best_clip['PixCorr']:.3f}")
    print(f"\n  full table -> {out_dir/'sweep.json'}")


if __name__ == "__main__":
    main()
