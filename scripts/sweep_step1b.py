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
from brainflow.step1.decoder_sgm import SDXLUnCLIPDecoder, quiet_benign_warnings
from brainflow.step1.metrics import pixcorr, ssim, CLIPMetric
from brainflow.metrics_full import evaluate_full, retrieval_metrics, format_comparison


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
    ap.add_argument("--blend-ws", type=float, nargs="+", default=[0.5],
                    help="prior/regression mix for cond_source=blend (1.0 = pure prior). "
                         "This knob was hardcoded at 0.5 and never swept.")
    ap.add_argument("--no-full-metrics", dest="full_metrics", action="store_false",
                    default=True, help="skip the 8-metric NSD table per config")
    ap.add_argument("--no-retrieval", dest="retrieval", action="store_false", default=True)
    ap.add_argument("--retrieval-pool", type=int, default=300)
    ap.add_argument("--out", type=str, default="outputs/step1c_sphere/sweep")
    return ap.parse_args()


def _load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg: TokenStep1Config = ckpt["cfg"]
    stats = TargetStats.from_dict(ckpt["stats"])
    model = TokenStep1Model(cfg, ckpt["voxels"])
    # Non-strict: a prior-only checkpoint (e.g. step1c_sphere, trained before the
    # low head existed) has no low_head.* weights. Drop the untrained head so the
    # sweep decodes token-only instead of crashing or using a random blur.
    res = model.load_state_dict(ckpt["model"], strict=False)
    if any("low_head" in k for k in res.missing_keys):
        model.low_head = None
        print("  checkpoint has no trained low_head -> token-only decodes")
    elif res.missing_keys:
        print(f"  [warn] {len(res.missing_keys)} missing keys (e.g. {res.missing_keys[:2]})")
    if "ema" in ckpt:                                   # prefer EMA weights
        sd = model.state_dict()
        for k, v in ckpt["ema"].items():
            if k in sd:
                sd[k].copy_(v)
        model.load_state_dict(sd, strict=False)
    return model.to(device).eval(), cfg, stats, ckpt["subjects"], ckpt["voxels"]


@torch.no_grad()
def _collect(loader, n):
    """A fixed subset of test batches (same images for every config = fair)."""
    out, tot = [], 0
    for b in loader:
        out.append((b["fmri"], int(b["subject"]), b["image"], b["emb"]))
        tot += b["fmri"].shape[0]
        if tot >= n:
            break
    return out


@torch.no_grad()
def _eval_config(model, subset, stats, decoder, clip_metric, device, *,
                 cond_source, cfg_scale, strength, steps, solver, n,
                 blend_w=None, full_metrics=True, retrieval=True, pool=300):
    if blend_w is not None:
        model.cfg.blend_w = blend_w
    preds, gts, tok_p, tok_g = [], [], [], []
    for fmri, subj, imgs, emb in subset:
        f = fmri.to(device)
        tok, cls_hat = model.predict_tokens(
            f, subj, stats, cond_source=cond_source, cfg_scale=cfg_scale,
            n_steps=steps, solver=solver)
        blur = model.predict_lowlevel(f, subj)
        preds.append(decoder.decode(tok, cls_hat, init_image=blur, strength=strength))
        gts.append(imgs)
        if retrieval:
            tok_p.append(tok.detach().to("cpu", torch.float16))
            tok_g.append(emb.to("cpu", torch.float16))
    pred = torch.cat(preds)[:n]
    gt = torch.cat(gts)[:n]
    m = {"legacy_PixCorr": pixcorr(pred, gt), "legacy_SSIM": ssim(pred, gt)}
    m.update({f"legacy_{k}": v for k, v in clip_metric.score(pred, gt).items()})
    if full_metrics:
        m.update(evaluate_full(pred, gt, device))
    else:
        m["PixCorr"] = m["legacy_PixCorr"]; m["SSIM"] = m["legacy_SSIM"]
        m["CLIP_2way"] = m["legacy_CLIP_2way"]
    if retrieval and tok_p:
        m.update(retrieval_metrics(torch.cat(tok_p)[:n], torch.cat(tok_g)[:n],
                                   batch_size=pool, device=device))
    return m


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    quiet_benign_warnings()
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
    # No trained low head -> the blur is None, so every strength is token-only;
    # collapse to strength=1.0 to avoid 4x redundant decodes.
    strengths = args.ll_strengths if model.low_head is not None else [1.0]
    configs = []
    for cond in args.cond_sources:
        scales = [1.0] if cond == "regression" else args.cfg_scales
        # blend_w only means anything for cond_source=blend
        blends = args.blend_ws if cond == "blend" else [None]
        for cfgs in scales:
            for bw in blends:
                for st in strengths:
                    configs.append((cond, cfgs, bw, st))
    print(f"  sweeping {len(configs)} configs x {args.n_images} images", flush=True)

    rows = []
    for cond, cfgs, bw, st in configs:
        m = _eval_config(model, subset, stats, decoder, clip_metric, device,
                         cond_source=cond, cfg_scale=cfgs, strength=st,
                         steps=args.steps, solver=args.solver, n=args.n_images,
                         blend_w=bw, full_metrics=args.full_metrics,
                         retrieval=args.retrieval, pool=args.retrieval_pool)
        row = {"cond_source": cond, "cfg_scale": cfgs, "blend_w": bw,
               "strength": st, **m}
        rows.append(row)
        extra = ""
        if "retrieval_fwd" in m:
            extra = f" ret={m['retrieval_fwd']:.3f}/{m['retrieval_bwd']:.3f}"
        print(f"  {cond:11s} cfg={cfgs:<4} bw={str(bw):<5} str={st:<4} | "
              f"PixCorr={m['PixCorr']:.3f} SSIM={m['SSIM']:.3f} "
              f"CLIP_2way={m['CLIP_2way']:.3f}{extra}", flush=True)

    best_pix = max(rows, key=lambda r: r["PixCorr"])
    best_clip = max(rows, key=lambda r: r["CLIP_2way"])
    summary = {"ckpt": args.ckpt, "n_images": args.n_images, "rows": rows,
               "best_PixCorr": best_pix, "best_CLIP_2way": best_clip}
    (out_dir / "sweep.json").write_text(json.dumps(summary, indent=2))

    print("\n=== BEST CONFIGS ===")
    print(f"  PixCorr : {best_pix['cond_source']} cfg={best_pix['cfg_scale']} "
          f"bw={best_pix['blend_w']} str={best_pix['strength']} -> "
          f"PixCorr={best_pix['PixCorr']:.3f} "
          f"SSIM={best_pix['SSIM']:.3f} CLIP_2way={best_pix['CLIP_2way']:.3f}")
    print(f"  CLIP_2way: {best_clip['cond_source']} cfg={best_clip['cfg_scale']} "
          f"bw={best_clip['blend_w']} str={best_clip['strength']} -> "
          f"CLIP_2way={best_clip['CLIP_2way']:.3f} "
          f"PixCorr={best_clip['PixCorr']:.3f}")
    if args.full_metrics:
        print("\n=== vs the published NSD table (CLIP-optimal config) ===")
        print(format_comparison(best_clip))
    print(f"\n  full table -> {out_dir/'sweep.json'}")


if __name__ == "__main__":
    main()
