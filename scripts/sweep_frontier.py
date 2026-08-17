from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from brainflow.tensor_cache import assert_tensor_cache_alignment
from brainflow.step1.model_tokens import TokenStep1Config, TokenStep1Model
from brainflow.step1.targets import TargetStats
from brainflow.step1.targets_bigg import build_or_load_bigg_targets
from brainflow.step1.data import build_step1_loaders
from brainflow.step1.decoder_sgm import SDXLUnCLIPDecoder, quiet_benign_warnings
from brainflow.step1.frontier import (decode_frontier, latent_frontier, parse_ks,
                                      pareto_front, predicted_frontier,
                                      predicted_resultant, format_surface)
from brainflow.metrics_full import format_comparison


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data-dir", type=str, default="./mindeyev2_cache")
    ap.add_argument("--tensor-cache", type=str, default="all_subjects_tensors.pt")
    ap.add_argument("--target-dir", type=str, default="./mindeyev2_cache")
    ap.add_argument("--mindeye-src", type=str, default="third_party/MindEyeV2/src")
    ap.add_argument("--ckpt-path", type=str,
                    default="third_party/unclip6_epoch0_step110000.ckpt")
    ap.add_argument("--ks", type=str, nargs="+", default=["0", "1", "2", "4", "8", "16"])
    ap.add_argument("--strengths", type=float, nargs="+", default=[1.0, 0.8, 0.7, 0.6, 0.5])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--solver", type=str, default="heun", choices=["euler", "heun"])
    ap.add_argument("--decode-steps", type=int, default=38)
    ap.add_argument("--cfg-scale", type=float, default=None)
    ap.add_argument("--cls-cfg-scale", type=float, default=None)
    ap.add_argument("--cls-vector-slot", type=int, nargs=2, default=[0, 1280])
    ap.add_argument("--max-images", type=int, default=200)
    ap.add_argument("--eval-subjects", type=int, nargs="+", default=None)
    ap.add_argument("--latent-only", action="store_true")
    ap.add_argument("--latent-batches", type=int, default=None)
    ap.add_argument("--no-full-metrics", dest="full_metrics", action="store_false",
                    default=True)
    ap.add_argument("--retrieval-pool", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out", type=str, default="outputs/frontier")
    return ap.parse_args()


def load_model(path, device, args):
    ckpt = torch.load(path, map_location="cpu")
    cfg: TokenStep1Config = ckpt["cfg"]
    cfg.n_steps = args.steps
    cfg.solver = args.solver
    if args.cfg_scale is not None:
        cfg.cfg_scale = args.cfg_scale
    if args.cls_cfg_scale is not None:
        cfg.cls_cfg_scale = args.cls_cfg_scale
    model = TokenStep1Model(cfg, ckpt["voxels"])
    res = model.load_state_dict(ckpt["model"], strict=False)
    if any("low_head" in k for k in res.missing_keys):
        model.low_head = None
    if "ema" in ckpt:
        sd = model.state_dict()
        for k, v in ckpt["ema"].items():
            if k in sd and sd[k].shape == v.shape:
                sd[k].copy_(v)
        model.load_state_dict(sd, strict=False)
    return model.to(device).eval(), cfg, ckpt


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    quiet_benign_warnings()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    hf_cache = data_dir / "hf_cache"
    ks = parse_ks(args.ks)

    model, cfg, ckpt = load_model(args.ckpt, device, args)
    stats = TargetStats.from_dict(ckpt["stats"])
    subjects = ckpt["subjects"]
    want = args.eval_subjects or [subjects[0]]

    tensors = assert_tensor_cache_alignment(data_dir / args.tensor_cache, torch.load(data_dir / args.tensor_cache, map_location="cpu"))
    targets = build_or_load_bigg_targets(tensors, subjects, args.target_dir, device,
                                         args.mindeye_src, hf_cache=hf_cache)
    bundle = build_step1_loaders(tensors, targets, subjects,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers)

    has_low = model.low_head is not None
    strengths = args.strengths if has_low else [1.0]
    print(f"ckpt={args.ckpt}")
    print(f"subjects={subjects} scoring={want} ks={ks} strengths={strengths}")
    print(f"flow_source={cfg.flow_source} flow_param={cfg.resolved_flow_param()} "
          f"cfg_scale={cfg.cfg_scale} low_head={'yes' if has_low else 'no'}")

    jsonl = out_dir / "frontier.jsonl"
    started = time.time()

    lat_points = latent_frontier(model, bundle.eval, stats, device, ks=ks,
                                 n_steps=args.steps, solver=args.solver,
                                 max_batches=args.latent_batches,
                                 retrieval_pool=args.retrieval_pool)
    with jsonl.open("a") as fh:
        for p in lat_points:
            fh.write(json.dumps({"stage": "latent", **p.flat()}) + "\n")

    print("\nlatent frontier (no diffusion pass)")
    print(f"{'K':>7} {'token_cos':>10} {'2way':>8} {'retr_fwd':>9} {'resultant':>10} "
          f"{'disp':>7} {'eff_rank':>9}")
    anchor_cos = None
    for p in lat_points:
        lab = "anchor" if p.k == 0 else str(p.k)
        if p.k == 0:
            anchor_cos = p.latent["token_cos"]
        print(f"{lab:>7} {p.latent['token_cos']:10.4f} {p.latent['two_way']:8.4f} "
              f"{p.latent['retr_fwd']:9.4f} {p.latent['resultant']:10.4f} "
              f"{p.latent['dispersion']:7.3f} {p.latent['eff_rank']:9.0f}")

    if anchor_cos is not None:
        pred = predicted_frontier(anchor_cos, [k for k in ks if k >= 1])
        print(f"\npredicted vs measured (Prop. 3, ||m||={anchor_cos:.4f})")
        print(f"{'K':>7} {'cos_pred':>9} {'cos_meas':>9} {'d_cos':>8} "
              f"{'res_pred':>9} {'res_meas':>9} {'d_res':>8}")
        for p in lat_points:
            if p.k >= 1 and p.k in pred:
                rp = predicted_resultant(anchor_cos, p.k)
                rm = p.latent["resultant"]
                print(f"{p.k:>7} {pred[p.k]:9.4f} {p.latent['token_cos']:9.4f} "
                      f"{p.latent['token_cos'] - pred[p.k]:+8.4f} "
                      f"{rp:9.4f} {rm:9.4f} {rm - rp:+8.4f}")
        print("a large positive d_cos means the transport is better calibrated than "
              "the independence assumption behind Prop. 3;")
        print("a large negative d_cos means the K draws are correlated, i.e. the "
              "posterior is narrower than it reports.")

    if args.latent_only:
        summary = {"ckpt": args.ckpt, "ks": ks, "latent": [p.flat() for p in lat_points],
                   "anchor_cos": anchor_cos, "seconds": time.time() - started}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {jsonl} and {out_dir/'summary.json'}")
        return

    decoder = SDXLUnCLIPDecoder(device, args.mindeye_src, args.ckpt_path,
                                num_steps=args.decode_steps,
                                cls_vector_slot=tuple(args.cls_vector_slot))

    n_cfg = len(ks) * len(strengths) * len(want)
    print(f"\ndecoding {n_cfg} configurations at <= {args.max_images} images each")

    def emit(pt):
        with jsonl.open("a") as fh:
            fh.write(json.dumps({"stage": "decode", **pt.flat()}) + "\n")
        lab = "anchor" if pt.k == 0 else f"K={pt.k}"
        print(f"  s{pt.subject:02d} {lab:>8} strength={pt.strength:.2f} "
              f"PixCorr={pt.image.get('PixCorr', float('nan')):.4f} "
              f"SSIM={pt.image.get('SSIM', float('nan')):.4f} "
              f"CLIP2way={pt.image.get('CLIP_2way', float('nan')):.4f} "
              f"Incep2way={pt.image.get('Inception_2way', float('nan')):.4f} "
              f"retr={pt.image.get('retrieval_fwd', float('nan')):.3f}", flush=True)

    dec_points = decode_frontier(model, bundle.eval, stats, decoder, device,
                                 ks=ks, strengths=strengths, n_steps=args.steps,
                                 solver=args.solver, max_images=args.max_images,
                                 subjects=want, full_metrics=args.full_metrics,
                                 retrieval_pool=args.retrieval_pool, on_point=emit)

    print("\nsurface")
    print(format_surface(dec_points))

    front = pareto_front(dec_points)
    print("\npareto front (PixCorr vs CLIP_2way)")
    for r in front:
        lab = "anchor" if r["k"] == 0 else f"K={r['k']}"
        print(f"  {lab:>8} strength={r['strength']:.2f} "
              f"PixCorr={r['image/PixCorr']:.4f} CLIP_2way={r['image/CLIP_2way']:.4f}")

    best = {}
    for key in ("image/PixCorr", "image/SSIM", "image/CLIP_2way",
                "image/Inception_2way", "image/retrieval_fwd"):
        rows = [r for r in (p.flat() for p in dec_points) if key in r]
        if rows:
            best[key] = max(rows, key=lambda r: r[key])

    summary = {
        "ckpt": args.ckpt,
        "config": {"ks": ks, "strengths": strengths, "steps": args.steps,
                   "solver": args.solver, "decode_steps": args.decode_steps,
                   "cfg_scale": cfg.cfg_scale, "flow_source": cfg.flow_source,
                   "flow_param": cfg.resolved_flow_param(),
                   "max_images": args.max_images, "subjects": want},
        "anchor_cos": anchor_cos,
        "latent": [p.flat() for p in lat_points],
        "decode": [p.flat() for p in dec_points],
        "pareto": front,
        "best": best,
        "seconds": time.time() - started,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if args.full_metrics and best.get("image/CLIP_2way"):
        top = best["image/CLIP_2way"]
        mean = {k.split("/", 1)[1]: v for k, v in top.items() if k.startswith("image/")}
        print("\nbest CLIP_2way configuration vs the published NSD table")
        print(format_comparison(mean))

    print(f"\nwrote {jsonl}")
    print(f"wrote {out_dir/'summary.json'}")
    print(f"elapsed {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
