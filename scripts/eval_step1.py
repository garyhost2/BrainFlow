from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from brainflow.tensor_cache import assert_tensor_cache_alignment
from brainflow.step1.model import Step1Config, Step1Model
from brainflow.step1.targets import TargetStats
from brainflow.step1.data import build_step1_loaders
from brainflow.step1.decoder import UnCLIPDecoder
from brainflow.step1.metrics import pixcorr, ssim, CLIPMetric

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data-dir", type=str, default="./mindeyev2_cache")
    ap.add_argument("--tensor-cache", type=str, default="all_subjects_tensors.pt")
    ap.add_argument("--target-cache", type=str, default="step1_targets_vith.pt")
    ap.add_argument("--cond-source", type=str, default="prior",
                    choices=["regression", "prior", "blend"])
    ap.add_argument("--cfg-scale", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=50, help="prior ODE steps")
    ap.add_argument("--solver", type=str, default="heun", choices=["euler", "heun"])
    ap.add_argument("--decode-steps", type=int, default=25, help="unCLIP diffusion steps")
    ap.add_argument("--guidance", type=float, default=10.0)
    ap.add_argument("--use-ema", action="store_true", default=True)
    ap.add_argument("--max-images", type=int, default=10_000)
    ap.add_argument("--out", type=str, default="outputs/step1/eval")
    return ap.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir); hf_cache = data_dir / "hf_cache"

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg: Step1Config = ckpt["cfg"]
    cfg.cond_source = args.cond_source
    cfg.cfg_scale = args.cfg_scale
    cfg.n_steps = args.steps
    cfg.solver = args.solver
    stats = TargetStats.from_dict(ckpt["stats"])
    subjects = ckpt["subjects"]
    voxels = ckpt["voxels"]

    model = Step1Model(cfg, voxels)
    model.load_state_dict(ckpt["model"])
    if args.use_ema and "ema" in ckpt:
        sd = model.state_dict()
        for k, v in ckpt["ema"].items():
            sd[k].copy_(v)
        model.load_state_dict(sd)
    model = model.to(device).eval()

    tensors = assert_tensor_cache_alignment(data_dir / args.tensor_cache, torch.load(data_dir / args.tensor_cache, map_location="cpu"))
    targets = torch.load(data_dir / args.target_cache, map_location="cpu")
    bundle = build_step1_loaders(tensors, targets, subjects, batch_size=64, num_workers=8)

    decoder = UnCLIPDecoder(device, hf_cache=hf_cache)
    clip_metric = CLIPMetric(device, hf_cache=hf_cache)

    preds, gts = [], []
    with torch.no_grad():
        for batch in tqdm(bundle.eval, desc="eval"):
            fmri = batch["fmri"].to(device, non_blocking=True)
            emb = model.predict_embedding(fmri, batch["subject"], stats)
            imgs = decoder.decode(emb, num_inference_steps=args.decode_steps,
                                  guidance_scale=args.guidance)
            preds.append(imgs); gts.append(batch["image"])
            if sum(p.shape[0] for p in preds) >= args.max_images:
                break
    pred = torch.cat(preds); gt = torch.cat(gts)

    metrics = {"PixCorr": pixcorr(pred, gt), "SSIM": ssim(pred, gt), "n": pred.shape[0]}
    metrics.update(clip_metric.score(pred, gt))
    print(json.dumps(metrics, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _save_grid(pred, gt, out_dir / "recon_grid.png")

def _save_grid(pred, gt, path, n=16):
    try:
        from torchvision.utils import save_image
        n = min(n, pred.shape[0])
        save_image(torch.cat([gt[:n], pred[:n]]), str(path), nrow=n)
        print(f"✓ grid -> {path}")
    except Exception as e:
        print(f"[grid skipped] {e}")

if __name__ == "__main__":
    main()
