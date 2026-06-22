from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from brainflow.step1.model import Step1Config, Step1Model
from brainflow.step1.targets import build_or_load_targets, TargetStats
from brainflow.step1.data import build_step1_loaders
from brainflow.step1.metrics import EMA, pixcorr, ssim, CLIPMetric

def setup_a100():

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="./mindeyev2_cache")
    ap.add_argument("--tensor-cache", type=str, default="all_subjects_tensors.pt")
    ap.add_argument("--target-cache", type=str, default="step1_targets_vith.pt")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.02)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--fmri-noise-std", type=float, default=0.05)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--eval-freq", type=int, default=5)
    ap.add_argument("--decode-eval", action="store_true", help="decode images during eval (slow)")
    ap.add_argument("--decode-n", type=int, default=64, help="#images to decode for image metrics")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--out", type=str, default="outputs/step1")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()

def lr_at(step, total_steps, warmup_steps, base_lr, min_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * prog))

@torch.no_grad()
def eval_embedding_cosine(model, loader, stats, device):
    model.eval()
    cos_sum, n = 0.0, 0
    for batch in loader:
        fmri = batch["fmri"].to(device, non_blocking=True)
        emb_raw = batch["emb"].to(device, non_blocking=True)
        subj = batch["subject"]
        pred_raw = model.predict_embedding(fmri, subj, stats, cond_source="regression")
        cos_sum += F.cosine_similarity(pred_raw, emb_raw, dim=-1).sum().item()
        n += fmri.shape[0]
    return cos_sum / max(1, n)

@torch.no_grad()
def eval_images(model, loader, stats, device, decoder, clip_metric, n_max, cond_source):
    model.eval()
    preds, gts = [], []
    got = 0
    for batch in loader:
        fmri = batch["fmri"].to(device, non_blocking=True)
        pred_raw = model.predict_embedding(fmri, batch["subject"], stats, cond_source=cond_source)
        imgs = decoder.decode(pred_raw, num_inference_steps=20, guidance_scale=10.0)
        preds.append(imgs); gts.append(batch["image"])
        got += imgs.shape[0]
        if got >= n_max:
            break
    pred = torch.cat(preds)[:n_max]; gt = torch.cat(gts)[:n_max]
    out = {"PixCorr": pixcorr(pred, gt), "SSIM": ssim(pred, gt)}
    out.update(clip_metric.score(pred, gt))
    return out, pred, gt

def main():
    args = parse_args()
    setup_a100()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    hf_cache = data_dir / "hf_cache"

    tensors = torch.load(data_dir / args.tensor_cache, map_location="cpu")
    targets = build_or_load_targets(
        tensors, args.subjects, data_dir / args.target_cache, device, hf_cache=hf_cache)
    stats: TargetStats = targets["_stats"]
    bundle = build_step1_loaders(
        tensors, targets, args.subjects, args.batch_size,
        num_workers=args.num_workers, fmri_noise_std=args.fmri_noise_std)

    cfg = Step1Config(subjects=args.subjects)
    model = Step1Model(cfg, bundle.voxels).to(device)
    if args.compile:
        model = torch.compile(model)
    ema = EMA(model, decay=args.ema_decay)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.95), fused=True)

    steps_per_epoch = len(bundle.train_sampler)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    decoder = None
    clip_metric = None
    if args.decode_eval:
        from brainflow.step1.decoder import UnCLIPDecoder
        decoder = UnCLIPDecoder(device, hf_cache=hf_cache)
        clip_metric = CLIPMetric(device, hf_cache=hf_cache)

    best_cos = -1.0
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        bundle.train_sampler.set_epoch(epoch)
        pbar = tqdm(bundle.train, desc=f"Ep{epoch}", mininterval=1.0)
        for batch in pbar:
            lr = lr_at(step, total_steps, warmup_steps, args.lr, args.min_lr)
            for g in opt.param_groups:
                g["lr"] = lr

            fmri = batch["fmri"].to(device, non_blocking=True)
            emb_raw = batch["emb"].to(device, non_blocking=True)
            target_std = stats.standardize(emb_raw)
            subj = batch["subject"]

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                ld = model.training_step(fmri, subj, target_std)
                loss = ld["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            ema.update(model)
            step += 1
            if step % 50 == 0:
                pbar.set_postfix(flow=f"{ld['flow']:.3f}", reg=f"{ld['reg']:.3f}",
                                 cos=f"{ld['cos']:.3f}", lr=f"{lr:.1e}")

        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            ema.store(model); ema.copy_to(model)
            cos = eval_embedding_cosine(model, bundle.eval, stats, device)
            msg = f"Ep{epoch:4d} | embed_cos={cos:.4f}"
            if args.decode_eval:
                im, pred, gt = eval_images(model, bundle.eval, stats, device,
                                           decoder, clip_metric, args.decode_n,
                                           cond_source=cfg.cond_source)
                msg += f" | PixCorr={im['PixCorr']:.3f} SSIM={im['SSIM']:.3f} " \
                       f"CLIP_cos={im['CLIP_cos']:.3f} CLIP_2way={im['CLIP_2way']:.3f}"
                _save_grid(pred, gt, out_dir / f"recon_ep{epoch}.png")
            ema.restore(model)
            print(msg)

            ckpt = {"model": model.state_dict(), "ema": ema.shadow,
                    "cfg": cfg, "stats": stats.to_dict(), "voxels": bundle.voxels,
                    "subjects": args.subjects, "epoch": epoch}
            torch.save(ckpt, out_dir / "last.pt")
            if cos > best_cos:
                best_cos = cos
                torch.save(ckpt, out_dir / "best_cos.pt")

    print(f"Done. best embed_cos={best_cos:.4f}. Checkpoints in {out_dir}")

def _save_grid(pred, gt, path, n=8):
    try:
        from torchvision.utils import save_image
        n = min(n, pred.shape[0])
        grid = torch.cat([gt[:n], pred[:n]], dim=0)
        save_image(grid, str(path), nrow=n)
    except Exception as e:
        print(f"[grid save skipped] {e}")

if __name__ == "__main__":
    main()
