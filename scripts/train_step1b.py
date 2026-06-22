from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from brainflow.step1.model_tokens import TokenStep1Config, TokenStep1Model
from brainflow.step1.targets_bigg import build_or_load_bigg_targets
from brainflow.step1.targets import TargetStats
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
    ap.add_argument("--target-dir", type=str, default="./mindeyev2_cache",
                    help="dir for per-subject bigG target files (step1b_bigg_s{N}.pt)")
    ap.add_argument("--mindeye-src", type=str, default="third_party/MindEyeV2/src")
    ap.add_argument("--ckpt-path", type=str, default="third_party/unclip6_epoch0_step110000.ckpt")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--enc-hidden", type=int, default=2048,
                    help="backbone width; 4096 matches MindEye2 capacity")
    ap.add_argument("--lambda-clip", type=float, default=1.0,
                    help="SoftCLIP contrastive weight (0 disables)")
    ap.add_argument("--clip-temp", type=float, default=0.006)
    ap.add_argument("--weight-decay", type=float, default=0.02)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--fmri-noise-std", type=float, default=0.05)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--eval-freq", type=int, default=5)
    ap.add_argument("--decode-eval", action="store_true")
    ap.add_argument("--decode-n", type=int, default=16)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--out", type=str, default="outputs/step1b")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()

def lr_at(step, total, warm, base, mn):
    if step < warm:
        return base * (step + 1) / max(1, warm)
    prog = (step - warm) / max(1, total - warm)
    return mn + 0.5 * (base - mn) * (1 + math.cos(math.pi * prog))

@torch.no_grad()
def eval_token_cosine(model, loader, stats, device):
    model.eval()
    cos_sum, n = 0.0, 0
    for batch in loader:
        fmri = batch["fmri"].to(device, non_blocking=True)
        tgt = batch["emb"].to(device, non_blocking=True)
        pred = model.predict_tokens(fmri, batch["subject"], stats, cond_source="regression")

        c = F.cosine_similarity(pred, tgt, dim=-1).mean(dim=-1)
        cos_sum += c.sum().item(); n += fmri.shape[0]
    return cos_sum / max(1, n)

def main():
    args = parse_args()
    setup_a100()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir); hf_cache = data_dir / "hf_cache"

    tensors = torch.load(data_dir / args.tensor_cache, map_location="cpu")
    targets = build_or_load_bigg_targets(
        tensors, args.subjects, args.target_dir, device, args.mindeye_src,
        hf_cache=hf_cache)
    stats: TargetStats = targets["_stats"]
    bundle = build_step1_loaders(tensors, targets, args.subjects, args.batch_size,
                                 num_workers=args.num_workers, fmri_noise_std=args.fmri_noise_std)

    cfg = TokenStep1Config(subjects=args.subjects, enc_hidden=args.enc_hidden,
                           lambda_clip=args.lambda_clip, clip_temp=args.clip_temp)
    model = TokenStep1Model(cfg, bundle.voxels).to(device)
    if args.compile:
        model = torch.compile(model)
    ema = EMA(model, decay=args.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.95), fused=True)

    spe = len(bundle.train_sampler)
    total_steps = spe * args.epochs
    warmup_steps = spe * args.warmup_epochs

    decoder = clip_metric = None
    if args.decode_eval:
        from brainflow.step1.decoder_sgm import SDXLUnCLIPDecoder
        decoder = SDXLUnCLIPDecoder(device, args.mindeye_src, args.ckpt_path)
        clip_metric = CLIPMetric(device, hf_cache=hf_cache)

    best_cos = -1.0
    best_clip = -1.0
    step = 0
    nan_skips = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        bundle.train_sampler.set_epoch(epoch)
        pbar = tqdm(bundle.train, desc=f"Ep{epoch}", mininterval=1.0)
        for batch in pbar:
            lr = lr_at(step, total_steps, warmup_steps, args.lr, args.min_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            fmri = batch["fmri"].to(device, non_blocking=True)
            tgt_raw = batch["emb"].to(device, non_blocking=True)
            tgt_std = stats.standardize(tgt_raw)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                ld = model.training_step(fmri, batch["subject"], tgt_std)
                loss = ld["loss"]
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if torch.isfinite(gnorm):
                opt.step(); ema.update(model)
            else:
                nan_skips += 1
            step += 1
            if step % 50 == 0:
                pbar.set_postfix(flow=f"{ld['flow']:.3f}", reg=f"{ld['reg']:.3f}",
                                 cos=f"{ld['cos']:.3f}", clip=f"{ld['clip']:.3f}",
                                 lr=f"{lr:.1e}", skip=nan_skips)

        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            ema.store(model); ema.copy_to(model)
            cos = eval_token_cosine(model, bundle.eval, stats, device)
            msg = f"Ep{epoch:4d} | token_cos={cos:.4f}"
            if args.decode_eval:

                n_subj = max(1, len(args.subjects))
                per_subj = max(1, args.decode_n // n_subj)
                preds, gts, got = [], [], {}
                with torch.no_grad():
                    for batch in bundle.eval:
                        s = int(batch["subject"])
                        if got.get(s, 0) >= per_subj:
                            continue
                        take = min(batch["fmri"].shape[0], per_subj - got.get(s, 0))
                        fmri = batch["fmri"][:take].to(device, non_blocking=True)
                        tok = model.predict_tokens(fmri, batch["subject"], stats,
                                                   cond_source=cfg.cond_source)
                        preds.append(decoder.decode(tok)); gts.append(batch["image"][:take])
                        got[s] = got.get(s, 0) + take
                        if len(got) == n_subj and all(v >= per_subj for v in got.values()):
                            break
                pred = torch.cat(preds); gt = torch.cat(gts)
                im = {"PixCorr": pixcorr(pred, gt), "SSIM": ssim(pred, gt)}
                im.update(clip_metric.score(pred, gt))
                msg += (f" | PixCorr={im['PixCorr']:.3f} SSIM={im['SSIM']:.3f} "
                        f"CLIP_cos={im['CLIP_cos']:.3f} CLIP_2way={im['CLIP_2way']:.3f}")
                _save_grid(pred, gt, out_dir / f"recon_ep{epoch}.png")
            ema.restore(model)
            if nan_skips:
                msg += f" | nan_skips={nan_skips}"
            print(msg)

            ckpt = {"model": model.state_dict(), "ema": ema.shadow, "cfg": cfg,
                    "stats": stats.to_dict(), "voxels": bundle.voxels,
                    "subjects": args.subjects, "epoch": epoch}
            torch.save(ckpt, out_dir / "last.pt")

            if args.decode_eval and im["CLIP_2way"] > best_clip:
                best_clip = im["CLIP_2way"]
                torch.save(ckpt, out_dir / "best_clip2way.pt")
            if cos > best_cos:
                best_cos = cos
                torch.save(ckpt, out_dir / "best_cos.pt")

    print(f"Done. best token_cos={best_cos:.4f}, best CLIP_2way={best_clip:.4f}. "
          f"Checkpoints in {out_dir}")

def _save_grid(pred, gt, path, n=8):
    try:
        from torchvision.utils import save_image
        n = min(n, pred.shape[0])
        save_image(torch.cat([gt[:n], pred[:n]]), str(path), nrow=n)
    except Exception as e:
        print(f"[grid skipped] {e}")

if __name__ == "__main__":
    main()
