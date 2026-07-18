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
from brainflow.step1.decoder_sgm import quiet_benign_warnings

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
    ap.add_argument("--lambda-clip", type=float, default=0.033,
                    help="SoftCLIP contrastive weight (0 disables)")
    ap.add_argument("--clip-temp", type=float, default=0.006)
    ap.add_argument("--geometry", type=str, default="sphere",
                    choices=["euclidean", "sphere"])
    ap.add_argument("--two-head", dest="two_head", action="store_true", default=True,
                    help="structured CLS + patch factorisation (paper default)")
    ap.add_argument("--no-two-head", dest="two_head", action="store_false",
                    help="patch-only ablation (no global CLS head)")
    ap.add_argument("--lambda-rcfm", type=float, default=1.0,
                    help="global CLS flow-matching weight")
    ap.add_argument("--cls-cfg-scale", type=float, default=3.0)
    ap.add_argument("--solver", type=str, default="heun", choices=["euler", "heun"])
    ap.add_argument("--uniform-t-prob", type=float, default=0.1)
    ap.add_argument("--lambda-radius", type=float, default=1.0)
    ap.add_argument("--lambda-low", type=float, default=1.0,
                    help="low-level (blurry-image) loss weight; 0 disables the pathway")
    ap.add_argument("--ll-strength", type=float, default=0.7,
                    help="img2img strength at decode (lower preserves more layout)")
    ap.add_argument("--ll-size", type=int, default=64,
                    help="blurry-image resolution the low head predicts (lower = more learnable)")
    ap.add_argument("--ll-loss", type=str, default="l1", choices=["l1", "huber", "mse"],
                    help="low-level loss; l1/huber are far less mean-seeking than mse")
    ap.add_argument("--no-low-level", dest="low_level", action="store_false", default=True)
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="train fraction held out for leakage-free selection (0 disables)")
    ap.add_argument("--weight-decay", type=float, default=0.02)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--fmri-noise-std", type=float, default=0.05)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--init-from", type=str, default=None,
                    help="warm-start model weights from a checkpoint (strict=False; "
                         "e.g. add the low-level head to an already-trained token model)")
    ap.add_argument("--freeze-token", action="store_true",
                    help="freeze the whole token model and train ONLY the low-level "
                         "head; select best.pt on PixCorr+SSIM (not val_cos). Use with "
                         "--init-from to add a low-level head without drifting the prior.")
    ap.add_argument("--eval-freq", type=int, default=5)
    ap.add_argument("--decode-eval", action="store_true")
    ap.add_argument("--decode-n", type=int, default=16)
    ap.add_argument("--cls-vector-slot", type=int, nargs=2, default=[0, 1280],
                    help="[lo hi] positions in the unCLIP 'vector' slot to fill with c_cls")
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
        pred, _ = model.predict_tokens(fmri, batch["subject"], stats, cond_source="regression")

        c = F.cosine_similarity(pred, tgt, dim=-1).mean(dim=-1)
        cos_sum += c.sum().item(); n += fmri.shape[0]
    return cos_sum / max(1, n)

def main():
    args = parse_args()
    setup_a100()
    quiet_benign_warnings()
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
                                 num_workers=args.num_workers, fmri_noise_std=args.fmri_noise_std,
                                 val_frac=args.val_frac, seed=args.seed)

    cfg = TokenStep1Config(subjects=args.subjects, enc_hidden=args.enc_hidden,
                           lambda_clip=args.lambda_clip, clip_temp=args.clip_temp,
                           geometry=args.geometry, uniform_t_prob=args.uniform_t_prob,
                           lambda_radius=args.lambda_radius, two_head=args.two_head,
                           lambda_rcfm=args.lambda_rcfm, cls_cfg_scale=args.cls_cfg_scale,
                           solver=args.solver, low_level=args.low_level,
                           lambda_low=args.lambda_low, ll_strength=args.ll_strength,
                           ll_size=args.ll_size, ll_loss=args.ll_loss)
    model = TokenStep1Model(cfg, bundle.voxels).to(device)
    if cfg.geometry == "sphere" and cfg.center_tokens:
        model.set_target_mean(stats.mean)
    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu")
        res = model.load_state_dict(ck["model"], strict=False)
        print(f"✓ warm-start from {args.init_from}: {len(res.missing_keys)} fresh params "
              f"(e.g. low_head), {len(res.unexpected_keys)} unexpected")

    low_only = bool(args.freeze_token)
    if args.freeze_token:
        if model.low_head is None:
            raise SystemExit("--freeze-token needs the low-level head (drop --no-low-level)")
        n_frozen = n_train = 0
        for name, p in model.named_parameters():
            train_it = name.startswith("low_head")
            p.requires_grad_(train_it)
            n_train += p.numel() if train_it else 0
            n_frozen += 0 if train_it else p.numel()
        print(f"✓ freeze-token: {n_frozen/1e6:.1f}M frozen, {n_train/1e6:.2f}M trainable "
              f"(low_head only); best.pt selected on PixCorr+SSIM")
    if args.compile:
        model = torch.compile(model)
    ema = EMA(model, decay=args.ema_decay)
    opt_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.95), fused=True)

    spe = len(bundle.train_sampler)
    total_steps = spe * args.epochs
    warmup_steps = spe * args.warmup_epochs

    decoder = clip_metric = None
    if args.decode_eval:
        from brainflow.step1.decoder_sgm import SDXLUnCLIPDecoder
        decoder = SDXLUnCLIPDecoder(device, args.mindeye_src, args.ckpt_path,
                                    cls_vector_slot=tuple(args.cls_vector_slot))
        clip_metric = CLIPMetric(device, hf_cache=hf_cache)

    best_cos = -1.0
    best_clip = -1.0
    best_img = -1.0
    step = 0
    nan_skips = 0
    for epoch in range(1, args.epochs + 1):
        # low_only: run the frozen backbone in eval mode (stable features; low_head
        # has no train/eval-sensitive layers) so the head learns a clean mapping.
        model.eval() if low_only else model.train()
        bundle.train_sampler.set_epoch(epoch)
        pbar = tqdm(bundle.train, desc=f"Ep{epoch}", mininterval=1.0)
        for batch in pbar:
            lr = lr_at(step, total_steps, warmup_steps, args.lr, args.min_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            fmri = batch["fmri"].to(device, non_blocking=True)
            tgt_raw = batch["emb"].to(device, non_blocking=True)
            tgt_std = stats.standardize(tgt_raw)
            tgt_cls = batch.get("cls")
            if tgt_cls is not None:
                tgt_cls = tgt_cls.to(device, non_blocking=True)
            tgt_img = batch["image"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                ld = model.training_step(fmri, batch["subject"], tgt_std,
                                         target_raw=tgt_raw, target_cls=tgt_cls,
                                         target_img=tgt_img, low_only=low_only)
                loss = ld["loss"]
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if torch.isfinite(gnorm):
                opt.step(); ema.update(model)
            else:
                nan_skips += 1
            step += 1
            if step % 50 == 0:
                pbar.set_postfix(flow=f"{float(ld.get('flow', 0)):.3f}",
                                 rcfm=f"{float(ld.get('rcfm', 0)):.3f}",
                                 reg=f"{float(ld.get('reg', 0)):.3f}",
                                 cos=f"{float(ld.get('cos', 0)):.3f}",
                                 low=f"{float(ld.get('low', 0)):.3f}",
                                 clip=f"{float(ld.get('clip', 0)):.3f}",
                                 lr=f"{lr:.1e}", skip=nan_skips)

        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            ema.store(model); ema.copy_to(model)
            # Honest selection: choose on a leakage-free split; TEST is report-only.
            sel_loader = bundle.val if bundle.val is not None else bundle.eval
            sel_name = "val" if bundle.val is not None else "test"
            if bundle.val is None and epoch == args.eval_freq:
                print("[warn] no validation split (--val-frac 0); selecting on TEST is leaky.")
            sel_cos = eval_token_cosine(model, sel_loader, stats, device)
            msg = f"Ep{epoch:4d} | {sel_name}_cos={sel_cos:.4f}"
            if bundle.val is not None:
                msg += f" test_cos={eval_token_cosine(model, bundle.eval, stats, device):.4f}"
            ema.restore(model)

            # Persist the checkpoint on the val-cos signal BEFORE any decoder
            # work. Decode-eval is heavy (SDXL UNet+VAE co-resident with the
            # training model + AdamW state) and only a progress signal; a crash
            # there must never throw away the trained weights -- it silently did
            # once (an OOM in the img2img VAE-encode killed a 10-epoch run before
            # a single checkpoint was written).
            ckpt = {"model": model.state_dict(), "ema": ema.shadow, "cfg": cfg,
                    "stats": stats.to_dict(), "voxels": bundle.voxels,
                    "subjects": args.subjects, "epoch": epoch}
            torch.save(ckpt, out_dir / "last.pt")
            # Default: select best.pt on token cosine. In --freeze-token mode the
            # token model is frozen (val_cos is constant), so best.pt is selected
            # on the low-level image metric (PixCorr+SSIM) after the decode below.
            if not low_only and sel_cos > best_cos:
                best_cos = sel_cos
                torch.save(ckpt, out_dir / "best.pt")

            # Best-effort image metrics: guard so an OOM/failure just skips this
            # epoch's decode instead of killing the run, and always restore the
            # training weights + free the cache afterwards.
            if args.decode_eval:
                im = None
                try:
                    ema.store(model); ema.copy_to(model)
                    n_subj = max(1, len(args.subjects))
                    per_subj = max(1, args.decode_n // n_subj)
                    preds, gts, got = [], [], {}
                    blur_sq, blur_n = 0.0, 0
                    # Honest selection: in --freeze-token mode best.pt is chosen on this
                    # metric, so decode the leakage-free VAL split (report-only TEST comes
                    # from the stage-2 full eval). Default token path decodes TEST purely
                    # as a progress readout (best.pt there is the val_cos checkpoint).
                    decode_loader = (bundle.val if (low_only and bundle.val is not None)
                                     else bundle.eval)
                    split_tag = "val" if decode_loader is bundle.val else "test"
                    with torch.no_grad():
                        for batch in decode_loader:
                            s = int(batch["subject"])
                            if got.get(s, 0) >= per_subj:
                                continue
                            take = min(batch["fmri"].shape[0], per_subj - got.get(s, 0))
                            fmri = batch["fmri"][:take].to(device, non_blocking=True)
                            tok, cls_hat = model.predict_tokens(fmri, batch["subject"], stats,
                                                                cond_source=cfg.cond_source)
                            blur = model.predict_lowlevel(fmri, batch["subject"])
                            if blur is not None:
                                # How good is the predicted blurry layout vs the GT-blurry
                                # target the head is trained on? (the img2img ceiling driver)
                                gtb = F.interpolate(batch["image"][:take].float().to(device),
                                                    cfg.ll_size, mode="bilinear",
                                                    align_corners=False).clamp(0, 1)
                                blur_sq += F.mse_loss(blur, gtb).item() * take
                                blur_n += take
                            preds.append(decoder.decode(tok, cls_hat, init_image=blur,
                                                        strength=cfg.ll_strength))
                            gts.append(batch["image"][:take])
                            got[s] = got.get(s, 0) + take
                            if len(got) == n_subj and all(v >= per_subj for v in got.values()):
                                break
                    pred = torch.cat(preds); gt = torch.cat(gts)
                    im = {"PixCorr": pixcorr(pred, gt), "SSIM": ssim(pred, gt)}
                    im.update(clip_metric.score(pred, gt))
                    msg += (f" | [{split_tag}] PixCorr={im['PixCorr']:.3f} SSIM={im['SSIM']:.3f} "
                            f"CLIP_cos={im['CLIP_cos']:.3f} CLIP_2way={im['CLIP_2way']:.3f}")
                    if blur_n:
                        msg += f" blur_mse={blur_sq / blur_n:.4f}"
                    _save_grid(pred, gt, out_dir / f"recon_ep{epoch}.png")
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    msg += f" | [decode-eval skipped: {type(e).__name__}: {e}]"
                finally:
                    ema.restore(model)
                    torch.cuda.empty_cache()
                # Legacy CLIP-2way checkpoint (no honest val split only), saved
                # AFTER the finally so ckpt's live tensors hold TRAINING weights
                # again -- not the EMA weights copy_to() swapped in for the decode.
                if bundle.val is None and im is not None and im["CLIP_2way"] > best_clip:
                    best_clip = im["CLIP_2way"]
                    torch.save(ckpt, out_dir / "best_clip2way.pt")
                # --freeze-token: token model is fixed, so best.pt IS the low-level
                # checkpoint -- select it on PixCorr+SSIM (the eval reloads EMA
                # weights, which is what this decode measured).
                if low_only and im is not None and (im["PixCorr"] + im["SSIM"]) > best_img:
                    best_img = im["PixCorr"] + im["SSIM"]
                    torch.save(ckpt, out_dir / "best.pt")
                    msg += "  ✓best.pt"

            if nan_skips:
                msg += f" | nan_skips={nan_skips}"
            print(msg)

    # Guarantee best.pt exists: in --freeze-token mode it is only written when an
    # image metric improves, so fall back to last.pt if no decode-eval ever ran.
    if not (out_dir / "best.pt").exists() and (out_dir / "last.pt").exists():
        import shutil
        shutil.copyfile(out_dir / "last.pt", out_dir / "best.pt")
        print("[best.pt] no selection metric fired; fell back to last.pt")
    if low_only:
        print(f"Done. best low-level PixCorr+SSIM={best_img:.4f}. Checkpoints in {out_dir}")
    else:
        print(f"Done. best {('val' if bundle.val is not None else 'test')}_cos={best_cos:.4f}. "
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
