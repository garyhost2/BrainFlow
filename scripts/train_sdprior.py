"""Phase 3 SDPrior training: frozen BrainEncoder + ClipPrior → CLIP embedding.

Usage (single GPU):
    BRAINFLOW_CONFIG=configs/phase3_sdprior.yaml \\
    INIT_FROM=outputs/mc_xl_v100_32/best_pc_v5.pt \\
    python -m scripts.train_sdprior

Usage (torchrun):
    BRAINFLOW_CONFIG=configs/phase3_sdprior.yaml \\
    INIT_FROM=outputs/mc_xl_v100_32/best_pc_v5.pt \\
    torchrun --standalone --nproc_per_node=1 -m scripts.train_sdprior
"""
from __future__ import annotations
import gc
import math
import os
import random
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch.set_float32_matmul_precision("high")
if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "enable_flash_sdp"):
    torch.backends.cuda.enable_flash_sdp(True)

from brainflow.clip_prior import ClipPrior
from brainflow.config import load_config
from brainflow.data import build_dataloaders, is_dist, is_main, rank, world_size
from brainflow.ema import EMA
from brainflow.models import BrainEncoder

load_dotenv()


# ── DDP helpers ────────────────────────────────────────────────────────────────

def setup_ddp(backend_cfg: str, init_method: str) -> torch.device:
    if "LOCAL_RANK" not in os.environ:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl" if backend_cfg == "auto" else backend_cfg
        device = torch.device(f"cuda:{local_rank}")
    else:
        backend = "gloo" if backend_cfg == "auto" else backend_cfg
        device = torch.device("cpu")
    timeout_sec = int(os.environ.get("DDP_TIMEOUT", "1800"))
    dist.init_process_group(backend=backend, init_method=init_method,
                            timeout=timedelta(seconds=timeout_sec))
    return device


def cleanup_ddp():
    if is_dist():
        dist.destroy_process_group()


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── CLIP stats fitting ─────────────────────────────────────────────────────────

def fit_clip_stats(prior: ClipPrior, train_loader, device: torch.device,
                   max_samples: int = 20000):
    """Compute CLIP CLS mean/std from training set, broadcast across ranks."""
    if is_main():
        print("[SDPrior] Fitting CLIP embedding statistics …")
    clips: list[torch.Tensor] = []
    n = 0
    for batch in train_loader:
        if n >= max_samples:
            break
        c = batch.get("clip_emb")
        if c is None:
            raise RuntimeError("'clip_emb' key missing from batch. Check data pipeline.")
        clips.append(c.float())
        n += c.shape[0]
    all_clips = torch.cat(clips, dim=0)[:max_samples]  # (N, 768)

    if is_main():
        prior.fit_stats(all_clips)
        print(f"[SDPrior] Stats fitted on {len(all_clips)} samples. "
              f"mean={prior.clip_mean.norm():.3f} std={prior.clip_std.mean():.3f}")

    if is_dist():
        dist.broadcast(prior.clip_mean.to(device), src=0)
        dist.broadcast(prior.clip_std.to(device), src=0)
        flag = torch.tensor([1.0 if prior._stats_fitted else 0.0], device=device)
        dist.broadcast(flag, src=0)
        if not is_main() and flag.item() > 0:
            prior._stats_fitted = True
        dist.barrier()


# ── Eval ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    brain_enc: BrainEncoder,
    prior: ClipPrior,
    loader,
    device: torch.device,
    cfg,
    n_batches: int = 32,
) -> dict[str, float]:
    """Sample CLIP embeddings and compute cosine similarity vs GT."""
    brain_enc.eval()
    prior.eval()
    cos_vals: list[float] = []

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        fmri = batch["fmri"].to(device)
        subject = batch["subject"].to(device)
        clip_gt = batch["clip_emb"].to(device).float()  # (B, 768)

        tokens, _ = brain_enc(fmri, subject)         # (B, N, D)
        sampled = prior.sample(
            tokens,
            n_steps=cfg.eval_ode_steps,
            cfg_scale=getattr(cfg, "cfg_scale", 1.5),
            normalize_output=True,
        )                                             # (B, 768) L2-normalized

        clip_gt_n = F.normalize(clip_gt, dim=-1)
        cos = F.cosine_similarity(sampled, clip_gt_n, dim=-1).mean().item()
        cos_vals.append(cos)

    brain_enc.train()
    prior.train()
    return {"CLIP_Cos": float(np.mean(cos_vals)) if cos_vals else 0.0}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    set_seed(42)
    cfg = load_config()
    if cfg.training_stage != "sdprior":
        raise ValueError(
            f"train_sdprior.py expects training_stage='sdprior', got {cfg.training_stage!r}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    device = setup_ddp(cfg.backend, cfg.init_method)

    if is_main():
        print(f"[SDPrior] world_size={world_size()} device={device} "
              f"subjects={cfg.subjects}")

    # Data — only needs fmri + clip (no latents, no clip_patches)
    train_loader, test_loader, eval_loader, train_sampler, voxels = build_dataloaders(cfg)
    if is_main():
        print(f"Train batches/rank: {len(train_loader)} | Eval: {len(eval_loader)}")

    # ── BrainEncoder (frozen) ──────────────────────────────────────────────────
    brain_enc = BrainEncoder(cfg, voxels).to(device)

    init_from = os.environ.get("INIT_FROM", "").strip()
    if init_from and Path(init_from).is_file():
        ckpt = torch.load(init_from, map_location="cpu", mmap=True)
        # Strip torch.compile prefix if present
        ckpt = {k.replace("._orig_mod.", "."): v for k, v in ckpt.items()}
        # Accept both plain keys (brain_enc.*) and bare keys
        enc_ckpt = {}
        for k, v in ckpt.items():
            if k.startswith("brain_enc."):
                enc_ckpt[k[len("brain_enc."):]] = v
            else:
                enc_ckpt[k] = v
        missing, unexpected = brain_enc.load_state_dict(enc_ckpt, strict=False)
        if is_main():
            loaded = len(enc_ckpt) - len(unexpected)
            print(f"[SDPrior] BrainEncoder warm-started from {init_from} "
                  f"({loaded}/{len(enc_ckpt)} tensors loaded, "
                  f"missing={len(missing)}, unexpected={len(unexpected)})")
    elif is_main():
        print(f"[SDPrior] WARNING: INIT_FROM={init_from!r} not found — random BrainEncoder init.")

    # Freeze BrainEncoder completely
    for p in brain_enc.parameters():
        p.requires_grad_(False)
    brain_enc.eval()

    # ── ClipPrior (trained) ────────────────────────────────────────────────────
    prior = ClipPrior(
        clip_dim=cfg.clip_dim,          # 768
        ctx_dim=cfg.brain_dim,          # 768
        dim=getattr(cfg, "prior_dim", 512),
        depth=getattr(cfg, "prior_blocks", 6),
        heads=getattr(cfg, "attn_heads", 8),
        dropout=cfg.enc_drop,
        cfg_drop=cfg.cfg_drop_prob,
    ).to(device)

    # Fit CLIP statistics before DDP wrap
    fit_clip_stats(prior, train_loader, device)

    # ── Resume support ─────────────────────────────────────────────────────────
    resume_path = cfg.output_dir / "resume.pt"
    start_epoch = 1
    best_cos = 0.0
    step = 0

    if resume_path.is_file() and is_main():
        print(f"[SDPrior] Found resume.pt — restoring checkpoint …")

    if resume_path.is_file():
        resume = torch.load(resume_path, map_location="cpu")
        # Sanity check: reject corrupted (NaN) checkpoints
        prior_sd = resume["prior"]
        n_nan = sum(1 for v in prior_sd.values() if not torch.isfinite(v).all())
        if n_nan > 0:
            if is_main():
                print(f"[SDPrior] WARNING: resume.pt has {n_nan} NaN tensors — "
                      f"ignoring corrupt checkpoint, starting fresh.")
            resume_path.unlink(missing_ok=True)
        else:
            prior.load_state_dict(prior_sd)
            start_epoch = resume.get("epoch", 0) + 1
            best_cos = resume.get("best_cos", 0.0)
            step = resume.get("step", 0)
            if prior.clip_mean.norm() < 1e-6 and "clip_mean" in resume:
                prior.clip_mean.copy_(resume["clip_mean"])
                prior.clip_std.copy_(resume["clip_std"])
                prior._stats_fitted = True
            if is_main():
                print(f"[SDPrior] Resumed from epoch {start_epoch - 1}, "
                      f"best_cos={best_cos:.4f}")

    n_prior = sum(p.numel() for p in prior.parameters() if p.requires_grad)
    if is_main():
        print(f"[SDPrior] ClipPrior params: {n_prior / 1e6:.1f}M (trainable)")

    # ── DDP wrap (prior only — BrainEncoder is frozen eval) ───────────────────
    if is_dist():
        prior_ddp = DDP(prior,
                        device_ids=[device.index] if device.type == "cuda" else None,
                        find_unused_parameters=False, broadcast_buffers=True)
        raw_prior = prior_ddp.module
    else:
        prior_ddp = prior
        raw_prior = prior

    # ── Optimizer + scheduler ─────────────────────────────────────────────────
    _ok = dict(lr=float(cfg.lr), weight_decay=float(cfg.weight_decay),
               betas=(0.9, 0.999), eps=1e-8)
    try:
        optimizer = AdamW(
            [p for p in prior_ddp.parameters() if p.requires_grad],
            fused=True, **_ok)
    except (TypeError, RuntimeError) as _e:
        if "fused" not in str(_e).lower() and "unexpected keyword" not in str(_e).lower():
            raise
        optimizer = AdamW(
            [p for p in prior_ddp.parameters() if p.requires_grad], **_ok)

    if resume_path.is_file():
        resume = torch.load(resume_path, map_location="cpu")
        if "optimizer" in resume:
            optimizer.load_state_dict(resume["optimizer"])

    def lr_lambda(epoch: int) -> float:
        if epoch < cfg.warmup_epochs:
            return epoch / max(1, cfg.warmup_epochs)
        progress = (epoch - cfg.warmup_epochs) / max(1, cfg.num_epochs - cfg.warmup_epochs)
        return max(float(cfg.min_lr) / float(cfg.lr),
                   0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if resume_path.is_file():
        resume = torch.load(resume_path, map_location="cpu")
        if "scheduler" in resume:
            scheduler.load_state_dict(resume["scheduler"])

    # fp16 on V100 (compute cap < 8.0), bf16 on Ampere+
    use_bf16 = False
    if device.type == "cuda":
        cc = torch.cuda.get_device_capability(device)
        use_bf16 = cc[0] >= 8
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and not use_bf16))
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

    if resume_path.is_file():
        resume = torch.load(resume_path, map_location="cpu")
        if "scaler" in resume:
            scaler.load_state_dict(resume["scaler"])

    ema = EMA(raw_prior, decay=cfg.ema_decay)
    if resume_path.is_file():
        resume = torch.load(resume_path, map_location="cpu")
        if "ema" in resume:
            ema.shadow = {k: v.clone() for k, v in resume["ema"].items()}

    # ── WandB ─────────────────────────────────────────────────────────────────
    use_wandb = False
    if is_main() and cfg.wandb_mode != "disabled":
        try:
            import wandb
            wandb_key = os.environ.get("WANDB_API_KEY")
            if wandb_key:
                wandb.login(key=wandb_key, anonymous="allow")
            wandb.init(
                project=cfg.wandb_project,
                name=f"{cfg.wandb_run_name}-sdprior",
                mode=cfg.wandb_mode,
                config={**{k: str(v) for k, v in cfg.__dict__.items()},
                        "stage": "sdprior", "world_size": world_size(),
                        "prior_params_M": round(n_prior / 1e6, 1)},
            )
            use_wandb = True
        except Exception as e:
            print(f"[wandb] disabled: {e}")

    # ── Training loop ──────────────────────────────────────────────────────────
    epochs_no_improve = 0

    for epoch in range(start_epoch, cfg.num_epochs + 1):
        prior_ddp.train()
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        losses: dict[str, list] = defaultdict(list)
        optimizer.zero_grad()
        pbar = tqdm(train_loader, leave=False, desc=f"Ep{epoch}[sdprior]",
                    disable=not is_main(), mininterval=1.0, miniters=50)

        for bi, batch in enumerate(pbar):
            fmri = batch["fmri"].to(device)
            subject = batch["subject"].to(device)
            clip_gt = batch["clip_emb"].to(device).float()  # (B, 768)

            with torch.no_grad():
                tokens, _ = brain_enc(fmri, subject)   # (B, N, D), frozen

            with torch.cuda.amp.autocast(enabled=device.type == "cuda",
                                         dtype=autocast_dtype):
                loss = raw_prior.flow_loss(clip_gt, tokens,
                                           sigma_min=float(cfg.sigma_min))

            # Skip NaN/Inf steps (fp16 overflow recovery)
            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue

            scaler.scale(loss / cfg.grad_accum).backward()

            if (bi + 1) % cfg.grad_accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in prior_ddp.parameters() if p.requires_grad],
                    cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if step >= cfg.ema_start:
                    ema.update(raw_prior)
                step += 1

            losses["loss"].append(loss.detach())
            if is_main() and bi % 50 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}",
                                 lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step()

        train_metrics = {
            f"train/{k}": float(torch.stack(v).mean().item())
            for k, v in losses.items()
        }
        train_metrics["train/lr"] = optimizer.param_groups[0]["lr"]
        train_metrics["epoch"] = epoch

        is_last = (epoch == cfg.num_epochs)
        do_eval = (epoch % cfg.eval_freq == 0 or epoch == 1 or is_last)

        if do_eval and is_main():
            gc.collect()
            torch.cuda.empty_cache()

            # Eval with EMA weights if available
            _backup_sd = None
            if step >= cfg.ema_start:
                _backup_sd = {k: v.clone() for k, v in raw_prior.state_dict().items()}
                ema.apply(raw_prior)

            eval_metrics = evaluate(
                brain_enc, raw_prior, eval_loader, device, cfg,
                n_batches=cfg.eval_batches,
            )

            if _backup_sd is not None:
                ema.restore(_backup_sd, raw_prior)

            all_metrics = {**train_metrics, **{f"eval/{k}": v for k, v in eval_metrics.items()}}
            print(f"[Ep {epoch:3d}] " +
                  " | ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in all_metrics.items()
                              if k != "epoch"))

            cos = eval_metrics.get("CLIP_Cos", 0.0)
            if cos > best_cos:
                best_cos = cos
                epochs_no_improve = 0
                if step >= cfg.ema_start:
                    _best_backup = {k: v.clone() for k, v in raw_prior.state_dict().items()}
                    ema.apply(raw_prior)
                torch.save(raw_prior.state_dict(),
                           cfg.output_dir / "best_clip_cos.pt")
                if step >= cfg.ema_start:
                    ema.restore(_best_backup, raw_prior)
                print(f"  [SDPrior] New best CLIP_Cos={best_cos:.4f} → best_clip_cos.pt")
            else:
                epochs_no_improve += 1

            # Save resume checkpoint every eval epoch
            resume_ckpt = {
                "epoch": epoch,
                "step": step,
                "prior": raw_prior.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "ema": {k: v.clone() for k, v in ema.shadow.items()},
                "best_cos": best_cos,
                "clip_mean": raw_prior.clip_mean.cpu(),
                "clip_std": raw_prior.clip_std.cpu(),
            }
            torch.save(resume_ckpt, cfg.output_dir / "resume.pt")

            if use_wandb:
                import wandb
                wandb.log(all_metrics, step=epoch)

            # Early stopping
            if (epoch >= cfg.min_epochs and
                    epochs_no_improve >= cfg.patience and
                    not is_last):
                print(f"[SDPrior] Early stopping at epoch {epoch} "
                      f"(no improvement for {epochs_no_improve} eval epochs)")
                torch.save(raw_prior.state_dict(),
                           cfg.output_dir / "early_stop.pt")
                break

    # ── Final save ────────────────────────────────────────────────────────────
    if is_main():
        torch.save(raw_prior.state_dict(), cfg.output_dir / "final_raw.pt")

        if step >= cfg.ema_start:
            _final_backup = {k: v.clone() for k, v in raw_prior.state_dict().items()}
            ema.apply(raw_prior)
        torch.save(raw_prior.state_dict(), cfg.output_dir / "final_ema.pt")
        if step >= cfg.ema_start:
            ema.restore(_final_backup, raw_prior)

        print(f"[SDPrior] Training done. Best CLIP_Cos={best_cos:.4f}")
        print(f"  Outputs in {cfg.output_dir}")
        print("  Files: best_clip_cos.pt  final_ema.pt  final_raw.pt")

    cleanup_ddp()


if __name__ == "__main__":
    main()
