"""Phase 2 DDP training entrypoint: FlowCLIPDiT (2A) + Joint (2B).

Stage 2A (FlowCLIPDiT only, BrainEncoder frozen):
    BRAINFLOW_CONFIG=configs/phase2_stage2a.yaml \\
    INIT_FROM=outputs/mc_xl_v100_32/best_pc_v5.pt \\
    torchrun --nproc_per_node=1 -m scripts.train_phase2

Stage 2B (joint fine-tune):
    BRAINFLOW_CONFIG=configs/phase2_stage2b.yaml \\
    INIT_FROM=outputs/phase2_stage2a/best_clip_cos.pt \\
    torchrun --nproc_per_node=2 -m scripts.train_phase2
"""
from __future__ import annotations
import os, sys, math, gc, random
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch.set_float32_matmul_precision("high")
if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "enable_flash_sdp"):
    torch.backends.cuda.enable_flash_sdp(True)

from brainflow.config import load_config
from brainflow.data import build_dataloaders, is_dist, is_main, rank, world_size
from brainflow.phase2_model import BrainFlowPhase2
from brainflow.ema import EMA
from brainflow.vae import FrozenVAE
from brainflow.metrics import evaluate

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


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Stage 2A eval: CLIP patch cosine similarity ────────────────────────────────

@torch.no_grad()
def evaluate_2a(raw_model: BrainFlowPhase2, loader, device, cfg,
                n_batches: int = 32) -> dict[str, float]:
    """Evaluate FlowCLIPDiT: sample patch tokens and compare cosine sim to GT."""
    raw_model.eval()
    cos_vals: list[float] = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        fmri = batch["fmri"].to(device)
        subject = batch["subject"].to(device)
        clip_patches = batch["clip_patches"].to(device).float()  # (B, 256, 1024)

        tokens, _ = raw_model.brain_enc(fmri, subject)
        sampled = raw_model.flow_clip.sample(
            tokens, n_steps=cfg.eval_ode_steps, solver=cfg.eval_solver)  # (B, 1024, 16, 16)
        B = sampled.shape[0]
        sampled_flat = sampled.permute(0, 2, 3, 1).reshape(B, 256, 1024)
        if raw_model.flow_clip._standardization_fitted:
            sampled_flat = raw_model.flow_clip.destandardize(sampled_flat)

        # Mean-pool over spatial tokens then compare
        sampled_cls = sampled_flat.mean(1)           # (B, 1024)
        gt_cls = clip_patches.mean(1)                # (B, 1024)
        cos = F.cosine_similarity(
            F.normalize(sampled_cls, dim=-1),
            F.normalize(gt_cls, dim=-1), dim=-1
        ).mean().item()
        cos_vals.append(cos)

    raw_model.train()
    return {"CLIP_Cos": float(np.mean(cos_vals)) if cos_vals else 0.0}


# ── CLIP standardization fitting ───────────────────────────────────────────────

def fit_standardization(raw_model: BrainFlowPhase2, train_loader,
                        device: torch.device, max_samples: int = 12000):
    """Fit clip_token_mean/std from training patch tokens, broadcast across ranks."""
    if is_main():
        print("[Phase2] Fitting CLIP patch token standardization …")
    patches_list: list[torch.Tensor] = []
    n_collected = 0
    for batch in train_loader:
        if n_collected >= max_samples:
            break
        cp = batch.get("clip_patches")
        if cp is None:
            raise RuntimeError("clip_patches missing from batch — check cfg.training_stage")
        patches_list.append(cp.float())
        n_collected += cp.shape[0]
    all_patches = torch.cat(patches_list, dim=0)[:max_samples]  # (N, 256, 1024)
    if is_main():
        raw_model.flow_clip.fit_standardization(all_patches)
        print(f"[Phase2] Standardization fitted on {len(all_patches)} samples.")
    # Broadcast fitted buffers from rank-0 to all ranks
    if is_dist():
        dist.broadcast(raw_model.flow_clip.clip_token_mean, src=0)
        dist.broadcast(raw_model.flow_clip.clip_token_std, src=0)
        # Sync flag
        flag = torch.tensor([1.0 if raw_model.flow_clip._standardization_fitted else 0.0],
                             device=device)
        dist.broadcast(flag, src=0)
        if not is_main() and flag.item() > 0:
            raw_model.flow_clip._standardization_fitted = True
        dist.barrier()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    set_seed(42)
    cfg = load_config()   # reads BRAINFLOW_CONFIG env var automatically
    stage = cfg.training_stage
    if stage not in ("2a", "2b"):
        raise ValueError(f"train_phase2.py expects training_stage 2a or 2b, got {stage!r}")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    device = setup_ddp(cfg.backend, cfg.init_method)
    if is_main():
        print(f"[Phase2 Stage {stage.upper()}] world_size={world_size()} device={device} "
              f"subjects={cfg.subjects}")

    train_loader, test_loader, eval_loader, train_sampler, voxels = build_dataloaders(cfg)
    if is_main():
        print(f"Train batches/rank: {len(train_loader)} | Test: {len(test_loader)}")

    # ── Build / warm-start model ───────────────────────────────────────────────
    init_from = os.environ.get("INIT_FROM", "").strip()
    raw_model = BrainFlowPhase2(cfg, voxels)
    if init_from and Path(init_from).is_file():
        ckpt = torch.load(init_from, map_location="cpu")
        # Strip torch.compile prefix if present
        ckpt = {k.replace("._orig_mod.", "."): v for k, v in ckpt.items()}
        missing, unexpected = raw_model.load_state_dict(ckpt, strict=False)
        if is_main():
            loaded = len(ckpt) - len(unexpected)
            print(f"[Phase2] Warm-started from {init_from} "
                  f"({loaded}/{len(ckpt)} tensors loaded)")
            if unexpected:
                print(f"  unexpected (skipped): {len(unexpected)} "
                      f"e.g. {unexpected[:3]}")
            if missing:
                print(f"  missing (random init): {len(missing)} "
                      f"e.g. {missing[:3]}")
    elif init_from and is_main():
        print(f"[Phase2] INIT_FROM={init_from!r} not found — random init.")

    # Set stage BEFORE DDP so frozen params are not tracked
    if stage == "2a":
        raw_model.set_stage_2a()
    else:
        raw_model.set_stage_2b()
    raw_model = raw_model.to(device)

    n_total = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    n_enc = sum(p.numel() for p in raw_model.brain_enc.parameters())
    n_fcd = sum(p.numel() for p in raw_model.flow_clip.parameters())
    n_funet = sum(p.numel() for p in raw_model.flow_vae.parameters())
    if is_main():
        print(f"Trainable: {n_total/1e6:.1f}M | enc={n_enc/1e6:.1f}M "
              f"clip_dit={n_fcd/1e6:.1f}M flow_unet={n_funet/1e6:.1f}M")

    # Fit CLIP standardization on rank-0, then broadcast
    if stage == "2a":
        fit_standardization(raw_model, train_loader, device)

    model: nn.Module = raw_model
    if is_dist():
        model = DDP(raw_model,
                    device_ids=[device.index] if device.type == "cuda" else None,
                    find_unused_parameters=False, broadcast_buffers=True)
        try:
            model._set_static_graph()
        except (AttributeError, RuntimeError):
            pass
        raw_model = model.module

    # A.5: torch.compile on FlowCLIPDiT (always active) — skip on V100 via env
    _disable_compile = os.environ.get("DISABLE_COMPILE", "0") == "1"
    if not _disable_compile and hasattr(torch, "compile"):
        try:
            raw_model.flow_clip = torch.compile(
                raw_model.flow_clip, mode="default", dynamic=False)
            if is_main():
                print("[compile] flow_clip compiled.")
        except Exception as e:
            if is_main():
                print(f"[compile] disabled: {e}")
    elif _disable_compile and is_main():
        print("[compile] disabled by DISABLE_COMPILE=1")

    ema = EMA(raw_model, decay=cfg.ema_decay)

    _optim_kwargs = dict(lr=float(cfg.lr), weight_decay=float(cfg.weight_decay),
                         betas=(0.9, 0.999), eps=1e-8)
    try:
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            fused=True, **_optim_kwargs)
    except (TypeError, RuntimeError) as _e:
        if "fused" not in str(_e).lower() and "unexpected keyword" not in str(_e).lower():
            raise
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad], **_optim_kwargs)

    def lr_lambda(epoch):
        if epoch < cfg.warmup_epochs:
            return epoch / max(1, cfg.warmup_epochs)
        progress = (epoch - cfg.warmup_epochs) / max(1, cfg.num_epochs - cfg.warmup_epochs)
        return max(cfg.min_lr / cfg.lr, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # fp16 + GradScaler on V100 (compute cap < 8.0); bf16 on Ampere+
    use_bf16 = False
    if device.type == "cuda":
        cc = torch.cuda.get_device_capability(device)
        use_bf16 = (cc[0] >= 8)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and not use_bf16))
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # VAE for Stage 2B eval
    eval_vae = None
    if stage == "2b":
        eval_vae = FrozenVAE(cache_dir=cfg.data_dir / "hf_cache").to(device)

    # WandB
    use_wandb = False
    if is_main() and cfg.wandb_mode != "disabled":
        try:
            import wandb
            wandb_key = os.environ.get("WANDB_API_KEY")
            if wandb_key:
                wandb.login(key=wandb_key, anonymous="allow")
            wandb.init(
                project=cfg.wandb_project,
                name=f"{cfg.wandb_run_name}-phase2-{stage}",
                mode=cfg.wandb_mode,
                config={**{k: str(v) for k, v in cfg.__dict__.items()},
                        "stage": stage, "world_size": world_size()},
            )
            use_wandb = True
        except Exception as e:
            print(f"[wandb] disabled: {e}")

    def _do_eval(full_eval=False):
        n_batches = cfg.eval_batches if not full_eval else 9999
        if stage == "2a":
            return evaluate_2a(raw_model, eval_loader, device, cfg, n_batches=n_batches)
        else:
            if eval_vae is None:
                return {}
            m = evaluate(raw_model, eval_vae, eval_loader, device, cfg,
                         n_batches=n_batches,
                         n_steps=cfg.eval_ode_steps,
                         solver=cfg.eval_solver)
            return m

    best_metric = 0.0
    best_pc = 0.0; best_clip_sim = 0.0; best_combined = 0.0
    step = 0; epochs_no_improve = 0; stopped_early = False

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        losses: dict[str, list] = defaultdict(list)
        optimizer.zero_grad()
        pbar = tqdm(train_loader, leave=False, desc=f"Ep{epoch}[{stage}]",
                    disable=not is_main(), mininterval=1.0, miniters=50)

        for bi, batch in enumerate(pbar):
            with torch.cuda.amp.autocast(enabled=device.type == "cuda",
                                         dtype=autocast_dtype):
                ld = raw_model.training_step(batch, device, epoch=epoch)
                loss = ld["loss"] / cfg.grad_accum

            scaler.scale(loss).backward()

            if (bi + 1) % cfg.grad_accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if step >= cfg.ema_start:
                    ema.update(raw_model)
                step += 1

            for k, v in ld.items():
                losses[k].append(v.detach())
            if is_main() and (bi % 50 == 0):
                postfix = {k: f"{v.item():.3f}" for k, v in ld.items() if k != "loss"}
                postfix["L"] = f"{ld['loss'].item():.3f}"
                pbar.set_postfix(**postfix)

        scheduler.step()

        train_metrics = {
            f"train/{k}": float(torch.stack(v).mean().item())
            for k, v in losses.items()
        }
        train_metrics["train/lr"] = optimizer.param_groups[0]["lr"]
        train_metrics["epoch"] = epoch

        is_last_epoch = (epoch == cfg.num_epochs)
        do_eval = (epoch % cfg.eval_freq == 0 or epoch == 1 or is_last_epoch)
        if do_eval and is_main():
            gc.collect(); torch.cuda.empty_cache()
            if step >= cfg.ema_start:
                orig = {k: v.clone() for k, v in raw_model.state_dict().items()}
                ema.apply(raw_model)
                m = _do_eval(full_eval=is_last_epoch)
                ema.restore(orig, raw_model)
                tag = "EMA"
            else:
                m = _do_eval(full_eval=is_last_epoch)
                tag = "raw"

            for k, v in m.items():
                train_metrics[f"test/{k}"] = v

            improved = False
            if stage == "2a":
                cos = m.get("CLIP_Cos", 0.0)
                if cos > best_metric:
                    best_metric = cos; improved = True
                    torch.save(raw_model.state_dict(),
                               cfg.output_dir / "best_clip_cos.pt")
                print(f"Ep{epoch:4d}/{cfg.num_epochs} | "
                      f"loss={train_metrics['train/loss']:.4f} "
                      f"cos={train_metrics.get('train/loss_cos', 0):.4f} | "
                      f"CLIP_Cos={cos:.4f} ({tag})")
            else:
                pc = m.get("PixCorr", 0.0)
                clip_s = m.get("CLIP_Sim", 0.0)
                ssim = m.get("SSIM", 0.0)
                combined = pc + 0.5 * ssim + 0.3 * clip_s
                if pc > best_pc:
                    best_pc = pc; improved = True
                    torch.save(raw_model.state_dict(), cfg.output_dir / "best_pc.pt")
                if clip_s > best_clip_sim:
                    best_clip_sim = clip_s; improved = True
                    torch.save(raw_model.state_dict(), cfg.output_dir / "best_clip.pt")
                if combined > best_combined:
                    best_combined = combined; improved = True
                    torch.save(raw_model.state_dict(), cfg.output_dir / "best_combined.pt")
                print(f"Ep{epoch:4d}/{cfg.num_epochs} | "
                      f"L={train_metrics['train/loss']:.4f} "
                      f"cfm={train_metrics.get('train/cfm', 0):.4f} | "
                      f"PC={pc:.3f} SSIM={ssim:.3f} CLIP={clip_s:.3f} ({tag})")

            epochs_no_improve = 0 if improved else epochs_no_improve + cfg.eval_freq
            if epoch >= cfg.min_epochs and epochs_no_improve >= cfg.patience:
                print(f"Early stopping at epoch {epoch} "
                      f"(no improvement for {epochs_no_improve} epochs)")
                stopped_early = True
                torch.save(raw_model.state_dict(), cfg.output_dir / "early_stop.pt")

        if is_dist():
            stop_t = torch.tensor([1 if stopped_early else 0], device=device)
            dist.broadcast(stop_t, src=0)
            stopped_early = bool(stop_t.item())
        if stopped_early:
            break

        if is_main() and use_wandb:
            import wandb
            wandb.log(train_metrics, step=epoch)

    if is_main():
        if step >= cfg.ema_start:
            orig = {k: v.clone() for k, v in raw_model.state_dict().items()}
            ema.apply(raw_model)
            torch.save(raw_model.state_dict(), cfg.output_dir / "final_ema.pt")
            ema.restore(orig, raw_model)
        torch.save(raw_model.state_dict(), cfg.output_dir / "final_raw.pt")
        if stage == "2a":
            print(f"Best CLIP_Cos = {best_metric:.4f}")
        else:
            print(f"Best: PC={best_pc:.4f} | CLIP={best_clip_sim:.4f} | "
                  f"Combined={best_combined:.4f}")
        if use_wandb:
            import wandb; wandb.finish()

    cleanup_ddp()


if __name__ == "__main__":
    main()
