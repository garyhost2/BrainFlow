"""DDP training entrypoint for BrainFlow v5.

    python -m scripts.train                                 # single GPU
    torchrun --nproc_per_node=$NGPUS -m scripts.train       # multi-GPU
"""
from __future__ import annotations
import os, sys, math, gc, random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brainflow.config import load_config
from brainflow.data import build_dataloaders, is_dist, is_main, rank, world_size
from brainflow.models import BrainFlowV5
from brainflow.ema import EMA
from brainflow.vae import FrozenVAE
from brainflow.metrics import evaluate

load_dotenv()


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
    dist.init_process_group(backend=backend, init_method=init_method)
    return device


def cleanup_ddp():
    if is_dist():
        dist.destroy_process_group()


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(42)
    cfg = load_config()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    device = setup_ddp(cfg.backend, cfg.init_method)
    if is_main():
        print(f"[BrainFlow v5] world_size={world_size()} | device={device} | "
              f"subjects={cfg.subjects}")

    train_loader, test_loader, train_sampler, voxels = build_dataloaders(cfg)
    if is_main():
        print(f"Voxels per subject: {voxels}")
        print(f"Train batches/rank: {len(train_loader)} | "
              f"Test batches/rank: {len(test_loader)}")

    model = BrainFlowV5(cfg, voxels).to(device)
    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_enc = sum(p.numel() for p in model.brain_enc.parameters())
    n_unet = sum(p.numel() for p in model.flow_unet.parameters())
    if is_main():
        print(f"Params: total={n_total/1e6:.1f}M | enc={n_enc/1e6:.1f}M | unet={n_unet/1e6:.1f}M")

    if is_dist():
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                    find_unused_parameters=True)
    raw_model = model.module if hasattr(model, "module") else model

    ema = EMA(raw_model, decay=cfg.ema_decay)

    optimizer = AdamW(model.parameters(), lr=cfg.lr,
                      weight_decay=cfg.weight_decay, betas=(0.9, 0.999), eps=1e-8)

    def lr_lambda(epoch):
        if epoch < cfg.warmup_epochs:
            return epoch / max(1, cfg.warmup_epochs)
        progress = (epoch - cfg.warmup_epochs) / max(1, cfg.num_epochs - cfg.warmup_epochs)
        return max(cfg.min_lr / cfg.lr, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    use_wandb = False
    if is_main() and cfg.wandb_mode != "disabled":
        try:
            import wandb
            wandb_key = os.environ.get("WANDB_API_KEY")
            if wandb_key:
                wandb.login(key=wandb_key, anonymous="allow")
            wandb.init(
                project=cfg.wandb_project, name=cfg.wandb_run_name,
                mode=cfg.wandb_mode,
                config={**cfg.__dict__, "n_total": n_total,
                        "n_enc": n_enc, "n_unet": n_unet,
                        "voxels": voxels, "world_size": world_size()},
            )
            use_wandb = True
        except Exception as e:
            print(f"[wandb] disabled: {e}")

    def _do_eval():
        vae = FrozenVAE(cache_dir=cfg.data_dir / "hf_cache")
        m = evaluate(raw_model, vae, test_loader, device, cfg, n_batches=cfg.eval_batches)
        del vae; gc.collect(); torch.cuda.empty_cache()
        return m

    best_pc = 0.0; best_clip = 0.0; best_combined = 0.0
    step = 0; epochs_no_improve = 0; stopped_early = False

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        losses = defaultdict(list)
        optimizer.zero_grad()
        pbar = tqdm(train_loader, leave=False, desc=f"Ep{epoch}", disable=not is_main())

        for bi, batch in enumerate(pbar):
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                ld = raw_model.training_step(batch, device)
                loss = ld["loss"] / cfg.grad_accum
            scaler.scale(loss).backward()

            if (bi + 1) % cfg.grad_accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
                if step >= cfg.ema_start: ema.update(raw_model)
                step += 1

            for k, v in ld.items(): losses[k].append(v.item())
            if is_main():
                pbar.set_postfix(c=f"{ld['cfm'].item():.3f}", a=f"{ld['align'].item():.3f}")

        scheduler.step()

        train_metrics = {f"train/{k}": float(np.mean(v)) for k, v in losses.items()}
        train_metrics["train/lr"] = optimizer.param_groups[0]["lr"]
        train_metrics["epoch"] = epoch

        do_eval = (epoch % cfg.eval_freq == 0 or epoch == 1 or epoch == cfg.num_epochs)
        if do_eval and is_main():
            gc.collect(); torch.cuda.empty_cache()
            if step >= cfg.ema_start:
                orig = {k: v.clone() for k, v in raw_model.state_dict().items()}
                ema.apply(raw_model)
                m = _do_eval()
                ema.restore(orig, raw_model)
                tag = "EMA"
            else:
                m = _do_eval()
                tag = "raw"

            for k, v in m.items(): train_metrics[f"test/{k}"] = v
            print(f"Ep{epoch:4d}/{cfg.num_epochs} | "
                  f"cfm={train_metrics['train/cfm']:.4f} "
                  f"align={train_metrics['train/align']:.4f} | "
                  f"PC={m['PixCorr']:.3f} SSIM={m['SSIM']:.3f} "
                  f"CLIP={m['CLIP_Sim']:.3f} ({tag})")

            improved = False
            combined = m["PixCorr"] + 0.3 * m["CLIP_Sim"]
            if m["PixCorr"] > best_pc:
                best_pc = m["PixCorr"]; improved = True
                torch.save(raw_model.state_dict(), cfg.output_dir / "best_pc_v5.pt")
            if m["CLIP_Sim"] > best_clip:
                best_clip = m["CLIP_Sim"]; improved = True
                torch.save(raw_model.state_dict(), cfg.output_dir / "best_clip_v5.pt")
            if combined > best_combined:
                best_combined = combined; improved = True
                torch.save(raw_model.state_dict(), cfg.output_dir / "best_combined_v5.pt")
            epochs_no_improve = 0 if improved else epochs_no_improve + cfg.eval_freq

            if epoch >= cfg.min_epochs and epochs_no_improve >= cfg.patience:
                print(f"⚠ Early stopping at epoch {epoch} "
                      f"(no improvement for {epochs_no_improve} epochs)")
                stopped_early = True
                torch.save(raw_model.state_dict(), cfg.output_dir / "early_stop_v5.pt")

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
            torch.save(raw_model.state_dict(), cfg.output_dir / "v5_ema_final.pt")
            ema.restore(orig, raw_model)
        torch.save(raw_model.state_dict(), cfg.output_dir / "v5_raw_final.pt")
        print(f"Best: PC={best_pc:.4f} | CLIP={best_clip:.4f} | Combined={best_combined:.4f}")
        if use_wandb:
            import wandb; wandb.finish()

    cleanup_ddp()


if __name__ == "__main__":
    main()
