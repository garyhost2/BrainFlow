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
from brainflow.config_overrides import apply_env_overrides
from brainflow.data import build_dataloaders, is_dist, is_main, world_size
from brainflow.models import BrainEncoder, CLIPPrior
from brainflow.ema import EMA

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

class PriorOnlyModel(nn.Module):
    def __init__(self, cfg, voxels_per_subject):
        super().__init__()

        cfg.method = "dit"
        self.cfg = cfg
        self.brain_enc = BrainEncoder(cfg, voxels_per_subject)
        self.clip_prior = CLIPPrior(cfg)
        self.lambda_align = float(os.environ.get("LAMBDA_ALIGN", "1.0"))
        self.lambda_prior = float(os.environ.get("LAMBDA_PRIOR", "1.0"))

    def encode_fmri(self, fmri, subject):
        return self.brain_enc(fmri, subject)

    def training_step(self, batch, device):
        fmri = batch["fmri"].to(device, non_blocking=True)
        clip_emb = batch["clip_emb"].to(device, non_blocking=True)
        subject = batch["subject"].to(device, non_blocking=True)

        _, cls_emb = self.encode_fmri(fmri, subject)

        if cls_emb.shape[-1] == clip_emb.shape[-1]:
            cos = F.cosine_similarity(cls_emb, clip_emb, dim=-1)
            loss_align = (1.0 - cos).mean()
        else:
            loss_align = torch.tensor(0.0, device=device)

        loss_prior = self.clip_prior.flow_loss(clip_emb, cls_emb)

        if not torch.isfinite(loss_prior) or not torch.isfinite(loss_align):
            raise RuntimeError(
                f"NaN/Inf in losses: align={loss_align.item()} prior={loss_prior.item()}")

        loss = self.lambda_align * loss_align + self.lambda_prior * loss_prior
        return {
            "loss": loss,
            "align": loss_align.detach(),
            "prior": loss_prior.detach(),
        }

    @torch.no_grad()
    def eval_step(self, batch, device, n_steps: int = 10):
        fmri = batch["fmri"].to(device, non_blocking=True)
        clip_emb = batch["clip_emb"].to(device, non_blocking=True)
        subject = batch["subject"].to(device, non_blocking=True)
        _, cls_emb = self.encode_fmri(fmri, subject)
        clip_pred = self.clip_prior.sample(cls_emb, n_steps=n_steps)
        clip_gt_norm = F.normalize(clip_emb, dim=-1)
        return F.cosine_similarity(clip_pred, clip_gt_norm, dim=-1).mean().item()

def main():
    set_seed(42)
    cfg = load_config()
    cfg = apply_env_overrides(cfg)

    cfg.method = "dit"

    if cfg.experiment_name in ("baseline", "v5"):
        cfg.experiment_name = "stage1_prior"

    cfg.output_dir = cfg.output_dir / cfg.experiment_name
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.experiment_name != "baseline":
        cfg.wandb_run_name = f"{cfg.wandb_run_name}-{cfg.experiment_name}"

    device = setup_ddp(cfg.backend, cfg.init_method)
    if is_main():
        print(f"[BrainFlow stage1-prior - {cfg.experiment_name.upper()}] "
              f"world_size={world_size()} | device={device} | subjects={cfg.subjects}")

    train_loader, _, eval_loader, train_sampler, voxels = build_dataloaders(cfg)
    if is_main():
        print(f"Voxels per subject: {voxels}")
        print(f"Train batches/rank: {len(train_loader)}")

    model = PriorOnlyModel(cfg, voxels).to(device)
    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_enc = sum(p.numel() for p in model.brain_enc.parameters())
    n_prior = sum(p.numel() for p in model.clip_prior.parameters())
    if is_main():
        print(f"Params: total={n_total/1e6:.1f}M | enc={n_enc/1e6:.1f}M | "
              f"prior={n_prior/1e6:.1f}M")
        print(f"lambda_align={model.lambda_align} lambda_prior={model.lambda_prior}")

    if is_dist():
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                    find_unused_parameters=False, broadcast_buffers=False)
        try:
            model._set_static_graph()
        except (AttributeError, RuntimeError):
            pass
    raw_model = model.module if hasattr(model, "module") else model

    ema = EMA(raw_model, decay=cfg.ema_decay)

    lr = float(os.environ.get("LR", str(cfg.lr)))
    _optim_kwargs = dict(lr=lr, weight_decay=cfg.weight_decay,
                         betas=(0.9, 0.999), eps=1e-8)
    try:
        optimizer = AdamW(model.parameters(), fused=True, **_optim_kwargs)
    except (TypeError, RuntimeError):
        optimizer = AdamW(model.parameters(), **_optim_kwargs)

    def lr_lambda(epoch):
        if epoch < cfg.warmup_epochs:
            return epoch / max(1, cfg.warmup_epochs)
        progress = (epoch - cfg.warmup_epochs) / max(1, cfg.num_epochs - cfg.warmup_epochs)
        return max(cfg.min_lr / lr, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_bf16 = False
    if device.type == "cuda":
        cc = torch.cuda.get_device_capability(device)
        use_bf16 = (cc[0] >= 8)

    use_amp = os.environ.get("USE_AMP", "0") == "1"
    scaler = torch.cuda.amp.GradScaler(
        enabled=(use_amp and device.type == "cuda" and not use_bf16))
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    if is_main():
        print(f"AMP enabled={use_amp} bf16={use_bf16}")

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
                        "n_enc": n_enc, "n_prior": n_prior,
                        "voxels": voxels, "world_size": world_size(),
                        "stage": "prior_pretrain"},
            )
            use_wandb = True
        except Exception as e:
            print(f"[wandb] disabled: {e}")

    best_clip_cos = -1.0
    step = 0

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        losses = defaultdict(list)
        optimizer.zero_grad()
        pbar = tqdm(train_loader, leave=False, desc=f"Ep{epoch}",
                    disable=not is_main(), mininterval=1.0, miniters=50)

        for bi, batch in enumerate(pbar):
            with torch.cuda.amp.autocast(
                    enabled=(use_amp and device.type == "cuda"),
                    dtype=autocast_dtype):
                ld = raw_model.training_step(batch, device)
                loss = ld["loss"] / cfg.grad_accum
            scaler.scale(loss).backward()

            if (bi + 1) % cfg.grad_accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
                if step >= cfg.ema_start: ema.update(raw_model)
                step += 1

            for k, v in ld.items():
                losses[k].append(v.detach())
            if is_main() and (bi % 50 == 0):
                pbar.set_postfix(a=f"{ld['align'].item():.3f}",
                                 p=f"{ld['prior'].item():.3f}")

        scheduler.step()

        train_metrics = {
            f"train/{k}": float(torch.stack(v).mean().item())
            for k, v in losses.items()
        }
        train_metrics["train/lr"] = optimizer.param_groups[0]["lr"]
        train_metrics["epoch"] = epoch

        do_eval = (epoch % cfg.eval_freq == 0 or epoch == 1
                   or epoch == cfg.num_epochs)
        if do_eval and is_main():
            gc.collect(); torch.cuda.empty_cache()
            if step >= cfg.ema_start:
                orig = {k: v.clone() for k, v in raw_model.state_dict().items()}
                ema.apply(raw_model)
                clip_cos = _eval_clip_cos(raw_model, eval_loader, device)
                ema.restore(orig, raw_model)
                tag = "EMA"
            else:
                clip_cos = _eval_clip_cos(raw_model, eval_loader, device)
                tag = "raw"

            train_metrics["test/clip_cos"] = clip_cos
            print(f"Ep{epoch:4d}/{cfg.num_epochs} | "
                  f"align={train_metrics['train/align']:.4f} "
                  f"prior={train_metrics['train/prior']:.4f} | "
                  f"clip_cos={clip_cos:.4f} ({tag})")

            if clip_cos > best_clip_cos:
                best_clip_cos = clip_cos
                torch.save(raw_model.state_dict(),
                           cfg.output_dir / "best_prior.pt")

        if is_main() and use_wandb:
            import wandb
            wandb.log(train_metrics, step=epoch)

    if is_main():
        if step >= cfg.ema_start:
            orig = {k: v.clone() for k, v in raw_model.state_dict().items()}
            ema.apply(raw_model)
            torch.save(raw_model.state_dict(), cfg.output_dir / "ema_final_prior.pt")
            ema.restore(orig, raw_model)
        torch.save(raw_model.state_dict(), cfg.output_dir / "raw_final_prior.pt")
        print(f"Best clip_cos = {best_clip_cos:.4f}")
        print(f"Checkpoints written to: {cfg.output_dir}")
        print("To warm-start full training, add to your training sbatch:")
        print(f"  export INIT_FROM={cfg.output_dir / 'best_prior.pt'}")
        if use_wandb:
            import wandb; wandb.finish()

    cleanup_ddp()

@torch.no_grad()
def _eval_clip_cos(model, loader, device, max_batches: int = 50, n_steps: int = 10):
    model.eval()
    cosines = []
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        c = model.eval_step(batch, device, n_steps=n_steps)
        cosines.append(c)
    model.train()
    return float(np.mean(cosines)) if cosines else 0.0

if __name__ == "__main__":
    main()
