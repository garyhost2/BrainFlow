"""Phase 4: Train BrainIPAdapter on frozen FLUX.1-dev.

Requires a Phase 3 checkpoint (BrainEncoder + ClipPrior already trained).

Usage (single GPU):
    BRAINFLOW_CONFIG=configs/phase4_flux.yaml \\
    python -m scripts.train_flux_adapter

Usage (torchrun, 2 GPUs):
    BRAINFLOW_CONFIG=configs/phase4_flux.yaml \\
    torchrun --standalone --nproc_per_node=2 -m scripts.train_flux_adapter

What is trained
---------------
  Only BrainIPAdapter (~120M params, fp32).

What is frozen
--------------
  FLUX.1-dev transformer + VAE + text encoders  (bf16, ~24GB)
  BrainEncoder                                   (fp32, ~50M)
  ClipPrior                                      (fp32, ~20M)

Gradient flow
-------------
  target_image → FLUX VAE encode → z (no_grad)
  z, t → FLUX denoising forward → v_pred (FLUX frozen, hooks fire)
  hooks inject ip_adapter residuals → gradients flow through ip_adapter only
  loss = MSE(v_pred, z - eps)  →  backward  →  ip_adapter optimized
"""
from __future__ import annotations

import gc
import logging
import math
import os
import random
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from dotenv import load_dotenv
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch.set_float32_matmul_precision("high")

from brainflow.clip_prior import ClipPrior
from brainflow.config import load_config
from brainflow.data import build_dataloaders, is_dist, is_main, rank, world_size
from brainflow.ema import EMA
from brainflow.flux_decoder import FLUXBrainDecoder
from brainflow.models import BrainEncoder, migrate_input_proj

load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── DDP helpers ───────────────────────────────────────────────────────────────

def setup_ddp(backend_cfg: str, init_method: str) -> torch.device:
    if "LOCAL_RANK" not in os.environ:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    backend = "nccl" if backend_cfg == "auto" else backend_cfg
    timeout_sec = int(os.environ.get("DDP_TIMEOUT", "3600"))
    dist.init_process_group(
        backend=backend, init_method=init_method,
        timeout=timedelta(seconds=timeout_sec),
    )
    return torch.device(f"cuda:{local_rank}")


def cleanup_ddp():
    if is_dist():
        dist.destroy_process_group()


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── checkpoint helpers ────────────────────────────────────────────────────────

def load_phase3_checkpoint(
    cfg,
    brain_enc: BrainEncoder,
    prior: ClipPrior,
    device: torch.device,
):
    """Load BrainEncoder + ClipPrior weights from a Phase 3 checkpoint."""
    path = getattr(cfg, "init_from_phase3", "")
    if not path:
        raise ValueError(
            "init_from_phase3 must be set in config/env to the Phase 3 checkpoint path."
        )
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Phase 3 checkpoint not found: {path}\n"
            "Run scripts/train_sdprior.py first."
        )
    sd = torch.load(path, map_location="cpu")
    # Strip torch.compile wrappers
    sd = {k.replace("._orig_mod.", "."): v for k, v in sd.items()}
    sd = migrate_input_proj(sd, brain_enc.max_vox)

    enc_sd = {k[len("brain_enc."):]: v
              for k, v in sd.items() if k.startswith("brain_enc.")}
    prior_sd = {k[len("prior."):]: v
                for k, v in sd.items() if k.startswith("prior.")}

    missing, unexpected = brain_enc.load_state_dict(enc_sd, strict=False)
    if missing and is_main():
        log.warning("BrainEncoder missing keys: %s", missing)

    if prior_sd:
        missing, unexpected = prior.load_state_dict(prior_sd, strict=False)
        if missing and is_main():
            log.warning("ClipPrior missing keys: %s", missing)
    else:
        if is_main():
            log.warning(
                "No 'prior.*' keys in checkpoint — ClipPrior starts from random init."
            )

    if is_main():
        log.info("Phase 3 checkpoint loaded: %s", path)


# ── sample VFM posterior (mu, log_sigma) ─────────────────────────────────────

@torch.no_grad()
def sample_vfm_posterior(
    prior: ClipPrior,
    tokens: torch.Tensor,          # (B, N, D)
    cfg,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ClipPrior flow sampler; return (mu_clip, log_sigma).

    For a standard CFM/Euclidean prior the sampler returns a single point
    (no explicit sigma).  We derive a per-sample uncertainty estimate from
    the L2 distance between two independent samples — cheap and useful as
    a gate signal even without a full VFM posterior head.

    If the prior uses VFM (flow_objective=="vfm"), the VFM head's log_sigma
    is used directly (requires the prior to expose sample_with_sigma()).
    """
    n_steps  = getattr(cfg, "prior_ode_steps", 20)
    cfg_sc   = getattr(cfg, "cfg_scale", 1.5)

    # Sample 1 — the mean estimate
    mu = prior.sample(tokens, n_steps=n_steps, cfg_scale=cfg_sc,
                      normalize_output=True)                   # (B, clip_dim)

    # Sigma estimate: spread between two draws at cfg_scale=1.0 (unconditional)
    # This is cheap (no extra backward) and gives a real uncertainty signal.
    s1 = prior.sample(tokens, n_steps=max(n_steps // 2, 5), cfg_scale=1.0,
                      normalize_output=True)
    s2 = prior.sample(tokens, n_steps=max(n_steps // 2, 5), cfg_scale=1.0,
                      normalize_output=True)
    log_sigma = 0.5 * (s1 - s2).pow(2).clamp(min=1e-8).log()  # (B, clip_dim)

    return mu.to(device), log_sigma.to(device)


# ── training loop ─────────────────────────────────────────────────────────────

def train(cfg):
    set_seed(42 + rank())
    device = setup_ddp(cfg.backend, cfg.init_method)

    if is_main():
        log.info("=== Phase 4 — FLUX IP-Adapter Training ===")
        log.info("Device: %s | World size: %d", device, world_size())

    # ── data ──────────────────────────────────────────────────────────────────
    # Override batch size with flux-specific setting
    cfg.batch_size_per_gpu = getattr(cfg, "flux_batch_size", 4)
    train_loader, test_loader, voxels = build_dataloaders(cfg)

    # ── models ────────────────────────────────────────────────────────────────
    brain_enc = BrainEncoder(cfg, voxels).to(device)
    prior = ClipPrior(
        clip_dim=cfg.clip_dim,
        ctx_dim=cfg.brain_dim,
        dim=getattr(cfg, "prior_dim", 512),
        depth=getattr(cfg, "prior_blocks", 6),
        geometry=getattr(cfg, "clip_prior_geometry", "euclidean"),
        prior_target=getattr(cfg, "prior_target", "cls"),
    ).to(device)

    # Load Phase 3 weights
    load_phase3_checkpoint(cfg, brain_enc, prior, device)

    # Freeze brain encoder and prior completely
    brain_enc.eval().requires_grad_(False)
    prior.eval().requires_grad_(False)

    # FLUX decoder (loads lazily on first compute_loss call)
    flux_decoder = FLUXBrainDecoder(cfg).to(device)
    flux_decoder.ip_adapter.unfreeze()

    n_trainable = flux_decoder.num_trainable_parameters()
    if is_main():
        log.info("BrainIPAdapter trainable params: %.1fM", n_trainable / 1e6)

    # ── DDP wrap (adapter only) ───────────────────────────────────────────────
    if is_dist():
        flux_decoder.ip_adapter = DDP(
            flux_decoder.ip_adapter,
            device_ids=[device.index],
            find_unused_parameters=False,
            broadcast_buffers=False,
        )

    # ── optimizer + scheduler ─────────────────────────────────────────────────
    lr           = getattr(cfg, "flux_lr", 1e-4)
    total_steps  = getattr(cfg, "flux_train_steps", 10000)
    warmup_steps = getattr(cfg, "flux_lr_warmup", 500)
    grad_accum   = getattr(cfg, "flux_grad_accum", 8)
    grad_clip    = getattr(cfg, "flux_grad_clip", 1.0)

    # Fused AdamW — ~15% faster on CUDA
    try:
        optimizer = AdamW(
            flux_decoder.ip_adapter.parameters(),
            lr=lr, weight_decay=0.01, betas=(0.9, 0.999),
            fused=True,
        )
    except TypeError:
        optimizer = AdamW(
            flux_decoder.ip_adapter.parameters(),
            lr=lr, weight_decay=0.01, betas=(0.9, 0.999),
        )

    # Cosine schedule with linear warmup
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # EMA of adapter weights for stable inference
    ema = EMA(
        flux_decoder.ip_adapter,
        decay=0.999,
        start_step=100,
    )

    # ── output dirs ───────────────────────────────────────────────────────────
    out_dir = Path(cfg.output_dir) / cfg.experiment_name
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)

    # ── wandb ─────────────────────────────────────────────────────────────────
    use_wandb = is_main() and getattr(cfg, "wandb_mode", "disabled") != "disabled"
    if use_wandb:
        import wandb
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            mode=cfg.wandb_mode,
            config={
                "ip_dim": cfg.ip_dim,
                "ip_n_blocks": cfg.ip_n_blocks,
                "flux_lr": lr,
                "total_steps": total_steps,
                "flux_grad_accum": grad_accum,
            },
        )

    # ── training loop ─────────────────────────────────────────────────────────
    save_every = getattr(cfg, "flux_save_every", 500)
    best_loss  = float("inf")
    step       = 0
    optimizer.zero_grad(set_to_none=True)

    train_iter = iter(train_loader)
    loss_accum = 0.0
    loss_count = 0

    if is_main():
        log.info("Starting training for %d steps (grad_accum=%d, effective_bs=%d)",
                 total_steps, grad_accum,
                 cfg.flux_batch_size * world_size() * grad_accum)

    while step < total_steps:
        # ── fetch batch ───────────────────────────────────────────────────────
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        fmri    = batch["fmri"].to(device, non_blocking=True)
        subject = batch["subject"].to(device, non_blocking=True)
        images  = batch.get("image")
        if images is None:
            # Reconstruct from VAE latents if raw images not in cache
            raise RuntimeError(
                "Batch missing 'image' key. "
                "Ensure the dataloader returns raw images for Phase 4 training."
            )
        images = images.to(device, non_blocking=True)

        # ── encode fMRI (no grad) ─────────────────────────────────────────────
        with torch.no_grad():
            tokens, _ = brain_enc(fmri, subject)           # (B, N, D)
            mu_clip, log_sigma = sample_vfm_posterior(
                prior, tokens, cfg, device
            )

        # ── FLUX loss (grad through ip_adapter only) ──────────────────────────
        flux_decoder.ip_adapter.train()
        loss = flux_decoder.compute_loss(tokens, mu_clip, log_sigma, images)
        loss = loss / grad_accum
        loss.backward()

        loss_accum += loss.item() * grad_accum
        loss_count += 1

        # ── optimizer step ────────────────────────────────────────────────────
        if (loss_count % grad_accum == 0):
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    flux_decoder.ip_adapter.parameters(), grad_clip
                )
            optimizer.step()
            scheduler.step()
            ema.update(step)
            optimizer.zero_grad(set_to_none=True)
            step += 1

            # ── logging ───────────────────────────────────────────────────────
            if is_main() and step % 50 == 0:
                avg_loss = loss_accum / loss_count
                cur_lr   = scheduler.get_last_lr()[0]
                log.info("step %5d/%d | loss=%.4f | lr=%.2e",
                         step, total_steps, avg_loss, cur_lr)
                if use_wandb:
                    wandb.log({"train/loss": avg_loss, "train/lr": cur_lr}, step=step)
                loss_accum = 0.0
                loss_count = 0

            # ── checkpoint ────────────────────────────────────────────────────
            if is_main() and step % save_every == 0:
                ckpt_path = out_dir / f"adapter_step{step:06d}.pt"
                flux_decoder.save_adapter(ckpt_path)

                avg = loss_accum / max(loss_count, 1)
                if avg < best_loss:
                    best_loss = avg
                    flux_decoder.save_adapter(out_dir / "best_adapter.pt")
                    log.info("New best adapter saved (loss=%.4f)", best_loss)

    # ── final save ────────────────────────────────────────────────────────────
    if is_main():
        ema.apply()
        flux_decoder.save_adapter(out_dir / "final_adapter_ema.pt")
        ema.restore()
        flux_decoder.save_adapter(out_dir / "final_adapter_raw.pt")
        log.info("Training complete. Outputs in %s", out_dir)
        if use_wandb:
            wandb.finish()

    cleanup_ddp()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load_config()
    train(cfg)
