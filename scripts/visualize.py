"""Generate and visualize reconstructions from a trained BrainFlow model.

Usage:
    python -m scripts.visualize --checkpoint outputs/best_combined_v5.pt --output viz.png
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brainflow.config import load_config
from brainflow.config_overrides import apply_env_overrides
from brainflow.data import build_dataloaders
from brainflow.models import BrainFlowV5, migrate_input_proj
from brainflow.vae import FrozenVAE


def tensor_to_pil(t):
    """Convert tensor [-1, 1] to PIL Image [0, 255]."""
    t = (t + 1) / 2  # [-1, 1] -> [0, 1]
    t = t.clamp(0, 1)
    t = (t * 255).byte()
    return t


def make_grid(images, nrow=8, padding=2):
    """Create image grid from batch of tensors."""
    B, C, H, W = images.shape
    ncol = (B + nrow - 1) // nrow
    
    # Create canvas
    canvas_h = ncol * (H + padding) - padding
    canvas_w = nrow * (W + padding) - padding
    canvas = torch.zeros(C, canvas_h, canvas_w, dtype=torch.uint8)
    
    for idx, img in enumerate(images):
        row = idx // nrow
        col = idx % nrow
        y = row * (H + padding)
        x = col * (W + padding)
        canvas[:, y:y+H, x:x+W] = tensor_to_pil(img)
    
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--output", type=str, default="visualizations.png",
                        help="Output image filename")
    parser.add_argument("--n_samples", type=int, default=32,
                        help="Number of samples to visualize")
    parser.add_argument("--cfg_scale", type=float, default=1.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--ode_steps", type=int, default=1,
                        help="Number of ODE integration steps")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on")
    args = parser.parse_args()

    # Load config
    cfg = load_config()
    cfg = apply_env_overrides(cfg)
    cfg.cfg_scale = args.cfg_scale
    cfg.ode_steps = args.ode_steps
    
    print(f"[Visualization Config]")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Output: {args.output}")
    print(f"  Samples: {args.n_samples}")
    print(f"  CFG scale: {args.cfg_scale}")
    print(f"  ODE steps: {args.ode_steps}")
    print()

    # Build dataloaders
    print("Loading data...")
    _, _, eval_loader, _, voxels = build_dataloaders(cfg)
    print(f"  Test batches: {len(eval_loader)}")
    print()

    # Load model
    device = torch.device(args.device)
    print("Loading model...")
    model = BrainFlowV5(cfg, voxels).to(device)
    
    ckpt = torch.load(args.checkpoint, map_location=device)
    # Strip torch.compile's "_orig_mod." prefix if present
    ckpt = {k.replace("._orig_mod.", "."): v for k, v in ckpt.items()}
    # Migrate old per-subject ModuleDict input_proj checkpoints to the new
    # shared zero-padded Linear (no-op for new checkpoints).
    if hasattr(model.brain_enc, "max_vox"):
        ckpt = migrate_input_proj(ckpt, model.brain_enc.max_vox)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    if missing:
        print(f"  [warn] missing keys: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        print(f"  [warn] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params/1e6:.1f}M")
    print()

    # Load VAE
    print("Loading VAE...")
    vae = FrozenVAE(cache_dir=cfg.data_dir / "hf_cache").to(device)
    print()

    # Generate samples
    print(f"Generating {args.n_samples} reconstructions...")
    
    gt_images = []
    pred_images = []
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Processing"):
            fmri = batch["fmri"].to(device)
            images = batch["image"].to(device)
            subject = batch["subject"].to(device)
            
            # Encode fMRI to tokens
            tokens, _ = model.encode_fmri(fmri, subject)
            
            # Sample latents via ODE
            pred_latents = model.sample(tokens, n_steps=cfg.ode_steps, cfg_scale=cfg.cfg_scale)
            
            # Decode to images
            pred_imgs = vae.decode(pred_latents)
            
            # Collect samples
            gt_images.append(images.cpu())
            pred_images.append(pred_imgs.cpu())
            
            # Check if we have enough samples
            total_samples = sum(x.shape[0] for x in gt_images)
            if total_samples >= args.n_samples:
                break
    
    # Concatenate and trim to exact n_samples
    gt_images = torch.cat(gt_images, dim=0)[:args.n_samples]
    pred_images = torch.cat(pred_images, dim=0)[:args.n_samples]
    
    print(f"Collected {gt_images.shape[0]} samples")
    print()

    # Create visualization grid: alternating rows of GT and predictions
    print("Creating visualization grid...")
    
    # Interleave: [GT_0, Pred_0, GT_1, Pred_1, ...]
    interleaved = []
    for gt, pred in zip(gt_images, pred_images):
        interleaved.append(gt)
        interleaved.append(pred)
    
    interleaved = torch.stack(interleaved, dim=0)
    
    # Create grid (2 columns: GT | Pred)
    grid = make_grid(interleaved, nrow=2, padding=4)
    
    # Convert to PIL and save
    # grid is (C, H, W), need (H, W, C) for PIL
    grid_np = grid.permute(1, 2, 0).numpy()
    pil_img = Image.fromarray(grid_np)
    
    pil_img.save(args.output)
    print(f"✓ Saved visualization to: {args.output}")
    print(f"  Grid size: {pil_img.size}")


if __name__ == "__main__":
    main()
