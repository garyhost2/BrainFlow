"""Evaluate a trained BrainFlow model on the test set.

Usage:
    python -m scripts.eval --checkpoint outputs/best_combined_v5.pt
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path

import torch
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brainflow.config import load_config
from brainflow.data import build_dataloaders
from brainflow.models import BrainFlowV5
from brainflow.vae import FrozenVAE
from brainflow.metrics import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--cfg_scale", type=float, default=1.0,
                        help="Classifier-free guidance scale for inference")
    parser.add_argument("--ode_steps", type=int, default=1,
                        help="Number of ODE integration steps")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on (cuda or cpu)")
    args = parser.parse_args()

    # Load config
    cfg = load_config()
    cfg.cfg_scale = args.cfg_scale
    cfg.ode_steps = args.ode_steps
    
    print(f"[Evaluation Config]")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  CFG scale: {args.cfg_scale}")
    print(f"  ODE steps: {args.ode_steps}")
    print(f"  Device: {args.device}")
    print()

    # Build dataloaders (we only need test loader)
    print("Loading data...")
    _, _, eval_loader, _, voxels = build_dataloaders(cfg)
    print(f"  Test batches: {len(eval_loader)}")
    print(f"  Voxels per subject: {voxels}")
    print()

    # Load model
    device = torch.device(args.device)
    print("Loading model...")
    model = BrainFlowV5(cfg, voxels).to(device)
    
    ckpt = torch.load(args.checkpoint, map_location=device)
    # Strip torch.compile's "_orig_mod." prefix if present
    ckpt = {k.replace("._orig_mod.", "."): v for k, v in ckpt.items()}
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

    # Evaluate
    print("Running evaluation...")
    with torch.no_grad():
        metrics = evaluate(model, vae, eval_loader, device, cfg, n_batches=9999)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()
