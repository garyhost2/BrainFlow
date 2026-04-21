"""BrainFlow v5.1 — Conditional Flow Matching for fMRI-to-Image Reconstruction.

v5.1: Phase 1 bugfix patch — data leak (B1), DDP (B3), trial averaging (B4),
      token dropout (B5), midpoint/Heun solver (B6), CFG/ODE defaults (B7),
      mixup fix, unsharded eval loader, GPU throughput improvements.
"""
from .config import load_config, Config
from .models import BrainFlowV5, BrainEncoder, FlowUNet
from .ema import EMA
from .vae import FrozenVAE

__all__ = [
    "load_config", "Config",
    "BrainFlowV5", "BrainEncoder", "FlowUNet",
    "EMA", "FrozenVAE",
]
