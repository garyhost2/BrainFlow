"""BrainFlow v5.1 — Conditional Flow Matching for fMRI-to-Image Reconstruction.

v5.1: Phase 1 bugfix patch — data leak (B1), DDP (B3), trial averaging (B4),
      token dropout (B5), midpoint/Heun solver (B6), CFG/ODE defaults (B7),
      mixup fix, unsharded eval loader, GPU throughput improvements.

v5.2 (Phase 2): Hierarchical VFM — Flow_CLIP DiT + VFM objective +
      CLIP-conditioned Flow_VAE + unified solver API.
"""
from .config import load_config, Config
from .models import BrainFlowV5, BrainEncoder, FlowUNet
from .ema import EMA
from .vae import FrozenVAE
from .vfm import vfm_loss, cfm_loss, velocity_from_posterior, flow_loss
from .flow_clip_dit import FlowCLIPDiT
from .flow_unet import FlowUNet as FlowUNetV2
from .solvers import solve, make_t_grid

__all__ = [
    "load_config", "Config",
    "BrainFlowV5", "BrainEncoder", "FlowUNet",
    "EMA", "FrozenVAE",
    # Phase 2
    "vfm_loss", "cfm_loss", "velocity_from_posterior", "flow_loss",
    "FlowCLIPDiT",
    "FlowUNetV2",
    "solve", "make_t_grid",
]
