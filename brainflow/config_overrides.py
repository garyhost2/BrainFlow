"""Configuration overrides from environment variables for experiments."""
from __future__ import annotations
import os


def apply_env_overrides(cfg):
    """Apply environment variable overrides to config.
    
    This allows launching different experiments without modifying config.yaml.
    
    Environment variables:
        EXPERIMENT_NAME: baseline | lpips | l1 | v6
        PERCEP_LOSS: none | lpips | l1
        LAMBDA_PERCEP: float (e.g., "0.1")
        USE_V6: "1" to enable V6 enhancements
    """
    # Experiment name
    if "EXPERIMENT_NAME" in os.environ:
        cfg.experiment_name = os.environ["EXPERIMENT_NAME"]
    
    # Perceptual loss type
    if "PERCEP_LOSS" in os.environ:
        cfg.percep_loss = os.environ["PERCEP_LOSS"]
    
    # Perceptual loss weight
    if "LAMBDA_PERCEP" in os.environ:
        cfg.lambda_percep = float(os.environ["LAMBDA_PERCEP"])
    
    # V6 enhancements - stronger alignment and perceptual supervision
    if "USE_V6" in os.environ and os.environ["USE_V6"] == "1":
        # V6-lite: Enhanced loss weights for better reconstruction
        cfg.lambda_align = 1.0  # Increase alignment loss (was 0.5)
        cfg.lambda_percep = 0.15  # Stronger perceptual loss (was 0.1)
        cfg.percep_loss = "lpips"  # Always use LPIPS for V6
        cfg.mixup_alpha = 0.1  # Reduce mixup (was 0.2) for cleaner gradients
        if not cfg.experiment_name.startswith("v6"):
            cfg.experiment_name = f"v6"
    
    return cfg
