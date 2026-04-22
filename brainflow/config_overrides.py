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
        BATCH_SIZE_PER_GPU: int (e.g., "32")
        GRAD_ACCUM: int (e.g., "6")
        N_TOKENS: int (e.g., "64")
        USE_V6: "1" to enable V6 enhancements (64 tokens + LPIPS)
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
    
    # Batch size and grad accumulation overrides for different GPU configs
    if "BATCH_SIZE_PER_GPU" in os.environ:
        cfg.batch_size_per_gpu = int(os.environ["BATCH_SIZE_PER_GPU"])
    if "GRAD_ACCUM" in os.environ:
        cfg.grad_accum = int(os.environ["GRAD_ACCUM"])
    
    # Number of tokens override
    if "N_TOKENS" in os.environ:
        cfg.n_tokens = int(os.environ["N_TOKENS"])
    
    # V6 enhancements - more tokens + perceptual supervision
    if "USE_V6" in os.environ and os.environ["USE_V6"] == "1":
        # V6: More expressive tokens + perceptual quality
        cfg.n_tokens = 64  # Increase from 16 to 64 for richer representation
        cfg.lambda_align = 0.1  # Keep baseline alignment weight
        cfg.lambda_percep = 0.15  # Stronger perceptual loss
        cfg.percep_loss = "lpips"  # Always use LPIPS for V6
        cfg.mixup_alpha = 0.1  # Reduce mixup for cleaner gradients
        if not cfg.experiment_name.startswith("v6"):
            cfg.experiment_name = f"v6"
    
    return cfg
