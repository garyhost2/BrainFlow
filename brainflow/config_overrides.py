"""Configuration overrides from environment variables for experiments."""
from __future__ import annotations
import os


def apply_env_overrides(cfg):
    """Apply environment variable overrides to config.
    
    This allows launching different experiments without modifying config.yaml.
    
    Environment variables:
        EXPERIMENT_NAME: baseline | lpips | l1 | v6 | v7
        PERCEP_LOSS: none | lpips | l1
        LAMBDA_PERCEP: float (e.g., "0.1")
        BATCH_SIZE_PER_GPU: int (e.g., "32")
        GRAD_ACCUM: int (e.g., "6")
        N_TOKENS: int (e.g., "64")
        UNET_BASE_CH: int (e.g., "192")
        N_ENC_BLOCKS: int (e.g., "6")
        USE_V6: "1" to enable V6 enhancements (64 tokens + LPIPS)
        USE_V7: "1" to enable V7 enhancements (compact model: 64 tokens, 192 base_ch, 6 enc blocks)
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
    
    # UNet architecture overrides
    if "UNET_BASE_CH" in os.environ:
        cfg.unet_base_ch = int(os.environ["UNET_BASE_CH"])
    # N_ENC_BLOCKS is kept for backwards compatibility; ENC_BLOCKS is canonical.
    # If both are set simultaneously, ENC_BLOCKS wins and a warning is emitted.
    if "N_ENC_BLOCKS" in os.environ:
        import warnings
        warnings.warn("N_ENC_BLOCKS is deprecated; use ENC_BLOCKS instead.", DeprecationWarning)
        cfg.enc_blocks = int(os.environ["N_ENC_BLOCKS"])

    # Encoder scaling
    if "ENC_HIDDEN" in os.environ:
        cfg.enc_hidden = int(os.environ["ENC_HIDDEN"])
    if "ENC_BLOCKS" in os.environ:
        cfg.enc_blocks = int(os.environ["ENC_BLOCKS"])

    # CLIP prior scaling (method == "dit")
    if "PRIOR_DIM" in os.environ:
        cfg.prior_dim = int(os.environ["PRIOR_DIM"])
    if "PRIOR_BLOCKS" in os.environ:
        cfg.prior_blocks = int(os.environ["PRIOR_BLOCKS"])
    if "LAMBDA_PRIOR" in os.environ:
        cfg.lambda_prior = float(os.environ["LAMBDA_PRIOR"])
    if "PRIOR_ODE_STEPS" in os.environ:
        cfg.prior_ode_steps = int(os.environ["PRIOR_ODE_STEPS"])
    
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
    
    # V7 enhancements - compact model with more tokens
    if "USE_V7" in os.environ and os.environ["USE_V7"] == "1":
        # V7: Compact architecture with richer representation
        cfg.n_tokens = 64  # More tokens for better representation
        cfg.lambda_align = 0.1  # Keep baseline alignment weight
        cfg.lambda_percep = 0.1  # Moderate perceptual loss
        cfg.percep_loss = "lpips"  # Use LPIPS for perceptual quality
        cfg.unet_base_ch = 192  # Smaller UNet (vs 256 baseline)
        cfg.n_enc_blocks = 6  # Fewer encoder blocks (vs 8 baseline)
        cfg.batch_size_per_gpu = 32  # Smaller batch for single GPU
        if not cfg.experiment_name.startswith("v7"):
            cfg.experiment_name = f"v7"

    # V8 enhancements - MindEye-v2 quality target: CLIP prior + larger token budget
    # Activate with: USE_V8=1 ./launch_experiment.sh v8
    if "USE_V8" in os.environ and os.environ["USE_V8"] == "1":
        cfg.n_tokens = 32          # More tokens for richer representation
        cfg.use_clip_prior = True  # C.1: CLIPPriorHead for CLIP prior injection
        cfg.lambda_prior = 0.3     # Stronger prior supervision
        cfg.lambda_pixel = 0.3     # C.2: Pixel-space L1 loss (t > 0.85 guard)
        cfg.lambda_align = 0.2     # Balanced alignment
        cfg.method = "baseline"    # CLIP prior + bigger tokens, standard Gaussian source
        if not cfg.experiment_name.startswith("v8"):
            cfg.experiment_name = "v8"

    # ── Method selection (NeurIPS experiments) ──
    # METHOD=baseline | hrf
    if "METHOD" in os.environ:
        cfg.method = os.environ["METHOD"].strip().lower()

    # SUBJECTS=1            (single subject)
    # SUBJECTS=1,2,5,7      (multi-subject)
    if "SUBJECTS" in os.environ:
        cfg.subjects = [int(s) for s in os.environ["SUBJECTS"].split(",") if s.strip()]

    # NUM_EPOCHS override (e.g., for smoke test)
    if "NUM_EPOCHS" in os.environ:
        cfg.num_epochs = int(os.environ["NUM_EPOCHS"])

    return cfg
