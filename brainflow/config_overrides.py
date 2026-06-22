from __future__ import annotations
import os

def apply_env_overrides(cfg):

    if "EXPERIMENT_NAME" in os.environ:
        cfg.experiment_name = os.environ["EXPERIMENT_NAME"]

    if "PERCEP_LOSS" in os.environ:
        cfg.percep_loss = os.environ["PERCEP_LOSS"]

    if "LAMBDA_PERCEP" in os.environ:
        cfg.lambda_percep = float(os.environ["LAMBDA_PERCEP"])

    if "BATCH_SIZE_PER_GPU" in os.environ:
        cfg.batch_size_per_gpu = int(os.environ["BATCH_SIZE_PER_GPU"])
    if "GRAD_ACCUM" in os.environ:
        cfg.grad_accum = int(os.environ["GRAD_ACCUM"])

    if "N_TOKENS" in os.environ:
        cfg.n_tokens = int(os.environ["N_TOKENS"])

    if "UNET_BASE_CH" in os.environ:
        cfg.unet_base_ch = int(os.environ["UNET_BASE_CH"])

    if "N_ENC_BLOCKS" in os.environ:
        import warnings
        warnings.warn("N_ENC_BLOCKS is deprecated; use ENC_BLOCKS instead.", DeprecationWarning)
        cfg.enc_blocks = int(os.environ["N_ENC_BLOCKS"])

    if "ENC_HIDDEN" in os.environ:
        cfg.enc_hidden = int(os.environ["ENC_HIDDEN"])
    if "ENC_BLOCKS" in os.environ:
        cfg.enc_blocks = int(os.environ["ENC_BLOCKS"])

    if "PRIOR_DIM" in os.environ:
        cfg.prior_dim = int(os.environ["PRIOR_DIM"])
    if "PRIOR_BLOCKS" in os.environ:
        cfg.prior_blocks = int(os.environ["PRIOR_BLOCKS"])
    if "LAMBDA_PRIOR" in os.environ:
        cfg.lambda_prior = float(os.environ["LAMBDA_PRIOR"])
    if "PRIOR_ODE_STEPS" in os.environ:
        cfg.prior_ode_steps = int(os.environ["PRIOR_ODE_STEPS"])

    if "USE_V6" in os.environ and os.environ["USE_V6"] == "1":

        cfg.n_tokens = 64
        cfg.lambda_align = 0.1
        cfg.lambda_percep = 0.15
        cfg.percep_loss = "lpips"
        cfg.mixup_alpha = 0.1
        if not cfg.experiment_name.startswith("v6"):
            cfg.experiment_name = f"v6"

    if "USE_V7" in os.environ and os.environ["USE_V7"] == "1":

        cfg.n_tokens = 64
        cfg.lambda_align = 0.1
        cfg.lambda_percep = 0.1
        cfg.percep_loss = "lpips"
        cfg.unet_base_ch = 192
        cfg.n_enc_blocks = 6
        cfg.batch_size_per_gpu = 32
        if not cfg.experiment_name.startswith("v7"):
            cfg.experiment_name = f"v7"

    if "USE_V8" in os.environ and os.environ["USE_V8"] == "1":
        cfg.n_tokens = 32
        cfg.use_clip_prior = True
        cfg.lambda_prior = 0.3
        cfg.lambda_pixel = 0.3
        cfg.lambda_align = 0.2
        cfg.method = "baseline"
        if not cfg.experiment_name.startswith("v8"):
            cfg.experiment_name = "v8"

    if "METHOD" in os.environ:
        cfg.method = os.environ["METHOD"].strip().lower()

    if "SUBJECTS" in os.environ:
        cfg.subjects = [int(s) for s in os.environ["SUBJECTS"].split(",") if s.strip()]

    if "NUM_EPOCHS" in os.environ:
        cfg.num_epochs = int(os.environ["NUM_EPOCHS"])

    return cfg
