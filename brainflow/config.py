"""Central config loader. Reads config.yaml and exposes a frozen dataclass."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import yaml


@dataclass
class Config:
    # Distribution
    backend: str = "auto"
    init_method: str = "env://"

    # Data
    hf_repo: str = "pscotti/mindeyev2"
    data_dir: Path = Path("./mindeyev2_cache")
    tensor_cache: str = "all_subjects_tensors.pt"
    clip_cache: str = "all_subjects_clip_v2.pt"
    clip_patch_cache: str = "all_subjects_clip_patches_v2.pt"  # Phase 2: ViT-L/14 patch tokens
    latent_cache: str = "all_subjects_latents_v2.pt"
    subjects: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8])
    img_size: int = 256
    max_train: int = 8859
    max_test: int = 982
    download_threads: int = 4
    force_rebuild: bool = False

    # DataLoader
    batch_size_per_gpu: int = 16
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last_train: bool = True
    drop_last_test: bool = False

    # Model
    enc_hidden: int = 1024
    enc_blocks: int = 4
    enc_drop: float = 0.25
    n_tokens: int = 16
    brain_dim: int = 768
    token_drop_prob: float = 0.1
    stochastic_depth: float = 0.1
    unet_base_ch: int = 128
    attn_heads: int = 8
    time_emb_dim: int = 512
    latent_ch: int = 4
    latent_res: int = 32
    clip_dim: int = 768

    # Training
    num_epochs: int = 150
    grad_accum: int = 4
    lr: float = 2e-4
    warmup_epochs: int = 10
    min_lr: float = 1e-5
    grad_clip: float = 0.5
    weight_decay: float = 0.05
    infonce_temp: float = 0.07
    lambda_align: float = 0.1
    lambda_cfm: float = 1.0
    lambda_percep: float = 0.15
    vfm_kl_weight: float = 0.0
    vfm_kl_anneal_epochs: int = 50
    percep_loss: str = "lpips"
    sigma_min: float = 1e-4
    cfg_drop_prob: float = 0.20
    cfg_scale: float = 2.0
    fmri_noise_std: float = 0.05
    fmri_mask_prob: float = 0.05
    mixup_alpha: float = 0.2
    ema_decay: float = 0.9995
    ema_start: int = 200
    eval_freq: int = 5
    eval_batches: int = 8
    patience: int = 25
    min_epochs: int = 60
    log_every_n_steps: int = 50

    # Inference
    ode_steps: int = 20
    eval_ode_steps: int = 10      # ODE steps for training-time eval (faster)
    eval_solver: str = "euler"    # solver for training-time eval: euler | midpoint | heun

    # C.1 CLIP Prior head (use_clip_prior)
    use_clip_prior: bool = False

    # C.2 Pixel-space L1 loss
    lambda_pixel: float = 0.0

    # Method (formulation): "baseline" | "hrf" | "sb" | "dit"
    #   baseline — standard CFM from Gaussian noise → VAE latent
    #   hrf      — Brain-manifold flow: source = projected CLS + small noise,
    #              + fixed double-gamma HRF velocity bias gated by sin(pi*t)
    #   sb       — I²SB-style noisy-bridge interpolant:
    #              xt = (1-t)x0 + t*z + σ(t)·ε, σ(t) = σ_max·√(t(1-t))
    #              Stochastic Euler-Maruyama sampler → uncertainty via K samples.
    #   dit      — DiT (Diffusion Transformer) backbone replacing UNet,
    #              + auxiliary unCLIP-style flow-matching prior on CLIP
    #              embeddings. Flow matching objective preserved.
    method: str = "baseline"
    hrf_len: int = 16            # HRF kernel length (matches n_tokens)
    sb_sigma_max: float = 0.5    # SB peak noise scale at t=0.5
    # DiT hyperparameters (method == "dit")
    dit_dim: int = 512
    dit_depth: int = 12
    dit_heads: int = 8
    dit_patch: int = 4
    lambda_prior: float = 0.2    # weight of the diffusion prior loss (also used by CLIPPriorHead)
    prior_ode_steps: int = 10    # sampling steps for the CLIP prior
    prior_dim: int = 512         # CLIPPrior hidden dim
    prior_blocks: int = 3        # CLIPPrior residual block count

    # ── Phase 2: VFM + Flow_CLIP + hierarchical conditioning ──────────────────
    # VFM objective toggle: "cfm" | "vfm"
    # "cfm" reproduces the existing CFM training loss exactly.
    # "vfm" uses Variational Flow Matching (rg-vfm, ICLR 2026).
    flow_objective: str = "cfm"

    # Flow_CLIP DiT hyperparameters
    # clip_dit_dim, clip_dit_depth, clip_dit_heads control the CLIP-space DiT.
    clip_dit_dim: int = 512
    clip_dit_depth: int = 6
    clip_dit_heads: int = 8
    # Cosine + MSE hybrid loss weight (lambda_cos * cosine_loss)
    lambda_cos: float = 0.1
    # Weight of auxiliary CLS head loss
    lambda_cls: float = 0.1
    # Weight of Flow_CLIP loss in the joint objective
    lambda_clip_flow: float = 0.2

    # Hierarchical conditioning: Flow_CLIP → Flow_VAE
    # Probability of sampling from Flow_CLIP (vs. teacher-forcing true CLIP tokens)
    # during Stage 2B joint training.  Linearly ramped 0 → clip_sample_prob_max
    # over the first clip_ramp_frac * num_epochs epochs.
    clip_sample_prob_max: float = 0.5
    clip_ramp_frac: float = 0.3   # fraction of epochs over which to ramp

    # Stage selector: "2a" | "2b" | "2c"
    # 2a: Flow_CLIP only, BrainEncoder frozen
    # 2b: joint fine-tune, ramped clip_sample_prob
    # 2c: inference sweep (CFG x solver x NFE)
    training_stage: str = "2b"

    # Subject filter (for specialist fine-tuning)
    # Set to [] for multi-subject training (default).
    subject_filter: List[int] = field(default_factory=list)

    # Output
    output_dir: Path = Path("./outputs")
    experiment_name: str = "baseline"

    # WandB
    wandb_project: str = "brainflow-v5"
    wandb_run_name: str = "v5-multisubject"
    wandb_mode: str = "online"  # "online" | "offline" | "disabled"

    # ── Item 1: Spherical geometry for ClipPrior ───────────────────────────────
    # "euclidean" — standard Gaussian CFM (default, preserves current behavior)
    # "sphere"    — Riemannian flow matching on S^(d-1) via SLERP + exp-map
    clip_prior_geometry: str = "euclidean"

    # ── Item 2: Time schedule for ODE integration ──────────────────────────────
    # "linear"       — uniform spacing (default)
    # "cosine"       — concentrates steps near t=0 and t=1
    # "logit_normal" — logistic-normal quantile spacing
    t_schedule: str = "linear"
    logit_normal_m: float = 0.0   # mean  for logit_normal schedule
    logit_normal_s: float = 1.0   # sigma for logit_normal schedule

    # ── Item 3: Minibatch OT coupling for VAE-latent CFM ──────────────────────
    # use_ot_coupling: enable entropic Sinkhorn coupling in _step_2b
    use_ot_coupling: bool = False
    ot_reg: float = 0.05    # Sinkhorn entropic regularization
    ot_iters: int = 50      # Sinkhorn iterations


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load config.yaml. If path is None, checks BRAINFLOW_CONFIG env var, then repo-root config.yaml."""
    if path is None:
        env_path = os.environ.get("BRAINFLOW_CONFIG", "").strip()
        path = Path(env_path) if env_path else Path(__file__).resolve().parent.parent / "config.yaml"
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    flat = {}
    for section, body in raw.items():
        if isinstance(body, dict):
            flat.update(body)
        else:
            flat[section] = body

    # Coerce path strings
    if "data_dir" in flat:
        flat["data_dir"] = Path(flat["data_dir"])
    if "output_dir" in flat:
        flat["output_dir"] = Path(flat["output_dir"])
    if "subjects" in flat:
        flat["subjects"] = list(flat["subjects"])

    # Force-coerce float fields — some PyYAML versions parse '1e-4' as a string
    _float_fields = {
        "lr", "min_lr", "weight_decay", "grad_clip", "ema_decay",
        "lambda_align", "lambda_cfm", "lambda_percep", "lambda_cos",
        "lambda_cls", "lambda_clip_flow", "sigma_min", "cfg_drop_prob",
        "cfg_scale", "fmri_noise_std", "fmri_mask_prob", "mixup_alpha",
        "infonce_temp", "clip_sample_prob_max", "clip_ramp_frac",
        "stochastic_depth", "enc_drop", "token_drop_prob",
        "ot_reg", "logit_normal_m", "logit_normal_s", "vfm_kl_weight",
    }
    for key in _float_fields:
        if key in flat and not isinstance(flat[key], float):
            flat[key] = float(flat[key])

    valid = {f.name for f in Config.__dataclass_fields__.values()}
    flat = {k: v for k, v in flat.items() if k in valid}
    cfg = Config(**flat)
    if (cfg.lambda_percep > 0) != (cfg.percep_loss != "none"):
        raise ValueError(
            "Invalid perceptual configuration: (lambda_percep > 0) must match "
            "(percep_loss != 'none'). Set percep_loss='lpips' (or 'l1') when "
            "lambda_percep > 0, or set percep_loss='none' with lambda_percep=0."
        )
    return cfg
