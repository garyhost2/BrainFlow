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
    clip_cache: str = "all_subjects_clip.pt"
    latent_cache: str = "all_subjects_latents.pt"
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
    lambda_percep: float = 0.1
    percep_loss: str = "none"
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
    lambda_prior: float = 0.2    # weight of the diffusion prior loss
    prior_ode_steps: int = 10    # sampling steps for the CLIP prior

    # Output
    output_dir: Path = Path("./outputs")
    experiment_name: str = "baseline"

    # WandB
    wandb_project: str = "brainflow-v5"
    wandb_run_name: str = "v5-multisubject"
    wandb_mode: str = "online"  # "online" | "offline" | "disabled"


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load config.yaml. If path is None, uses repo-root config.yaml."""
    if path is None:
        path = Path(__file__).resolve().parent.parent / "config.yaml"
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

    valid = {f.name for f in Config.__dataclass_fields__.values()}
    flat = {k: v for k, v in flat.items() if k in valid}
    return Config(**flat)
