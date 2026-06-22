from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import yaml

@dataclass
class Config:

    backend: str = "auto"
    init_method: str = "env://"

    hf_repo: str = "pscotti/mindeyev2"
    data_dir: Path = Path("./mindeyev2_cache")
    tensor_cache: str = "all_subjects_tensors.pt"
    clip_cache: str = "all_subjects_clip_v2.pt"
    clip_patch_cache: str = "all_subjects_clip_patches_v2.pt"
    latent_cache: str = "all_subjects_latents_v2.pt"
    subjects: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 8])
    img_size: int = 256
    max_train: int = 8859
    max_test: int = 982
    download_threads: int = 4
    force_rebuild: bool = False

    batch_size_per_gpu: int = 16
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last_train: bool = True
    drop_last_test: bool = False

    enc_hidden: int = 1024
    enc_blocks: int = 4
    enc_drop: float = 0.15
    stem_drop: float = 0.15
    n_tokens: int = 64
    brain_dim: int = 1024
    enc_token_attn_layers: int = 0
    enc_token_attn_heads: int = 8
    token_drop_prob: float = 0.1
    stochastic_depth: float = 0.1
    unet_base_ch: int = 128
    attn_heads: int = 8
    time_emb_dim: int = 512
    latent_ch: int = 4
    latent_res: int = 32
    clip_dim: int = 768
    use_subject_heads: bool = True
    subject_head_rank: int = 16
    use_ridge_init: bool = True
    ridge_lambda: float = 1e3
    ridge_init_cache: str = "ridge_init.pt"

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

    ode_steps: int = 20
    eval_ode_steps: int = 10
    eval_solver: str = "euler"

    use_clip_prior: bool = True

    lambda_pixel: float = 0.3
    percep_t_min: float = 0.5
    percep_t_power: float = 1.0
    use_mixco: bool = True
    mixco_alpha: float = 0.3
    softclip_start_epoch: int = -1
    softclip_temp: float = 0.006

    method: str = "baseline"
    hrf_len: int = 16
    sb_sigma_max: float = 0.5

    dit_dim: int = 512
    dit_depth: int = 12
    dit_heads: int = 8
    dit_patch: int = 4
    lambda_prior: float = 0.3
    prior_ode_steps: int = 10
    prior_dim: int = 512
    prior_blocks: int = 3
    prior_target: str = "cls"
    decoder_model_id: str = "stabilityai/stable-diffusion-2-1-unclip"
    decoder_num_inference_steps: int = 30
    decoder_guidance_scale: float = 7.5

    flow_objective: str = "cfm"

    clip_dit_dim: int = 512
    clip_dit_depth: int = 6
    clip_dit_heads: int = 8

    lambda_cos: float = 0.1

    lambda_cls: float = 0.1

    lambda_clip_flow: float = 0.2

    clip_sample_prob_max: float = 0.5
    clip_ramp_frac: float = 0.3

    training_stage: str = "2b"

    subject_filter: List[int] = field(default_factory=list)

    flux_model_id: str = "black-forest-labs/FLUX.1-dev"

    ip_dim: int = 1024
    ip_n_blocks: int = 19

    flux_train_steps: int = 10000
    flux_lr: float = 1e-4
    flux_lr_warmup: int = 500
    flux_grad_clip: float = 1.0
    flux_grad_ckpt: bool = False
    flux_offload_t5: bool = True
    flux_batch_size: int = 4
    flux_grad_accum: int = 8
    flux_save_every: int = 500

    flux_steps: int = 28
    flux_guidance: float = 3.5
    flux_height: int = 512
    flux_width: int = 512

    init_from_phase3: str = ""

    output_dir: Path = Path("./outputs")
    experiment_name: str = "baseline"

    wandb_project: str = "brainflow-v5"
    wandb_run_name: str = "v5-multisubject"
    wandb_mode: str = "online"

    clip_prior_geometry: str = "euclidean"

    t_schedule: str = "linear"
    logit_normal_m: float = 0.0
    logit_normal_s: float = 1.0

    use_ot_coupling: bool = False
    ot_reg: float = 0.05
    ot_iters: int = 50

def load_config(path: str | os.PathLike | None = None) -> Config:
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

    if "data_dir" in flat:
        flat["data_dir"] = Path(flat["data_dir"])
    if "output_dir" in flat:
        flat["output_dir"] = Path(flat["output_dir"])
    if "subjects" in flat:
        flat["subjects"] = list(flat["subjects"])

    _float_fields = {
        "lr", "min_lr", "weight_decay", "grad_clip", "ema_decay",
        "lambda_align", "lambda_cfm", "lambda_percep", "lambda_cos",
        "lambda_cls", "lambda_clip_flow", "sigma_min", "cfg_drop_prob",
        "cfg_scale", "fmri_noise_std", "fmri_mask_prob", "mixup_alpha",
        "infonce_temp", "clip_sample_prob_max", "clip_ramp_frac",
        "stochastic_depth", "enc_drop", "token_drop_prob",
        "stem_drop", "lambda_pixel", "lambda_prior", "percep_t_min",
        "percep_t_power", "mixco_alpha", "ot_reg", "logit_normal_m",
        "logit_normal_s", "vfm_kl_weight", "ridge_lambda", "softclip_temp",
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
