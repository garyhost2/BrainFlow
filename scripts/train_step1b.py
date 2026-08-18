from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from brainflow.tensor_cache import assert_tensor_cache_alignment
from brainflow.step1.model_tokens import TokenStep1Config, TokenStep1Model
from brainflow.step1.targets_bigg import build_or_load_bigg_targets
from brainflow.step1.targets import TargetStats
from brainflow.step1.data import build_step1_loaders
from brainflow.step1.metrics import EMA, pixcorr, ssim, CLIPMetric
from brainflow.step1.decoder_sgm import quiet_benign_warnings
from brainflow.step1.instrument import (latent_diagnostics, train_val_gap,
                                        grad_norms, write_manifest)

def setup_a100():
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="./mindeyev2_cache")
    ap.add_argument("--tensor-cache", type=str, default="all_subjects_tensors.pt")
    ap.add_argument("--target-dir", type=str, default="./mindeyev2_cache",
                    help="dir for per-subject bigG target files (step1b_bigg_s{N}.pt)")
    ap.add_argument("--mindeye-src", type=str, default="third_party/MindEyeV2/src")
    ap.add_argument("--ckpt-path", type=str, default="third_party/unclip6_epoch0_step110000.ckpt")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--enc-hidden", type=int, default=2048,
                    help="backbone width; 4096 matches MindEye2 capacity")
    ap.add_argument("--shared-encoder", dest="shared_encoder",
                    action="store_true", default=True,
                    help="one input projection for all subjects + a per-subject "
                         "rank-r residual (default). Cross-subject transfer at the "
                         "input stage; without it a second subject only feeds the trunk.")
    ap.add_argument("--per-subject-encoder", dest="shared_encoder",
                    action="store_false",
                    help="legacy: a full nn.Linear(V_s, enc_hidden) per subject")
    ap.add_argument("--subject-rank", type=int, default=16,
                    help="LoRA rank of the per-subject residual (--shared-encoder)")
    ap.add_argument("--lambda-clip", type=float, default=0.033,
                    help="SoftCLIP contrastive weight (0 disables)")
    ap.add_argument("--clip-temp", type=float, default=0.006)
    ap.add_argument("--lambda-cos", type=float, default=0.5,
                    help="per-token angular fidelity on the regression anchor")
    ap.add_argument("--lambda-reg", type=float, default=1.0,
                    help="regression-anchor weight (drives cond_source=regression)")
    ap.add_argument("--enc-drop", type=float, default=0.15,
                    help="backbone dropout; raise to fight the observed overfitting")
    ap.add_argument("--mixup-pct", type=float, default=0.0,
                    help="fraction of epochs trained with BiMixCo voxel mixing "
                         "(MindEye2 uses 0.33). 0 disables.")
    ap.add_argument("--mixco-beta", type=float, default=0.15)
    ap.add_argument("--mixco-s-thresh", type=float, default=0.5)
    ap.add_argument("--geometry", type=str, default="sphere",
                    choices=["euclidean", "sphere"])
    # --- R-XFM ------------------------------------------------------------
    ap.add_argument("--flow-source", type=str, default="anchor_var",
                    choices=["noise", "anchor_det", "anchor_var"],
                    help="what the flow transports FROM. 'noise' is the legacy "
                         "step1c behaviour (uniform on the sphere, ~90 deg from "
                         "every target); the anchor sources start at the "
                         "regression anchor, so v==0 is the identity and the "
                         "prior is bounded below by cond_source=regression.")
    ap.add_argument("--flow-param", type=str, default="auto",
                    choices=["auto", "endpoint", "velocity"],
                    help="what the DiT emits. 'velocity' (legacy) trains mse(v, u_t), "
                         "which by the identity u_t = Log_zt(z1)/(1-t) weights the "
                         "endpoint error by 1/(1-t)^2 -- 100x at t=0.9, 2500x at "
                         "t=0.98 -- i.e. capacity goes where the task is trivial and "
                         "is starved at t->0 where inference starts. 'endpoint' "
                         "predicts zhat1 and scores 1-cos(zhat1, z1), uniform in t.")
    ap.add_argument("--cfg-scale", type=float, default=None,
                    help="classifier-free guidance for the PATCH prior. Default "
                         "resolves to 1.0 for anchor sources (guidance is incoherent "
                         "when z0 is itself brain-derived, since the 'unconditional' "
                         "branch drops the brain tokens) and 3.0 for --flow-source noise.")
    ap.add_argument("--center-mode", type=str, default="global",
                    choices=["global", "per_position"],
                    help="how the token mean is pooled before the polar split. "
                         "'global' averages over images AND all 256 positions, so "
                         "every centred direction keeps a content-free positional "
                         "offset that inflates every cosine in the repo. Run "
                         "scripts/check_centering.py to measure the split first.")
    ap.add_argument("--select-on", type=str, default="two_way",
                    choices=["two_way", "retrieval", "cos"],
                    help="signal for best.pt. 'cos' (legacy) calls predict_tokens "
                         "with cond_source='regression', so it scores z0 and is "
                         "structurally blind to the flow; it is also a pointwise "
                         "functional while every deployed metric is a pairwise "
                         "ranking one. 'two_way'/'retrieval' come from the latent "
                         "diagnostics and are computed on the flow output.")
    ap.add_argument("--anchor-jitter-rad", type=float, default=0.15,
                    help="expected geodesic displacement (radians) of a draw from "
                         "q(z0|fMRI). Keep well below the measured anchor error "
                         "(log anchor_theta_deg): cos(z0,z1) ~ cos(jitter)*cos(theta).")
    ap.add_argument("--lambda-kl", type=float, default=0.1,
                    help="wrapped-Gaussian KL, nats per tangent dim. Stops sigma "
                         "collapsing to 0; it is a floor, not a strong prior.")
    ap.add_argument("--lambda-clip-tok", type=float, default=0.0,
                    help="token-level SoftCLIP on pooled predicted tokens (the "
                         "256x1664 sequence the decoder actually consumes)")
    ap.add_argument("--cls-cond-mode", type=str, default="jitter",
                    choices=["teacher", "jitter", "sampled", "none"],
                    help="how the patch prior sees c_cls at TRAIN time. 'teacher' "
                         "is the legacy exposure-bias trap: ground truth at train, "
                         "a sampled estimate at inference.")
    ap.add_argument("--cls-jitter-rad", type=float, default=0.30,
                    help="geodesic displacement in RADIANS applied to the true "
                         "c_cls at train time. Calibrate as arccos(cls_cos_hat_true) "
                         "from the logged val/cls_cos_hat_true. (Replaces "
                         "--cls-jitter-kappa, which was a per-coordinate sigma: 0.02 "
                         "there realised 0.02*sqrt(1279) = 0.72 rad = 41 deg.)")
    ap.add_argument("--stochastic-source", action="store_true",
                    help="draw z0 ~ q at inference instead of using the mode")
    ap.add_argument("--diag-batches", type=int, default=8,
                    help="batches per latent-diagnostic pass (train and val each)")
    ap.add_argument("--decode-steps", type=int, default=20,
                    help="ODE steps used inside the latent diagnostics")
    ap.add_argument("--gradnorm-every", type=int, default=200,
                    help="steps between per-term gradient-norm probes (0 disables). "
                         "Costs one backward per term, so keep it sparse.")
    ap.add_argument("--two-head", dest="two_head", action="store_true", default=True,
                    help="structured CLS + patch factorisation (paper default)")
    ap.add_argument("--no-two-head", dest="two_head", action="store_false",
                    help="patch-only ablation (no global CLS head)")
    ap.add_argument("--lambda-rcfm", type=float, default=1.0,
                    help="global CLS flow-matching weight")
    ap.add_argument("--cls-cfg-scale", type=float, default=3.0)
    ap.add_argument("--solver", type=str, default="heun", choices=["euler", "heun"])
    ap.add_argument("--uniform-t-prob", type=float, default=0.1)
    ap.add_argument("--logit-normal-m", type=float, default=0.0,
                    help="mean of the logit-normal flow-time schedule: "
                         "t = sigmoid(N(m, s)). m=0 centres t at 0.5, which puts only "
                         "~2%% of training mass below t=0.1. With an informative source "
                         "that is the wrong place: zt = slerp(z0, z1, t) is already at "
                         "cosine 0.78 at t=0.5 and 0.99 at t=0.9, so the model trains on "
                         "inputs that are nearly the answer, while the sampler STARTS at "
                         "t=0 where the input is the anchor. Negative m moves mass to low "
                         "t (m=-1.5 puts the median at t=0.18).")
    ap.add_argument("--logit-normal-s", type=float, default=1.0,
                    help="std of the logit-normal flow-time schedule")
    ap.add_argument("--lambda-radius", type=float, default=1.0)
    ap.add_argument("--lambda-low", type=float, default=1.0,
                    help="low-level (blurry-image) loss weight; 0 disables the pathway")
    ap.add_argument("--ll-strength", type=float, default=0.7,
                    help="img2img strength at decode (lower preserves more layout)")
    ap.add_argument("--ll-size", type=int, default=64,
                    help="blurry-image resolution the low head predicts (lower = more learnable)")
    ap.add_argument("--ll-loss", type=str, default="l1", choices=["l1", "huber", "mse"],
                    help="low-level loss; l1/huber are far less mean-seeking than mse")
    ap.add_argument("--ll-target", type=str, default="rgb", choices=["rgb", "latent"],
                    help="what the low head regresses: a blurry RGB image, or the "
                         "VAE latent E(blur(GT)) the decoder actually starts from. "
                         "'latent' needs the cache from scripts/build_latent_targets.py")
    ap.add_argument("--no-low-level", dest="low_level", action="store_false", default=True)
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="train fraction held out for leakage-free selection (0 disables)")
    ap.add_argument("--weight-decay", type=float, default=0.02)
    ap.add_argument("--warmup-epochs", type=int, default=5)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--fmri-noise-std", type=float, default=0.05)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--init-from", type=str, default=None,
                    help="warm-start model weights from a checkpoint (strict=False; "
                         "e.g. add the low-level head to an already-trained token model)")
    ap.add_argument("--freeze-token", action="store_true",
                    help="freeze the whole token model and train ONLY the low-level "
                         "head; select best.pt on PixCorr+SSIM (not val_cos). Use with "
                         "--init-from to add a low-level head without drifting the prior.")
    ap.add_argument("--freeze-anchor", action="store_true",
                    help="freeze the encoder, anchor, radius and log-sigma heads and "
                         "train ONLY the flow prior. Trained jointly, the anchor keeps "
                         "improving under the flow, so the flow's own source is "
                         "non-stationary and it converges to something ~0.12 from every "
                         "target. Use with --init-from to give the prior a fixed source.")
    ap.add_argument("--eval-freq", type=int, default=5)
    ap.add_argument("--decode-eval", action="store_true")
    ap.add_argument("--decode-n", type=int, default=16)
    ap.add_argument("--cls-vector-slot", type=int, nargs=2, default=[0, 1280],
                    help="[lo hi] positions in the unCLIP 'vector' slot to fill with c_cls")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--out", type=str, default="outputs/step1b")
    ap.add_argument("--seed", type=int, default=0)
    # Remote tracking. Off unless --wandb-project is given, so nothing about the
    # existing jobs changes. diagnostics.jsonl stays the source of truth; wandb
    # is a mirror, because the only other way to read a run is to be on the
    # cluster's filesystem.
    ap.add_argument("--wandb-project", type=str, default=None,
                    help="W&B project. Unset disables tracking entirely.")
    ap.add_argument("--wandb-entity", type=str, default=None)
    ap.add_argument("--wandb-name", type=str, default=None,
                    help="Run name. Defaults to the basename of --out.")
    ap.add_argument("--wandb-mode", type=str, default="online",
                    choices=["online", "offline"],
                    help="'offline' writes to <out>/wandb and needs a later "
                         "`wandb sync`; use it if the compute nodes lose egress.")
    return ap.parse_args()

def lr_at(step, total, warm, base, mn):
    if step < warm:
        return base * (step + 1) / max(1, warm)
    prog = (step - warm) / max(1, total - warm)
    return mn + 0.5 * (base - mn) * (1 + math.cos(math.pi * prog))

_WANDB = None


def _wandb_start(args, cfg, out_dir):
    """Return a live W&B run, or None. Never fatal: a tracking failure must not
    kill a 12-hour job, so every path here degrades to local-only logging."""
    global _WANDB
    if not args.wandb_project:
        return None
    try:
        import os
        import wandb
    except ImportError:
        print("[warn] --wandb-project set but wandb is not installed; "
              "logging to diagnostics.jsonl only.")
        return None
    if args.wandb_mode == "online" and not os.environ.get("WANDB_API_KEY"):
        print("[warn] WANDB_API_KEY unset; falling back to --wandb-mode offline. "
              f"Sync later with: wandb sync {out_dir}/wandb/latest-run")
        args.wandb_mode = "offline"
    try:
        _WANDB = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=args.wandb_name or Path(out_dir).name,
            mode=args.wandb_mode, dir=str(out_dir),
            config={**vars(args), **{f"cfg.{k}": v for k, v in vars(cfg).items()
                                     if isinstance(v, (int, float, str, bool))}})
        print(f"✓ W&B {args.wandb_mode}: {getattr(_WANDB, 'url', out_dir)}")
    except Exception as e:
        print(f"[warn] wandb.init failed ({e}); logging to diagnostics.jsonl only.")
        _WANDB = None
    return _WANDB


def _wandb_log(row: dict):
    """No explicit step: train rows land many times per epoch while eval rows
    land once, so an epoch-valued step would be non-monotonic and wandb would
    drop the out-of-order ones. Every row carries `epoch`, so use that as the
    x-axis in the UI."""
    if _WANDB is None:
        return
    try:
        _WANDB.log({k: v for k, v in row.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)})
    except Exception:
        pass


def _append_metrics(out_dir, row: dict):
    """One JSON object per eval, appended. paper_report.py reads the tail."""
    import json
    with open(Path(out_dir) / "diagnostics.jsonl", "a") as f:
        f.write(json.dumps({k: (float(v) if isinstance(v, (int, float)) else v)
                            for k, v in row.items()}) + "\n")
    _wandb_log(row)


@torch.no_grad()
def per_position_mean(targets, subjects, token_len, token_dim, chunk=512):
    """Mean bigG token PER POSITION, pooled over images and subjects: (N, d).

    ``targets_bigg._accum_stats`` reshapes to (-1, d) before accumulating, so
    ``stats.mean`` is a single vector averaged over positions as well as images.
    Streaming here in chunks keeps the (n_img, 256, 1664) mmap off the heap.
    """
    total = torch.zeros(token_len, token_dim, dtype=torch.float64)
    count = 0
    for s in subjects:
        e = targets[f"emb_train_{s}"]
        for i in range(0, len(e), chunk):
            block = e[i:i + chunk].float()
            total += block.sum(dim=0).double()
            count += block.shape[0]
    if count == 0:
        raise ValueError("no training embeddings found for per-position centring")
    return (total / count).float()


def _migrate_input_proj(model, state, label):
    """Rewrite a per-subject ``input_proj`` ModuleDict into the shared layout.

    Old checkpoints hold ``backbone.input_proj.<sid>.weight`` of shape
    (h, V_sid); the shared encoder wants one ``backbone.input_proj.weight`` of
    shape (h, max_vox). The key NAME differs, so ``strict=False`` would drop it
    without a word and the warm start would silently begin from a random input
    layer -- the one stage that matters most for a prior we are trying to reuse.

    Only one subject's matrix can seed a shared one. Preference order:

    1. the WIDEST subject that is also in this run -- column j then refers to the
       same voxel in the checkpoint and in the data we are about to feed, so the
       seeded weights are meaningful and not merely a warm shape;
    2. otherwise the widest subject in the checkpoint -- voxel correspondence is
       lost, but a trained projection is still a better starting point than noise.

    Mirrors :func:`brainflow.models.migrate_input_proj`, which does the same for
    the v5 ``brain_enc.`` prefix. Kept separate so ``step1`` does not take a
    dependency on the legacy module for forty lines of key rewriting.
    """
    bb = getattr(model, "backbone", None)
    if bb is None or not getattr(bb, "shared_encoder", False):
        return state
    if "backbone.input_proj.weight" in state:
        return state                                    # already shared
    old = {}
    for k, v in state.items():
        m = re.fullmatch(r"backbone\.input_proj\.(\d+)\.(weight|bias)", k)
        if m:
            old.setdefault(int(m.group(1)), {})[m.group(2)] = v
    if not old:
        return state
    usable = sorted((s for s in old if "weight" in old[s]),
                    key=lambda s: old[s]["weight"].shape[-1], reverse=True)
    state = {k: v for k, v in state.items()
             if not re.fullmatch(r"backbone\.input_proj\.\d+\.(weight|bias)", k)}
    in_run = [s for s in usable if s in bb.voxels_per_subject]
    sid = in_run[0] if in_run else usable[0]
    if not in_run:
        print(f"  [{label}] input_proj: checkpoint subjects {usable} are not in "
              f"this run ({sorted(bb.voxels_per_subject)}); seeding from {sid} "
              f"anyway, voxel correspondence is lost")
    w = model.state_dict()["backbone.input_proj.weight"].clone()
    src = old[sid]["weight"]
    n = min(src.shape[-1], w.shape[-1])
    if src.shape[0] != w.shape[0]:
        print(f"  [{label}] input_proj: enc_hidden {src.shape[0]} != {w.shape[0]}; "
              f"shared layer starts fresh")
        return state
    w[:, :n] = src[:, :n].to(w.dtype)
    state["backbone.input_proj.weight"] = w
    if "bias" in old[sid]:
        state["backbone.input_proj.bias"] = old[sid]["bias"]
    extra = "" if len(usable) == 1 else f" (discarded subjects {[s for s in usable if s != sid]})"
    print(f"  [{label}] input_proj: seeded shared layer from subject {sid}, "
          f"{n}/{w.shape[-1]} columns{extra}")
    return state


def _load_shape_safe(model, state, label):
    """load_state_dict(strict=False) that also drops SHAPE-mismatched entries.

    ``tgt_mean`` is (d,) under center_mode='global' and (N, d) under
    'per_position', so a warm-start across the two would raise instead of simply
    starting that buffer fresh.
    """
    own = model.state_dict()
    dropped = [k for k, v in state.items()
               if k in own and tuple(own[k].shape) != tuple(v.shape)]
    if dropped:
        print(f"  [{label}] shape mismatch, kept fresh: {', '.join(dropped)}")
    keep = {k: v for k, v in state.items() if k not in dropped}
    return model.load_state_dict(keep, strict=False)


@torch.no_grad()
def eval_token_cosine(model, loader, stats, device):
    model.eval()
    cos_sum, n = 0.0, 0
    for batch in loader:
        fmri = batch["fmri"].to(device, non_blocking=True)
        tgt = batch["emb"].to(device, non_blocking=True)
        pred, _ = model.predict_tokens(fmri, batch["subject"], stats, cond_source="regression")

        c = F.cosine_similarity(pred, tgt, dim=-1).mean(dim=-1)
        cos_sum += c.sum().item(); n += fmri.shape[0]
    return cos_sum / max(1, n)

def main():
    args = parse_args()
    setup_a100()
    quiet_benign_warnings()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir); hf_cache = data_dir / "hf_cache"

    tensors = assert_tensor_cache_alignment(data_dir / args.tensor_cache, torch.load(data_dir / args.tensor_cache, map_location="cpu"))
    targets = build_or_load_bigg_targets(
        tensors, args.subjects, args.target_dir, device, args.mindeye_src,
        hf_cache=hf_cache)
    stats: TargetStats = targets["_stats"]
    lat_stats = None
    if args.low_level and args.ll_target == "latent":
        from brainflow.step1.targets_latent import build_or_load_latent_targets
        # decoder=None: refuse to build a 2 GB/subject cache inside a training job.
        targets |= build_or_load_latent_targets(
            tensors, args.subjects, args.target_dir, decoder=None, ll_size=args.ll_size)
        lat_stats = targets["_lat_stats"]
        print(f"✓ blurry-latent targets loaded (ll_size={args.ll_size}, "
              f"channel std={lat_stats['std'].tolist()})")
    bundle = build_step1_loaders(tensors, targets, args.subjects, args.batch_size,
                                 num_workers=args.num_workers, fmri_noise_std=args.fmri_noise_std,
                                 val_frac=args.val_frac, seed=args.seed)

    # CFG default depends on the flow source: with an anchor source the
    # "unconditional" branch drops the brain tokens while z0 is itself
    # brain-derived, so the two branches do not differ in the way guidance assumes.
    cfg_scale = args.cfg_scale
    if cfg_scale is None:
        cfg_scale = 3.0 if args.flow_source == "noise" else 1.0
        print(f"✓ --cfg-scale unset; resolved to {cfg_scale} for "
              f"--flow-source {args.flow_source}")

    cfg = TokenStep1Config(subjects=args.subjects, enc_hidden=args.enc_hidden,
                           shared_encoder=args.shared_encoder,
                           subject_rank=args.subject_rank,
                           lambda_clip=args.lambda_clip, clip_temp=args.clip_temp,
                           geometry=args.geometry, uniform_t_prob=args.uniform_t_prob,
                           logit_normal_m=args.logit_normal_m,
                           logit_normal_s=args.logit_normal_s,
                           lambda_radius=args.lambda_radius, two_head=args.two_head,
                           lambda_rcfm=args.lambda_rcfm, cls_cfg_scale=args.cls_cfg_scale,
                           solver=args.solver, low_level=args.low_level,
                           lambda_low=args.lambda_low, ll_strength=args.ll_strength,
                           ll_size=args.ll_size, ll_loss=args.ll_loss,
                           ll_target=args.ll_target,
                           lambda_cos=args.lambda_cos, lambda_reg=args.lambda_reg,
                           enc_drop=args.enc_drop, mixup_pct=args.mixup_pct,
                           mixco_beta=args.mixco_beta, mixco_s_thresh=args.mixco_s_thresh,
                           flow_source=args.flow_source,
                           flow_param=args.flow_param,
                           cfg_scale=cfg_scale,
                           center_mode=args.center_mode,
                           anchor_jitter_rad=args.anchor_jitter_rad,
                           lambda_kl=args.lambda_kl,
                           lambda_clip_tok=args.lambda_clip_tok,
                           cls_cond_mode=args.cls_cond_mode,
                           cls_jitter_rad=args.cls_jitter_rad,
                           stochastic_source=args.stochastic_source)
    _wandb_start(args, cfg, out_dir)
    model = TokenStep1Model(cfg, bundle.voxels).to(device)
    if cfg.geometry == "sphere" and cfg.center_tokens:
        model.set_target_mean(
            per_position_mean(targets, args.subjects, cfg.token_len, cfg.token_dim)
            if cfg.center_mode == "per_position" else stats.mean)
    if lat_stats is not None and model.low_head is not None:
        model.low_head.set_latent_stats(lat_stats["mean"], lat_stats["std"])
    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu")
        state = _migrate_input_proj(model, ck["model"], "init-from")
        res = _load_shape_safe(model, state, "init-from")
        print(f"✓ warm-start from {args.init_from}: {len(res.missing_keys)} fresh params "
              f"(e.g. low_head), {len(res.unexpected_keys)} unexpected")

    low_only = bool(args.freeze_token)
    if args.freeze_token:
        if model.low_head is None:
            raise SystemExit("--freeze-token needs the low-level head (drop --no-low-level)")
        n_frozen = n_train = 0
        for name, p in model.named_parameters():
            train_it = name.startswith("low_head")
            p.requires_grad_(train_it)
            n_train += p.numel() if train_it else 0
            n_frozen += 0 if train_it else p.numel()
        print(f"✓ freeze-token: {n_frozen/1e6:.1f}M frozen, {n_train/1e6:.2f}M trainable "
              f"(low_head only); best.pt selected on PixCorr+SSIM")
    if args.freeze_anchor:
        if args.freeze_token:
            raise SystemExit("--freeze-anchor and --freeze-token are mutually exclusive: "
                             "one trains the flow prior, the other the low-level head.")
        # Everything that defines the SOURCE is frozen: the encoder and anchor fix mu,
        # radius_head fixes the norm, logs_head fixes the posterior width. Leaving any
        # of them trainable would keep the source moving, which is the thing this flag
        # exists to rule out.
        trainable_prefixes = ("prior", "cls_prior")
        n_frozen = n_train = 0
        for name, p_ in model.named_parameters():
            train_it = name.startswith(trainable_prefixes)
            p_.requires_grad_(train_it)
            n_train += p_.numel() if train_it else 0
            n_frozen += 0 if train_it else p_.numel()
        if n_train == 0:
            raise SystemExit("--freeze-anchor left nothing trainable: no `prior` module.")
        print(f"✓ freeze-anchor: {n_frozen/1e6:.1f}M frozen, {n_train/1e6:.2f}M trainable "
              f"(flow prior only); the source is now stationary")
    write_manifest(out_dir, args=args, cfg=cfg, model=model, voxels=bundle.voxels)
    if cfg.geometry == "sphere" and cfg.flow_source != "noise":
        print(f"✓ R-XFM source={cfg.flow_source} param={cfg.resolved_flow_param()} "
              f"jitter={cfg.anchor_jitter_rad} rad | cls_cond={cfg.cls_cond_mode}"
              f"({cfg.cls_jitter_rad} rad) | cfg_scale={cfg.cfg_scale} "
              f"| center={cfg.center_mode} | select_on={args.select_on}")
        if cfg.cfg_scale != 1.0:
            print("  [warn] classifier-free guidance with an anchor source is "
                  "incoherent: the 'unconditional' branch drops the brain tokens "
                  "while z0 is itself brain-derived. Prefer --cfg-scale 1.0.")
        if cfg.flow_source == "anchor_var" and cfg.lambda_kl <= 0:
            print("  [warn] --lambda-kl 0 with anchor_var: nothing floors sigma, so "
                  "the flow will drive it to 0 and the posterior degenerates to the "
                  "deterministic anchor (equivalent to --flow-source anchor_det).")
    if args.lambda_reg > 0:
        print(f"  [warn] --lambda-reg {args.lambda_reg}: for unit vectors "
              f"mse(a,b) == (2/d)*(1-cos(a,b)), so `reg` is `cos` scaled by "
              f"1/{cfg.token_dim // 2}. It is very likely not doing what you think.")
    if args.compile:
        model = torch.compile(model)
    ema = EMA(model, decay=args.ema_decay)
    opt_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.95), fused=True)

    spe = len(bundle.train_sampler)
    total_steps = spe * args.epochs
    warmup_steps = spe * args.warmup_epochs

    decoder = clip_metric = None
    if args.decode_eval:
        from brainflow.step1.decoder_sgm import SDXLUnCLIPDecoder
        decoder = SDXLUnCLIPDecoder(device, args.mindeye_src, args.ckpt_path,
                                    cls_vector_slot=tuple(args.cls_vector_slot))
        clip_metric = CLIPMetric(device, hf_cache=hf_cache)

    best_cos = -1.0
    best_clip = -1.0
    best_img = -1.0
    step = 0
    nan_skips = 0
    for epoch in range(1, args.epochs + 1):
        # low_only: run the frozen backbone in eval mode (stable features; low_head
        # has no train/eval-sensitive layers) so the head learns a clean mapping.
        model.eval() if low_only else model.train()
        bundle.train_sampler.set_epoch(epoch)
        # BiMixCo runs only for the first mixup_pct of training, then the
        # contrastive term reverts to clean SoftCLIP (MindEye2's schedule): the
        # mix is a regulariser for the underfit phase, not the final objective.
        use_mixco = (not low_only) and epoch <= int(args.mixup_pct * args.epochs)
        if use_mixco and epoch == 1:
            print(f"✓ BiMixCo on for epochs 1-{int(args.mixup_pct * args.epochs)} "
                  f"(beta={args.mixco_beta}, {args.mixco_s_thresh:.0%} of each batch)")
        pbar = tqdm(bundle.train, desc=f"Ep{epoch}", mininterval=1.0)
        for batch in pbar:
            lr = lr_at(step, total_steps, warmup_steps, args.lr, args.min_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            fmri = batch["fmri"].to(device, non_blocking=True)
            tgt_raw = batch["emb"].to(device, non_blocking=True)
            tgt_std = stats.standardize(tgt_raw)
            tgt_cls = batch.get("cls")
            if tgt_cls is not None:
                tgt_cls = tgt_cls.to(device, non_blocking=True)
            tgt_img = batch["image"].to(device, non_blocking=True)
            tgt_lat = batch.get("lat")
            if tgt_lat is not None:
                tgt_lat = tgt_lat.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            probe = (args.gradnorm_every and step % args.gradnorm_every == 0
                     and not low_only)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                ld = model.training_step(fmri, batch["subject"], tgt_std,
                                         target_raw=tgt_raw, target_cls=tgt_cls,
                                         target_img=tgt_img, low_only=low_only,
                                         use_mixco=use_mixco, target_lat=tgt_lat,
                                         keep_term_graph=probe)
                loss = ld["loss"]

            # Per-term gradient norms: loss WEIGHTS are uninterpretable when the
            # terms have different natural scales (`reg` is `cos` scaled by 2/d).
            # Measured, not guessed. autograd.grad does not touch .grad, and every
            # probe retains the graph, so loss.backward() below is unaffected.
            if probe:
                gn = grad_norms({k: v for k, v in ld.items() if k != "loss"},
                                model.parameters())
                print("  gradnorm | " + "  ".join(
                    f"{k.split('/')[-1]}={v:.3e}" for k, v in sorted(gn.items())))
                _append_metrics(out_dir, {"step": step, **gn})

            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if torch.isfinite(gnorm):
                opt.step(); ema.update(model)
            else:
                nan_skips += 1
            step += 1
            if step % 50 == 0:
                pbar.set_postfix(flow=f"{float(ld.get('flow', 0)):.3f}",
                                 rcfm=f"{float(ld.get('rcfm', 0)):.3f}",
                                 reg=f"{float(ld.get('reg', 0)):.3f}",
                                 cos=f"{float(ld.get('cos', 0)):.3f}",
                                 low=f"{float(ld.get('low', 0)):.3f}",
                                 clip=f"{float(ld.get('clip', 0)):.3f}",
                                 lr=f"{lr:.1e}", skip=nan_skips)
                _wandb_log({f"train/{k}": float(v) for k, v in ld.items()}
                           | {"train/lr": lr, "train/grad_norm": float(gnorm),
                              "train/nan_skips": nan_skips,
                              "epoch": epoch, "step": step})

        if epoch % args.eval_freq == 0 or epoch == args.epochs:
            ema.store(model); ema.copy_to(model)
            # Honest selection: choose on a leakage-free split; TEST is report-only.
            sel_loader = bundle.val if bundle.val is not None else bundle.eval
            sel_name = "val" if bundle.val is not None else "test"
            if bundle.val is None and epoch == args.eval_freq:
                print("[warn] no validation split (--val-frac 0); selecting on TEST is leaky.")
            sel_cos = eval_token_cosine(model, sel_loader, stats, device)
            msg = f"Ep{epoch:4d} | {sel_name}_cos={sel_cos:.4f}"

            # Latent-space diagnostics: seconds, no decode. `delta_cos` is what
            # the flow ADDS over its own starting point -- if it is <= 0 the prior
            # is dead weight no matter what the decoded metrics say.
            diag = latent_diagnostics(model, sel_loader, device,
                                      n_steps=args.decode_steps,
                                      solver=args.solver, max_batches=args.diag_batches,
                                      prefix="val/")
            diag_tr = latent_diagnostics(model, bundle.train, device,
                                         n_steps=args.decode_steps,
                                         solver=args.solver,
                                         max_batches=args.diag_batches, prefix="train/")
            diag |= diag_tr
            diag |= train_val_gap(diag_tr, {k: v for k, v in diag.items()
                                            if k.startswith("val/")})
            if "val/delta_cos" in diag:
                msg += (f" | Dcos={diag['val/delta_cos']:+.4f}"
                        f" anchor={diag['val/anchor_cos']:.4f}"
                        f" flow={diag['val/flow_cos']:.4f}"
                        f" 2way={diag.get('val/two_way', float('nan')):.3f}"
                        f" retr={diag.get('val/retr_fwd_300', float('nan')):.3f}")
                if "val/cls_cos_hat_true" in diag:
                    msg += f" clsCos={diag['val/cls_cos_hat_true']:.3f}"
                if "val/sigma_mean" in diag:
                    k_tan = cfg.token_dim - 1
                    msg += f" jit={diag['val/sigma_mean'] * (k_tan ** 0.5):.3f}rad"
            _append_metrics(out_dir, {"epoch": epoch, **diag})
            if bundle.val is not None:
                msg += f" test_cos={eval_token_cosine(model, bundle.eval, stats, device):.4f}"
            ema.restore(model)

            # Persist the checkpoint on the val-cos signal BEFORE any decoder
            # work. Decode-eval is heavy (SDXL UNet+VAE co-resident with the
            # training model + AdamW state) and only a progress signal; a crash
            # there must never throw away the trained weights -- it silently did
            # once (an OOM in the img2img VAE-encode killed a 10-epoch run before
            # a single checkpoint was written).
            ckpt = {"model": model.state_dict(), "ema": ema.shadow, "cfg": cfg,
                    "stats": stats.to_dict(), "voxels": bundle.voxels,
                    "subjects": args.subjects, "epoch": epoch}
            torch.save(ckpt, out_dir / "last.pt")
            # Selection signal. `sel_cos` calls predict_tokens with
            # cond_source="regression", so under R-XFM it scores exactly z0 and is
            # structurally blind to delta_cos -- the quantity this branch exists to
            # produce. It is also a POINTWISE functional (per-image, per-token inner
            # product) while every deployed metric (CLIP-2way, retrieval) is a
            # PAIRWISE ranking one; write the predictor as xhat = a*x + b*mbar + e
            # and the pairwise metric is invariant to b after centring while the
            # pointwise one is increasing in it, so the two can and did move in
            # opposite directions (run 347831: val_cos fell from epoch 10 while
            # decoded CLIP_2way climbed to epoch 110). Default to the flow-aware
            # ranking statistic; --select-on cos restores the old behaviour.
            sel_key = {"two_way": "val/two_way",
                       "retrieval": "val/retr_fwd_300"}.get(args.select_on)
            sel_val = sel_cos if sel_key is None else diag.get(sel_key)
            if sel_val is None:
                if epoch == args.eval_freq:
                    print(f"[warn] --select-on {args.select_on} unavailable "
                          f"({sel_key} missing from diagnostics); falling back to cos.")
                sel_val = sel_cos
            if not low_only and sel_val > best_cos:
                best_cos = sel_val
                torch.save(ckpt, out_dir / "best.pt")
                msg += f"  ✓best.pt({args.select_on}={sel_val:.4f})"

            # Best-effort image metrics: guard so an OOM/failure just skips this
            # epoch's decode instead of killing the run, and always restore the
            # training weights + free the cache afterwards.
            if args.decode_eval:
                im = None
                try:
                    ema.store(model); ema.copy_to(model)
                    n_subj = max(1, len(args.subjects))
                    per_subj = max(1, args.decode_n // n_subj)
                    preds, gts, got = [], [], {}
                    blur_sq, blur_n = 0.0, 0
                    # Always decode the leakage-free VAL split when one exists: both
                    # best.pt (--freeze-token) and best_clip2way.pt are selected on
                    # this metric, so it must not be TEST. TEST is reported once, by
                    # the stage-2 full eval.
                    decode_loader = bundle.val if bundle.val is not None else bundle.eval
                    split_tag = "val" if decode_loader is bundle.val else "test"
                    with torch.no_grad():
                        for batch in decode_loader:
                            s = int(batch["subject"])
                            if got.get(s, 0) >= per_subj:
                                continue
                            take = min(batch["fmri"].shape[0], per_subj - got.get(s, 0))
                            fmri = batch["fmri"][:take].to(device, non_blocking=True)
                            tok, cls_hat = model.predict_tokens(fmri, batch["subject"], stats,
                                                                cond_source=cfg.cond_source)
                            blur = model.predict_lowlevel(fmri, batch["subject"])
                            lat = model.predict_low_latent(fmri, batch["subject"])
                            # How good is the predicted layout vs the target the head is
                            # trained on? (the img2img ceiling driver). Reported as
                            # blur_mse in both modes so the two are directly comparable
                            # -- in latent mode it is MSE in STANDARDISED latent space,
                            # where 1.0 is what predicting the mean scores.
                            if blur is not None:
                                gtb = F.interpolate(batch["image"][:take].float().to(device),
                                                    cfg.ll_size, mode="bilinear",
                                                    align_corners=False).clamp(0, 1)
                                blur_sq += F.mse_loss(blur, gtb).item() * take
                                blur_n += take
                            elif lat is not None and batch.get("lat") is not None:
                                gl = model.low_head.standardize(
                                    batch["lat"][:take].float().to(device))
                                blur_sq += F.mse_loss(
                                    model.low_head.standardize(lat), gl).item() * take
                                blur_n += take
                            preds.append(decoder.decode(tok, cls_hat, init_image=blur,
                                                        init_latent=lat,
                                                        strength=cfg.ll_strength))
                            gts.append(batch["image"][:take])
                            got[s] = got.get(s, 0) + take
                            if len(got) == n_subj and all(v >= per_subj for v in got.values()):
                                break
                    pred = torch.cat(preds); gt = torch.cat(gts)
                    im = {"PixCorr": pixcorr(pred, gt), "SSIM": ssim(pred, gt)}
                    im.update(clip_metric.score(pred, gt))
                    msg += (f" | [{split_tag}] PixCorr={im['PixCorr']:.3f} SSIM={im['SSIM']:.3f} "
                            f"CLIP_cos={im['CLIP_cos']:.3f} CLIP_2way={im['CLIP_2way']:.3f}")
                    if blur_n:
                        msg += f" blur_mse={blur_sq / blur_n:.4f}"
                    _save_grid(pred, gt, out_dir / f"recon_ep{epoch}.png")
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    msg += f" | [decode-eval skipped: {type(e).__name__}: {e}]"
                finally:
                    ema.restore(model)
                    torch.cuda.empty_cache()
                # CLIP-2way checkpoint, saved AFTER the finally so ckpt's live
                # tensors hold TRAINING weights again -- not the EMA weights
                # copy_to() swapped in for the decode.
                #
                # This used to be gated on `bundle.val is None`, so a run with
                # --val-frac (now the default) produced NO CLIP-selected
                # checkpoint at all -- and best.pt selects on sel_cos, which in
                # run 347831 fell monotonically from epoch 10 while the decoded
                # CLIP_2way kept climbing to epoch 110. The two signals
                # disagree, so keep a checkpoint for each.
                if im is not None and im["CLIP_2way"] > best_clip:
                    best_clip = im["CLIP_2way"]
                    torch.save(ckpt, out_dir / "best_clip2way.pt")
                    msg += "  ✓best_clip2way.pt"
                # --freeze-token: token model is fixed, so best.pt IS the low-level
                # checkpoint -- select it on PixCorr+SSIM (the eval reloads EMA
                # weights, which is what this decode measured).
                if low_only and im is not None and (im["PixCorr"] + im["SSIM"]) > best_img:
                    best_img = im["PixCorr"] + im["SSIM"]
                    torch.save(ckpt, out_dir / "best.pt")
                    msg += "  ✓best.pt"

            if nan_skips:
                msg += f" | nan_skips={nan_skips}"
            print(msg)

    # Guarantee best.pt exists: in --freeze-token mode it is only written when an
    # image metric improves, so fall back to last.pt if no decode-eval ever ran.
    if not (out_dir / "best.pt").exists() and (out_dir / "last.pt").exists():
        import shutil
        shutil.copyfile(out_dir / "last.pt", out_dir / "best.pt")
        print("[best.pt] no selection metric fired; fell back to last.pt")
    if low_only:
        print(f"Done. best low-level PixCorr+SSIM={best_img:.4f}. Checkpoints in {out_dir}")
    else:
        split = "val" if bundle.val is not None else "test"
        print(f"Done. best {split}/{args.select_on}={best_cos:.4f}. "
              f"Checkpoints in {out_dir}")
    if _WANDB is not None:
        try:
            _WANDB.summary["best_selection"] = float(best_img if low_only else best_cos)
            _WANDB.finish()
        except Exception:
            pass

def _save_grid(pred, gt, path, n=8):
    try:
        from torchvision.utils import save_image
        n = min(n, pred.shape[0])
        save_image(torch.cat([gt[:n], pred[:n]]), str(path), nrow=n)
    except Exception as e:
        print(f"[grid skipped] {e}")

if __name__ == "__main__":
    main()
