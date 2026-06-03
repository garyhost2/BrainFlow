"""Phase 4 evaluation: BrainEncoder + ClipPrior + FLUXBrainDecoder.

Computes the full 8-metric suite + CLIP_Cos + CLIP_2way on the NSD test set.

Usage:
    BRAINFLOW_CONFIG=configs/phase4_flux.yaml \\
    PHASE3_CKPT=outputs/phase3_sdprior/best_clip_cos.pt \\
    ADAPTER_CKPT=outputs/phase4_flux/best_adapter.pt \\
    python -m scripts.eval_flux [--subject 1] [--n-batches 50] [--out-dir outputs/phase4_flux]

Flags (all optional, env vars take priority):
    --subject      Subject ID to evaluate (default: 1)
    --n-batches    Max eval batches (-1 = full test set, default: -1)
    --out-dir      Directory to write full_metrics.json
    --flux-steps   FLUX inference steps (default: from config)
    --guidance     FLUX guidance scale (default: from config)
    --height       Output image height in pixels (default: 512)
    --width        Output image width  in pixels (default: 512)

Outputs
-------
  {out_dir}/flux_metrics.json  — all metrics
  {out_dir}/flux_samples.png   — 4×4 grid (predicted | GT)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brainflow.clip_prior import ClipPrior
from brainflow.config import load_config
from brainflow.data import build_dataloaders
from brainflow.flux_decoder import FLUXBrainDecoder
from brainflow.metrics_full import evaluate_full, two_way_identification
from brainflow.models import BrainEncoder, migrate_input_proj

load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Phase 4 FLUX eval")
    p.add_argument("--subject",    type=int,   default=None)
    p.add_argument("--n-batches",  type=int,   default=-1)
    p.add_argument("--out-dir",    type=str,   default=None)
    p.add_argument("--flux-steps", type=int,   default=None)
    p.add_argument("--guidance",   type=float, default=None)
    p.add_argument("--height",     type=int,   default=None)
    p.add_argument("--width",      type=int,   default=None)
    p.add_argument("--phase3-ckpt", type=str,  default=None)
    p.add_argument("--adapter-ckpt", type=str, default=None)
    return p.parse_args()


# ── model loading ─────────────────────────────────────────────────────────────

def load_models(cfg, device: torch.device, args):
    # ── BrainEncoder ──────────────────────────────────────────────────────────
    # We need voxel counts to instantiate; build dataloaders first
    _, test_loader, voxels = build_dataloaders(cfg)

    brain_enc = BrainEncoder(cfg, voxels).to(device)

    # ── ClipPrior ─────────────────────────────────────────────────────────────
    prior = ClipPrior(
        clip_dim=cfg.clip_dim,
        ctx_dim=cfg.brain_dim,
        dim=getattr(cfg, "prior_dim", 512),
        depth=getattr(cfg, "prior_blocks", 6),
        geometry=getattr(cfg, "clip_prior_geometry", "euclidean"),
        prior_target=getattr(cfg, "prior_target", "cls"),
    ).to(device)

    # ── Load Phase 3 checkpoint ───────────────────────────────────────────────
    phase3_path = (
        args.phase3_ckpt
        or os.environ.get("PHASE3_CKPT")
        or getattr(cfg, "init_from_phase3", "")
    )
    if not phase3_path or not Path(phase3_path).exists():
        raise FileNotFoundError(
            f"Phase 3 checkpoint not found: {phase3_path}\n"
            "Set --phase3-ckpt, PHASE3_CKPT env var, or init_from_phase3 in config."
        )
    sd = torch.load(phase3_path, map_location="cpu")
    sd = {k.replace("._orig_mod.", "."): v for k, v in sd.items()}
    sd = migrate_input_proj(sd, brain_enc.max_vox)

    enc_sd   = {k[len("brain_enc."):]: v for k, v in sd.items() if k.startswith("brain_enc.")}
    prior_sd = {k[len("prior."):]: v     for k, v in sd.items() if k.startswith("prior.")}

    brain_enc.load_state_dict(enc_sd, strict=False)
    if prior_sd:
        prior.load_state_dict(prior_sd, strict=False)
    else:
        log.warning("No 'prior.*' keys in Phase 3 checkpoint — ClipPrior at random init.")

    brain_enc.eval().requires_grad_(False)
    prior.eval().requires_grad_(False)
    log.info("Phase 3 models loaded from %s", phase3_path)

    # ── FLUXBrainDecoder ──────────────────────────────────────────────────────
    flux_decoder = FLUXBrainDecoder(cfg).to(device)

    adapter_path = (
        args.adapter_ckpt
        or os.environ.get("ADAPTER_CKPT")
        or str(Path(cfg.output_dir) / cfg.experiment_name / "best_adapter.pt")
    )
    if not Path(adapter_path).exists():
        raise FileNotFoundError(
            f"Adapter checkpoint not found: {adapter_path}\n"
            "Set --adapter-ckpt, ADAPTER_CKPT env var, or run train_flux_adapter.py first."
        )
    flux_decoder.load_adapter(adapter_path)
    flux_decoder.ip_adapter.eval().requires_grad_(False)
    log.info("IP-Adapter loaded from %s", adapter_path)

    return brain_enc, prior, flux_decoder, test_loader


# ── generation loop ───────────────────────────────────────────────────────────

@torch.no_grad()
def generate_predictions(
    brain_enc: BrainEncoder,
    prior: ClipPrior,
    flux_decoder: FLUXBrainDecoder,
    loader,
    cfg,
    device: torch.device,
    args,
) -> tuple[list, list, list, list]:
    """Run full test set through the pipeline.

    Returns
    -------
    pred_images   : list of (3, H, W) tensors (float32, [0,1])
    gt_images     : list of (3, H, W) tensors (float32, [0,1])
    pred_clips    : list of (clip_dim,) tensors (L2-normalised)
    gt_clips      : list of (clip_dim,) tensors (L2-normalised)
    """
    n_steps  = args.flux_steps  or getattr(cfg, "flux_steps", 28)
    guidance = args.guidance    or getattr(cfg, "flux_guidance", 3.5)
    height   = args.height      or getattr(cfg, "flux_height", 512)
    width    = args.width       or getattr(cfg, "flux_width", 512)
    n_prior  = getattr(cfg, "prior_ode_steps", 20)
    cfg_sc   = getattr(cfg, "cfg_scale", 1.5)
    max_b    = args.n_batches if args.n_batches > 0 else int(1e9)

    pred_images, gt_images, pred_clips, gt_clips = [], [], [], []

    for i, batch in enumerate(loader):
        if i >= max_b:
            break

        fmri    = batch["fmri"].to(device)
        subject = batch["subject"].to(device)
        clip_gt = F.normalize(batch["clip_emb"].to(device).float(), dim=-1)
        img_gt  = batch["image"].to(device).float()               # (B,3,H,W)

        # Encode fMRI → brain tokens
        tokens, _ = brain_enc(fmri, subject)                      # (B, N, D)

        # ClipPrior → (mu_clip, log_sigma)
        mu_clip = prior.sample(tokens, n_steps=n_prior,
                               cfg_scale=cfg_sc, normalize_output=True)
        # Uncertainty: spread between two low-step draws
        s1 = prior.sample(tokens, n_steps=max(n_prior // 2, 5),
                          cfg_scale=1.0, normalize_output=True)
        s2 = prior.sample(tokens, n_steps=max(n_prior // 2, 5),
                          cfg_scale=1.0, normalize_output=True)
        log_sigma = 0.5 * (s1 - s2).pow(2).clamp(min=1e-8).log()

        # Generate images via FLUX
        images = flux_decoder.generate(
            tokens, mu_clip, log_sigma,
            height=height, width=width,
            n_steps=n_steps, guidance=guidance,
        )                                                          # (B,3,H,W)

        for j in range(images.shape[0]):
            pred_images.append(images[j].cpu())
            gt_images.append(img_gt[j].cpu())
            pred_clips.append(mu_clip[j].cpu())
            gt_clips.append(clip_gt[j].cpu())

        if (i + 1) % 10 == 0:
            log.info("  Processed %d batches (%d images total)…",
                     i + 1, len(pred_images))

    return pred_images, gt_images, pred_clips, gt_clips


# ── visualisation helper ──────────────────────────────────────────────────────

def save_sample_grid(pred_images, gt_images, path: Path, n: int = 8):
    """Save an n×2 grid: predicted (left) | ground truth (right)."""
    try:
        import torchvision.utils as vutils
        n = min(n, len(pred_images))
        pairs = []
        for i in range(n):
            pairs += [pred_images[i], gt_images[i]]
        grid = vutils.make_grid(
            torch.stack(pairs), nrow=2, padding=4, normalize=False
        )
        from torchvision.transforms.functional import to_pil_image
        to_pil_image(grid).save(path)
        log.info("Sample grid saved → %s", path)
    except Exception as e:
        log.warning("Could not save sample grid: %s", e)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config()

    # Subject filter
    if args.subject is not None:
        cfg.subjects = [args.subject]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Evaluating on device: %s", device)

    brain_enc, prior, flux_decoder, test_loader = load_models(cfg, device, args)

    log.info("Generating predictions…")
    pred_images, gt_images, pred_clips, gt_clips = generate_predictions(
        brain_enc, prior, flux_decoder, test_loader, cfg, device, args
    )
    log.info("Generated %d samples total.", len(pred_images))

    # ── 8-metric evaluation ───────────────────────────────────────────────────
    log.info("Computing full 8-metric evaluation…")
    metrics = evaluate_full(pred_images, gt_images)

    # ── CLIP metrics ──────────────────────────────────────────────────────────
    pred_c = torch.stack(pred_clips)
    gt_c   = torch.stack(gt_clips)
    clip_cos  = float(F.cosine_similarity(
        F.normalize(pred_c, dim=-1),
        F.normalize(gt_c, dim=-1),
        dim=-1,
    ).mean())
    clip_2way = two_way_identification(pred_c, gt_c)

    metrics["CLIP_Cos"]  = clip_cos
    metrics["CLIP_2way"] = clip_2way

    # ── print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 52)
    print(f"  Phase 4 FLUX Eval  |  subject={args.subject or 'all'}  |  n={len(pred_images)}")
    print("=" * 52)
    col_w = 20
    for k, v in sorted(metrics.items()):
        print(f"  {k:<{col_w}} {v:.4f}")
    print("=" * 52 + "\n")

    # MindEye v2 reference (subject 1, from paper Table 1)
    ref = {
        "PixCorr": 0.323, "SSIM": 0.421,
        "AlexNet(2)": 0.922, "AlexNet(5)": 0.976,
        "Inception": 0.867, "CLIP": 0.936,
        "EffNet-B": 0.673, "SwAV": 0.459,
    }
    print("  vs. MindEye v2 (subj-1 reference):")
    for k, rv in sorted(ref.items()):
        ours = metrics.get(k, float("nan"))
        delta = ours - rv
        marker = "▲" if delta > 0 else "▼"
        print(f"  {k:<{col_w}} ours={ours:.4f}  ref={rv:.4f}  {marker}{abs(delta):.4f}")
    print()

    # ── save JSON ─────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir or Path(cfg.output_dir) / cfg.experiment_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "flux_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics saved → %s", json_path)

    # ── sample grid ───────────────────────────────────────────────────────────
    save_sample_grid(pred_images, gt_images, out_dir / "flux_samples.png", n=8)


if __name__ == "__main__":
    main()
