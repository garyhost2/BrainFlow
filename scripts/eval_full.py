from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from brainflow.metrics_full import evaluate_full

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Full 8-metric BrainFlow evaluation (or --self-test for CI)."
    )
    p.add_argument("--ckpt", type=str, default=None,
                   help="Path to model checkpoint (.pt).  Required unless --self-test.")
    p.add_argument("--subject", type=int, default=1,
                   help="NSD subject ID (default: 1).")
    p.add_argument("--ode-steps", type=int, default=20,
                   help="ODE integration steps (default: 20).")
    p.add_argument("--solver", type=str, default="euler",
                   choices=["euler", "midpoint", "heun", "dpm_pp_2m"],
                   help="ODE solver (default: euler).")
    p.add_argument("--schedule", type=str, default="linear",
                   choices=["linear", "cosine", "logit_normal"],
                   help="Time schedule for make_t_grid (default: linear).")
    p.add_argument("--cfg-scale", type=float, default=None,
                   help="CFG scale override (default: value from config.yaml).")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Directory to write full_metrics.json (default: outputs/<exp>/).")
    p.add_argument("--self-test", action="store_true",
                   help="Run evaluate_full on random tensors for CI (no GPU/data needed).")
    return p

_SKIP_FOR_SELF_TEST = [
    "AlexNet(2)", "AlexNet(5)", "Inception", "CLIP", "EffNet-B", "SwAV"
]

def run_self_test() -> int:
    print("=== BrainFlow eval_full self-test (CPU, no data/checkpoint) ===")
    B, C, H, W = 4, 3, 64, 64
    pred = torch.rand(B, C, H, W)
    target = torch.rand(B, C, H, W)
    device = torch.device("cpu")

    metrics = evaluate_full(pred, target, device, skip=_SKIP_FOR_SELF_TEST)

    print("\nSelf-test results:")
    for name, val in metrics.items():
        print(f"  {name:15s}: {val:.6f}")

    expected_keys = {"PixCorr", "SSIM"}
    missing = expected_keys - metrics.keys()
    if missing:
        print(f"\nFAIL: missing keys: {missing}", file=sys.stderr)
        return 1
    if not all(isinstance(v, float) and (v == v) for v in metrics.values()):
        print("\nFAIL: non-finite metric value", file=sys.stderr)
        return 1
    print("\nSelf-test PASSED ✓")
    return 0

def run_full_eval(args: argparse.Namespace) -> int:
    from brainflow.config import load_config
    from brainflow.data import build_dataloaders
    from brainflow.vae import FrozenVAE
    from brainflow.solvers import make_t_grid

    cfg = load_config()
    if args.cfg_scale is not None:
        cfg.cfg_scale = args.cfg_scale

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg.subject_filter = [args.subject]
    _, test_loader, eval_loader, _, voxels = build_dataloaders(cfg)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 1

    state = torch.load(ckpt_path, map_location="cpu")
    state = {k.replace("._orig_mod.", "."): v for k, v in state.items()}

    is_phase2 = any(k.startswith("flow_clip.") for k in state.keys())

    if is_phase2:
        from brainflow.phase2_model import BrainFlowPhase2
        model = BrainFlowPhase2(cfg, voxels).to(device).eval()
        model.load_state_dict(state, strict=False)
    else:
        from brainflow.models import BrainFlowV5, migrate_input_proj
        model = BrainFlowV5(cfg, voxels).to(device).eval()
        state = migrate_input_proj(state, model.brain_enc.max_vox)
        model.load_state_dict(state, strict=False)

    print(f"Loaded {'Phase 2' if is_phase2 else 'Phase 1'} checkpoint: {ckpt_path}")

    vae = FrozenVAE(cache_dir=cfg.data_dir / "hf_cache").to(device)

    t_grid = make_t_grid(args.ode_steps, device=device, schedule=args.schedule)

    all_pred, all_target = [], []
    n_batches = 0

    with torch.no_grad():
        for batch in eval_loader:
            fmri = batch["fmri"].to(device)
            subject = batch["subject"].to(device)
            images = batch["image"].to(device)

            tokens, cls = model.encode_fmri(fmri, subject) if hasattr(model, "encode_fmri") \
                else model.brain_enc(fmri, subject)

            latents = model.sample(
                tokens,
                n_steps=args.ode_steps,
                cfg_scale=cfg.cfg_scale,
                solver=args.solver,
            )
            pred_imgs = vae.decode(latents).clamp(0.0, 1.0)

            all_pred.append(pred_imgs.cpu())
            all_target.append(images.cpu())
            n_batches += 1

    pred_all = torch.cat(all_pred, dim=0)
    target_all = torch.cat(all_target, dim=0)
    print(f"Evaluated {pred_all.shape[0]} samples from {n_batches} batches.")

    print("\nComputing metrics (this may take a while for GPU metrics)...")
    metrics = evaluate_full(pred_all, target_all, device)

    print("\n" + "=" * 50)
    print(f"Full 8-Metric Evaluation — Subject {args.subject}")
    print("=" * 50)
    for name, val in metrics.items():
        print(f"  {name:15s}: {val:.6f}")
    print("=" * 50)

    out_dir = Path(args.out_dir) if args.out_dir else (cfg.output_dir / cfg.experiment_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "full_metrics.json"
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults written to: {out_file}")

    return 0

def main() -> None:
    args = _build_parser().parse_args()

    if args.self_test:
        sys.exit(run_self_test())

    if args.ckpt is None:
        print("ERROR: --ckpt is required unless --self-test is specified.", file=sys.stderr)
        sys.exit(1)

    sys.exit(run_full_eval(args))

if __name__ == "__main__":
    main()
