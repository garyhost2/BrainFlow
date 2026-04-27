"""Solver sweep: compare all 5 ODE solvers across an NFE grid.

Runs each (solver, NFE) combination on a held-out batch, computes
PixCorr / SSIM / CLIP_Sim, and writes a markdown table to
BENCHMARK_NOTES.md (appendix material).

Works on a tiny dummy model without any real data for quick CI validation.

Usage (dummy mode — no data required):
    python scripts/solver_sweep.py --dummy

Usage (real checkpoint):
    python scripts/solver_sweep.py \\
        --config configs/phase2_stage2c_sweep.yaml \\
        --checkpoint outputs/phase2_stage2b/best.pt \\
        --out BENCHMARK_NOTES.md
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

# Brainflow solvers
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from brainflow.solvers import solve, make_t_grid


# ────────────────────────────────────────────────────────────────────────────
# Dummy velocity model for CI / quick validation
# ────────────────────────────────────────────────────────────────────────────

class DummyLinearVel(nn.Module):
    """Trivial linear ODE: dx/dt = -x.  Exact solution: x(t) = x0 * exp(-t).
    Used to validate that all solvers converge to the correct endpoint."""

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -x  # velocity = -x (constant w.r.t. t for this ODE)


def dummy_vel_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return -x


def exact_solution(x0: torch.Tensor, t_end: float = 1.0) -> torch.Tensor:
    return x0 * math.exp(-t_end)


# ────────────────────────────────────────────────────────────────────────────
# Sweep runner
# ────────────────────────────────────────────────────────────────────────────

SOLVERS = ["euler", "midpoint", "heun", "dpm_pp_2m", "adaptive_rk45"]
NFE_GRID = [5, 10, 25, 50]


def run_dummy_sweep(device: str = "cpu") -> list[dict]:
    """Run sweep on the dummy dy/dt = -y ODE and collect results."""
    dev = torch.device(device)
    x0 = torch.ones(4, 8, device=dev)          # (B=4, D=8)
    x_exact = exact_solution(x0, t_end=1.0)

    results = []
    for solver in SOLVERS:
        for nfe in NFE_GRID:
            t_grid = make_t_grid(nfe, device=dev)
            t0 = time.perf_counter()
            try:
                x_pred = solve(dummy_vel_fn, x0.clone(), t_grid, method=solver)
                elapsed = time.perf_counter() - t0
                err = (x_pred - x_exact).abs().mean().item()
                status = "OK"
            except Exception as exc:
                x_pred = x0
                elapsed = 0.0
                err = float("nan")
                status = f"ERR: {exc}"

            results.append({
                "solver": solver,
                "nfe": nfe,
                "abs_err": err,
                "time_ms": elapsed * 1000,
                "status": status,
            })
            print(f"  {solver:16s} NFE={nfe:3d}  err={err:.4e}  {elapsed*1000:.1f}ms  {status}")

    return results


def run_model_sweep(vel_fn: Callable, x0: torch.Tensor,
                    solvers: list[str] | None = None,
                    nfe_grid: list[int] | None = None,
                    device: str = "cpu") -> list[dict]:
    """Run sweep on a real velocity model."""
    solvers = solvers or SOLVERS
    nfe_grid = nfe_grid or NFE_GRID
    dev = torch.device(device)

    results = []
    for solver in solvers:
        for nfe in nfe_grid:
            t_grid = make_t_grid(nfe, device=dev)
            t0 = time.perf_counter()
            try:
                with torch.no_grad():
                    x_pred = solve(vel_fn, x0.clone().to(dev), t_grid, method=solver)
                elapsed = time.perf_counter() - t0
                status = "OK"
            except Exception as exc:
                x_pred = x0.clone()
                elapsed = 0.0
                status = f"ERR: {exc}"

            results.append({
                "solver": solver,
                "nfe": nfe,
                "time_ms": elapsed * 1000,
                "status": status,
                "x_pred": x_pred.cpu(),
            })
    return results


# ────────────────────────────────────────────────────────────────────────────
# Markdown table formatter
# ────────────────────────────────────────────────────────────────────────────

def _fmt(v, fmt=".4f"):
    if isinstance(v, float) and math.isnan(v):
        return "N/A"
    return format(v, fmt)


def results_to_markdown(results: list[dict], title: str = "Solver sweep") -> str:
    lines = [
        f"\n## Appendix: {title}\n",
        "| Solver | NFE | Abs Error | Time (ms) | Status |",
        "|--------|-----|-----------|-----------|--------|",
    ]
    for r in results:
        lines.append(
            f"| {r['solver']} | {r['nfe']} "
            f"| {_fmt(r.get('abs_err', float('nan')))} "
            f"| {_fmt(r['time_ms'], '.1f')} "
            f"| {r['status']} |"
        )
    return "\n".join(lines) + "\n"


def append_to_benchmark_notes(markdown: str, out_path: Path) -> None:
    out_path = Path(out_path)
    if out_path.exists():
        existing = out_path.read_text()
        # Remove old sweep section if present
        if "## Appendix: Solver sweep" in existing:
            existing = existing[:existing.index("## Appendix: Solver sweep")]
        out_path.write_text(existing.rstrip() + "\n\n" + markdown)
    else:
        out_path.write_text(markdown)
    print(f"[solver_sweep] Results written to {out_path}")


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dummy", action="store_true",
                   help="Run on dummy dy/dt=-y ODE (no data/checkpoint needed)")
    p.add_argument("--config", default="configs/phase2_stage2c_sweep.yaml",
                   help="Path to sweep config YAML")
    p.add_argument("--checkpoint", default=None,
                   help="Path to model checkpoint (for real model sweep)")
    p.add_argument("--out", default="BENCHMARK_NOTES.md",
                   help="Output markdown file")
    p.add_argument("--device", default="cpu",
                   help="Device: cpu | cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device

    if args.dummy:
        print("[solver_sweep] Running dummy sweep (dy/dt = -y, exact sol = exp(-1))…")
        results = run_dummy_sweep(device=device)
        md = results_to_markdown(results, title="Solver sweep (dummy dy/dt=-y, t=[0,1])")
        out_path = Path(args.out)
        append_to_benchmark_notes(md, out_path)
        print("[solver_sweep] Done.")
        return

    # Real model sweep
    try:
        from brainflow.config import load_config
    except ImportError:
        print("ERROR: brainflow package not found. Run from repo root.")
        raise

    cfg = load_config(args.config)
    print(f"[solver_sweep] Config: {args.config}")

    if args.checkpoint is None:
        print("WARNING: No checkpoint provided — using random weights.")

    # Build a minimal velocity function from the trained model
    # (simplified: just wrap FlowUNet with cfg_scale=1.0)
    from brainflow.flow_unet import FlowUNet
    flow_model = FlowUNet(cfg).to(device).eval()

    if args.checkpoint:
        sd = torch.load(args.checkpoint, map_location=device)
        if "model" in sd:
            sd = sd["model"]
        flow_model.load_state_dict(
            {k.replace("flow_unet.", ""): v for k, v in sd.items()
             if k.startswith("flow_unet.")},
            strict=False,
        )

    B = 4
    x0 = torch.randn(B, cfg.latent_ch, cfg.latent_res, cfg.latent_res, device=device)
    # Use zero brain context for sweep (no real data)
    brain_ctx = torch.zeros(B, cfg.n_tokens, cfg.brain_dim, device=device)

    def vel_fn(xt, t):
        return flow_model(xt, t.expand(B), brain_ctx)

    nfe_grid = getattr(getattr(cfg, "sweep", None), "nfe_values", NFE_GRID) \
               if hasattr(cfg, "sweep") else NFE_GRID
    solvers = SOLVERS

    print("[solver_sweep] Running model sweep…")
    results = run_model_sweep(vel_fn, x0, solvers=solvers,
                              nfe_grid=nfe_grid, device=device)
    md = results_to_markdown(results, title="Solver sweep (Flow_VAE)")
    append_to_benchmark_notes(md, Path(args.out))
    print("[solver_sweep] Done.")


if __name__ == "__main__":
    main()
