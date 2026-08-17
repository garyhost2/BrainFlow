from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


def _cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.float(), b.float(), dim=-1).mean(dim=-1)


@torch.no_grad()
def effective_rank(x: torch.Tensor, energy: float = 0.99, max_rows: int = 512) -> float:
    f = x.float().flatten(1)
    if f.shape[0] > max_rows:
        f = f[:max_rows]
    if f.shape[0] < 2:
        return float("nan")
    f = f - f.mean(0, keepdim=True)
    s = torch.linalg.svdvals(f.to(torch.float32))
    p = s ** 2
    total = p.sum()
    if total <= 0:
        return 0.0
    c = torch.cumsum(p / total, dim=0)
    return float((c < energy).sum().item() + 1)


@torch.no_grad()
def dispersion(x: torch.Tensor, max_rows: int = 512) -> float:
    f = x.float().flatten(1)
    if f.shape[0] > max_rows:
        f = f[:max_rows]
    if f.shape[0] < 2:
        return float("nan")
    denom = f.norm(dim=-1).pow(2).mean().sqrt().clamp_min(1e-12)
    return float((f - f.mean(0, keepdim=True)).norm(dim=-1).pow(2).mean().sqrt() / denom)


@torch.no_grad()
def latent_diagnostics(model, loader, device, *, n_steps: int = 20,
                       solver: str = "heun", max_batches: int | None = None,
                       prefix: str = "") -> dict[str, float]:
    core = model.module if hasattr(model, "module") else model
    was_training = core.training
    core.eval()

    anchor_cos, flow_cos, cls_cos = [], [], []
    radius_pred, radius_true = [], []
    sig_mean, sig_min, sig_max = [], [], []
    pred_pool, targ_pool = [], []
    seen = 0

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        fmri = batch["fmri"].to(device, non_blocking=True)
        tgt = batch["emb"].to(device, non_blocking=True).float()
        subject = batch["subject"]

        diag = core.diagnose(fmri, subject, tgt,
                             target_cls=batch.get("cls"),
                             n_steps=n_steps, solver=solver)

        anchor_cos.append(diag["anchor_cos"].cpu())
        flow_cos.append(diag["flow_cos"].cpu())
        if diag.get("cls_cos") is not None:
            cls_cos.append(diag["cls_cos"].cpu())
        if diag.get("logr_pred") is not None:
            radius_pred.append(diag["logr_pred"].flatten().cpu())
            radius_true.append(diag["logr_true"].flatten().cpu())
        if diag.get("sigma") is not None:
            s = diag["sigma"]
            sig_mean.append(float(s.mean())); sig_min.append(float(s.min()))
            sig_max.append(float(s.max()))

        pred_pool.append(diag["z_flow"].flatten(1).half().cpu())
        targ_pool.append(F.normalize(tgt, dim=-1).flatten(1).half().cpu())
        seen += fmri.shape[0]

    if was_training:
        core.train()
    if seen == 0:
        return {}

    a = torch.cat(anchor_cos); f_ = torch.cat(flow_cos)
    out = {
        "anchor_cos": float(a.mean()),
        "anchor_theta_deg": float(torch.rad2deg(a.clamp(-1, 1).arccos()).mean()),
        "flow_cos": float(f_.mean()),
        "delta_cos": float((f_ - a).mean()),
        "delta_cos_win_rate": float((f_ > a).float().mean()),
    }
    if cls_cos:
        out["cls_cos_hat_true"] = float(torch.cat(cls_cos).mean())
    if radius_pred:
        rp = torch.cat(radius_pred).float(); rt = torch.cat(radius_true).float()
        rp = rp - rp.mean(); rt = rt - rt.mean()
        denom = (rp.norm() * rt.norm()).clamp_min(1e-8)
        out["radius_corr"] = float((rp * rt).sum() / denom)
    if sig_mean:
        out["sigma_mean"] = sum(sig_mean) / len(sig_mean)
        out["sigma_min"] = min(sig_min)
        out["sigma_max"] = max(sig_max)

    P = torch.cat(pred_pool); T = torch.cat(targ_pool)
    from .metrics import retrieval_metrics, two_way_identification
    r = retrieval_metrics(P, T, batch_size=300, device=torch.device("cpu"))
    out["retr_fwd_300"] = r["retrieval_fwd"]
    out["retr_bwd_300"] = r["retrieval_bwd"]
    out["retr_pools"] = float(r["retrieval_pools"])
    out["two_way"] = two_way_identification(P.float(), T.float())
    out["eff_rank"] = effective_rank(P)
    out["dispersion"] = dispersion(P)
    out["n_samples"] = float(seen)

    return {f"{prefix}{k}": v for k, v in out.items()}


def train_val_gap(train_d: dict, val_d: dict) -> dict[str, float]:
    out = {}
    for k, v in train_d.items():
        base = k.split("/", 1)[-1]
        vk = f"val/{base}"
        if vk in val_d and isinstance(v, (int, float)):
            out[f"gap/{base}"] = float(v) - float(val_d[vk])
    return out


@torch.no_grad()
def _global_norm(grads) -> float:
    tot = 0.0
    for g in grads:
        if g is not None:
            tot += float(g.detach().float().pow(2).sum())
    return math.sqrt(tot)


def grad_norms(terms: dict[str, torch.Tensor], params) -> dict[str, float]:
    params = [p for p in params if p.requires_grad]
    out = {}
    items = [(k, v) for k, v in terms.items()
             if torch.is_tensor(v) and v.requires_grad and v.numel() == 1]
    for name, term in items:
        g = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
        out[f"gradnorm/{name}"] = _global_norm(g)
    return out


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True).strip())
    except Exception:
        return False


def param_report(model) -> dict[str, float]:
    core = model.module if hasattr(model, "module") else model
    out, seen = {}, set()
    for name, mod in core.named_children():
        n = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        out[f"params_M/{name}"] = n / 1e6
        seen.update(id(p) for p in mod.parameters())
    out["params_M/total"] = sum(
        p.numel() for p in core.parameters() if p.requires_grad) / 1e6
    return out


def write_manifest(out_dir: Path, *, args, cfg, model, voxels: dict,
                   extra: dict | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_d = asdict(cfg) if is_dataclass(cfg) else dict(cfg)
    args_d = vars(args) if hasattr(args, "__dict__") else dict(args)
    payload = {
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in args_d.items()},
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg_d.items()},
        "voxels_per_subject": {str(k): int(v) for k, v in voxels.items()},
        "params": param_report(model),
        "torch": torch.__version__,
        "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"),
    }
    payload["config_hash"] = hashlib.blake2b(
        json.dumps(payload["config"], sort_keys=True, default=str).encode(),
        digest_size=8).hexdigest()
    if extra:
        payload |= extra
    p = out_dir / "manifest.json"
    p.write_text(json.dumps(payload, indent=2, default=str))
    return p
