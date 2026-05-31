from __future__ import annotations

from pathlib import Path

import torch

from .config import Config


def ridge_init_path(cfg: Config) -> Path:
    raw = Path(getattr(cfg, "ridge_init_cache", "ridge_init.pt"))
    return raw if raw.is_absolute() else cfg.data_dir / raw


def _solve_ridge(x: torch.Tensor, y: torch.Tensor, ridge_lambda: float) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.float()
    y = y.float()
    x_mean = x.mean(dim=0, keepdim=True)
    y_mean = y.mean(dim=0, keepdim=True)
    x_centered = x - x_mean
    y_centered = y - y_mean

    n_samples, n_features = x_centered.shape
    if n_samples >= n_features:
        gram = x_centered.T @ x_centered
        gram.diagonal().add_(ridge_lambda)
        w = torch.linalg.solve(gram, x_centered.T @ y_centered)
    else:
        gram = x_centered @ x_centered.T
        gram.diagonal().add_(ridge_lambda)
        alpha = torch.linalg.solve(gram, y_centered)
        w = x_centered.T @ alpha
    b = (y_mean - x_mean @ w).squeeze(0)
    return w, b


def compute_ridge_init(tensors: dict, clips: dict, cfg: Config) -> dict:
    subjects = list(getattr(cfg, "subjects", []))
    voxels = tensors.get("voxels", {})
    max_vox = max(int(voxels[s]) for s in subjects)
    x_chunks, y_chunks = [], []
    for subj in subjects:
        fmri = tensors[f"fmri_train_{subj}"].float()
        if fmri.shape[1] < max_vox:
            fmri = torch.nn.functional.pad(fmri, (0, max_vox - fmri.shape[1]))
        x_chunks.append(fmri)
        y_chunks.append(clips[f"clip_train_{subj}"].float())
    x = torch.cat(x_chunks, dim=0)
    y = torch.cat(y_chunks, dim=0)
    weight, bias = _solve_ridge(x, y, float(getattr(cfg, "ridge_lambda", 1e3)))
    return {
        "weight": weight.T.contiguous(),
        "bias": bias.contiguous(),
        "max_vox": max_vox,
        "clip_dim": y.shape[1],
        "subjects": subjects,
        "ridge_lambda": float(getattr(cfg, "ridge_lambda", 1e3)),
    }


def save_ridge_init(tensors: dict, clips: dict, cfg: Config) -> Path:
    payload = compute_ridge_init(tensors, clips, cfg)
    path = ridge_init_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def load_ridge_init_state(cfg: Config, max_vox: int, clip_dim: int) -> dict | None:
    path = ridge_init_path(cfg)
    if not path.is_file():
        print(f"[ridge_init] no cached ridge init at {path}; using zero-init skip path.")
        return None
    payload = torch.load(path, map_location="cpu")
    weight = payload.get("weight")
    bias = payload.get("bias")
    if weight is None or bias is None:
        print(f"[ridge_init] missing weight/bias in {path}; using zero-init skip path.")
        return None
    if tuple(weight.shape) != (clip_dim, max_vox):
        print(
            f"[ridge_init] shape mismatch in {path}: got {tuple(weight.shape)}, "
            f"expected {(clip_dim, max_vox)}; using zero-init skip path."
        )
        return None
    if tuple(bias.shape) != (clip_dim,):
        print(
            f"[ridge_init] bias mismatch in {path}: got {tuple(bias.shape)}, "
            f"expected {(clip_dim,)}; using zero-init skip path."
        )
        return None
    print(f"[ridge_init] loaded ridge init from {path}.")
    return {"weight": weight.float(), "bias": bias.float()}
