from __future__ import annotations

import torch
import torch.nn.functional as F

@torch.no_grad()
def _sinkhorn_log(
    C: torch.Tensor,
    reg: float = 0.05,
    n_iters: int = 50,
) -> torch.Tensor:
    B = C.shape[0]
    log_K = -C / reg

    log_mu = -math.log(B) * torch.ones(B, device=C.device, dtype=C.dtype)

    log_u = torch.zeros(B, device=C.device, dtype=C.dtype)
    log_v = torch.zeros(B, device=C.device, dtype=C.dtype)

    for _ in range(n_iters):

        log_u = log_mu - torch.logsumexp(log_K + log_v[None, :], dim=1)
        log_v = log_mu - torch.logsumexp(log_K + log_u[:, None], dim=0)

    log_pi = log_K + log_u[:, None] + log_v[None, :]
    return log_pi.exp()

import math

@torch.no_grad()
def ot_minibatch_coupling(
    x0: torch.Tensor,
    x1: torch.Tensor,
    reg: float = 0.05,
    n_iters: int = 50,
    p: int = 2,
) -> torch.Tensor:
    if p != 2:
        raise ValueError(f"Only p=2 (squared Euclidean) is supported, got p={p}")

    B = x0.shape[0]

    x0_flat = x0.detach().float().reshape(B, -1)
    x1_flat = x1.detach().float().reshape(B, -1)

    diff = x0_flat[:, None, :] - x1_flat[None, :, :]
    C = (diff ** 2).sum(dim=-1)

    C = C / (C.max().clamp(min=1e-8))

    try:
        import ot as pot

        M = C.cpu().numpy()
        a = (torch.ones(B, dtype=torch.float64) / B).numpy()
        pi = torch.from_numpy(pot.sinkhorn(a, a, M, reg=reg, numItermax=n_iters)).to(C.device)
    except ImportError:
        pi = _sinkhorn_log(C, reg=reg, n_iters=n_iters)

    assignment = pi.argmax(dim=1)

    return x0[assignment].detach()
