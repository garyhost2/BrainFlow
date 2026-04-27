"""Variational Flow Matching (VFM) objective.

Implements the VFM formulation from:
  "Riemannian Flow Matching on General Geometries" (rg-vfm, ICLR 2026).

Key idea: instead of predicting the velocity directly, the network predicts
the *parameters of a Gaussian posterior* q(x1 | xt, t) over the flow endpoint
x1. The velocity is then recovered as:

    v(xt, t) = (E_q[x1] - xt) / (1 - t)

with numerical guards near t=1. The VFM loss is the negative ELBO on x1:

    L_VFM = ||mu - x1||^2 / (2 * sigma^2) + log(sigma)

which reduces to the standard CFM MSE loss when sigma → 0 (deterministic limit).

Toggle via `cfg.flow_objective ∈ {"cfm", "vfm"}` so CFM remains a clean ablation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_EPS_T = 1e-3   # guard against (1-t) underflow near t=1
_LOG_SIGMA_MIN = -10.0
_LOG_SIGMA_MAX = 2.0


class VFMHead(nn.Module):
    """Thin wrapper that splits the raw network output into (mu, log_sigma).

    The network is expected to produce `2 * out_channels` feature maps (or
    `2 * out_dim` vectors) — first half is the mean, second half is log_sigma.

    For spatial outputs (B, C, H, W) the split is along channel dim=1.
    For vector outputs (B, D) the split is along dim=-1.
    """

    def __init__(self, spatial: bool = True):
        super().__init__()
        self.spatial = spatial

    def forward(self, raw: torch.Tensor):
        dim = 1 if self.spatial else -1
        mu, log_sigma = raw.chunk(2, dim=dim)
        log_sigma = log_sigma.clamp(_LOG_SIGMA_MIN, _LOG_SIGMA_MAX)
        return mu, log_sigma


def vfm_loss(mu: torch.Tensor, log_sigma: torch.Tensor,
             x1: torch.Tensor) -> torch.Tensor:
    """Variational endpoint loss (negative ELBO on x1 posterior).

    Args:
        mu:         predicted posterior mean  (same shape as x1)
        log_sigma:  predicted posterior log-std (same shape as x1)
        x1:         ground-truth flow endpoint

    Returns:
        scalar loss = mean over batch and spatial/channel dims of
            [ (mu - x1)^2 / (2 * exp(2*log_sigma)) + log_sigma ]
    """
    sigma = log_sigma.exp()
    nll = 0.5 * ((mu - x1) ** 2) / (sigma ** 2) + log_sigma
    return nll.mean()


def cfm_loss(v_pred: torch.Tensor, v_target: torch.Tensor) -> torch.Tensor:
    """Standard conditional flow matching MSE loss (clean ablation baseline)."""
    return F.mse_loss(v_pred, v_target)


def velocity_from_posterior(mu: torch.Tensor, xt: torch.Tensor,
                             t: torch.Tensor) -> torch.Tensor:
    """Recover velocity from posterior mean via the linear interpolant identity.

    v(xt, t) = (E_q[x1] - xt) / (1 - t)

    Near t=1 we clamp the denominator to _EPS_T for numerical stability.

    Args:
        mu:  posterior mean prediction  (B, ...) — same layout as xt
        xt:  current state on trajectory (B, ...)
        t:   time in [0, 1]             (B,)

    Returns:
        velocity  (B, ...)
    """
    # Broadcast t to match spatial/channel layout of xt
    for _ in range(xt.dim() - 1):
        t = t.unsqueeze(-1)
    denom = (1.0 - t).clamp(min=_EPS_T)
    return (mu - xt) / denom


def flow_loss(raw_pred: torch.Tensor, xt: torch.Tensor, t: torch.Tensor,
              x1: torch.Tensor, flow_objective: str = "vfm",
              spatial: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Unified loss dispatcher for VFM / CFM.

    In VFM mode:
        - raw_pred has 2× channels/features: (mu, log_sigma)
        - Loss = vfm_loss(mu, log_sigma, x1)
        - Velocity = velocity_from_posterior(mu, xt, t)

    In CFM mode:
        - raw_pred has 1× channels/features (direct velocity prediction)
        - Loss = cfm_loss(raw_pred, x1 - x0) where ut = x1 - x0
        Note: caller must pass ut (= x1 - x0) as x1 in CFM mode.

    Args:
        raw_pred:         raw network output
        xt:               interpolated state at time t
        t:                time values (B,)
        x1:               flow endpoint targets (latent / CLIP token grid)
                          In CFM mode, pass ut = x1 - x0 here.
        flow_objective:   "vfm" | "cfm"
        spatial:          True for image-like (B,C,H,W), False for vector (B,D)

    Returns:
        (loss, v_pred)  — loss scalar and predicted velocity
    """
    if flow_objective == "vfm":
        head = VFMHead(spatial=spatial)
        mu, log_sigma = head(raw_pred)
        loss = vfm_loss(mu, log_sigma, x1)
        v_pred = velocity_from_posterior(mu, xt, t)
        return loss, v_pred
    elif flow_objective == "cfm":
        # raw_pred IS the velocity; x1 should be ut = x1 - x0
        loss = cfm_loss(raw_pred, x1)
        return loss, raw_pred
    else:
        raise ValueError(f"Unknown flow_objective: {flow_objective!r}. Use 'vfm' or 'cfm'.")
