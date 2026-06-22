from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS_T = 1e-3
_LOG_SIGMA_MIN = -10.0
_LOG_SIGMA_MAX = 2.0

class VFMHead(nn.Module):

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
    sigma = log_sigma.exp()
    nll = 0.5 * ((mu - x1) ** 2) / (sigma ** 2) + log_sigma
    return nll.mean()

def cfm_loss(v_pred: torch.Tensor, v_target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(v_pred, v_target)

def velocity_from_posterior(mu: torch.Tensor, xt: torch.Tensor,
                             t: torch.Tensor) -> torch.Tensor:

    for _ in range(xt.dim() - 1):
        t = t.unsqueeze(-1)
    denom = (1.0 - t).clamp(min=_EPS_T)
    return (mu - xt) / denom

def flow_loss(raw_pred: torch.Tensor, xt: torch.Tensor, t: torch.Tensor,
              x1: torch.Tensor, flow_objective: str = "vfm",
              spatial: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    if flow_objective == "vfm":
        head = VFMHead(spatial=spatial)
        mu, log_sigma = head(raw_pred)
        loss = vfm_loss(mu, log_sigma, x1)
        v_pred = velocity_from_posterior(mu, xt, t)
        return loss, v_pred
    elif flow_objective == "cfm":

        loss = cfm_loss(raw_pred, x1)
        return loss, raw_pred
    else:
        raise ValueError(f"Unknown flow_objective: {flow_objective!r}. Use 'vfm' or 'cfm'.")
