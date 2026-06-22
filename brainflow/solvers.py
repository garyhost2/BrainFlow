from __future__ import annotations

import math
from typing import Callable

import torch

VelocityFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
"""velocity_fn(xt: Tensor[B,...], t: Tensor[B]) -> v: Tensor[B,...]"""

def euler(vel_fn: VelocityFn, x0: torch.Tensor,
          t_grid: torch.Tensor) -> torch.Tensor:
    x = x0
    for i in range(len(t_grid) - 1):
        dt = t_grid[i + 1] - t_grid[i]
        t = t_grid[i].expand(x.shape[0])
        v = vel_fn(x, t)
        x = x + v * dt
    return x

def midpoint(vel_fn: VelocityFn, x0: torch.Tensor,
             t_grid: torch.Tensor) -> torch.Tensor:
    x = x0
    for i in range(len(t_grid) - 1):
        dt = t_grid[i + 1] - t_grid[i]
        t_mid = (t_grid[i] + t_grid[i + 1]) / 2.0
        t = t_mid.expand(x.shape[0])
        v = vel_fn(x, t)
        x = x + v * dt
    return x

def heun(vel_fn: VelocityFn, x0: torch.Tensor,
         t_grid: torch.Tensor) -> torch.Tensor:
    x = x0
    for i in range(len(t_grid) - 1):
        dt = t_grid[i + 1] - t_grid[i]
        t_cur = t_grid[i].expand(x.shape[0])
        t_next = t_grid[i + 1].expand(x.shape[0])
        v1 = vel_fn(x, t_cur)
        x_euler = x + v1 * dt
        v2 = vel_fn(x_euler, t_next)
        x = x + (v1 + v2) * (dt / 2)
    return x

def dpm_pp_2m(vel_fn: VelocityFn, x0: torch.Tensor,
              t_grid: torch.Tensor) -> torch.Tensor:
    x = x0
    v_prev = None

    for i in range(len(t_grid) - 1):
        dt = t_grid[i + 1] - t_grid[i]
        t = t_grid[i].expand(x.shape[0])
        v_cur = vel_fn(x, t)

        if v_prev is None:

            x = x + v_cur * dt
        else:

            x = x + dt * (1.5 * v_cur - 0.5 * v_prev)

        v_prev = v_cur

    return x

def adaptive_rk45(vel_fn: VelocityFn, x0: torch.Tensor,
                  t_grid: torch.Tensor,
                  rtol: float = 1e-4,
                  atol: float = 1e-4,
                  adjoint: bool = False) -> torch.Tensor:
    try:
        import torchdiffeq
    except ImportError as exc:
        raise ImportError(
            "torchdiffeq is required for adaptive_rk45. "
            "Install with: pip install torchdiffeq"
        ) from exc

    B = x0.shape[0]
    flat_shape = x0.shape

    def odefunc(t_scalar: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        t = t_scalar.expand(B)
        return vel_fn(y.reshape(flat_shape), t).reshape(y.shape)

    y0 = x0.reshape(B, -1)
    t_span = torch.stack([t_grid[0], t_grid[-1]]).to(x0.device)

    if adjoint:
        ys = torchdiffeq.odeint_adjoint(
            odefunc, y0, t_span, rtol=rtol, atol=atol, method="dopri5",
        )
    else:
        ys = torchdiffeq.odeint(
            odefunc, y0, t_span, rtol=rtol, atol=atol, method="dopri5",
        )

    return ys[-1].reshape(flat_shape)

_SOLVERS = {
    "euler": euler,
    "midpoint": midpoint,
    "heun": heun,
    "dpm_pp_2m": dpm_pp_2m,
    "adaptive_rk45": adaptive_rk45,
}

def solve(
    model: VelocityFn,
    x0: torch.Tensor,
    t_grid: torch.Tensor,
    method: str = "euler",
    **kwargs,
) -> torch.Tensor:
    if method not in _SOLVERS:
        raise ValueError(
            f"Unknown solver: {method!r}. "
            f"Available: {list(_SOLVERS)}"
        )
    return _SOLVERS[method](model, x0, t_grid, **kwargs)

def make_t_grid(
    n_steps: int,
    device: torch.device | str = "cpu",
    schedule: str = "linear",
    logit_normal_m: float = 0.0,
    logit_normal_s: float = 1.0,
) -> torch.Tensor:
    _VALID = ("linear", "cosine", "logit_normal")
    if schedule not in _VALID:
        raise ValueError(
            f"Unknown schedule: {schedule!r}. Choose one of {_VALID}."
        )

    N = n_steps

    if schedule == "linear":
        return torch.linspace(0.0, 1.0, N + 1, device=device)

    if schedule == "cosine":
        i = torch.arange(N + 1, dtype=torch.float64, device=device)
        t = (1.0 - torch.cos(i * math.pi / N)) / 2.0

        t[0] = 0.0
        t[-1] = 1.0
        return t.float()

    eps = 1e-6
    u = torch.linspace(eps, 1.0 - eps, N + 1, dtype=torch.float64, device=device)

    z = torch.erfinv(2.0 * u - 1.0) * math.sqrt(2.0)
    t = torch.sigmoid(logit_normal_m + logit_normal_s * z)

    t_min, t_max = t[0], t[-1]
    t = (t - t_min) / (t_max - t_min).clamp(min=1e-12)
    t[0] = 0.0
    t[-1] = 1.0
    return t.float()
