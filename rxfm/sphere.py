from __future__ import annotations

import math

import torch
import torch.nn.functional as F

EPS = 1e-6


def random_sphere(shape, device, dtype=torch.float32):
    return F.normalize(torch.randn(shape, device=device, dtype=dtype), dim=-1)


def project_tangent(v, z):
    return v - (v * z).sum(-1, keepdim=True) * z


def exp_map(z, w):
    w = project_tangent(w, z)
    n = w.norm(dim=-1, keepdim=True)
    safe = n.clamp_min(EPS)
    out = torch.cos(n) * z + torch.sin(n) * (w / safe)
    return torch.where(n < EPS, z, out)


def log_map(z, y):
    p = project_tangent(y, z)
    pn = p.norm(dim=-1, keepdim=True)
    dot = (z * y).sum(-1, keepdim=True)
    theta = torch.atan2(pn, dot)
    out = theta * p / pn.clamp_min(EPS)
    return torch.where((pn < EPS) & (dot > 0), torch.zeros_like(out), out)


def _angle(z0, z1):
    cos = (z0 * z1).sum(-1, keepdim=True).clamp(-1 + EPS, 1 - EPS)
    theta = cos.arccos()
    return theta, theta.sin().clamp_min(EPS)


def slerp(z0, z1, t):
    theta, sin = _angle(z0, z1)
    val = (((1 - t) * theta).sin() * z0 + (t * theta).sin() * z1) / sin
    near = theta < EPS
    return torch.where(near, F.normalize((1 - t) * z0 + t * z1, dim=-1), val)


def slerp_velocity(z0, z1, t):
    theta, sin = _angle(z0, z1)
    val = (theta / sin) * (-((1 - t) * theta).cos() * z0 + (t * theta).cos() * z1)
    near = theta < EPS
    return torch.where(near, z1 - z0, val)


def polar_encode(x, mean):
    c = x - mean
    r = c.norm(dim=-1, keepdim=True).clamp_min(EPS)
    return c / r, r.squeeze(-1)


def polar_decode(z, r, mean):
    return mean + r.unsqueeze(-1) * z


def tangent_noise(z, kappa):
    if kappa <= 0:
        return z
    w = project_tangent(torch.randn_like(z) * kappa, z)
    return exp_map(z, w)


def sigma_for_angle(rad: float, k: int) -> float:
    return rad / math.sqrt(max(1, k))


def tangent_noise_rad(z, rad):
    if rad <= 0:
        return z
    return tangent_noise(z, sigma_for_angle(rad, z.shape[-1] - 1))
