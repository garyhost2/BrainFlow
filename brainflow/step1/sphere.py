"""Geometry of the sphere and the oblique manifold (product of spheres).

Every routine here is the concrete realisation of a formula from
``docs/paper`` (sections 4-5, "Going curved" and "The sphere and the oblique
manifold"). Tensors carry the unit directions in their last dimension, so a
batch of token directions is ``(B, N, d)`` and a single CLS direction is
``(B, d)``; all operations broadcast over the leading dims and act on ``dim=-1``.

Conventions (matched to the paper and the test-suite):
    * ``project_tangent(v, z)`` -- argument order is (vector, base point).
    * ``exp_map(z, w)``         -- base point first, tangent step second.
    * Singular ends (theta -> 0 coincident, theta -> pi antipodal) are guarded
      by ``EPS`` clamps that modify the maps only on a measure-zero set, exactly
      the NaN-hardening described in section 5.7 of the paper.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Slerp / log singularities are clamped with this epsilon (paper: _SLERP_EPS).
EPS = 1e-6


def random_sphere(shape, device, dtype=torch.float32):
    """Uniform sample on S^{d-1} as g/||g||, g ~ N(0, I) (paper, Prop. 5.13)."""
    return F.normalize(torch.randn(shape, device=device, dtype=dtype), dim=-1)


def project_tangent(v, z):
    """Orthogonal projector P_z v = v - <z,v> z onto T_z S^{d-1} (eq. 5.3)."""
    return v - (v * z).sum(-1, keepdim=True) * z


def exp_map(z, w):
    """Exp_z(w) = cos|w| z + sin|w| w/|w|; lands on the sphere exactly (eq. 5.x).

    ``w`` is re-projected to the tangent space first so the caller may pass a raw
    (possibly slightly radial) increment without leaving the manifold.
    """
    w = project_tangent(w, z)
    n = w.norm(dim=-1, keepdim=True)
    safe = n.clamp_min(EPS)
    out = torch.cos(n) * z + torch.sin(n) * (w / safe)
    # |w| ~ 0  =>  Exp_z(w) -> z (the limit), avoid 0/0.
    return torch.where(n < EPS, z, out)


def log_map(z, y):
    """Log_z(y) = theta * P_z y / ||P_z y|| with theta = arccos<z,y> (eq. 5.x)."""
    p = project_tangent(y, z)
    pn = p.norm(dim=-1, keepdim=True).clamp_min(EPS)
    theta = (z * y).sum(-1, keepdim=True).clamp(-1 + EPS, 1 - EPS).arccos()
    return theta * p / pn


def _angle(z0, z1):
    cos = (z0 * z1).sum(-1, keepdim=True).clamp(-1 + EPS, 1 - EPS)
    theta = cos.arccos()
    return theta, theta.sin().clamp_min(EPS)


def slerp(z0, z1, t):
    """Geodesic interpolant sin((1-t)θ)/sinθ z0 + sin(tθ)/sinθ z1 (eq. 5.slerp).

    Falls back to the (normalised) flat chord when the two endpoints are nearly
    collinear, where the closed form is a removable 0/0.
    """
    theta, sin = _angle(z0, z1)
    val = (((1 - t) * theta).sin() * z0 + (t * theta).sin() * z1) / sin
    near = theta < EPS
    return torch.where(near, F.normalize((1 - t) * z0 + t * z1, dim=-1), val)


def slerp_velocity(z0, z1, t):
    """d/dt slerp = (θ/sinθ)(-cos((1-t)θ) z0 + cos(tθ) z1); tangent, norm θ."""
    theta, sin = _angle(z0, z1)
    val = (theta / sin) * (-((1 - t) * theta).cos() * z0 + (t * theta).cos() * z1)
    near = theta < EPS
    return torch.where(near, z1 - z0, val)


def polar_encode(x, mean):
    """Centered polar split X_i = mu + r_i z_i  ->  (z_i in S^{d-1}, r_i>0)."""
    c = x - mean
    r = c.norm(dim=-1, keepdim=True).clamp_min(EPS)
    return c / r, r.squeeze(-1)


def polar_decode(z, r, mean):
    """Inverse polar map X_i = mu + r_i z_i (lossless, paper Prop. 6.lossless)."""
    return mean + r.unsqueeze(-1) * z


def tangent_noise(z, kappa):
    """On-manifold augmentation: rotate z by a tangent Gaussian (paper eq. 9.augment).

    Draw eta ~ N(0, kappa^2 I), project to T_z, and Exp back, giving a pure
    geodesic rotation (||z|| stays 1) -- the prior's actual error type, unlike a
    Euclidean perturbation which knocks z off the sphere.
    """
    if kappa <= 0:
        return z
    w = project_tangent(torch.randn_like(z) * kappa, z)
    return exp_map(z, w)
