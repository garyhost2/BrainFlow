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

import math

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
    """Log_z(y) = theta * P_z y / ||P_z y|| with theta the angle between z and y.

    theta comes from ``atan2(||P_z y||, <z,y>)``, not ``arccos<z,y>``. The arccos
    form needs its argument clamped away from 1 to stay differentiable, and that
    clamp puts a FLOOR under the output: at y == z it returned
    ``arccos(1 - 1e-6) = 1.41e-3`` radians in a numerically-arbitrary direction
    instead of 0. That is harmless when Log is only used to measure a real
    displacement, but ``flow_param="endpoint"`` evaluates ``Log_{z_t}(zhat1)``
    with ``zhat1 == z_t`` at initialisation, where the spurious 1.4e-3 became a
    per-step drift that broke the identity-flow bound the whole design rests on.

    ``atan2`` is exact and differentiable at theta = 0 and stays well-conditioned
    near theta = pi, so no clamping of the angle is needed at all.
    """
    p = project_tangent(y, z)
    pn = p.norm(dim=-1, keepdim=True)
    dot = (z * y).sum(-1, keepdim=True)
    theta = torch.atan2(pn, dot)
    out = theta * p / pn.clamp_min(EPS)
    # Coincident (theta -> 0): the limit is the zero tangent vector. Antipodal
    # (theta -> pi) is genuinely undefined -- every direction is a geodesic -- so
    # leave whatever direction the projection produced, with the correct norm pi.
    return torch.where((pn < EPS) & (dot > 0), torch.zeros_like(out), out)


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

    ``kappa`` is a PER-COORDINATE sigma, so the realised geodesic displacement is
    ``kappa * sqrt(d - 1)`` radians and therefore grows with the ambient width.
    Callers that want to specify an ANGLE must use :func:`tangent_noise_rad`; this
    stays the raw primitive.
    """
    if kappa <= 0:
        return z
    w = project_tangent(torch.randn_like(z) * kappa, z)
    return exp_map(z, w)


def sigma_for_angle(rad: float, k: int) -> float:
    """Per-coordinate tangent sigma giving an expected geodesic step of ``rad``.

    An isotropic tangent Gaussian N(0, sigma^2 I_k) has expected norm ~ sigma*sqrt(k),
    and on the sphere that norm *is* the geodesic distance travelled by Exp. So the
    interpretable knob is the displacement in radians and sigma follows.
    """
    return rad / math.sqrt(max(1, k))


def tangent_noise_rad(z, rad):
    """Geodesic rotation of ``z`` by ``rad`` RADIANS, independent of dim(z).

    ``tangent_noise(z, 0.02)`` on S^1279 displaces by 0.02*sqrt(1279) = 0.72 rad
    (41 degrees), not 0.02 -- a 36x error that silently scales with the token
    width. Every angular knob in the config is specified in radians and routed
    through here so the two conventions cannot be mixed up again.
    """
    if rad <= 0:
        return z
    return tangent_noise(z, sigma_for_angle(rad, z.shape[-1] - 1))
