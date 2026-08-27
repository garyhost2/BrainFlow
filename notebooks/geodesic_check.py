# %% [code]
import os, csv, json, math
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "/kaggle/working/geodesic" if os.path.isdir("/kaggle/working") else "runs/geodesic"
os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
REG = []
EPS = 1e-12


def log(**kw):
    print(" ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
                   for k, v in kw.items()), flush=True)


def reg(name, ok, predicts, **kw):
    REG.append(dict(tag="PASS" if ok else "WARN", name=name, predicts=predicts,
                    detail={k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                            for k, v in kw.items()}))
    log(reg=REG[-1]["tag"], name=name, **kw)


def save_csv(name, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(OUT, "results", name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)


def flush_reg():
    with open(os.path.join(OUT, "results", "register.json"), "w") as f:
        json.dump(dict(register=REG), f, indent=1)


def normalize(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(EPS)


def angle(z0, z1):
    return (z0 * z1).sum(-1, keepdim=True).clamp(-1 + 1e-14, 1 - 1e-14).arccos()


def slerp(z0, z1, s):
    th = angle(z0, z1)
    sn = th.sin().clamp_min(EPS)
    return (((1 - s) * th).sin() * z0 + (s * th).sin() * z1) / sn


# %% [code]
# Coefficients of the interpolant in the plane spanned by the endpoints.
#   geodesic:  z_s = p(s) z0 + q(s) z1,  p = sin((1-s)th)/sin th,  q = sin(s th)/sin th
#   flat:      z_s = (1-s) z0 + s z1
def coef_geo(s, th):
    sn = math.sin(th)
    return math.sin((1 - s) * th) / sn, math.sin(s * th) / sn


def coef_flat(s, th):
    return 1.0 - s, s


# Given the state at time t and the auxiliary state at time alpha*t, both linear in
# (z0, z1), the regression target z1 is recovered by inverting a 2x2 system. The
# sensitivity of that recovery to the state is |p_a / (p_t q_a - p_a q_t)|.
def sens_closed(t, alpha, th, geodesic):
    cf = coef_geo if geodesic else coef_flat
    p_t, q_t = cf(t, th)
    p_a, q_a = cf(alpha * t, th)
    det = p_t * q_a - p_a * q_t
    return abs(p_a / det) if abs(det) > 0 else float("inf")


def sens_geo_formula(t, alpha, th):
    # closed form obtained from the product-to-sum identity: the determinant
    # collapses to -sin((1-alpha) t th)/sin th
    return math.sin((1 - alpha * t) * th) / math.sin((1 - alpha) * t * th)


def sens_flat_formula(t, alpha):
    return (1 - alpha * t) / (t * (1 - alpha))


# %% [code]
# 1. the collapsed determinant identity, symbolically checked on a grid
rows, err_det, err_sens = [], 0.0, 0.0
for th in [0.05, 0.3, 0.8, 1.5, 2.4, 3.0]:
    for alpha in [0.0, 0.25, 0.5, 0.85]:
        for t in [1e-4, 1e-3, 1e-2, 0.05, 0.2, 0.5, 0.9]:
            p_t, q_t = coef_geo(t, th)
            p_a, q_a = coef_geo(alpha * t, th)
            det = p_t * q_a - p_a * q_t
            det_cf = -math.sin((1 - alpha) * t * th) / math.sin(th)
            err_det = max(err_det, abs(det - det_cf))
            s_direct = sens_closed(t, alpha, th, True)
            s_form = sens_geo_formula(t, alpha, th)
            err_sens = max(err_sens, abs(s_direct - s_form) / max(s_form, 1e-30))
            rows.append(dict(theta=th, alpha=alpha, t=t, det=det, det_closed=det_cf,
                             sens_geodesic=s_form, sens_flat=sens_flat_formula(t, alpha),
                             ratio=s_form / sens_flat_formula(t, alpha),
                             sinc_theta=math.sin(th) / th))
save_csv("sensitivity.csv", rows)
reg("geo_determinant_collapses", err_det < 1e-12,
    "the 2x2 determinant of the geodesic recovery system equals -sin((1-alpha) t theta)/sin theta",
    max_abs_err=err_det, n=len(rows))
reg("geo_sensitivity_closed_form", err_sens < 1e-12,
    "the recovery sensitivity on the sphere is sin((1-alpha t) theta)/sin((1-alpha) t theta)",
    max_rel_err=err_sens, n=len(rows))

# %% [code]
# 2. the same 1/t divergence as the flat case, up to the bounded factor sin(theta)/theta
worst_ratio, worst_lim = 0.0, 0.0
order_exact, order_alpha = [], []
for th in [0.05, 0.3, 0.8, 1.5, 2.4, 3.0]:
    for alpha in [0.0, 0.25, 0.5, 0.85]:
        d1 = abs(sens_geo_formula(1e-4, alpha, th) / sens_flat_formula(1e-4, alpha)
                 - math.sin(th) / th)
        d2 = abs(sens_geo_formula(1e-5, alpha, th) / sens_flat_formula(1e-5, alpha)
                 - math.sin(th) / th)
        # t falls by ten between the two, so the fitted order is log10(d1/d2)
        order = math.log10(d1 / d2) if d2 > 0 else float("nan")
        (order_exact if alpha == 0.0 else order_alpha).append(order)
        worst_ratio = max(worst_ratio, d2 / max(math.sin(th) / th, 1e-300))
        # t * sensitivity tends to the finite limit sin(theta)/((1-alpha) theta)
        lim = math.sin(th) / ((1 - alpha) * th)
        worst_lim = max(worst_lim, abs(1e-6 * sens_geo_formula(1e-6, alpha, th) - lim) / lim)
reg("geo_matches_flat_up_to_sinc",
    worst_ratio < 1e-3 and min(order_exact) > 1.9 and min(order_alpha) > 0.9,
    "as t goes to zero the geodesic amplification is the flat amplification times "
    "sin(theta)/theta, a factor bounded in (0,1]. the remainder is second order in t for the "
    "exact-source model (alpha = 0), where the numerator carries no t at all, and first order "
    "for the alpha family; either way the 1/t divergence is unchanged",
    max_rel_dev_at_1e5=worst_ratio,
    fitted_order_alpha0=min(order_exact), fitted_order_alpha_positive=min(order_alpha))
reg("geo_divergence_is_one_over_t", worst_lim < 1e-3,
    "t times the sensitivity converges to sin(theta)/((1-alpha) theta), so the sensitivity "
    "diverges exactly at rate 1/t on the sphere as it does in the plane",
    max_rel_dev_from_limit=worst_lim)

# %% [code]
# 3. finite-difference check against the actual spherical interpolant in R^d
def recover_z1(zt, a, t, alpha, th):
    p_t, q_t = coef_geo(t, th)
    p_a, q_a = coef_geo(alpha * t, th)
    det = p_t * q_a - p_a * q_t
    return (p_t * a - p_a * zt) / det


torch.manual_seed(0)
d = 64
fd_rec, fd_sens = 0.0, 0.0
rows2 = []
for trial in range(40):
    z0 = normalize(torch.randn(1, d))
    z1 = normalize(torch.randn(1, d))
    th = float(angle(z0, z1))
    for alpha in [0.0, 0.5]:
        for t in [1e-3, 1e-2, 0.1, 0.5]:
            zt = slerp(z0, z1, torch.tensor([[t]]))
            a = slerp(z0, z1, torch.tensor([[alpha * t]]))
            rec = recover_z1(zt, a, t, alpha, th)
            fd_rec = max(fd_rec, float((rec - z1).norm()))
            # perturb the state inside the plane and measure the response
            h = 1e-7
            dirv = normalize(torch.randn(1, d))
            rec2 = recover_z1(zt + h * dirv, a, t, alpha, th)
            emp = float((rec2 - rec).norm()) / h
            cf = sens_geo_formula(t, alpha, th)
            fd_sens = max(fd_sens, abs(emp - cf) / cf)
            rows2.append(dict(trial=trial, theta=th, alpha=alpha, t=t,
                              recon_err=float((rec - z1).norm()),
                              sens_empirical=emp, sens_closed=cf))
save_csv("finite_difference.csv", rows2)
reg("geo_recovery_is_exact_on_path", fd_rec < 1e-9,
    "on the geodesic path the regression target is recovered exactly from the state and the "
    "auxiliary input, which is why the augmented objective can reach a lower training loss",
    max_recon_err=fd_rec, n=len(rows2))
reg("geo_sensitivity_matches_finite_difference", fd_sens < 1e-5,
    "the closed-form amplification agrees with a finite-difference measurement of the "
    "recovery map in ambient dimension 64",
    max_rel_err=fd_sens, n=len(rows2))

# %% [code]
# 4. the truncation experiment ranks epsilon by the accumulated amplification, so what
#    must survive is the ordering, not the value. theta at the operating point is
#    arccos(0.2637) = 1.304, the angle between the source and the target.
TRUNC = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35]
TH_OP = math.acos(0.2637)


def accum(eps, alpha, th, geodesic, floor=1e-3, n=20001):
    lo = max(eps, floor)
    g = np.linspace(lo, 1.0, n)
    y = np.array([sens_geo_formula(float(t), alpha, th) if geodesic
                  else sens_flat_formula(float(t), alpha) for t in g])
    return float(np.trapz(y, g))


def rank(v):
    return np.argsort(np.argsort(np.asarray(v, float)))


rows3 = []
for th in [0.3, 0.8, TH_OP, 1.8, 3.0]:
    for alpha in [0.0, 0.5]:
        ig = [accum(e, alpha, th, True) for e in TRUNC]
        iff = [accum(e, alpha, th, False) for e in TRUNC]
        same = bool((rank(ig) == rank(iff)).all())
        rho = float(np.corrcoef(np.log(ig), np.log(iff))[0, 1])
        rows3.append(dict(theta=th, alpha=alpha, ranks_agree=int(same), log_corr=rho,
                          operating_point=int(abs(th - TH_OP) < 1e-9)))
save_csv("truncation_ordering.csv", rows3)
allsame = all(r["ranks_agree"] for r in rows3)
reg("geo_truncation_ordering_is_identical", allsame,
    "over the tested epsilon grid the accumulated amplification orders the truncation levels "
    "identically on the sphere and in the plane, at every angle tested including the operating "
    "point, so the reported rank correlation does not depend on which interpolant is assumed",
    n_settings=len(rows3), theta_operating=TH_OP,
    min_log_corr=min(r["log_corr"] for r in rows3))
op = [r for r in rows3 if r["operating_point"]]
reg("geo_shape_matches_at_operating_point", all(r["log_corr"] > 0.99 for r in op),
    "at the angle the experiments actually run at, the two amplification curves agree in "
    "shape and not only in ordering; the agreement degrades only for near-antipodal "
    "endpoints, which the model does not produce",
    log_corr=[round(r["log_corr"], 5) for r in op],
    log_corr_at_theta_3=[round(r["log_corr"], 5) for r in rows3 if r["theta"] == 3.0])

flush_reg()
npass = sum(1 for g in REG if g["tag"] == "PASS")
log(passed=npass, total=len(REG), out=OUT)
