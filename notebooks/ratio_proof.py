# %% [code]
import os, csv, math, json, time
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "/kaggle/working/ratio" if os.path.isdir("/kaggle/working") else "runs/ratio"
os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
REG, TIMES = [], {}


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
        json.dump(dict(register=REG, times_min=TIMES), f, indent=1)


def tick(k):
    TIMES[k] = time.time()


def tock(k):
    TIMES[k] = (time.time() - TIMES[k]) / 60.0
    log(stage=k, minutes=TIMES[k])


def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)


def sym(M):
    return 0.5 * (M + M.transpose(-1, -2))


def skew(M):
    return 0.5 * (M - M.transpose(-1, -2))


def fro(M):
    return M.flatten(1).norm(dim=1).clamp_min(1e-30)


def spd_batch(B, n, dev, gen):
    A = torch.randn(B, n, n, generator=gen, device=dev, dtype=torch.float64)
    S = A @ A.transpose(-1, -2) / n + 0.35 * torch.eye(n, device=dev, dtype=torch.float64)
    return S


def msqrt_inv(S):
    w, Q = torch.linalg.eigh(S)
    w = w.clamp_min(1e-12)
    return Q @ torch.diag_embed(w.rsqrt()) @ Q.transpose(-1, -2)


def project_offspan(R, S0, S1):
    B0 = S0 / fro(S0)[:, None, None]
    p1 = S1 - (S1 * B0).flatten(1).sum(1)[:, None, None] * B0
    B1 = p1 / fro(p1)[:, None, None]
    R = R - (R * B0).flatten(1).sum(1)[:, None, None] * B0
    R = R - (R * B1).flatten(1).sum(1)[:, None, None] * B1
    return R


def rescale_psd(S0, S1, C, margin=0.92):
    Ki = msqrt_inv(S0) @ C @ msqrt_inv(S1)
    smax = torch.linalg.matrix_norm(Ki, ord=2).clamp_min(1e-30)
    g = torch.clamp(margin / smax, max=1.0)
    return C * g[:, None, None]


def endpoint_map(S0, S1, C, steps):
    B, n, _ = S0.shape
    I = torch.eye(n, device=S0.device, dtype=S0.dtype).expand(B, n, n).clone()
    M = I.clone()
    S = C + C.transpose(-1, -2)
    Ct = C.transpose(-1, -2)
    ts = np.linspace(0.0, 1.0, steps + 1)

    def Kof(t):
        V = (1 - t) ** 2 * S0 + t * (1 - t) * S + t ** 2 * S1
        G = (1 - t) * (Ct - S0) + t * (S1 - C)
        return torch.linalg.solve(V.transpose(-1, -2), G.transpose(-1, -2)).transpose(-1, -2)

    Ka = Kof(float(ts[0]))
    for i in range(steps):
        dt = ts[i + 1] - ts[i]
        Kb = Kof(float(ts[i + 1]))
        Mp = M + dt * (Ka @ M)
        M = M + dt * 0.5 * (Ka @ M + Kb @ Mp)
        Ka = Kb
    return M


def build(S0, S1, eps_off, eps_skew, gen, margin=0.92):
    B, n, _ = S0.shape
    dev, dt = S0.device, S0.dtype
    sg = 0.30 + 0.25 * torch.rand(B, 1, 1, generator=gen, device=dev, dtype=dt)
    ta = 0.30 + 0.25 * torch.rand(B, 1, 1, generator=gen, device=dev, dtype=dt)
    S_clean = sg * S0 + ta * S1
    R = sym(torch.randn(B, n, n, generator=gen, device=dev, dtype=dt))
    R = project_offspan(R, S0, S1)
    R = R / fro(R)[:, None, None] * fro(S_clean)[:, None, None]
    W = skew(torch.randn(B, n, n, generator=gen, device=dev, dtype=dt))
    W = W / fro(W)[:, None, None] * fro(S_clean)[:, None, None]
    S = S_clean + eps_off * R
    C = 0.5 * S + eps_skew * W
    C = rescale_psd(S0, S1, C, margin)
    return C


def violations(S0, S1, C):
    S = C + C.transpose(-1, -2)
    N = skew(C)
    Sp = project_offspan(S, S0, S1)
    return (fro(Sp) / fro(S)).cpu().numpy(), (fro(N) / fro(C)).cpu().numpy()


set_seed(0)
GEN = torch.Generator(device=DEVICE)
GEN.manual_seed(0)
log(device=DEVICE, torch=torch.__version__, out=OUT)

# %% [code]
TQ = np.linspace(0.0, 1.0, 20001)


def simpson(y):
    return float(np.trapz(y, TQ)) if len(TQ) % 2 == 0 else float(
        (TQ[1] - TQ[0]) / 3.0 * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum()))


def pair_AB(ai, bi, si, aj, bj, sj):
    t = TQ
    vi = (1 - t) ** 2 * ai + 2 * t * (1 - t) * si + t ** 2 * bi
    vj = (1 - t) ** 2 * aj + 2 * t * (1 - t) * sj + t ** 2 * bj
    dvi = -2 * (1 - t) * ai + 2 * (1 - 2 * t) * si + 2 * t * bi
    root = np.sqrt(vi * vj)
    A = simpson(((1 - 2 * t) - t * (1 - t) * dvi / vi) / root)
    Bq = simpson(1.0 / root)
    dvj = -2 * (1 - t) * aj + 2 * (1 - 2 * t) * sj + 2 * t * bj
    A_ibp = simpson(-0.5 * t * (1 - t) * (dvi / vi - dvj / vj) / root)
    return A, Bq, A_ibp


tick("pairwise")
n = 10
a = 0.35 + 4.0 * torch.rand(n, generator=GEN)
lam = 1.7
b = 0.35 + 4.0 * torch.rand(n, generator=GEN)
b[:3] = lam * a[:3]
sg, ta = 0.42, 0.47
s = sg * a + ta * b
S0 = torch.diag(a)[None]
S1 = torch.diag(b)[None]
C0 = torch.diag(s)[None]
Mi = torch.sqrt(b / a)

cases, rows = [], []
for _ in range(15):
    i = int(torch.randint(0, n, (1,), generator=GEN))
    j = int(torch.randint(0, n, (1,), generator=GEN))
    if i == j:
        j = (i + 1) % n
    cases.append((i, j, "p"))
    cases.append((i, j, "q"))
cases += [(0, 1, "p"), (0, 2, "p"), (4, 4, "p")]

EPSFD = 1e-5
STEPS = 4000
err_form, err_ibp, prop_zero = 0.0, 0.0, 0.0
for (i, j, kind) in cases:
    E = torch.zeros(1, n, n)
    if kind == "p":
        E[0, i, j] = 1.0
        E[0, j, i] = 1.0
    else:
        E[0, i, j] = 1.0
        E[0, j, i] = -1.0
    Mp = endpoint_map(S0, S1, C0 + EPSFD * E, STEPS)
    Mm = endpoint_map(S0, S1, C0 - EPSFD * E, STEPS)
    D_fd = ((Mp - Mm) / (2 * EPSFD))[0]
    D_th = torch.zeros(n, n)
    for (u, v_) in ([(i, j)] if i == j else [(i, j), (j, i)]):
        p = 0.5 * (E[0, u, v_] + E[0, v_, u]).item()
        q = 0.5 * (E[0, u, v_] - E[0, v_, u]).item()
        A, Bq, A2 = pair_AB(a[u].item(), b[u].item(), s[u].item(),
                            a[v_].item(), b[v_].item(), s[v_].item())
        D_th[u, v_] = math.sqrt(b[u].item() / a[v_].item()) * (p * A - q * Bq)
        err_ibp = max(err_ibp, abs(A - A2))
    den = max(float(D_th.norm()), 1e-12)
    e = float((D_fd - D_th).norm()) / max(den, 1e-9)
    if den < 1e-9:
        prop_zero = max(prop_zero, float(D_fd.norm()))
    else:
        err_form = max(err_form, e)
    rows.append(dict(i=i, j=j, kind=kind, fro_th=den, fro_fd=float(D_fd.norm()), rel_err=e))
save_csv("pairwise.csv", rows)
reg("ratio_pairwise_formula_matches_ode", err_form < 5e-3 and err_ibp < 1e-8,
    "in the commuting model the response acts pairwise, with the symmetric response given by "
    "the mismatch-rate integral and the skew response by the positive kernel, matching the "
    "finite-differenced endpoint map",
    max_rel_err=err_form, max_ibp_identity_err=err_ibp, n_cases=len(cases))
reg("ratio_sym_vanishes_when_variances_proportional", prop_zero < 5e-6,
    "a symmetric perturbation on a pair whose interpolant variances are proportional in time "
    "produces no first-order response, including off the span and on the diagonal",
    max_fd_norm_on_null_cases=prop_zero)
tock("pairwise")
flush_reg()

# %% [code]
tick("equal_marginals")
n = 12
Sd = spd_batch(1, n, DEVICE, GEN)
c_tot = 0.85
C0 = 0.5 * c_tot * Sd + 0.5 * c_tot * Sd


def i_of_c(c):
    t = TQ
    return simpson(1.0 / (1 - 2 * (1 - c) * t * (1 - t)))


Esym = sym(torch.randn(1, n, n, generator=GEN, device=DEVICE))
Esym = Esym / fro(Esym)[:, None, None]
Eskw = skew(torch.randn(1, n, n, generator=GEN, device=DEVICE))
Eskw = Eskw / fro(Eskw)[:, None, None]

Mp = endpoint_map(Sd, Sd, C0 + EPSFD * Esym, STEPS)
Mm = endpoint_map(Sd, Sd, C0 - EPSFD * Esym, STEPS)
D_sym = float(((Mp - Mm) / (2 * EPSFD)).norm())
Mp = endpoint_map(Sd, Sd, C0 + EPSFD * Eskw, STEPS)
Mm = endpoint_map(Sd, Sd, C0 - EPSFD * Eskw, STEPS)
D_skw = ((Mp - Mm) / (2 * EPSFD))[0]
D_pred = -i_of_c(c_tot) * (Eskw[0] @ torch.linalg.inv(Sd[0]))
rel = float((D_skw - D_pred).norm() / D_pred.norm())
reg("ratio_equal_marginals_sym_response_is_zero", D_sym < 1e-5,
    "with homogeneous marginals every symmetric coupling perturbation, in or out of the span, "
    "moves the endpoint map by zero at first order",
    fd_norm=D_sym, skew_norm_same_setup=float(D_skw.norm()))
reg("ratio_equal_marginals_skew_response_closed_form", rel < 1e-3,
    "with homogeneous marginals the skew response is the explicit closed form given by the "
    "arctangent integral times N Sigma inverse",
    rel_err=rel, i_of_c=i_of_c(c_tot))
tock("equal_marginals")
flush_reg()

# %% [code]
tick("scaling")
n, B = 16, 64
STEPS_SW = 1500
EPSV = 0.05
DELTAS = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
S0 = spd_batch(B, n, DEVICE, GEN)
S1full = spd_batch(B, n, DEVICE, GEN)
rows = []
ratios = []
for d in DELTAS:
    S1 = S0 + d * (S1full - S0)
    gen_d = torch.Generator(device=DEVICE)
    gen_d.manual_seed(1234)
    C_cl = build(S0, S1, 0.0, 0.0, gen_d)
    M0 = endpoint_map(S0, S1, C_cl, STEPS_SW)
    resp = {}
    for fam, eo, es in [("off", EPSV, 0.0), ("skw", 0.0, EPSV)]:
        gen_d = torch.Generator(device=DEVICE)
        gen_d.manual_seed(1234)
        C = build(S0, S1, eo, es, gen_d)
        M = endpoint_map(S0, S1, C, STEPS_SW)
        drift = (fro(M - M0) / fro(M0)).cpu().numpy()
        vo, vs = violations(S0, S1, C)
        viol = vo if fam == "off" else vs
        resp[fam] = np.median(drift / np.clip(viol, 1e-12, None))
    ratios.append(resp["skw"] / resp["off"])
    rows.append(dict(delta=d, resp_off=float(resp["off"]), resp_skw=float(resp["skw"]),
                     ratio=float(ratios[-1])))
    log(delta=d, resp_off=float(resp["off"]), resp_skw=float(resp["skw"]),
        ratio=float(ratios[-1]))
save_csv("scaling.csv", rows)
lx, ly = np.log10(DELTAS), np.log10(ratios)
Amat = np.stack([lx, np.ones_like(lx)], 1)
slope, icpt = np.linalg.lstsq(Amat, ly, rcond=None)[0]
resid = float(np.sqrt(np.mean((Amat @ np.array([slope, icpt]) - ly) ** 2)))
reg("ratio_scales_as_inverse_marginal_mismatch", -1.3 < slope < -0.7 and resid < 0.12,
    "the skew-to-span response ratio diverges as the marginals homogenize, scaling as one over "
    "the marginal separation, so the measured 6.585 is a property of the sampled ensemble at "
    "its own separation and not a universal constant",
    slope=float(slope), rms_resid=resid,
    ratio_at_delta_1=float(ratios[-1]), ratio_at_delta_005=float(ratios[0]))
reg("ratio_at_full_separation_matches_paper", 4.5 < ratios[-1] < 9.0,
    "at the ensemble's own separation the sweep reproduces the measured ratio of about six "
    "and a half",
    ratio=float(ratios[-1]), paper_value=6.585)
tock("scaling")
flush_reg()
npass = sum(1 for g in REG if g["tag"] == "PASS")
log(passed=npass, total=len(REG))
