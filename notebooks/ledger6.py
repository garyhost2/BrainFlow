# %% [code]
import os, csv, math, json, time
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

FAST = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "/kaggle/working/ledger6" if os.path.isdir("/kaggle/working") else "runs/ledger6"
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
os.makedirs(os.path.join(OUT, "results"), exist_ok=True)

STAGES = os.environ.get(
    "LEDGER_STAGES",
    "flat,sigflat,core,grid,noise,fix,metric,threshold,guidance,discrete,scaling,lr2,real"
).split(",")

D_OP, A_OP, BETA_OP, J_OP = 256, 0.37, 0.8, 0.15
STEPS = 600 if FAST else 3000
STEPS_LONG = 2400 if FAST else 12000
BATCH = 256
HIDDEN = 512
BASE_WIDTH = 256
BLOCKS = 3
BASE_LR = 3e-4
ODE_K = 32
EVAL_N = 384 if FAST else 768
KDIV = 4 if FAST else 6
EPS = 1e-6

SIG_POLAR = [0.0, 0.01, 0.03] if FAST else [0.0, 0.005, 0.01, 0.03, 0.06, 0.1, 0.2, 0.4]
RHO_CPL = [0.0, 1.0] if FAST else [0.0, 0.25, 0.5, 0.75, 1.0]
BETAS = [0.0, 0.6, 0.95] if FAST else [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95]
K_LIST = [1, 8, 32] if FAST else [1, 2, 4, 8, 16, 32, 64]
ALPHAS = [0.0, 0.5, 0.9] if FAST else [0.0, 0.25, 0.5, 0.7, 0.85, 0.95]
SIG_AUX = [0.0, 0.3, 1.0] if FAST else [0.0, 0.05, 0.1, 0.2, 0.35, 0.6, 1.0, 1.6]
TRUNC = [0.0, 0.1] if FAST else [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35]
CHURNS = [0.0, 0.05, 0.15]
WIDTHS = [256, 512] if FAST else [256, 512, 1024, 2048]
LR_GRID = [1e-4, 3e-4] if FAST else [3e-5, 1e-4, 3e-4, 1e-3]
GUID_W = [1.0, 2.0, 3.0]
LAMBDAS = [0.0, 0.5, 1.0] if FAST else [0.0, 0.25, 0.5, 0.75, 1.0]
A_GRID = [0.37] if FAST else [0.15, 0.25, 0.37, 0.5, 0.7, 0.9]
J_GRID = [0.15] if FAST else [0.0, 0.05, 0.15, 0.3, 0.5, 0.8]
SEEDS = [0, 1] if FAST else [0, 1, 2]
SEEDS_SCALE = [0] if FAST else [0, 1, 2, 3, 4]
ORACLE_STEPS = 1200 if FAST else 3500
ORACLE_STEPS_GRID = 800 if FAST else 1800

LR2_GRID = [1e-4, 1e-3]
LR2_WIDTHS = [256, 512] if FAST else [256, 512, 1024, 2048]
LR2_SEEDS = [0, 1] if FAST else [0, 1, 2, 3, 4]

DIST_NCOND = 8 if FAST else 16
DIST_NPER = 64 if FAST else 128
DIST_KNN = 5
CLIP_N = 800 if FAST else 6000
CLIP_STEPS = 500 if FAST else 3000

TRAPZ = getattr(np, "trapezoid", np.trapz)
REG = []
TIMES = {}


def log(**kw):
    print(" ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}"
                   for k, v in kw.items()), flush=True)


def reg(name, ok, predicts, **kw):
    REG.append(dict(tag="PASS" if ok else "WARN", name=name, predicts=predicts,
                    detail={k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                            for k, v in kw.items()}))
    log(reg=REG[-1]["tag"], name=name, **kw)


def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)


def tick(name):
    TIMES[name] = time.time()


def tock(name):
    dt = time.time() - TIMES[name]
    TIMES[name] = dt
    log(stage=name, minutes=dt / 60.0)


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


def save_fig(name):
    plt.savefig(os.path.join(OUT, "figures", name), dpi=150, bbox_inches="tight")
    plt.close()


def flush_reg():
    with open(os.path.join(OUT, "results", "register.json"), "w") as f:
        json.dump(dict(register=REG, times_min={k: v / 60.0 for k, v in TIMES.items()
                                                if isinstance(v, float) and v < 1e8}), f, indent=1)


def spearman(a, b):
    def rk(v):
        return np.argsort(np.argsort(np.asarray(v, float))).astype(float)
    ra, rb = rk(a), rk(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = math.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


set_seed(0)
log(device=DEVICE, torch=torch.__version__, out=OUT, fast=FAST, stages=len(STAGES),
    gpu=torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu")

# %% [code]
def normalize(x):
    return F.normalize(x, dim=-1)


def random_sphere(shape, device):
    return normalize(torch.randn(shape, device=device))


def project_tangent(v, z):
    return v - (v * z).sum(-1, keepdim=True) * z


def rand_tangent_dir(z):
    return normalize(project_tangent(torch.randn_like(z), z))


def exp_map(z, w):
    w = project_tangent(w, z)
    n = w.norm(dim=-1, keepdim=True)
    out = torch.cos(n) * z + torch.sin(n) * (w / n.clamp_min(EPS))
    return normalize(torch.where(n < EPS, z, out))


def _angle(z0, z1):
    cs = (z0 * z1).sum(-1, keepdim=True).clamp(-1 + EPS, 1 - EPS)
    th = cs.arccos()
    return th, th.sin().clamp_min(EPS)


def slerp(z0, z1, t):
    th, sn = _angle(z0, z1)
    val = (((1 - t) * th).sin() * z0 + (t * th).sin() * z1) / sn
    return torch.where(th < EPS, normalize((1 - t) * z0 + t * z1), val)


def slerp_velocity(z0, z1, t):
    th, sn = _angle(z0, z1)
    val = (th / sn) * (-((1 - t) * th).cos() * z0 + (t * th).cos() * z1)
    return torch.where(th < EPS, z1 - z0, val)


def resultant_A(d, kappa):
    if kappa <= 0:
        return 0.0
    nu = d / 2.0
    acc = 0.0
    for k in range(int(max(500, 2.2 * kappa + 100)), 0, -1):
        acc = 1.0 / (2.0 * (nu + k) / kappa + acc)
    return 1.0 / (2.0 * nu / kappa + acc)


_KC = {}


def kappa_from_A(d, A):
    key = (d, round(A, 6))
    if key in _KC:
        return _KC[key]
    if A <= 0:
        _KC[key] = 0.0
        return 0.0
    lo, hi = 1e-4, 10.0
    while resultant_A(d, hi) < A and hi < 1e8:
        hi *= 2
    for _ in range(140):
        mid = 0.5 * (lo + hi)
        if resultant_A(d, mid) < A:
            lo = mid
        else:
            hi = mid
    _KC[key] = 0.5 * (lo + hi)
    return _KC[key]


_BETA_CACHE = {}


@torch.no_grad()
def sample_vmf(mu, kappa, over=4):
    B, d = mu.shape
    if kappa <= 0:
        return random_sphere((B, d), mu.device)
    key = (d, str(mu.device))
    if key not in _BETA_CACHE:
        a = torch.tensor(0.5 * (d - 1), device=mu.device)
        _BETA_CACHE[key] = torch.distributions.Beta(a, a)
    bd = _BETA_CACHE[key]
    b = (d - 1) / (2 * kappa + math.sqrt(4 * kappa ** 2 + (d - 1) ** 2))
    x0 = (1 - b) / (1 + b)
    cc = kappa * x0 + (d - 1) * math.log(max(1 - x0 * x0, 1e-300))
    m = over * B
    Z = bd.sample((m,)).double()
    Wc = (1 - (1 + b) * Z) / (1 - (1 - b) * Z)
    U = torch.rand(m, device=mu.device).double()
    ok = (kappa * Wc + (d - 1) * torch.log((1 - x0 * Wc).clamp_min(1e-300)) - cc) >= U.log()
    score = ok.double() + 1e-3 * torch.rand(m, device=mu.device).double()
    Wf = Wc[score.topk(B).indices].float().unsqueeze(-1).clamp(-1 + 1e-7, 1 - 1e-7)
    return normalize(Wf * mu + torch.sqrt((1 - Wf * Wf).clamp_min(0)) * rand_tangent_dir(mu))


_kap_op = kappa_from_A(D_OP, A_OP)
_mu = random_sphere((20000, D_OP), DEVICE)
log(geometry="ready", kappa_op=_kap_op,
    vmf_A_measured=float((sample_vmf(_mu, _kap_op) * _mu).sum(-1).mean()), A_target=A_OP)

# %% [code]
@dataclass
class Task:
    d: int = D_OP
    A: float = A_OP
    beta: float = BETA_OP
    jit: float = J_OP
    kappa: float = 0.0
    mix_lambda: float = None
    rho_couple: float = 1.0
    sig_polar: float = 0.0

    def resolve(self):
        self.kappa = kappa_from_A(self.d, self.A)
        return self


def ladder(A, beta, j):
    s = math.sqrt(max(0.0, 1 - beta * beta))
    return dict(src=A * (s * math.cos(j) + beta * math.sin(j)), est_c=A * s, est_m=A,
                samp_cond=A * A * s * s, samp_aug=A * A)


def law_markov(A, beta):
    s2 = max(0.0, 1 - beta * beta)
    return A * (A * s2 + beta * math.sqrt(max(0.0, 1 - A * A * s2)))


def beta_star(A, j):
    if A >= math.cos(j):
        return float("nan")
    f = lambda b: law_markov(A, b) - ladder(A, b, j)["src"]
    lo, hi = 0.0, 0.99
    if f(lo) * f(hi) > 0:
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


@torch.no_grad()
def make_batch(task, B, device, c=None):
    if c is None:
        c = random_sphere((B, task.d), device)
    xi_s = rand_tangent_dir(c)
    if task.rho_couple >= 1.0:
        xi_t = xi_s
    else:
        r = task.rho_couple
        xi_t = normalize(project_tangent(
            r * xi_s + math.sqrt(max(0.0, 1 - r * r)) * rand_tangent_dir(c), c))
    s = math.sqrt(max(0.0, 1 - task.beta ** 2))
    m = normalize(s * c + task.beta * xi_t)
    z1 = sample_vmf(m, task.kappa)
    if task.sig_polar > 0:
        phi = (task.jit + task.sig_polar * torch.randn(B, 1, device=device)).clamp(1e-3, math.pi - 1e-3)
    else:
        phi = torch.full((B, 1), float(task.jit), device=device)
    z0 = normalize(torch.cos(phi) * c + torch.sin(phi) * xi_s)
    if task.mix_lambda is not None:
        z0 = slerp(random_sphere((B, task.d), device), z0,
                   torch.full((B, 1), float(task.mix_lambda), device=device))
    return c, xi_s, xi_t, m, z0, z1


task0 = Task().resolve()
L0 = ladder(A_OP, BETA_OP, J_OP)
LAW0 = law_markov(A_OP, BETA_OP)
BSTAR = beta_star(A_OP, J_OP)
S_OP = math.sqrt(1 - BETA_OP ** 2)
AS_OP = A_OP * S_OP
log(kappa=task0.kappa, **{k: float(v) for k, v in L0.items()}, law=LAW0, beta_star=BSTAR, As=AS_OP)

# %% [code]
def w1_1d(a, b):
    n = min(a.numel(), b.numel())
    return float((a.flatten()[:n].sort().values - b.flatten()[:n].sort().values).abs().mean())


def energy_distance(x, y):
    nx, ny = x.shape[0], y.shape[0]
    return float(2 * torch.cdist(x, y).mean()
                 - torch.cdist(x, x).sum() / max(1, nx * (nx - 1))
                 - torch.cdist(y, y).sum() / max(1, ny * (ny - 1)))


def coverage(real, fake, k=DIST_KNN):
    k = min(k, real.shape[0] - 1)
    dr = torch.cdist(real, real)
    dr.fill_diagonal_(float("inf"))
    rad = dr.topk(k, largest=False).values[:, -1]
    return float((torch.cdist(real, fake) < rad[:, None]).any(1).float().mean())


@torch.no_grad()
def dist_report(sample_fn, task, n_cond=DIST_NCOND, n_per=DIST_NPER, seed=1234):
    set_seed(seed)
    cb = random_sphere((n_cond, task.d), DEVICE)
    c = cb.repeat_interleave(n_per, 0)
    _, xs, xt, m, z0, _ = make_batch(task, c.shape[0], DEVICE, c=c)
    xf = sample_fn(c, xs, xt, m, z0)
    _, _, _, _, _, ra = make_batch(task, c.shape[0], DEVICE, c=c)
    _, _, _, _, _, rb = make_batch(task, c.shape[0], DEVICE, c=c)
    acc = {k: [] for k in ["w1_u", "w1_null", "energy", "energy_null", "cov", "cov_null",
                           "polar_mu", "polar_sd", "polar_sd_true"]}
    for i in range(n_cond):
        sl = slice(i * n_per, (i + 1) * n_per)
        ci = cb[i:i + 1]
        g, r1, r2 = xf[sl], ra[sl], rb[sl]
        pg, p1, p2 = (g * ci).sum(-1), (r1 * ci).sum(-1), (r2 * ci).sum(-1)
        acc["w1_u"].append(w1_1d(pg, p1))
        acc["w1_null"].append(w1_1d(p2, p1))
        acc["energy"].append(energy_distance(g, r1))
        acc["energy_null"].append(energy_distance(r2, r1))
        acc["cov"].append(coverage(r1, g))
        acc["cov_null"].append(coverage(r1, r2))
        acc["polar_mu"].append(float(pg.mean()))
        acc["polar_sd"].append(float(pg.std()))
        acc["polar_sd_true"].append(float(p1.std()))
    return {k: float(np.mean(v)) for k, v in acc.items()}


log(metrics="ready")

# %% [code]
if "flat" in STAGES:
    tick("flat")
    rng = np.random.default_rng(0)

    def flat_map(S0, S1, C, n=4000):
        d = S0.shape[0]
        ts = np.linspace(0, 1, n + 1)
        M = np.eye(d)
        res = 0.0
        for i in range(n):
            t = 0.5 * (ts[i] + ts[i + 1])
            dt = ts[i + 1] - ts[i]
            V = (1 - t) ** 2 * S0 + t * (1 - t) * (C + C.T) + t ** 2 * S1
            G = (1 - t) * (C.T - S0) + t * (S1 - C)
            Vp = -2 * (1 - t) * S0 + (1 - 2 * t) * (C + C.T) + 2 * t * S1
            res = max(res, float(np.abs(G - 0.5 * Vp - 0.5 * (C.T - C)).max()))
            M = (np.eye(d) + dt * (G @ np.linalg.inv(V))) @ M
        return M, res

    def brenier(S0, S1):
        def ms(S):
            w, Q = np.linalg.eigh(S)
            return Q @ np.diag(np.sqrt(np.clip(w, 0, None))) @ Q.T
        r = ms(S0)
        ri = np.linalg.inv(r)
        return ri @ ms(r @ S1 @ r) @ ri

    def mineig(S0, S1, C):
        return float(np.linalg.eigvalsh(np.block([[S0, C], [C.T, S1]])).min())

    rows = []
    worst_id = 0.0
    for _ in range(300):
        dd = int(rng.integers(1, 6))
        L = rng.normal(size=(2 * dd, 2 * dd))
        Jm = L @ L.T + 0.3 * np.eye(2 * dd)
        S0, S1, C = Jm[:dd, :dd], Jm[dd:, dd:], Jm[:dd, dd:]
        t = float(rng.uniform(0.05, 0.95))
        G = (1 - t) * (C.T - S0) + t * (S1 - C)
        Vp = -2 * (1 - t) * S0 + (1 - 2 * t) * (C + C.T) + 2 * t * S1
        worst_id = max(worst_id, float(np.abs(G - 0.5 * Vp - 0.5 * (C.T - C)).max()))
    reg("L1_identity", worst_id < 1e-10,
        "G(t) - V'(t)/2 equals (C^T-C)/2 exactly, for any joint Gaussian",
        max_resid=worst_id)

    d3 = 3
    Q = np.linalg.qr(rng.normal(size=(d3, d3)))[0]
    Q2 = np.linalg.qr(rng.normal(size=(d3, d3)))[0]
    S0 = np.diag([1.0, 0.4, 0.15])
    S1c = np.diag([0.8, 0.5, 0.25])
    S1n = Q @ np.diag([0.8, 0.5, 0.25]) @ Q.T
    SK = rng.normal(size=(d3, d3))
    SK = 0.5 * (SK - SK.T)
    E = Q2 @ np.diag([1.0, 0.6, 0.3]) @ Q2.T
    arms = {
        "sym_span_commuting": (S0, S1c, lambda a: a * 0.5 * (S0 + S1c), 1, 1),
        "sym_span_noncommuting": (S0, S1n, lambda a: a * 0.5 * (S0 + S1n), 1, 1),
        "sym_offspan": (S0, S1c, lambda a: a * 0.35 * E, 1, 0),
        "skew_span": (S0, S1n, lambda a: 0.15 * (S0 + S1n) + a * SK, 0, 1),
        "skew_offspan": (S0, S1c, lambda a: 0.105 * E + a * SK, 0, 0),
    }
    drift = {}
    for arm, (A0, A1, Cf, hi, hii) in arms.items():
        Mref = None
        for a in [0.0, 0.1, 0.2, 0.3, 0.45, 0.6]:
            C = Cf(a)
            if mineig(A0, A1, C) <= 1e-6:
                continue
            M, res = flat_map(A0, A1, C)
            if Mref is None:
                Mref = M
            dr = float(np.linalg.norm(M - Mref) / np.linalg.norm(Mref))
            T = brenier(A0, A1)
            rows.append(dict(arm=arm, a=a, hyp_i=hi, hyp_ii=hii, drift=dr, resid=res,
                             to_brenier=float(np.linalg.norm(M - T) / np.linalg.norm(T)),
                             asym=float(np.linalg.norm(M - M.T)),
                             commutator=float(np.abs(A0 @ A1 - A1 @ A0).max()),
                             pushfwd=float(np.abs(M @ A0 @ M.T - A1).max())))
            drift[arm] = max(drift.get(arm, 0.0), dr)
        log(arm=arm, hyp_i=hi, hyp_ii=hii, max_drift=drift.get(arm, float("nan")))
    save_csv("flat_arms.csv", rows)

    TOL = 5e-3
    inv = max(drift["sym_span_commuting"], drift["sym_span_noncommuting"])
    reg("T411_both_hyp_invariant", inv < TOL,
        "hypotheses (i) and (ii) together give coupling-independence, and marginal commutation "
        "is irrelevant to it",
        commuting=drift["sym_span_commuting"], noncommuting=drift["sym_span_noncommuting"])
    reg("T411_hyp_ii_fails_moves", drift["sym_offspan"] > 10 * inv,
        "C symmetric but S outside span{S0,S1} breaks invariance even with commuting marginals",
        drift=drift["sym_offspan"], invariant_arms=inv, ratio=drift["sym_offspan"] / inv)
    reg("T411_hyp_i_fails_moves", drift["skew_span"] > 10 * inv,
        "S in span but C asymmetric breaks invariance, separating (i) from (ii)",
        drift=drift["skew_span"], ratio=drift["skew_span"] / inv)
    reg("T411_neither_moves", drift["skew_offspan"] > 10 * inv,
        "with both hypotheses violated the map moves",
        drift=drift["skew_offspan"], ratio=drift["skew_offspan"] / inv)
    br = [r["to_brenier"] for r in rows if r["arm"] == "sym_span_noncommuting"]
    reg("C413_noncommuting_not_brenier", max(br) > 0.01,
        "commuting marginals buy the Brenier map, which is separate from invariance",
        to_brenier=max(br))
    pf = max(r["pushfwd"] for r in rows)
    reg("P49_marginal_matching_any_skew", pf < 1e-3,
        "marginal matching holds with no hypothesis on the skew part of C",
        max_pushforward_err=pf)
    asym_nc = max(r["asym"] for r in rows if r["arm"] == "sym_span_noncommuting")
    reg("tangency_independent_of_invariance", asym_nc > 0.01,
        "under (i) and (ii) the field is a gradient field iff the marginals commute, so "
        "invariance without tangency is realisable",
        map_asymmetry=asym_nc, drift=drift["sym_span_noncommuting"])

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    for arm in arms:
        sel = [r for r in rows if r["arm"] == arm]
        if sel:
            ax.semilogy([r["a"] for r in sel], [max(r["drift"], 1e-16) for r in sel], "o-", label=arm)
    ax.axhline(TOL, ls="--", c="k", lw=1)
    ax.set_xlabel("coupling strength")
    ax.set_ylabel("relative endpoint-map drift")
    ax.legend(fontsize=7)
    save_fig("F1_flat_arms.png")
    tock("flat")
    flush_reg()

# %% [code]
if "sigflat" in STAGES:
    tick("sigflat")
    rng = np.random.default_rng(1)
    dF, nF = D_OP, 120000
    kap = kappa_from_A(dF, A_OP)
    cvec = np.zeros(dF)
    cvec[0] = 1.0
    Pm = np.eye(dF) - np.outer(cvec, cvec)

    def wood_W(kappa, d, size, rng):
        b = (d - 1) / (2 * kappa + np.sqrt(4 * kappa ** 2 + (d - 1) ** 2))
        x0 = (1 - b) / (1 + b)
        cc = kappa * x0 + (d - 1) * np.log(1 - x0 * x0)
        out = np.empty(size)
        f = 0
        while f < size:
            m = (size - f) * 2 + 64
            Z = rng.beta((d - 1) / 2.0, (d - 1) / 2.0, m)
            W = (1 - (1 + b) * Z) / (1 - (1 - b) * Z)
            U = rng.random(m)
            keep = kappa * W + (d - 1) * np.log(np.maximum(1 - x0 * W, 1e-300)) - cc >= np.log(U)
            tk = W[keep][: size - f]
            out[f:f + tk.size] = tk
            f += tk.size
        return out

    def un(x):
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    def ab(S):
        a = float(cvec @ S @ cvec)
        b = float(np.trace(Pm @ S) / (dF - 1))
        return a, b, float(np.abs(S - (a * np.outer(cvec, cvec) + b * Pm)).max())

    def map2d(S0, S1, C, ns=4000):
        M = np.eye(2)
        ts = np.linspace(0, 1, ns + 1)
        for i in range(ns):
            t = 0.5 * (ts[i] + ts[i + 1])
            dt = ts[i + 1] - ts[i]
            V = (1 - t) ** 2 * S0 + t * (1 - t) * (C + C.T) + t ** 2 * S1
            G = (1 - t) * (C.T - S0) + t * (S1 - C)
            M = (np.eye(2) + dt * (G @ np.linalg.inv(V))) @ M
        return M

    rows = []
    for sig in SIG_POLAR:
        xi = un(rng.standard_normal((nF, dF)) @ Pm.T)
        phi = J_OP + sig * rng.standard_normal(nF)
        z0 = np.cos(phi)[:, None] * cvec[None, :] + np.sin(phi)[:, None] * xi
        m = un(S_OP * cvec + BETA_OP * xi)
        W = wood_W(kap, dF, nF, rng)
        v = rng.standard_normal((nF, dF))
        v = un(v - (v * m).sum(-1, keepdims=True) * m)
        z1 = W[:, None] * m + np.sqrt(np.maximum(1 - W ** 2, 0))[:, None] * v
        S0f, S1f = np.cov(z0.T), np.cov(z1.T)
        Cf = (z0 - z0.mean(0)).T @ (z1 - z1.mean(0)) / nF
        a0, b0, _ = ab(S0f)
        a1, b1, _ = ab(S1f)
        ac, bc, _ = ab(Cf)
        det = a0 * b1 - a1 * b0
        gp = math.sqrt(a1 / a0) if a0 > 1e-14 else float("inf")
        go = float(map2d(np.diag([a0, b0]), np.diag([a1, b1]), np.diag([ac, bc]))[0, 0]) \
            if a0 > 1e-14 else float("inf")
        rows.append(dict(sig_ang=sig, sd0=float(np.cos(phi).std()), sd1=float(z1[:, 0].std()),
                         a0=a0, b0=b0, a1=a1, b1=b1, pencil_det=det, gain_pred=gp, gain_ode=go,
                         polar_mean_target=float(z1[:, 0].mean()),
                         csym=float(np.abs(Cf - Cf.T).max())))
        log(sig_ang=sig, sd0=rows[-1]["sd0"], a0=a0, det=det, gain=go,
            polar_mu=rows[-1]["polar_mean_target"])
    save_csv("sigma_flat.csv", rows)

    pos = [r for r in rows if r["sig_ang"] > 0]
    reg("A41_restored_by_sigma", all(r["a0"] > 1e-12 for r in pos),
        "any sigma_polar > 0 restores rank along c, so V(0) is positive definite",
        min_a0=min(r["a0"] for r in pos), a0_at_zero=rows[0]["a0"])
    reg("pencil_nondegenerate_everywhere", all(abs(r["pencil_det"]) > 1e-12 for r in rows),
        "hypotheses (i) and (ii) hold at every sigma including zero",
        min_absdet=min(abs(r["pencil_det"]) for r in rows),
        sign_change=int(min(r["pencil_det"] for r in rows) < 0 < max(r["pencil_det"] for r in rows)))
    gr = [abs(r["gain_ode"] * r["sd0"] / r["sd1"] - 1) for r in pos]
    reg("gain_law", max(gr) < 0.02,
        "polar gain equals sd(<z1,c>)/sd(<z0,c>), diverging as sigma tends to zero",
        max_rel_err=max(gr))
    pm = [r["polar_mean_target"] for r in rows]
    reg("polar_mean_is_As", max(abs(p - AS_OP) for p in pm) < 0.005,
        "the mean polar landing is As at every sigma, so the sigma to zero limit identifies "
        "the collapse point",
        max_dev=max(abs(p - AS_OP) for p in pm), As=AS_OP)

    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ax[0].loglog([r["sig_ang"] for r in pos], [r["gain_ode"] for r in pos], "o-")
    ax[0].set_xlabel("sigma_polar")
    ax[0].set_ylabel("polar gain")
    ax[1].semilogx([r["sig_ang"] for r in pos], [r["pencil_det"] for r in pos], "o-")
    ax[1].axhline(0, c="k", lw=0.8)
    ax[1].set_xlabel("sigma_polar")
    ax[1].set_ylabel("pencil determinant")
    save_fig("F2_sigma_flat.png")
    tock("sigflat")
    flush_reg()

# %% [code]
class TimeEmb(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.register_buffer("freq", torch.exp(torch.linspace(0, math.log(200.0), dim // 2)))

    def forward(self, t):
        a = t[:, None] * self.freq[None, :]
        return torch.cat([a.sin(), a.cos()], -1)


class Block(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.n = nn.LayerNorm(w, elementwise_affine=False)
        self.net = nn.Sequential(nn.Linear(w, 4 * w), nn.SiLU(), nn.Linear(4 * w, w))
        self.ada = nn.Linear(w, 3 * w)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, h, te):
        sh, sc, g = self.ada(F.silu(te)).chunk(3, -1)
        return h + g * self.net(self.n(h) * (1 + sc) + sh)


class FieldNet(nn.Module):
    def __init__(self, d, width, blocks, see_source, mup=True, base_width=BASE_WIDTH):
        super().__init__()
        self.d, self.see_source, self.mup = d, see_source, mup
        self.mult = width / base_width
        self.temb = TimeEmb(64)
        self.tproj = nn.Linear(64, width)
        self.inp = nn.Linear(d * (3 if see_source else 2), width)
        self.blocks = nn.ModuleList([Block(width) for _ in range(blocks)])
        self.out = nn.Linear(width, d)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.null_cond = nn.Parameter(torch.zeros(d))

    def param_groups(self, base_lr=BASE_LR):
        if not self.mup:
            return [dict(params=list(self.parameters()), lr=base_lr)]
        inp = list(self.inp.parameters()) + list(self.tproj.parameters())
        outp = list(self.out.parameters())
        ids = {id(p) for p in inp + outp}
        hid = [p for p in self.parameters() if id(p) not in ids]
        return [dict(params=inp, lr=base_lr),
                dict(params=hid, lr=base_lr / self.mult),
                dict(params=outp, lr=base_lr / self.mult)]

    def raw(self, z, t, c, aux=None, drop=None):
        cc = c if drop is None else torch.where(drop[:, None], self.null_cond.expand_as(c), c)
        parts = [z, cc]
        if self.see_source:
            parts.append(aux if aux is not None else torch.zeros_like(z))
        h = self.inp(torch.cat(parts, -1))
        te = self.tproj(self.temb(t))
        for b in self.blocks:
            h = b(h, te)
        return self.out(h)


def make_vel(net, c, aux, guidance=1.0):
    def vel(z, tq):
        t = torch.full((z.shape[0],), float(tq), device=z.device)
        vc = project_tangent(net.raw(z, t, c, aux), z)
        if guidance == 1.0:
            return vc
        dr = torch.ones(z.shape[0], dtype=torch.bool, device=z.device)
        vu = project_tangent(net.raw(z, t, c, aux, dr), z)
        return vu + guidance * (vc - vu)
    return vel


@torch.no_grad()
def integrate(vel_fn, z0, steps=ODE_K, heun=True, churn=0.0, t_start=0.0):
    z = z0.clone()
    ts = np.linspace(t_start, 1.0, steps + 1)
    for i in range(steps):
        t, tn = float(ts[i]), float(ts[i + 1])
        dt = tn - t
        v = vel_fn(z, t)
        if heun:
            zp = exp_map(z, dt * v)
            v = 0.5 * (v + project_tangent(vel_fn(zp, tn), z))
        z = exp_map(z, dt * v)
        if churn > 0:
            z = exp_map(z, churn * math.sqrt(abs(dt)) * rand_tangent_dir(z))
    return z


def sf_ode(net, steps=ODE_K, churn=0.0, guidance=1.0, t_start=0.0, aux_sigma=0.0):
    def f(c, xs, xt, m, z0):
        aux = None
        if net.see_source:
            aux = z0 if aux_sigma <= 0 else exp_map(z0, aux_sigma * rand_tangent_dir(z0))
        return integrate(make_vel(net, c, aux, guidance), z0, steps, churn=churn, t_start=t_start)
    return f


def sf_alpha(net, alpha, steps=ODE_K):
    def f(c, xs, xt, m, z0):
        B = c.shape[0]
        ts = np.linspace(0.0, 1.0, steps + 1)
        cache = {0.0: z0.clone()}
        z = z0.clone()
        for i in range(steps):
            t, tn = float(ts[i]), float(ts[i + 1])
            key = min(cache, key=lambda k: abs(k - alpha * t))
            tt = torch.full((B,), t, device=z.device)
            v = project_tangent(net.raw(z, tt, c, cache[key]), z)
            z = exp_map(z, (tn - t) * v)
            cache[tn] = z.clone()
        return z
    return f


def sf_identity():
    return lambda c, xs, xt, m, z0: z0


def sf_const_c():
    return lambda c, xs, xt, m, z0: c


def sf_bayes_m():
    return lambda c, xs, xt, m, z0: m


def sf_true_post(task):
    def f(c, xs, xt, m, z0):
        return make_batch(task, c.shape[0], c.device, c=c)[5]
    return f


@torch.no_grad()
def evaluate(sample_fn, task, n=EVAL_N, k_div=KDIV):
    cb = random_sphere((n, task.d), DEVICE)
    c = cb.repeat_interleave(k_div, 0)
    cc, xs, xt, m, z0, z1 = make_batch(task, c.shape[0], DEVICE, c=c)
    xf = sample_fn(cc, xs, xt, m, z0)
    pc = (xf * cc).sum(-1)
    ps = (xf * xs).sum(-1)
    pt = (xf * xt).sum(-1)
    g = normalize(xf).view(n, k_div, -1)
    pw = torch.einsum("nkd,nld->nkl", g, g)
    off = (pw.sum((1, 2)) - k_div) / (k_div * (k_div - 1))
    return dict(fid=float((xf * z1).sum(-1).mean()), align_c=float(pc.mean()),
                align_xi_s=float(ps.mean()), align_xi_t=float(pt.mean()),
                align_m=float((xf * m).sum(-1).mean()),
                span_src=float((pc ** 2 + ps ** 2).mean()),
                span_tgt=float((pc ** 2 + pt ** 2).mean()),
                polar_sd_out=float(pc.std()),
                paircos=float(off.mean()),
                win_vs_src=float(((xf * z1).sum(-1) > (z0 * z1).sum(-1)).float().mean()))


def train_field(task, see_source, steps=STEPS, seed=0, p_uncond=0.0, tag="", width=HIDDEN,
                mup=True, lr=BASE_LR, aux_alpha=None, aux_sigma=0.0, early_stop=False):
    set_seed(seed)
    net = FieldNet(task.d, width, BLOCKS, see_source, mup=mup).to(DEVICE)
    opt = torch.optim.AdamW(net.param_groups(lr))
    every = max(200, steps // 8)
    best = dict(w1=float("inf"), state=None, step=-1)
    run_t, t0 = None, time.time()
    for step in range(1, steps + 1):
        c, xs, xt, m, z0, z1 = make_batch(task, BATCH, DEVICE)
        t = torch.rand(BATCH, device=DEVICE)
        zt = slerp(z0, z1, t[:, None])
        aux = None
        if see_source:
            if aux_alpha is not None:
                aux = slerp(z0, z1, (aux_alpha * t)[:, None])
            elif aux_sigma > 0:
                aux = exp_map(z0, aux_sigma * rand_tangent_dir(z0))
            else:
                aux = z0
        drop = (torch.rand(BATCH, device=DEVICE) < p_uncond) if p_uncond > 0 else None
        pred = project_tangent(net.raw(zt, t, c, aux, drop), zt)
        loss = F.mse_loss(pred, slerp_velocity(z0, z1, t[:, None]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        ld = loss.detach()
        run_t = ld if run_t is None else 0.95 * run_t + 0.05 * ld
        if early_stop and (step % every == 0 or step == steps):
            sf = sf_alpha(net, aux_alpha) if aux_alpha is not None else sf_ode(net, aux_sigma=aux_sigma)
            w1 = dist_report(sf, task, n_cond=6, n_per=64, seed=99)["w1_u"]
            if w1 < best["w1"]:
                best = dict(w1=w1, step=step,
                            state={k: v.detach().clone() for k, v in net.state_dict().items()})
    if early_stop and best["state"] is not None:
        net.load_state_dict(best["state"])
    run = float(run_t)
    log(tag=tag, steps=steps, loss=run, secs=time.time() - t0, es_step=best["step"])
    return net, run, best["step"]


log(model="ready")

# %% [code]
class OraNet(nn.Module):
    def __init__(self, nin, nout):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(nin, 96), nn.SiLU(),
                               nn.Linear(96, 96), nn.SiLU(), nn.Linear(96, nout))

    def forward(self, x):
        return self.f(x)


def fit_oracle1(task, steps=ORACLE_STEPS, batch=2048, seed=0):
    set_seed(seed + 777)
    ora = OraNet(2, 1).to(DEVICE)
    opt = torch.optim.AdamW(ora.parameters(), lr=2e-3)
    fl = None
    for _ in range(steps):
        c, xs, xt, m, z0, z1 = make_batch(task, batch, DEVICE)
        t = torch.rand(batch, device=DEVICE)
        zt = slerp(z0, z1, t[:, None])
        ut = slerp_velocity(z0, z1, t[:, None])
        u = (zt * c).sum(-1, keepdim=True)
        v = ora(torch.cat([u, t[:, None]], -1))[:, 0:1] * project_tangent(c, zt)
        loss = ((v - ut) ** 2).sum(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        ld = loss.detach()
        fl = ld if fl is None else 0.97 * fl + 0.03 * ld
    return ora, float(fl)


def sf_oracle1(ora, steps=64, t_start=0.0):
    def f(c, xs, xt, m, z0):
        def vel(z, tq):
            t = torch.full((z.shape[0], 1), float(tq), device=z.device)
            u = (z * c).sum(-1, keepdim=True)
            return ora(torch.cat([u, t], -1))[:, 0:1] * project_tangent(c, z)
        return integrate(vel, z0, steps=steps, t_start=t_start)
    return f


if "core" in STAGES:
    tick("core")
    net_c1, loss_c1, _ = train_field(task0, False, tag="c1")
    net_c2, loss_c2, _ = train_field(task0, True, tag="c2")
    ora1, fl1 = fit_oracle1(task0)
    net_o3, loss_o3, es_o3 = train_field(task0, True, steps=STEPS_LONG, width=1024,
                                        early_stop=True, tag="oracle3")

    F0 = {}
    for name, sf in [("identity", sf_identity()), ("const_c", sf_const_c()),
                     ("bayes_m", sf_bayes_m()), ("true_posterior", sf_true_post(task0)),
                     ("oracle1", sf_oracle1(ora1)), ("c1_ode", sf_ode(net_c1)),
                     ("c2_ode", sf_ode(net_c2)), ("oracle3_ode", sf_ode(net_o3))]:
        ev = evaluate(sf, task0)
        dv = dist_report(sf, task0)
        F0[name] = {**ev, **dv}
        log(cfg=name, fid=ev["fid"], align_c=ev["align_c"], span_src=ev["span_src"],
            paircos=ev["paircos"], polar_sd=dv["polar_sd"], w1=dv["w1_u"], w1_null=dv["w1_null"])
    save_csv("F0.csv", [dict(name=k, **v) for k, v in F0.items()])

    reg("oracle1_matches_law", abs(F0["oracle1"]["fid"] - LAW0) < 0.02,
        "oracle1 is the exact population marginal field so it lands on the closed-form law",
        oracle1=F0["oracle1"]["fid"], law=LAW0)
    reg("C55_azimuth_conserved_population", F0["oracle1"]["span_src"] > 0.99,
        "azimuth conservation is exact for the population field",
        span_src=F0["oracle1"]["span_src"])
    reg("C55_violated_by_training", F0["c1_ode"]["span_src"] < 0.95,
        "the trained field leaks out of the plane and that leakage is the fidelity gap",
        span_src=F0["c1_ode"]["span_src"], fid_gap=LAW0 - F0["c1_ode"]["fid"])
    reg("ring_paircos_population",
        abs(F0["oracle1"]["paircos"] - F0["oracle1"]["align_c"] ** 2) < 0.005,
        "the population image is a ring, so pairwise cosine equals align_c squared",
        paircos=F0["oracle1"]["paircos"], align_c_sq=F0["oracle1"]["align_c"] ** 2)
    reg("ring_paircos_not_trained",
        abs(F0["c1_ode"]["paircos"] - F0["c1_ode"]["align_c"] ** 2) > 0.02,
        "the trained image is not a ring",
        paircos=F0["c1_ode"]["paircos"], align_c_sq=F0["c1_ode"]["align_c"] ** 2)
    reg("polar_collapse_at_sigma0",
        F0["oracle1"]["polar_sd"] < 0.2 * F0["true_posterior"]["polar_sd"],
        "with a deterministic source the polar coordinate collapses to a single value",
        out=F0["oracle1"]["polar_sd"], true=F0["true_posterior"]["polar_sd"])
    reg("M2_no_better_than_M1", F0["oracle3_ode"]["fid"] <= F0["oracle1"]["fid"] + 0.01,
        "the source-conditioned model does not beat the marginal one even with double width, "
        "four times the steps and checkpoint selection on a distributional criterion",
        oracle3=F0["oracle3_ode"]["fid"], oracle1=F0["oracle1"]["fid"], c2=F0["c2_ode"]["fid"],
        es_step=es_o3, of_steps=STEPS_LONG)
    reg("cosine_ranks_sampler_last",
        F0["true_posterior"]["fid"] == min(v["fid"] for v in F0.values()),
        "cosine to a single target ranks a correct sampler last",
        true_post=F0["true_posterior"]["fid"])
    tock("core")
    flush_reg()

# %% [code]
if "grid" in STAGES:
    tick("grid")
    rows = []
    for sig in SIG_POLAR:
        for rho in RHO_CPL:
            tk = Task(d=D_OP, A=A_OP, beta=BETA_OP, jit=J_OP, rho_couple=rho, sig_polar=sig).resolve()
            ora, fl = fit_oracle1(tk)
            ev = evaluate(sf_oracle1(ora), tk)
            dv = dist_report(sf_oracle1(ora), tk)
            cs = random_sphere((8192, D_OP), DEVICE)
            _, _, _, _, z0c, z1c = make_batch(tk, 8192, DEVICE, c=cs)
            rows.append(dict(sig_polar=sig, rho=rho, arm="oracle1", floor=fl, **ev, **dv,
                             sd0=float((z0c * cs).sum(-1).std()),
                             sd1=float((z1c * cs).sum(-1).std()),
                             mu0=float((z0c * cs).sum(-1).mean()),
                             mu1=float((z1c * cs).sum(-1).mean())))
            log(sig_polar=sig, rho=rho, fid=ev["fid"], align_c=ev["align_c"],
                polar_sd=dv["polar_sd"], polar_sd_true=dv["polar_sd_true"],
                span_src=ev["span_src"], paircos=ev["paircos"], w1=dv["w1_u"])
    for sig in ([0.0, 0.1] if FAST else [0.0, 0.03, 0.1, 0.4]):
        for rho in ([0.0, 1.0] if FAST else [0.0, 0.5, 1.0]):
            tk = Task(d=D_OP, A=A_OP, beta=BETA_OP, jit=J_OP, rho_couple=rho, sig_polar=sig).resolve()
            n1, lo, _ = train_field(tk, False, tag=f"grid_c1_s{sig}_r{rho}")
            ev = evaluate(sf_ode(n1), tk)
            dv = dist_report(sf_ode(n1), tk)
            rows.append(dict(sig_polar=sig, rho=rho, arm="trained_c1", floor=lo, **ev, **dv))
            log(sig_polar=sig, rho=rho, arm="trained", fid=ev["fid"], span_src=ev["span_src"],
                polar_sd=dv["polar_sd"], w1=dv["w1_u"])
    save_csv("grid_sigma_rho.csv", rows)

    orc = [r for r in rows if r["arm"] == "oracle1"]
    tk_ctl = Task(d=D_OP, A=A_OP, beta=BETA_OP, jit=J_OP, rho_couple=1.0, sig_polar=0.0).resolve()
    ctl = []
    for sd in [0, 1, 2, 3, 4]:
        ora_c, _ = fit_oracle1(tk_ctl, seed=sd)
        ctl.append(evaluate(sf_oracle1(ora_c), tk_ctl)["align_c"])
    refit_sd = float(np.std(ctl))
    refit_floor = max(ctl) - min(ctl)
    log(ctl_align_c=str([round(v, 5) for v in ctl]), refit_sd=refit_sd, refit_range=refit_floor)

    drifts = []
    mono = 0
    for sig in SIG_POLAR:
        sel = sorted([r for r in orc if r["sig_polar"] == sig], key=lambda r: r["rho"])
        if len(sel) > 1:
            v = [r["align_c"] for r in sel]
            drifts.append(max(v) - min(v))
            if all(v[i] > v[i + 1] for i in range(len(v) - 1)) or \
               all(v[i] < v[i + 1] for i in range(len(v) - 1)):
                mono += 1
    reg("grid_coupling_invariance", max(drifts) < max(0.01, 2 * refit_floor),
        "the conditional second moments are co-diagonal, so if the flat criterion transferred "
        "the endpoint map would be invariant under a coupling sweep at every sigma_polar; "
        "align_c characterises the map because the azimuth is conserved",
        max_align_c_spread_over_rho=max(drifts), refit_range=refit_floor, refit_sd=refit_sd)
    reg("grid_no_monotone_rho_trend", mono == 0,
        "if the spread were refit noise the ordering in rho would not be monotone",
        monotone_rows=mono, total_rows=len(drifts), p_per_row=1.0 / 60.0)
    fsp = []
    for sig in SIG_POLAR:
        sel = [r for r in orc if r["sig_polar"] == sig]
        if len(sel) > 1:
            fsp.append(max(r["fid"] for r in sel) - min(r["fid"] for r in sel))
    reg("grid_fid_is_not_the_map", max(fsp) > 0.1,
        "fid falls with rho at fixed map because the target private factor decorrelates from "
        "the source one, so fid cannot test the invariance and align_c is the right statistic",
        max_fid_spread_over_rho=max(fsp))
    marg = []
    for sig in SIG_POLAR:
        sel = [r for r in orc if r["sig_polar"] == sig]
        marg.append(max(abs(r["mu1"] - sel[0]["mu1"]) for r in sel))
    reg("grid_marginals_pinned_under_rho", max(marg) < 0.01,
        "the rho construction leaves both conditional marginals fixed",
        max_mu1_drift=max(marg))
    pol = [r for r in orc if r["sig_polar"] > 0 and r["rho"] == 1.0]
    rel = [abs(r["polar_sd"] / max(r["polar_sd_true"], 1e-9) - 1) for r in pol]
    reg("grid_polar_gain_law", float(np.median(rel)) < 0.25,
        "for sigma_polar > 0 transport is possible and the output polar spread matches the "
        "target polar spread, as the flat gain law predicts",
        median_rel_err=float(np.median(rel)), n=len(pol))
    z = [r for r in orc if r["sig_polar"] == 0 and r["rho"] == 1.0][0]
    reg("grid_collapse_only_at_zero", z["polar_sd"] < 0.25 * z["polar_sd_true"],
        "at sigma_polar = 0 the image is a ring and the polar spread collapses",
        out=z["polar_sd"], target=z["polar_sd_true"])
    sp = [r["span_src"] for r in orc]
    reg("grid_C55_holds_all_sigma", min(sp) > 0.99,
        "azimuth conservation does not depend on sigma_polar",
        min_span_src=min(sp))
    am = [abs(r["align_c"] - AS_OP) for r in orc if r["rho"] == 1.0]
    reg("grid_polar_mean_As", max(am) < 0.03,
        "the mean polar landing stays at As across the whole sigma sweep",
        max_dev=max(am), As=AS_OP)

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
    for rho in RHO_CPL:
        sel = sorted([r for r in orc if r["rho"] == rho], key=lambda r: r["sig_polar"])
        ax[0].semilogx([max(r["sig_polar"], 1e-3) for r in sel], [r["align_c"] for r in sel],
                       "o-", label=f"rho={rho}")
    ax[0].axhline(AS_OP, ls="--", c="k", lw=1)
    ax[0].set_xlabel("sigma_polar")
    ax[0].set_ylabel("align_c (endpoint map)")
    ax[0].legend(fontsize=7)
    sel = sorted([r for r in orc if r["rho"] == 1.0], key=lambda r: r["sig_polar"])
    ax[1].semilogx([max(r["sig_polar"], 1e-3) for r in sel], [r["polar_sd"] for r in sel], "o-")
    ax[1].semilogx([max(r["sig_polar"], 1e-3) for r in sel], [r["polar_sd_true"] for r in sel], "s--")
    ax[1].set_xlabel("sigma_polar")
    ax[1].set_ylabel("polar sd (out vs target)")
    ax[2].semilogx([max(r["sig_polar"], 1e-3) for r in sel], [r["w1_u"] for r in sel], "o-")
    ax[2].semilogx([max(r["sig_polar"], 1e-3) for r in sel], [r["w1_null"] for r in sel], "s--")
    ax[2].set_xlabel("sigma_polar")
    ax[2].set_ylabel("W1 (model vs null)")
    save_fig("F3_grid_sigma_rho.png")
    tock("grid")
    flush_reg()

# %% [code]
if "noise" in STAGES:
    tick("noise")
    rows = []
    fam = {}
    for kind, grid in [("alpha", ALPHAS), ("sigma_aux", SIG_AUX)]:
        for sd in SEEDS[:2]:
            pts = []
            for g in grid:
                if kind == "alpha":
                    net, lo, _ = train_field(task0, True, seed=sd, aux_alpha=g, tag=f"a{g}s{sd}")
                    sf = sf_alpha(net, g)
                else:
                    net, lo, _ = train_field(task0, True, seed=sd, aux_sigma=g, tag=f"n{g}s{sd}")
                    sf = sf_ode(net, aux_sigma=g)
                ev = evaluate(sf, task0)
                dv = dist_report(sf, task0)
                rows.append(dict(kind=kind, knob=g, seed=sd, final_loss=lo, **ev, **dv))
                pts.append([ev["fid"], ev["span_src"], ev["paircos"], dv["w1_u"]])
                log(kind=kind, knob=g, seed=sd, fid=ev["fid"], loss=lo, w1=dv["w1_u"],
                    span_src=ev["span_src"])
            fam[(kind, sd)] = np.array(pts)
    save_csv("noise_equivalence.csv", rows)

    allp = np.concatenate(list(fam.values()), 0)
    mu, sg = allp.mean(0), allp.std(0) + 1e-9
    nf = {k: (v - mu) / sg for k, v in fam.items()}

    def cdist_curve(P, Q):
        D = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=-1)
        return 0.5 * (D.min(1).mean() + D.min(0).mean())

    ss = SEEDS[:2]
    dx = float(np.mean([cdist_curve(nf[("alpha", a)], nf[("sigma_aux", b)]) for a in ss for b in ss]))
    dw = float(np.mean([cdist_curve(nf[(k, ss[0])], nf[(k, ss[1])]) for k in ["alpha", "sigma_aux"]])) \
        if len(ss) > 1 else float("nan")
    reg("noise_equivalence", dx < 1.5 * dw,
        "the alpha family and the auxiliary-noise family trace one curve, so the mechanism is "
        "ill-conditioning of the recovery map and nothing else",
        cross=dx, within=dw)
    al = [r for r in rows if r["kind"] == "alpha" and r["seed"] == ss[0]]
    reg("loss_anticorrelates_with_quality",
        spearman([r["final_loss"] for r in al], [r["fid"] for r in al]) > 0.5,
        "along the alpha family the training loss rises while sampled fidelity rises",
        spearman=spearman([r["final_loss"] for r in al], [r["fid"] for r in al]))
    aw = [r for r in rows if r["kind"] == "alpha" and r["seed"] == ss[0]]
    reg("alpha_w1_monotone",
        all(aw[i]["w1_u"] >= aw[i + 1]["w1_u"] for i in range(len(aw) - 1)) or
        all(aw[i]["w1_u"] <= aw[i + 1]["w1_u"] for i in range(len(aw) - 1)),
        "the distributional error is monotone along the alpha family, as fidelity is",
        w1_by_alpha=str([round(r["w1_u"], 5) for r in aw]),
        argmin_alpha=min(aw, key=lambda r: r["w1_u"])["knob"])

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for kind, mk in [("alpha", "o-"), ("sigma_aux", "s--")]:
        sel = [r for r in rows if r["kind"] == kind and r["seed"] == ss[0]]
        ax[0].plot([r["fid"] for r in sel], [r["w1_u"] for r in sel], mk, label=kind)
        ax[1].plot([r["knob"] for r in sel], [r["w1_u"] for r in sel], mk, label=kind)
    ax[0].set_xlabel("fid")
    ax[0].set_ylabel("W1")
    ax[0].legend(fontsize=8)
    ax[1].set_xlabel("knob")
    ax[1].set_ylabel("W1")
    ax[1].legend(fontsize=8)
    save_fig("F4_noise_equivalence.png")
    tock("noise")
    flush_reg()

# %% [code]
if "fix" in STAGES:
    tick("fix")
    rows = []
    b1 = evaluate(sf_ode(net_c1), task0)["fid"]
    b2 = evaluate(sf_ode(net_c2), task0)["fid"]
    for eps in TRUNC:
        sf = sf_ode(net_c2, t_start=eps)
        ev = evaluate(sf, task0)
        dv = dist_report(sf, task0)
        lo = max(eps, 1e-3)
        gridt = np.linspace(lo, 1.0, 400)
        rows.append(dict(remedy="truncation", knob=eps, amp_int=float(TRAPZ(1.0 / gridt, gridt)),
                         **ev, **dv))
        log(remedy="truncation", eps=eps, fid=ev["fid"], w1=dv["w1_u"], amp_int=rows[-1]["amp_int"])
    for ch in CHURNS:
        sf = sf_ode(net_c2, churn=ch)
        ev = evaluate(sf, task0)
        dv = dist_report(sf, task0)
        rows.append(dict(remedy="churn", knob=ch, amp_int=float("nan"), **ev, **dv))
        log(remedy="churn", knob=ch, fid=ev["fid"], w1=dv["w1_u"])
    for sg in ([0.0, 0.3] if FAST else [0.0, 0.1, 0.3, 0.6, 1.0]):
        ns, _, _ = train_field(task0, True, aux_sigma=sg, tag=f"fix_aux{sg}")
        sf = sf_ode(ns, aux_sigma=sg)
        ev = evaluate(sf, task0)
        dv = dist_report(sf, task0)
        rows.append(dict(remedy="aux_noise", knob=sg, amp_int=float("nan"), **ev, **dv))
        log(remedy="aux_noise", knob=sg, fid=ev["fid"], w1=dv["w1_u"])
    save_csv("fixes.csv", rows)

    best = max(rows, key=lambda r: r["fid"])
    reg("a_remedy_recovers_M2", best["fid"] >= b1 - 0.01,
        "the source-conditioned deficit is ill-conditioning, so attacking the singularity "
        "recovers it",
        best=best["remedy"], knob=best["knob"], fid=best["fid"], c1=b1, c2=b2)
    tr = [r for r in rows if r["remedy"] == "truncation"]
    reg("truncation_tracks_amplification",
        abs(spearman([r["amp_int"] for r in tr], [-r["fid"] for r in tr])) > 0.8,
        "endpoint error follows the amplification integral in shape, not merely monotonically",
        spearman=spearman([r["amp_int"] for r in tr], [-r["fid"] for r in tr]))

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for rem in ["truncation", "churn", "aux_noise"]:
        sel = [r for r in rows if r["remedy"] == rem]
        ax.plot([r["knob"] for r in sel], [r["fid"] for r in sel], "o-", label=rem)
    ax.axhline(b1, ls="--", c="tab:red", lw=1)
    ax.axhline(b2, ls=":", c="k", lw=1)
    ax.set_xlabel("remedy strength")
    ax.set_ylabel("fid")
    ax.legend(fontsize=7)
    save_fig("F5_fixes.png")
    tock("fix")
    flush_reg()

# %% [code]
if "metric" in STAGES:
    tick("metric")
    rows = []
    for name, sf in [("identity", sf_identity()), ("const_c", sf_const_c()),
                     ("bayes_m", sf_bayes_m()), ("oracle1", sf_oracle1(ora1)),
                     ("c1_K1", sf_ode(net_c1, steps=1)), ("c1_K64", sf_ode(net_c1, steps=64)),
                     ("c2_ode", sf_ode(net_c2)), ("true_posterior", sf_true_post(task0))]:
        ev = evaluate(sf, task0)
        dv = dist_report(sf, task0)
        rows.append(dict(family="estimator", name=name, knob=float("nan"), **ev, **dv))
        log(fam="estimator", name=name, fid=ev["fid"], w1=dv["w1_u"], paircos=ev["paircos"])
    for K in K_LIST:
        for ch in CHURNS:
            sf = sf_ode(net_c1, steps=K, churn=ch)
            ev = evaluate(sf, task0)
            dv = dist_report(sf, task0)
            rows.append(dict(family="steps", name=f"K{K}_ch{ch}", knob=K, churn=ch, **ev, **dv))
            log(fam="steps", K=K, churn=ch, fid=ev["fid"], w1=dv["w1_u"], paircos=ev["paircos"])
    save_csv("metric_ordering.csv", rows)

    est = [r for r in rows if r["family"] == "estimator"]
    stp = [r for r in rows if r["family"] == "steps" and r.get("churn") == 0.0]
    for nm, sel in [("estimator", est), ("steps", stp)]:
        rho = spearman([r["fid"] for r in sel], [r["w1_u"] for r in sel])
        reg(f"ordering_flip_{nm}", rho > 0.4,
            "higher cosine goes with worse distributional fit, so the two metrics invert",
            spearman=rho, n=len(sel))
    rho2 = spearman([r["knob"] for r in stp], [r["fid"] for r in stp])
    reg("steps_trade", rho2 < -0.5,
        "more integration steps lower cosine and raise variety",
        spearman_K_fid=rho2,
        spearman_K_paircos=spearman([r["knob"] for r in stp], [r["paircos"] for r in stp]))

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    y = np.arange(len(est))
    ax.barh(y - 0.2, [r["fid"] for r in est], 0.4, label="cosine")
    ax.barh(y + 0.2, [-r["w1_u"] for r in est], 0.4, label="-W1")
    ax.set_yticks(y)
    ax.set_yticklabels([r["name"] for r in est], fontsize=8)
    ax.axvline(0, c="k", lw=0.8)
    ax.legend(fontsize=8)
    save_fig("F6_ordering.png")
    tock("metric")
    flush_reg()

# %% [code]
if "threshold" in STAGES:
    tick("threshold")
    rows = []
    for A in A_GRID:
        for j in J_GRID:
            pred = beta_star(A, j)
            exists = A < math.cos(j)
            meas, prev = float("nan"), None
            for b in BETAS:
                tk = Task(d=D_OP, A=A, beta=b, jit=j).resolve()
                ora, _ = fit_oracle1(tk, steps=ORACLE_STEPS_GRID, batch=1024)
                fl = evaluate(sf_oracle1(ora), tk, n=256, k_div=3)["fid"]
                idf = evaluate(sf_identity(), tk, n=256, k_div=3)["fid"]
                rows.append(dict(A=A, j=j, beta=b, flow=fl, identity=idf, delta=fl - idf,
                                 law=law_markov(A, b), src=ladder(A, b, j)["src"]))
                if prev is not None and prev[1] <= 0 < fl - idf:
                    meas = prev[0] + (b - prev[0]) * (-prev[1]) / ((fl - idf) - prev[1])
                prev = (b, fl - idf)
            rows.append(dict(A=A, j=j, beta=float("nan"), beta_star_pred=pred,
                             beta_star_meas=meas, exists_pred=int(exists),
                             exists_meas=int(np.isfinite(meas)), degenerate=int(j == 0.0)))
            log(A=A, j=j, bstar_pred=pred, bstar_meas=meas, exists_pred=int(exists))
    save_csv("beta_star.csv", rows)

    pr = [r for r in rows if "beta_star_pred" in r]
    both = [(r["beta_star_pred"], r["beta_star_meas"]) for r in pr
            if np.isfinite(r["beta_star_pred"]) and np.isfinite(r["beta_star_meas"])]
    err = max(abs(p - m) for p, m in both) if both else float("nan")
    reg("beta_star_matches", err < 0.06,
        "beta star is decidable in advance from A and j alone",
        max_abs_err=err, n=len(both))
    nz = [r for r in pr if r["degenerate"] == 0]
    agree = sum(1 for r in nz if r["exists_pred"] == r["exists_meas"])
    reg("existence_criterion_A_lt_cosj", agree == len(nz),
        "beta star exists iff A < cos j, so the no-crossing region is predicted not discovered; "
        "j = 0 is excluded because the field vanishes at z0 = c and the trajectory is stationary",
        agree=agree, total=len(nz), excluded_j0=len(pr) - len(nz))

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for j in J_GRID:
        ax.plot(A_GRID, [beta_star(A, j) for A in A_GRID], "o-", label=f"j={j}")
    ax.set_xlabel("A")
    ax.set_ylabel("beta*")
    ax.legend(fontsize=7)
    save_fig("F7_beta_star.png")
    tock("threshold")
    flush_reg()

# %% [code]
if "guidance" in STAGES:
    tick("guidance")
    rows = []
    for lam in LAMBDAS:
        tk = Task(d=D_OP, A=A_OP, beta=BETA_OP, jit=J_OP, mix_lambda=lam).resolve()
        for sd in SEEDS[:2]:
            ng, _, _ = train_field(tk, False, seed=sd, p_uncond=0.1, tag=f"cfg_l{lam}_s{sd}")
            base = evaluate(sf_ode(ng, guidance=1.0), tk)
            for w in GUID_W:
                sf = sf_ode(ng, guidance=w)
                ev = evaluate(sf, tk)
                dv = dist_report(sf, tk)
                rows.append(dict(lam=lam, w=w, seed=sd, dfid=ev["fid"] - base["fid"], **ev, **dv))
                log(lam=lam, w=w, seed=sd, dfid=ev["fid"] - base["fid"], fid=ev["fid"],
                    align_c=ev["align_c"], w1=dv["w1_u"])
    save_csv("guidance.csv", rows)

    def dm(lam, w):
        v = [r["dfid"] for r in rows if r["lam"] == lam and r["w"] == w]
        return float(np.mean(v)), float(np.std(v))
    lo, hi = min(LAMBDAS), max(LAMBDAS)
    a, sa = dm(lo, 3.0)
    b, sb = dm(hi, 3.0)
    reg("guidance_sign_flip", a > 0 > b and abs(a - b) > 2 * max(sa, sb, 1e-6),
        "guidance is specific to starting from noise and reverses sign with an informative start",
        gain_at_noise=a, sd_noise=sa, gain_at_informative=b, sd_informative=sb)

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for w in GUID_W:
        mus = [dm(l, w)[0] for l in LAMBDAS]
        sds = [dm(l, w)[1] for l in LAMBDAS]
        ax.errorbar(LAMBDAS, mus, yerr=sds, fmt="o-", capsize=3, label=f"w={w}")
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xlabel("lambda")
    ax.set_ylabel("change in cosine")
    ax.legend(fontsize=8)
    save_fig("F8_guidance.png")
    tock("guidance")
    flush_reg()

# %% [code]
class DiscreteTask:
    def __init__(self, K=8, M=32, L=16, V=12, beta_d=0.6, eta=0.05, gamma=0.35, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.K, self.M, self.L, self.V = K, M, L, V
        self.beta_d, self.eta, self.gamma = beta_d, eta, gamma
        self.MASK = V
        self.n_xi = int(round(beta_d * L))
        self.tab_c = torch.randint(0, V, (K, L), generator=g)
        self.tab_x = torch.randint(0, V, (K, M, L), generator=g)
        self.xi_pos = torch.zeros(L, dtype=torch.bool)
        self.xi_pos[: self.n_xi] = True

    def draw(self, B, device, c=None):
        c = torch.randint(0, self.K, (B,), device=device) if c is None else c
        xi = torch.randint(0, self.M, (B,), device=device)
        base = self.tab_c.to(device)[c]
        priv = self.tab_x.to(device)[c, xi]
        msk = self.xi_pos.to(device)[None, :].expand(B, -1)
        clean = torch.where(msk, priv, base)
        x1 = torch.where(torch.rand(B, self.L, device=device) < self.eta,
                         torch.randint(0, self.V, (B, self.L), device=device), clean)
        x0 = torch.where(torch.rand(B, self.L, device=device) < self.eta,
                         torch.randint(0, self.V, (B, self.L), device=device), clean)
        x0 = torch.where(torch.rand(B, self.L, device=device) < self.gamma,
                         torch.randint(0, self.V, (B, self.L), device=device), x0)
        return c, xi, x0, x1


class MaskDenoiser(nn.Module):
    def __init__(self, task, width=256, see_source=False):
        super().__init__()
        self.t, self.see_source = task, see_source
        nin = task.L * (task.V + 1) * (2 if see_source else 1) + task.K + 1
        self.f = nn.Sequential(nn.Linear(nin, width), nn.SiLU(),
                               nn.Linear(width, width), nn.SiLU(),
                               nn.Linear(width, task.L * task.V))

    def forward(self, xt, c, t, x0=None):
        B = xt.shape[0]
        parts = [F.one_hot(xt, self.t.V + 1).float().view(B, -1)]
        if self.see_source:
            parts.append(F.one_hot(x0, self.t.V + 1).float().view(B, -1))
        parts += [F.one_hot(c, self.t.K).float(), t[:, None]]
        return self.f(torch.cat(parts, -1)).view(B, self.t.L, self.t.V)


def train_discrete(task, see_source, steps=2000, seed=0, width=256, start_at_source=False):
    set_seed(seed)
    net = MaskDenoiser(task, width, see_source).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    for _ in range(steps):
        c, xi, x0, x1 = task.draw(512, DEVICE)
        t = torch.rand(512, device=DEVICE)
        keep = torch.rand(512, task.L, device=DEVICE) < t[:, None]
        if start_at_source:
            xt = torch.where(keep, x1, x0)
        else:
            xt = torch.where(keep, x1, torch.full_like(x1, task.MASK))
        lg = net(xt, c, t, x0)
        loss = F.cross_entropy(lg[~keep], x1[~keep]) if (~keep).any() else lg.sum() * 0
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return net


@torch.no_grad()
def sample_discrete(net, task, c, x0, steps=8, start_at_source=False):
    B = c.shape[0]
    xt = x0.clone() if start_at_source else torch.full((B, task.L), task.MASK, device=c.device)
    done = torch.zeros(B, task.L, dtype=torch.bool, device=c.device)
    for i in range(steps):
        t = torch.full((B,), i / steps, device=c.device)
        lg = net(xt, c, t, x0)
        pred = torch.distributions.Categorical(logits=lg).sample()
        n_rev = max(1, int(round(task.L * (i + 1) / steps)))
        conf = lg.log_softmax(-1).max(-1).values
        conf = torch.where(done, torch.full_like(conf, -1e9), conf)
        idx = conf.topk(min(n_rev, task.L), -1).indices
        nx = xt.clone()
        nx.scatter_(1, idx, pred.gather(1, idx))
        xt = torch.where(done, xt, nx)
        done = done.scatter(1, idx, True)
    return torch.where(xt == task.MASK, torch.zeros_like(xt), xt)


@torch.no_grad()
def eval_discrete(fn, task, n_cond=8, n_per=64):
    a, tv, ss = [], [], []
    for k in range(n_cond):
        c = torch.full((n_per,), k % task.K, device=DEVICE, dtype=torch.long)
        _, _, x0, x1 = task.draw(n_per, DEVICE, c=c)
        g = fn(c, x0)
        a.append(float((g == x1).float().mean()))
        _, _, _, r = task.draw(n_per, DEVICE, c=c)
        tv.append(float(0.5 * (F.one_hot(g, task.V).float().mean(0)
                               - F.one_hot(r, task.V).float().mean(0)).abs().sum(-1).mean()))
        ss.append(float((g[:, None, :] == g[None, :, :]).float().mean()))
    return dict(acc=float(np.mean(a)), tv=float(np.mean(tv)), selfsim=float(np.mean(ss)))


if "discrete" in STAGES:
    tick("discrete")
    rows = []
    BD = [0.2, 0.7] if FAST else [0.1, 0.25, 0.4, 0.55, 0.7, 0.85]
    DSTEPS = 800 if FAST else 2000
    for bd in BD:
        dt = DiscreteTask(beta_d=bd)
        arms_d = {}
        for nm, see, sas in [("d1_mask_start_label_only", False, False),
                             ("d2_mask_start_sees_source", True, False),
                             ("d3_source_start_label_only", False, True),
                             ("d4_source_start_sees_source", True, True)]:
            nd = train_discrete(dt, see, steps=DSTEPS, start_at_source=sas)
            arms_d[nm] = eval_discrete(
                lambda c, x0, n=nd, s=sas: sample_discrete(n, dt, c, x0, 8, start_at_source=s), dt)
        idn = eval_discrete(lambda c, x0: x0, dt)
        rows.append(dict(beta_d=bd, arm="identity", steps=8, **idn))
        for nm, r in arms_d.items():
            rows.append(dict(beta_d=bd, arm=nm, steps=8, **r))
        log(beta_d=bd, ident=idn["acc"],
            **{k.split("_")[0]: v["acc"] for k, v in arms_d.items()})
    dt = DiscreteTask(beta_d=0.7)
    idn7 = eval_discrete(lambda c, x0: x0, dt)
    nds = train_discrete(dt, True, steps=DSTEPS)
    for K in [1, 2, 4, 8, 16]:
        r = eval_discrete(lambda c, x0, kk=K: sample_discrete(nds, dt, c, x0, kk), dt)
        rows.append(dict(beta_d=0.7, arm="steps", steps=K, identity=idn7["acc"], **r))
        log(K=K, acc=r["acc"], tv=r["tv"], selfsim=r["selfsim"], identity=idn7["acc"])
    save_csv("discrete.csv", rows)

    def dget(arm):
        return {r["beta_d"]: r["acc"] for r in rows if r["arm"] == arm}
    d1, d2, d3, d4 = (dget("d1_mask_start_label_only"), dget("d2_mask_start_sees_source"),
                      dget("d3_source_start_label_only"), dget("d4_source_start_sees_source"))
    idm = dget("identity")
    crossed = [b for b in sorted(d1) if d1[b] > idm[b]]
    reg("discrete_threshold", 0 < len(crossed) < len(d1),
        "the do-nothing crossover reappears in masked diffusion, so the decision rule is not "
        "an artifact of the continuous sphere",
        crossed=str(crossed), all_beta=str(sorted(d1)))
    reg("discrete_source_as_conditioning_helps",
        all(d2[b] > d1[b] for b in sorted(d1)),
        "when the source is conditioning only and not the sampling start, using it helps",
        mean_gain=float(np.mean([d2[b] - d1[b] for b in sorted(d1)])))
    reg("discrete_source_as_start_hurts",
        all(d4[b] < d3[b] for b in sorted(d3)),
        "when the source is also the sampling start, conditioning on it hurts, mirroring the "
        "continuous case",
        mean_gain=float(np.mean([d4[b] - d3[b] for b in sorted(d3)])),
        d3=str([round(d3[b], 4) for b in sorted(d3)]),
        d4=str([round(d4[b], 4) for b in sorted(d4)]))
    st = [r for r in rows if r["arm"] == "steps"]
    st_beats = all(r["acc"] > r["identity"] for r in st)
    sp_a = spearman([r["steps"] for r in st], [r["acc"] for r in st])
    sp_d = spearman([r["steps"] for r in st], [1 - r["selfsim"] for r in st])
    reg("discrete_steps_trade", st_beats and (sp_a < 0 < sp_d),
        "more unmasking steps lower token accuracy and raise diversity, measured on an arm that "
        "beats the do-nothing baseline; a step-count trade-off read off a model that loses to "
        "doing nothing is not interpretable, which is why the arm is checked first",
        sp_acc=sp_a, sp_div=sp_d, beats_identity=int(st_beats),
        arm_acc=float(np.mean([r["acc"] for r in st])), identity=idn7["acc"])

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.9))
    bs = sorted(idm)
    ax[0].plot(bs, [idm[b] for b in bs], "k--", label="identity")
    for nm, dd in [("d1 mask start, label only", d1), ("d2 mask start, sees source", d2),
                   ("d3 source start, label only", d3), ("d4 source start, sees source", d4)]:
        ax[0].plot(bs, [dd[b] for b in bs], "o-", label=nm)
    ax[0].set_xlabel("beta_d")
    ax[0].set_ylabel("token accuracy")
    ax[0].legend(fontsize=6)
    ax[1].plot([r["steps"] for r in st], [r["acc"] for r in st], "o-")
    ax[1].set_xlabel("unmasking steps")
    ax[1].set_ylabel("accuracy")
    save_fig("F9_discrete.png")
    tock("discrete")
    flush_reg()

# %% [code]
if "scaling" in STAGES:
    tick("scaling")
    rows_lr = []
    for mup in [False, True]:
        for lr in LR_GRID:
            for w in [BASE_WIDTH, WIDTHS[-1]]:
                net, lo, _ = train_field(task0, True, width=w, mup=mup, lr=lr,
                                         steps=STEPS // 2, tag=f"lr_mup{int(mup)}_w{w}_lr{lr}")
                sf = sf_ode(net)
                ev = evaluate(sf, task0, n=256, k_div=4)
                dv = dist_report(sf, task0, n_cond=6, n_per=64)
                rows_lr.append(dict(mup=int(mup), lr=lr, width=w, final_loss=lo, **ev, **dv))
                log(mup=int(mup), width=w, lr=lr, loss=lo, fid=ev["fid"], w1=dv["w1_u"])
    save_csv("lr_transfer.csv", rows_lr)

    def best_lr(mup, w):
        sel = [r for r in rows_lr if r["mup"] == int(mup) and r["width"] == w]
        return min(sel, key=lambda r: r["w1_u"])["lr"]

    reg("mup_transfers_lr", best_lr(True, BASE_WIDTH) == best_lr(True, WIDTHS[-1]),
        "under maximal-update parameterisation the optimal learning rate is width-independent",
        mup_base=best_lr(True, BASE_WIDTH), mup_wide=best_lr(True, WIDTHS[-1]),
        sp_base=best_lr(False, BASE_WIDTH), sp_wide=best_lr(False, WIDTHS[-1]))

    rows = []
    for mup in [False, True]:
        lr = best_lr(mup, BASE_WIDTH)
        for sd in SEEDS_SCALE:
            for w in WIDTHS:
                for see in [False, True]:
                    net, lo, es = train_field(task0, see, width=w, mup=mup, lr=lr, seed=sd,
                                              early_stop=True,
                                              tag=f"sc_mup{int(mup)}_w{w}_c{int(see)+1}_s{sd}")
                    sf = sf_ode(net)
                    ev = evaluate(sf, task0)
                    dv = dist_report(sf, task0)
                    rows.append(dict(mup=int(mup), width=w, seed=sd, arm="c2" if see else "c1",
                                     lr=lr, final_loss=lo, es_step=es, **ev, **dv))
                    log(mup=int(mup), width=w, arm="c2" if see else "c1", seed=sd,
                        fid=ev["fid"], w1=dv["w1_u"], loss=lo, es=es, span_src=ev["span_src"])
            save_csv("scaling.csv", rows)
            flush_reg()
        net4, lo4, es4 = train_field(task0, True, width=HIDDEN, mup=mup, lr=lr,
                                     steps=STEPS_LONG, early_stop=True, tag=f"sc4x_mup{int(mup)}")
        sf = sf_ode(net4)
        ev4 = evaluate(sf, task0)
        dv4 = dist_report(sf, task0)
        rows.append(dict(mup=int(mup), width=HIDDEN, seed=0, arm="c2_4x", lr=lr,
                         final_loss=lo4, es_step=es4, **ev4, **dv4))
        log(mup=int(mup), arm="c2_4x", fid=ev4["fid"], w1=dv4["w1_u"], loss=lo4, es=es4)
    save_csv("scaling.csv", rows)

    def agg(mup, arm, w, key="fid"):
        sel = [r for r in rows if r["mup"] == int(mup) and r["arm"] == arm and r["width"] == w]
        return float(np.mean([r[key] for r in sel])), float(np.std([r[key] for r in sel]))

    for mup in [False, True]:
        f0, s0 = agg(mup, "c2", WIDTHS[0])
        f1, s1 = agg(mup, "c2", WIDTHS[-1])
        g0, t0 = agg(mup, "c1", WIDTHS[0])
        g1, t1 = agg(mup, "c1", WIDTHS[-1])
        l0, _ = agg(mup, "c2", WIDTHS[0], "final_loss")
        l1, _ = agg(mup, "c2", WIDTHS[-1], "final_loss")
        w0, _ = agg(mup, "c2", WIDTHS[0], "w1_u")
        w1v, _ = agg(mup, "c2", WIDTHS[-1], "w1_u")
        tag = "mup" if mup else "sp"
        gap0, gap1 = g0 - f0, g1 - f1
        gse = math.sqrt((s0 ** 2 + t0 ** 2 + s1 ** 2 + t1 ** 2) / max(len(SEEDS_SCALE), 1))
        reg(f"scaling_{tag}_capacity_closes_gap", gap1 < gap0 - gse,
            "if the difference between the two models is estimation error then capacity closes it",
            gap_narrow=gap0, gap_wide=gap1, gap_se=gse,
            c2_narrow=f0, c2_wide=f1, c1_narrow=g0, c1_wide=g1)
        reg(f"scaling_{tag}_exceeds_seed_spread", abs(f1 - f0) > 2 * max(s0, s1, 1e-9),
            "whatever its direction the width effect is larger than seed noise",
            delta=f1 - f0, two_sd=2 * max(s0, s1), n_seeds=len(SEEDS_SCALE))
        reg(f"scaling_{tag}_loss_falls", l1 < l0,
            "the objective is optimised better at larger width",
            loss_narrow=l0, loss_wide=l1)
        reg(f"scaling_{tag}_w1_falls", w1v < w0,
            "if the difference were estimation error the distributional fit would improve too",
            w1_narrow=w0, w1_wide=w1v)
        reg(f"scaling_{tag}_seed_variance_controlled", max(s0, s1) < 0.05,
            "seed variance stays small enough for the width effect to be readable",
            sd_narrow=s0, sd_wide=s1)

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.0))
    for mup, mk in [(False, "s--"), (True, "o-")]:
        for arm, col in [("c1", "tab:red"), ("c2", "tab:purple")]:
            mus = [agg(mup, arm, w)[0] for w in WIDTHS]
            sds = [agg(mup, arm, w)[1] for w in WIDTHS]
            ax[0].errorbar(WIDTHS, mus, yerr=sds, fmt=mk, c=col, capsize=3,
                           label=f"{arm}_{'mup' if mup else 'sp'}")
        ax[1].plot(WIDTHS, [agg(mup, "c2", w, "w1_u")[0] for w in WIDTHS], mk,
                   label=f"w1_c2_{'mup' if mup else 'sp'}")
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("width")
    ax[0].set_ylabel("fid")
    ax[0].legend(fontsize=7)
    ax[1].set_xscale("log", base=2)
    ax[1].set_xlabel("width")
    ax[1].set_ylabel("W1")
    ax[1].legend(fontsize=7)
    save_fig("F10_scaling.png")
    tock("scaling")
    flush_reg()

# %% [code]
if "lr2" in STAGES:
    tick("lr2")
    rows_lr2 = []
    for lr in LR2_GRID:
        for sd in LR2_SEEDS:
            for w in LR2_WIDTHS:
                for see in [False, True]:
                    net, lo, es = train_field(task0, see, width=w, mup=True, lr=lr, seed=sd,
                                              early_stop=True,
                                              tag=f"lr2_{lr}_w{w}_c{int(see)+1}_s{sd}")
                    sf = sf_ode(net)
                    ev = evaluate(sf, task0)
                    dv = dist_report(sf, task0)
                    rows_lr2.append(dict(lr=lr, width=w, seed=sd, arm="c2" if see else "c1",
                                         final_loss=lo, es_step=es, **ev, **dv))
                    log(lr=lr, width=w, arm="c2" if see else "c1", seed=sd, fid=ev["fid"],
                        w1=dv["w1_u"], loss=lo, es=es, span_src=ev["span_src"])
            save_csv("lr2_scaling.csv", rows_lr2)
            flush_reg()
    save_csv("lr2_scaling.csv", rows_lr2)

    def a2(lr, arm, w, key="fid"):
        sel = [r for r in rows_lr2 if r["lr"] == lr and r["arm"] == arm and r["width"] == w]
        v = [r[key] for r in sel]
        return float(np.mean(v)), float(np.std(v)), len(v)

    def gap2(lr, w):
        m1, s1, n1 = a2(lr, "c1", w)
        m2, s2, n2 = a2(lr, "c2", w)
        return m1 - m2, math.sqrt(s1 ** 2 / max(n1, 1) + s2 ** 2 / max(n2, 1))

    wlo, whi = LR2_WIDTHS[0], LR2_WIDTHS[-1]
    for lr in LR2_GRID:
        for w in LR2_WIDTHS:
            g, gs = gap2(lr, w)
            m1, s1, _ = a2(lr, "c1", w)
            m2, s2, _ = a2(lr, "c2", w)
            log(lr=lr, width=w, c1=m1, c1_sd=s1, c2=m2, c2_sd=s2, gap=g, gap_se=gs)

    trend = {lr: a2(lr, "c2", whi)[0] - a2(lr, "c2", wlo)[0] for lr in LR2_GRID}
    reg("lr2_c2_trend_sign_agrees_across_lr",
        all(t < 0 for t in trend.values()) or all(t > 0 for t in trend.values()),
        "the direction of the width effect on the source-conditioned model is a property of "
        "the objective and not of the learning rate",
        **{f"trend_lr{lr}": trend[lr] for lr in LR2_GRID})

    flat_g = {}
    for lr in LR2_GRID:
        glo, slo = gap2(lr, wlo)
        ghi, shi = gap2(lr, whi)
        flat_g[lr] = (ghi - glo, 2 * max(slo, shi))
    reg("lr2_gap_flat_in_width_at_both_lr",
        all(abs(d) < tol for d, tol in flat_g.values()),
        "the difference between the two models does not close with width at either learning rate",
        **{f"d_gap_lr{lr}": flat_g[lr][0] for lr in LR2_GRID},
        **{f"tol_lr{lr}": flat_g[lr][1] for lr in LR2_GRID})

    cross = []
    for w in LR2_WIDTHS:
        g0, s0 = gap2(LR2_GRID[0], w)
        g1, s1 = gap2(LR2_GRID[1], w)
        cross.append((w, abs(g0 - g1), 2 * max(s0, s1)))
    reg("lr2_gap_is_lr_independent", all(d < tol for _, d, tol in cross),
        "at matched width the difference between the two models is the same at both learning "
        "rates, so it is not a tuning artifact",
        max_abs_diff=max(d for _, d, _ in cross),
        widths=str([w for w, _, _ in cross]),
        diffs=str([round(d, 5) for _, d, _ in cross]))

    lossfall = {lr: a2(lr, "c2", whi, "final_loss")[0] - a2(lr, "c2", wlo, "final_loss")[0]
                for lr in LR2_GRID}
    reg("lr2_loss_falls_with_width_at_both_lr", all(v < 0 for v in lossfall.values()),
        "the training objective is optimised better at larger width regardless of learning rate",
        **{f"dloss_lr{lr}": lossfall[lr] for lr in LR2_GRID})

    w1fall = {lr: a2(lr, "c2", whi, "w1_u")[0] - a2(lr, "c2", wlo, "w1_u")[0] for lr in LR2_GRID}
    reg("lr2_w1_falls_with_width_at_both_lr", all(v < 0 for v in w1fall.values()),
        "the distributional error also improves with width, so the fidelity decrease is not a "
        "general degradation",
        **{f"dw1_lr{lr}": w1fall[lr] for lr in LR2_GRID})

    bw = {(lr, w): a2(lr, "c2", w, "w1_u")[0] for lr in LR2_GRID for w in [wlo, whi]}
    amin_lo = min(LR2_GRID, key=lambda lr: bw[(lr, wlo)])
    amin_hi = min(LR2_GRID, key=lambda lr: bw[(lr, whi)])
    reg("lr2_mup_transfer_resolved", amin_lo == amin_hi,
        "with more seeds the distributional criterion selects the same learning rate at both "
        "widths",
        argmin_narrow=amin_lo, argmin_wide=amin_hi,
        w1_narrow=str([round(bw[(lr, wlo)], 5) for lr in LR2_GRID]),
        w1_wide=str([round(bw[(lr, whi)], 5) for lr in LR2_GRID]))

    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.0))
    for lr, mk in zip(LR2_GRID, ["o-", "s--"]):
        for arm, col in [("c1", "tab:red"), ("c2", "tab:purple")]:
            mus = [a2(lr, arm, w)[0] for w in LR2_WIDTHS]
            sds = [a2(lr, arm, w)[1] for w in LR2_WIDTHS]
            ax[0].errorbar(LR2_WIDTHS, mus, yerr=sds, fmt=mk, c=col, capsize=3,
                           label=f"{arm} lr={lr}")
        gs = [gap2(lr, w) for w in LR2_WIDTHS]
        ax[1].errorbar(LR2_WIDTHS, [g for g, _ in gs], yerr=[s for _, s in gs], fmt=mk,
                       capsize=3, label=f"lr={lr}")
        ax[2].plot(LR2_WIDTHS, [a2(lr, "c2", w, "w1_u")[0] for w in LR2_WIDTHS], mk,
                   label=f"w1 c2 lr={lr}")
    for a in ax:
        a.set_xscale("log", base=2)
        a.set_xlabel("width")
        a.legend(fontsize=7)
    ax[0].set_ylabel("fid")
    ax[1].set_ylabel("c1 - c2")
    ax[1].axhline(0, c="k", lw=0.8)
    ax[2].set_ylabel("W1")
    save_fig("F12_lr2_scaling.png")
    tock("lr2")
    flush_reg()

# %% [code]
if "real" in STAGES:
    tick("real")
    try:
        try:
            import open_clip
        except Exception:
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch"], check=True)
            import open_clip
        import torchvision
        from torchvision import transforms as T

        model, _, preproc = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k")
        model = model.to(DEVICE).eval()
        root = "/kaggle/working/cifar" if os.path.isdir("/kaggle/working") else "runs/cifar"
        ds = torchvision.datasets.CIFAR10(root=root, train=True, download=True)

        @torch.no_grad()
        def embed(imgs):
            out = []
            for i in range(0, len(imgs), 256):
                x = torch.stack([preproc(p) for p in imgs[i:i + 256]]).to(DEVICE)
                out.append(normalize(model.encode_image(x).float()))
            return torch.cat(out)

        idx = np.random.default_rng(0).choice(len(ds), CLIP_N, replace=False)
        imgs = [ds[i][0] for i in idx]
        labels = torch.tensor([ds[i][1] for i in idx], device=DEVICE)
        Z1 = embed(imgs)
        blur = T.GaussianBlur(9, sigma=(0.8, 3.0))
        Z0 = embed([blur(p) for p in imgs])
        proto = normalize(torch.stack([Z1[labels == k].mean(0) for k in range(10)]))
        C = proto[labels]
        perm = torch.randperm(CLIP_N, device=DEVICE)
        floor = float((Z1 * Z1[perm]).sum(-1).mean())
        A_hat = float(torch.stack([Z1[labels == k].mean(0).norm() for k in range(10)]).mean())
        resid = normalize(project_tangent(Z1, C))
        beta_hat = float((1 - (Z1 * C).sum(-1) ** 2).clamp_min(0).sqrt().mean())
        j_hat = float((torch.arccos((Z0 * C).sum(-1).clamp(-1, 1))
                       - torch.arccos((Z1 * C).sum(-1).clamp(-1, 1))).mean())
        probe = nn.Linear(resid.shape[1], 10).to(DEVICE)
        po = torch.optim.AdamW(probe.parameters(), lr=1e-2)
        set_seed(4321)
        pperm = torch.randperm(CLIP_N, device=DEVICE)
        pcut = int(0.8 * CLIP_N)
        ptr, pte = pperm[:pcut], pperm[pcut:]
        for _ in range(400):
            l = F.cross_entropy(probe(resid[ptr].detach()), labels[ptr])
            po.zero_grad(set_to_none=True)
            l.backward()
            po.step()
        probe_acc = float((probe(resid[pte]).argmax(-1) == labels[pte]).float().mean())
        probe_acc_in = float((probe(resid[ptr]).argmax(-1) == labels[ptr]).float().mean())
        pred_bstar = beta_star(A_hat, max(j_hat, 0.01))
        log(floor=floor, A_hat=A_hat, beta_hat=beta_hat, j_hat=j_hat, pred_bstar=pred_bstar,
            probe_acc=probe_acc, probe_acc_in=probe_acc_in,
            exists=int(A_hat < math.cos(max(j_hat, 0.01))))

        reg("clip_xi_perp_c", probe_acc < 0.15,
            "the private residual carries no label information, as the synthetic model assumes; "
            "scored on a held-out split because a linear probe on this feature width has of the "
            "order of one parameter per sample and its in-sample accuracy is not an estimate of "
            "label content",
            probe_acc=probe_acc, probe_acc_in=probe_acc_in, chance=0.1,
            n_train=int(0.8 * CLIP_N), n_test=CLIP_N - int(0.8 * CLIP_N))
        reg("clip_registered_beta_star",
            (not np.isfinite(pred_bstar)) or beta_hat > pred_bstar,
            "estimated from the embeddings this data sits above the threshold so the flow should "
            "beat returning the blurred embedding",
            beta_hat=beta_hat, pred_bstar=pred_bstar)

        clip_rows = []
        for see in [False, True]:
            set_seed(0)
            rnet = FieldNet(Z1.shape[1], 512, 3, see).to(DEVICE)
            ro = torch.optim.AdamW(rnet.param_groups(1e-4))
            ntr = int(0.8 * CLIP_N)
            for step in range(CLIP_STEPS):
                b = torch.randint(0, ntr, (256,), device=DEVICE)
                t = torch.rand(256, device=DEVICE)
                zt = slerp(Z0[b], Z1[b], t[:, None])
                pr = project_tangent(rnet.raw(zt, t, C[b], Z0[b] if see else None), zt)
                loss = F.mse_loss(pr, slerp_velocity(Z0[b], Z1[b], t[:, None]))
                ro.zero_grad(set_to_none=True)
                loss.backward()
                ro.step()
            te = torch.arange(ntr, CLIP_N, device=DEVICE)
            aux = Z0[te] if see else None
            out = integrate(make_vel(rnet, C[te], aux), Z0[te], steps=32)
            got = float((out * Z1[te]).sum(-1).mean())
            clip_rows.append(dict(arm="c2" if see else "c1", flow=got))
            log(clip_arm="c2" if see else "c1", flow=got)
        src = float((Z0[torch.arange(int(0.8 * CLIP_N), CLIP_N, device=DEVICE)] *
                     Z1[torch.arange(int(0.8 * CLIP_N), CLIP_N, device=DEVICE)]).sum(-1).mean())
        pro = float((C[torch.arange(int(0.8 * CLIP_N), CLIP_N, device=DEVICE)] *
                     Z1[torch.arange(int(0.8 * CLIP_N), CLIP_N, device=DEVICE)]).sum(-1).mean())
        g1 = [r["flow"] for r in clip_rows if r["arm"] == "c1"][0]
        g2 = [r["flow"] for r in clip_rows if r["arm"] == "c2"][0]
        log(floor=floor, src=src, flow_c1=g1, flow_c2=g2, proto=pro)
        reg("clip_flow_beats_source", g1 > src,
            "the registered threshold prediction transfers to real embeddings",
            floor=floor, src=src, flow=g1, proto=pro,
            frac_of_range=(g1 - floor) / max(pro - floor, 1e-9))
        reg("clip_source_as_start_hurts", g2 < g1,
            "conditioning on the source while starting there also hurts on real embeddings",
            c1=g1, c2=g2)
        save_csv("clip.csv", [dict(floor=floor, A_hat=A_hat, beta_hat=beta_hat, j_hat=j_hat,
                                   pred_bstar=pred_bstar, probe_acc=probe_acc,
                                   src=src, flow_c1=g1, flow_c2=g2, proto=pro)])
        fig, ax = plt.subplots(figsize=(6.6, 3.6))
        nm = ["random pair", "blurred source", "flow c1", "flow c2", "class prototype"]
        vv = [floor, src, g1, g2, pro]
        ax.bar(nm, vv)
        for i, v in enumerate(vv):
            ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
        ax.set_ylabel("cosine to clean embedding")
        ax.tick_params(axis="x", labelsize=7)
        save_fig("F11_clip.png")
    except Exception as e:
        reg("clip_ran", False, "open_clip and CIFAR-10 are reachable", err=repr(e)[:200])
    tock("real")
    flush_reg()

# %% [code]
flush_reg()
npass = sum(1 for g in REG if g["tag"] == "PASS")
log(passed=npass, total=len(REG))
for k, v in TIMES.items():
    if isinstance(v, float) and v < 1e8:
        log(stage_time=k, minutes=v / 60.0)
log(total_minutes=sum(v / 60.0 for v in TIMES.values() if isinstance(v, float) and v < 1e8))
for g in REG:
    if g["tag"] == "WARN":
        log(contradicted=g["name"], predicted=g["predicts"], observed=json.dumps(g["detail"]))
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(dict(passed=npass, total=len(REG), stages=STAGES, fast=FAST,
                   d=D_OP, A=A_OP, beta=BETA_OP, j=J_OP, law=LAW0, beta_star=BSTAR,
                   times_min={k: v / 60.0 for k, v in TIMES.items()
                              if isinstance(v, float) and v < 1e8},
                   register=REG), f, indent=1)
log(out=OUT, files=len(os.listdir(os.path.join(OUT, "results")))
    + len(os.listdir(os.path.join(OUT, "figures"))))
