# %% [code]
import os, csv, math, json, time
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

FAST = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "/kaggle/working/retest" if os.path.isdir("/kaggle/working") else "runs/retest"
os.makedirs(os.path.join(OUT, "results"), exist_ok=True)

D_OP, A_OP, BETA_OP, J_OP = 256, 0.37, 0.8, 0.15
STEPS = 600 if FAST else 3000
BATCH = 256
HIDDEN = 512
BASE_WIDTH = 256
BLOCKS = 3
BASE_LR = 3e-4
ODE_K = 32
EVAL_N = 384 if FAST else 768
KDIV = 4 if FAST else 6
EPS = 1e-6

ALPHAS = [0.0, 0.5, 0.95] if FAST else [0.0, 0.25, 0.5, 0.7, 0.85, 0.95]
SIG_AUX = [0.0, 0.2, 1.0] if FAST else [0.0, 0.1, 0.2, 0.35, 0.6, 1.0]
SEEDS = [0, 1]
CAP_EPS = [1.0, 0.5, 0.25] if FAST else [1.0, 0.7, 0.5, 0.35, 0.25]
RHO_CPL = [0.0, 1.0] if FAST else [0.0, 0.25, 0.5, 0.75, 1.0]
ORACLE_STEPS = 800 if FAST else 1800

DIST_NCOND = 8 if FAST else 16
DIST_NPER = 64 if FAST else 128
DIST_KNN = 5

REG, TIMES = [], {}


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


def tick(k):
    TIMES[k] = time.time()


def tock(k):
    TIMES[k] = (time.time() - TIMES[k]) / 60.0
    log(stage=k, minutes=TIMES[k])


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


set_seed(0)
log(device=DEVICE, torch=torch.__version__, out=OUT, fast=FAST,
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


_mu = random_sphere((20000, D_OP), DEVICE)
log(geometry="ready", kappa_op=kappa_from_A(D_OP, A_OP),
    vmf_A_measured=float((sample_vmf(_mu, kappa_from_A(D_OP, A_OP)) * _mu).sum(-1).mean()),
    A_target=A_OP)

# %% [code]
@dataclass
class Task:
    d: int = D_OP
    A: float = A_OP
    beta: float = BETA_OP
    jit: float = J_OP
    kappa: float = 0.0
    rho_couple: float = 1.0
    sig_polar: float = 0.0

    def resolve(self):
        self.kappa = kappa_from_A(self.d, self.A)
        return self


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
    return c, xi_s, xi_t, m, z0, z1


def cap_task(eps, rho=1.0, A0=A_OP, beta0=BETA_OP, j0=J_OP):
    s0 = math.sqrt(max(0.0, 1 - beta0 ** 2))
    A = math.cos(eps * math.acos(min(max(A0, -1.0), 1.0)))
    s = math.cos(eps * math.acos(min(max(s0, -1.0), 1.0)))
    beta = math.sqrt(max(0.0, 1 - s * s))
    return Task(d=D_OP, A=A, beta=beta, jit=eps * j0, rho_couple=rho).resolve()


task0 = Task().resolve()
for e in CAP_EPS:
    tk = cap_task(e)
    log(cap_eps=e, A=tk.A, beta=tk.beta, jit=tk.jit, kappa=tk.kappa,
        angle_c_to_m=float(math.acos(min(max(math.sqrt(max(0.0, 1 - tk.beta ** 2)), -1.0), 1.0))),
        angle_spread_z1=float(math.acos(min(max(tk.A, -1.0), 1.0))))

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
    acc = {k: [] for k in ["w1_u", "w1_null", "energy", "cov", "polar_sd"]}
    for i in range(n_cond):
        sl = slice(i * n_per, (i + 1) * n_per)
        ci = cb[i:i + 1]
        g, r1, r2 = xf[sl], ra[sl], rb[sl]
        pg, p1, p2 = (g * ci).sum(-1), (r1 * ci).sum(-1), (r2 * ci).sum(-1)
        acc["w1_u"].append(w1_1d(pg, p1))
        acc["w1_null"].append(w1_1d(p2, p1))
        acc["energy"].append(energy_distance(g, r1))
        acc["cov"].append(coverage(r1, g))
        acc["polar_sd"].append(float(pg.std()))
    return {k: float(np.mean(v)) for k, v in acc.items()}


@torch.no_grad()
def evaluate(sample_fn, task, n=EVAL_N, k_div=KDIV):
    cb = random_sphere((n, task.d), DEVICE)
    c = cb.repeat_interleave(k_div, 0)
    cc, xs, xt, m, z0, z1 = make_batch(task, c.shape[0], DEVICE, c=c)
    xf = sample_fn(cc, xs, xt, m, z0)
    pc = (xf * cc).sum(-1)
    ps = (xf * xs).sum(-1)
    g = normalize(xf).view(n, k_div, -1)
    pw = torch.einsum("nkd,nld->nkl", g, g)
    off = (pw.sum((1, 2)) - k_div) / (k_div * (k_div - 1))
    return dict(fid=float((xf * z1).sum(-1).mean()), align_c=float(pc.mean()),
                span_src=float((pc ** 2 + ps ** 2).mean()), paircos=float(off.mean()),
                polar_sd_out=float(pc.std()))


log(metrics="ready")

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


@torch.no_grad()
def integrate(vel_fn, z0, steps=ODE_K, heun=True, t_start=0.0):
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
    return z


def sf_ode(net, steps=ODE_K, heun=True, aux_sigma=0.0):
    def f(c, xs, xt, m, z0):
        aux = None
        if net.see_source:
            aux = z0 if aux_sigma <= 0 else exp_map(z0, aux_sigma * rand_tangent_dir(z0))

        def vel(z, tq):
            t = torch.full((z.shape[0],), float(tq), device=z.device)
            return project_tangent(net.raw(z, t, c, aux), z)
        return integrate(vel, z0, steps, heun=heun)
    return f


def sf_alpha(net, alpha, steps=ODE_K, heun=True):
    @torch.no_grad()
    def f(c, xs, xt, m, z0):
        B = c.shape[0]
        ts = np.linspace(0.0, 1.0, steps + 1)
        cache = {0.0: z0.clone()}
        z = z0.clone()
        for i in range(steps):
            t, tn = float(ts[i]), float(ts[i + 1])
            dt = tn - t
            k = min(cache, key=lambda q: abs(q - alpha * t))
            tt = torch.full((B,), t, device=z.device)
            v = project_tangent(net.raw(z, tt, c, cache[k]), z)
            if heun:
                zp = exp_map(z, dt * v)
                kn = min(cache, key=lambda q: abs(q - alpha * tn))
                ttn = torch.full((B,), tn, device=z.device)
                v = 0.5 * (v + project_tangent(net.raw(zp, ttn, c, cache[kn]), z))
            z = exp_map(z, dt * v)
            cache[tn] = z.clone()
        return z
    return f


def train_field(task, see_source, steps=STEPS, seed=0, tag="", width=HIDDEN,
                lr=BASE_LR, aux_alpha=None, aux_sigma=0.0):
    set_seed(seed)
    net = FieldNet(task.d, width, BLOCKS, see_source).to(DEVICE)
    opt = torch.optim.AdamW(net.param_groups(lr))
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
        pred = project_tangent(net.raw(zt, t, c, aux), zt)
        loss = F.mse_loss(pred, slerp_velocity(z0, z1, t[:, None]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        ld = loss.detach()
        run_t = ld if run_t is None else 0.95 * run_t + 0.05 * ld
    run = float(run_t)
    log(tag=tag, steps=steps, loss=run, secs=time.time() - t0)
    return net, run


log(model="ready")

# %% [code]
tick("integrator")
rows = []
nets = {}
for kind, grid in [("alpha", ALPHAS), ("sigma_aux", SIG_AUX)]:
    for sd in SEEDS:
        for g in grid:
            if kind == "alpha":
                net, lo = train_field(task0, True, seed=sd, aux_alpha=g, tag=f"a{g}s{sd}")
            else:
                net, lo = train_field(task0, True, seed=sd, aux_sigma=g, tag=f"n{g}s{sd}")
            nets[(kind, g, sd)] = (net, lo)
            for hn in [True, False]:
                sf = sf_alpha(net, g, heun=hn) if kind == "alpha" else \
                     sf_ode(net, heun=hn, aux_sigma=g)
                ev = evaluate(sf, task0)
                dv = dist_report(sf, task0)
                rows.append(dict(kind=kind, knob=g, seed=sd, heun=int(hn), final_loss=lo,
                                 **ev, **dv))
                log(kind=kind, knob=g, seed=sd, heun=int(hn), fid=ev["fid"], w1=dv["w1_u"],
                    span_src=ev["span_src"])
        save_csv("integrator.csv", rows)
save_csv("integrator.csv", rows)


def pick(kind, knob, sd, hn, key):
    v = [r[key] for r in rows if r["kind"] == kind and r["knob"] == knob
         and r["seed"] == sd and r["heun"] == int(hn)]
    return float(np.mean(v)) if v else float("nan")


la = float(np.mean([nets[("alpha", 0.0, s)][1] for s in SEEDS]))
ls = float(np.mean([nets[("sigma_aux", 0.0, s)][1] for s in SEEDS]))
reg("anchor_is_the_same_model", abs(la - ls) < 0.05 * max(la, ls),
    "at alpha=0 and sigma_aux=0 the auxiliary input is z0 in both branches, so the two "
    "families share one trained model and their terminal losses must agree",
    loss_alpha0=la, loss_sigma0=ls, rel_diff=abs(la - ls) / max(la, ls))

seed_spread = float(np.mean([
    abs(pick(k, 0.0, SEEDS[0], True, "w1_u") - pick(k, 0.0, SEEDS[1], True, "w1_u"))
    for k in ["alpha", "sigma_aux"]])) if len(SEEDS) > 1 else float("nan")
mis = abs(float(np.mean([pick("alpha", 0.0, s, False, "w1_u") for s in SEEDS]))
          - float(np.mean([pick("sigma_aux", 0.0, s, True, "w1_u") for s in SEEDS])))
mat = abs(float(np.mean([pick("alpha", 0.0, s, True, "w1_u") for s in SEEDS]))
          - float(np.mean([pick("sigma_aux", 0.0, s, True, "w1_u") for s in SEEDS])))
reg("anchor_splits_under_mismatched_integrator", mis > 3 * max(seed_spread, 1e-9),
    "the original comparison sampled the alpha family with Euler and the auxiliary-noise "
    "family with Heun, so the shared model reports two different distributional errors",
    gap_mismatched=mis, seed_spread=seed_spread)
reg("anchor_agrees_under_matched_integrator", mat < 3 * max(seed_spread, 1e-9),
    "under one integrator the shared model reports one distributional error, which is the "
    "test that the offset was the sampler and not the auxiliary law",
    gap_matched=mat, gap_mismatched=mis, seed_spread=seed_spread,
    shrink=mis / max(mat, 1e-9))

FEAT = ["fid", "span_src", "paircos", "w1_u"]


def curve(kind, sd, hn):
    grid = ALPHAS if kind == "alpha" else SIG_AUX
    return np.array([[pick(kind, g, sd, hn, f) for f in FEAT] for g in grid])


def chamfer(P, Q):
    D = np.linalg.norm(P[:, None, :] - Q[None, :, :], axis=-1)
    return 0.5 * (D.min(1).mean() + D.min(0).mean())


for label, ha, hs in [("mismatched", False, True), ("matched_heun", True, True),
                      ("matched_euler", False, False)]:
    fam = {}
    for sd in SEEDS:
        fam[("alpha", sd)] = curve("alpha", sd, ha)
        fam[("sigma_aux", sd)] = curve("sigma_aux", sd, hs)
    allp = np.concatenate(list(fam.values()), 0)
    mu, sg = allp.mean(0), allp.std(0) + 1e-9
    nf = {k: (v - mu) / sg for k, v in fam.items()}
    cross = float(np.mean([chamfer(nf[("alpha", a)], nf[("sigma_aux", b)])
                           for a in SEEDS for b in SEEDS]))
    within = float(np.mean([chamfer(nf[(k, SEEDS[0])], nf[(k, SEEDS[1])])
                            for k in ["alpha", "sigma_aux"]])) if len(SEEDS) > 1 else float("nan")
    ok = cross < 1.5 * within
    reg(f"noise_equivalence_{label}", ok if label != "mismatched" else (not ok),
        "the two auxiliary families trace one curve once they are sampled with the same "
        "integrator; under mismatched integrators they do not"
        if label != "mismatched" else
        "the original mismatched comparison is reproduced and still fails",
        cross=cross, within=within, ratio=cross / max(within, 1e-9))
tock("integrator")
flush_reg()

# %% [code]
class OraNet(nn.Module):
    def __init__(self, nin=2, nout=1):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(nin, 96), nn.SiLU(),
                               nn.Linear(96, 96), nn.SiLU(), nn.Linear(96, nout))

    def forward(self, x):
        return self.f(x)


def fit_oracle1(task, steps=ORACLE_STEPS, batch=1024, seed=0):
    set_seed(seed + 777)
    ora = OraNet().to(DEVICE)
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


def sf_oracle1(ora, steps=64):
    def f(c, xs, xt, m, z0):
        def vel(z, tq):
            t = torch.full((z.shape[0], 1), float(tq), device=z.device)
            u = (z * c).sum(-1, keepdim=True)
            return ora(torch.cat([u, t], -1))[:, 0:1] * project_tangent(c, z)
        return integrate(vel, z0, steps=steps)
    return f


tick("cap")
crows = []
for e in CAP_EPS:
    thetas = []
    for rho in RHO_CPL:
        tk = cap_task(e, rho=rho)
        ora, fl = fit_oracle1(tk)
        ev = evaluate(sf_oracle1(ora), tk)
        th = math.acos(min(max(ev["align_c"], -1.0), 1.0))
        thetas.append(th)
        crows.append(dict(cap_eps=e, rho=rho, floor=fl, theta=th, **ev))
        log(cap_eps=e, rho=rho, align_c=ev["align_c"], theta=th, span_src=ev["span_src"], floor=fl)
    sp = max(thetas) - min(thetas)
    crows.append(dict(cap_eps=e, rho=float("nan"), theta_spread=sp,
                      theta_mean=float(np.mean(thetas)), rel_spread=sp / max(np.mean(thetas), 1e-12)))
    log(cap_eps=e, theta_spread=sp, theta_mean=float(np.mean(thetas)),
        rel_spread=sp / max(np.mean(thetas), 1e-12))
    save_csv("cap_scaling.csv", crows)
save_csv("cap_scaling.csv", crows)

sm = [r for r in crows if "theta_spread" in r]
x = np.log10([r["cap_eps"] for r in sm])
y = np.log10([max(r["theta_spread"], 1e-16) for r in sm])
Amat = np.stack([x, np.ones_like(x)], 1)
slope, icept = np.linalg.lstsq(Amat, y, rcond=None)[0]
resid = float(np.sqrt(np.mean((Amat @ np.array([slope, icept]) - y) ** 2)))
log(law="cap", slope=float(slope), intercept=float(icept), rms_resid=resid, n=len(sm))

reg("cap_spread_is_a_power_law", resid < 0.25,
    "the spread of the endpoint map over the coupling is a power law in the cap radius, so "
    "the failure of transfer is quantitative and not categorical",
    slope=float(slope), rms_resid=resid, n=len(sm))
reg("cap_transfer_in_flat_limit", slope > 1.15,
    "if the flat criterion is recovered as the sphere flattens then the spread must vanish "
    "faster than the cap radius itself, that is with exponent strictly above one; exponent "
    "one means the spread is a fixed fraction of the configuration and transfer fails",
    slope=float(slope), threshold=1.15)
rel = [r["rel_spread"] for r in sm]
reg("cap_relative_spread_shrinks", rel[-1] < 0.5 * rel[0],
    "the spread measured in units of the configuration scale shrinks as the cap closes",
    rel_at_largest=rel[0], rel_at_smallest=rel[-1], eps=str(CAP_EPS))
tock("cap")
flush_reg()

# %% [code]
flush_reg()
npass = sum(1 for g in REG if g["tag"] == "PASS")
log(passed=npass, total=len(REG), total_minutes=sum(TIMES.values()))
for g in REG:
    if g["tag"] == "WARN":
        log(contradicted=g["name"], predicted=g["predicts"], observed=json.dumps(g["detail"]))
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(dict(passed=npass, total=len(REG), times_min=TIMES, register=REG), f, indent=1)
log(out=OUT, files=len(os.listdir(os.path.join(OUT, "results"))))
