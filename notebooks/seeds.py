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
OUT = "/kaggle/working/seeds" if os.path.isdir("/kaggle/working") else "runs/seeds"
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

SEEDS_FIELD = [0, 1, 2, 3] if FAST else [0, 1, 2, 3, 4, 5, 6, 7]
SEEDS_ORACLE = [0, 1, 2, 3] if FAST else list(range(10))
CAP_EPS = [1.0, 0.5, 0.25] if FAST else [1.0, 0.5, 0.25]
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


def halves(v):
    v = list(v)
    h = len(v) // 2
    return v[:h], v[h:]


set_seed(0)
log(device=DEVICE, torch=torch.__version__, out=OUT, fast=FAST,
    gpu=torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
    n_field_seeds=len(SEEDS_FIELD), n_oracle_seeds=len(SEEDS_ORACLE))

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
_mu = random_sphere((20000, D_OP), DEVICE)
log(geometry="ready", kappa_op=task0.kappa,
    vmf_A_measured=float((sample_vmf(_mu, task0.kappa) * _mu).sum(-1).mean()), A_target=A_OP)

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
    acc = {k: [] for k in ["w1_u", "w1_null", "energy", "cov"]}
    for i in range(n_cond):
        sl = slice(i * n_per, (i + 1) * n_per)
        ci = cb[i:i + 1]
        g, r1, r2 = xf[sl], ra[sl], rb[sl]
        pg, p1, p2 = (g * ci).sum(-1), (r1 * ci).sum(-1), (r2 * ci).sum(-1)
        acc["w1_u"].append(w1_1d(pg, p1))
        acc["w1_null"].append(w1_1d(p2, p1))
        acc["energy"].append(energy_distance(g, r1))
        acc["cov"].append(coverage(r1, g))
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
                span_src=float((pc ** 2 + ps ** 2).mean()), paircos=float(off.mean()))


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
    def __init__(self, d, width, blocks, see_source, base_width=BASE_WIDTH):
        super().__init__()
        self.d, self.see_source = d, see_source
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
def integrate(vel_fn, z0, steps=ODE_K, heun=True):
    z = z0.clone()
    ts = np.linspace(0.0, 1.0, steps + 1)
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


def train_field(task, see_source, steps=STEPS, seed=0, tag="", width=HIDDEN,
                lr=BASE_LR, aux_sigma=0.0):
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
            aux = z0 if aux_sigma <= 0 else exp_map(z0, aux_sigma * rand_tangent_dir(z0))
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


log(model="ready")

# %% [code]
tick("field_seeds")
frows = []
CONFIGS = [("exact_z0", 0.0), ("noised_1.0", 1.0)]
for nm, sg in CONFIGS:
    for sd in SEEDS_FIELD:
        net, lo = train_field(task0, True, seed=sd, aux_sigma=sg, tag=f"{nm}_s{sd}")
        sf = sf_ode(net, heun=True, aux_sigma=sg)
        ev = evaluate(sf, task0)
        dv = dist_report(sf, task0)
        frows.append(dict(config=nm, aux_sigma=sg, seed=sd, final_loss=lo, **ev, **dv))
        log(config=nm, seed=sd, fid=ev["fid"], w1=dv["w1_u"], span_src=ev["span_src"])
    save_csv("field_seeds.csv", frows)
save_csv("field_seeds.csv", frows)


def col(nm, key):
    return [r[key] for r in frows if r["config"] == nm]


for key in ["fid", "w1_u", "span_src"]:
    for nm, _ in CONFIGS:
        v = col(nm, key)
        log(config=nm, metric=key, mean=float(np.mean(v)), sd=float(np.std(v)),
            rng=float(max(v) - min(v)), n=len(v))

stab = {}
for key in ["fid", "w1_u"]:
    ok = True
    for nm, _ in CONFIGS:
        v = col(nm, key)
        a, b = halves(v)
        sf_, sa, sb = np.std(v), np.std(a), np.std(b)
        stab[f"{nm}_{key}"] = (float(sf_), float(sa), float(sb))
        if sf_ <= 0 or abs(sa - sf_) / sf_ > 0.6 or abs(sb - sf_) / sf_ > 0.6:
            ok = False
    reg(f"seed_sd_stable_{key}", ok,
        "the seed standard deviation is estimated well enough that each half of the seeds "
        "reproduces the pooled estimate, so it is usable as a null for the family comparison",
        **{k: v[0] for k, v in stab.items() if k.endswith(key)}, n_seeds=len(SEEDS_FIELD))

for key in ["fid", "w1_u", "span_src"]:
    a, b = col(CONFIGS[0][0], key), col(CONFIGS[1][0], key)
    eff = abs(float(np.mean(a)) - float(np.mean(b)))
    noise = float(max(np.std(a), np.std(b)))
    reg(f"knob_effect_exceeds_seed_noise_{key}", eff > 3 * max(noise, 1e-12),
        "the auxiliary-noise knob moves the model far more than the seed does, so a "
        "family-to-family comparison is measurable at this seed count",
        effect=eff, seed_sd=noise, ratio=eff / max(noise, 1e-12))
tock("field_seeds")
flush_reg()

# %% [code]
tick("oracle_seeds")
orows = []
for e in CAP_EPS:
    tk = cap_task(e, rho=1.0)
    th = []
    for sd in SEEDS_ORACLE:
        ora, fl = fit_oracle1(tk, seed=sd)
        ev = evaluate(sf_oracle1(ora), tk)
        t_ = math.acos(min(max(ev["align_c"], -1.0), 1.0))
        th.append(t_)
        orows.append(dict(cap_eps=e, seed=sd, floor=fl, theta=t_, **ev))
    log(cap_eps=e, theta_mean=float(np.mean(th)), theta_sd=float(np.std(th)),
        theta_range=float(max(th) - min(th)), n=len(th))
    orows.append(dict(cap_eps=e, seed=-1, theta_sd=float(np.std(th)),
                      theta_range=float(max(th) - min(th)), theta_mean=float(np.mean(th))))
    save_csv("oracle_seeds.csv", orows)
save_csv("oracle_seeds.csv", orows)

sm = [r for r in orows if r.get("seed") == -1]
x = np.log10([r["cap_eps"] for r in sm])
y = np.log10([max(r["theta_sd"], 1e-16) for r in sm])
Am = np.stack([x, np.ones_like(x)], 1)
nslope, nic = np.linalg.lstsq(Am, y, rcond=None)[0]
nresid = float(np.sqrt(np.mean((Am @ np.array([nslope, nic]) - y) ** 2)))
log(law="refit_noise", slope=float(nslope), intercept=float(nic), rms_resid=nresid, n=len(sm))

for key in ["theta"]:
    ok = True
    for e in CAP_EPS:
        v = [r[key] for r in orows if r["cap_eps"] == e and r.get("seed", -1) >= 0]
        a, b = halves(v)
        s = np.std(v)
        if s <= 0 or abs(np.std(a) - s) / s > 0.6 or abs(np.std(b) - s) / s > 0.6:
            ok = False
    reg("refit_sd_stable", ok,
        "the refit standard deviation of the endpoint map is estimated well enough that each "
        "half of the seeds reproduces the pooled estimate",
        n_seeds=len(SEEDS_ORACLE), eps=str(CAP_EPS))

reg("refit_noise_shrinks_with_cap", nslope > 0.8,
    "the refit noise floor shrinks at least proportionally to the cap radius, so it does not "
    "become the dominant term at small cap and the exponent fitted for the coupling spread is "
    "not an artifact of a flat noise floor",
    noise_slope=float(nslope), rms_resid=nresid)

e_min = min(CAP_EPS)
tks = []
for rho in RHO_CPL:
    tk = cap_task(e_min, rho=rho)
    ora, fl = fit_oracle1(tk, seed=0)
    ev = evaluate(sf_oracle1(ora), tk)
    t_ = math.acos(min(max(ev["align_c"], -1.0), 1.0))
    tks.append(t_)
    orows.append(dict(cap_eps=e_min, rho=rho, seed=-2, theta=t_, floor=fl, **ev))
    log(cap_eps=e_min, rho=rho, theta=t_, align_c=ev["align_c"])
save_csv("oracle_seeds.csv", orows)

sig = max(tks) - min(tks)
noise_sd = [r["theta_sd"] for r in sm if r["cap_eps"] == e_min][0]
reg("signal_exceeds_refit_noise_at_smallest_cap", sig > 3 * max(noise_sd, 1e-12),
    "at the smallest cap the spread of the endpoint map over the coupling still stands clear "
    "of the refit noise, so the cap-scaling exponent is measured on signal and not on noise",
    cap_eps=e_min, rho_spread=sig, refit_sd=noise_sd, ratio=sig / max(noise_sd, 1e-12))
tock("oracle_seeds")
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
