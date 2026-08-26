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
OUT = "/kaggle/working/lr3" if os.path.isdir("/kaggle/working") else "runs/lr3"
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
os.makedirs(os.path.join(OUT, "results"), exist_ok=True)

D_OP, A_OP, BETA_OP, J_OP = 256, 0.37, 0.8, 0.15
STEPS = 600 if FAST else 3000
BATCH = 256
BASE_WIDTH = 256
BLOCKS = 3
ODE_K = 32
EVAL_N = 384 if FAST else 768
KDIV = 4 if FAST else 6
EPS = 1e-6

LRS = [1e-4, 3e-4] if FAST else [1e-4, 3e-4, 1e-3]
WIDTHS = [256, 512] if FAST else [256, 512, 1024, 2048]
SEEDS = [0, 1] if FAST else [0, 1, 2, 3, 4]

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
    plt.savefig(os.path.join(OUT, "figures", name + ".png"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(OUT, "figures", name + ".pdf"), bbox_inches="tight")
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
log(device=DEVICE, torch=torch.__version__, out=OUT, fast=FAST,
    cells=len(LRS) * len(WIDTHS), trainings=len(LRS) * len(WIDTHS) * len(SEEDS) * 2,
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


@dataclass
class Task:
    d: int = D_OP
    A: float = A_OP
    beta: float = BETA_OP
    jit: float = J_OP
    kappa: float = 0.0

    def resolve(self):
        self.kappa = kappa_from_A(self.d, self.A)
        return self


@torch.no_grad()
def make_batch(task, B, device, c=None):
    if c is None:
        c = random_sphere((B, task.d), device)
    xi_s = rand_tangent_dir(c)
    xi_t = xi_s
    s = math.sqrt(max(0.0, 1 - task.beta ** 2))
    m = normalize(s * c + task.beta * xi_t)
    z1 = sample_vmf(m, task.kappa)
    phi = torch.full((B, 1), float(task.jit), device=device)
    z0 = normalize(torch.cos(phi) * c + torch.sin(phi) * xi_s)
    return c, xi_s, xi_t, m, z0, z1


task0 = Task().resolve()
log(geometry="ready", kappa=task0.kappa)

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

    def param_groups(self, base_lr):
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


def make_vel(net, c, aux):
    def vel(z, tq):
        t = torch.full((z.shape[0],), float(tq), device=z.device)
        return project_tangent(net.raw(z, t, c, aux), z)
    return vel


@torch.no_grad()
def integrate(vel_fn, z0, steps=ODE_K):
    z = z0.clone()
    ts = np.linspace(0.0, 1.0, steps + 1)
    for i in range(steps):
        t, tn = float(ts[i]), float(ts[i + 1])
        dt = tn - t
        v = vel_fn(z, t)
        zp = exp_map(z, dt * v)
        v = 0.5 * (v + project_tangent(vel_fn(zp, tn), z))
        z = exp_map(z, dt * v)
    return z


def sf_ode(net, steps=ODE_K):
    def f(c, xs, xt, m, z0):
        aux = z0 if net.see_source else None
        return integrate(make_vel(net, c, aux), z0, steps)
    return f


def sf_identity():
    return lambda c, xs, xt, m, z0: z0


def sf_true_post(task):
    def f(c, xs, xt, m, z0):
        return make_batch(task, c.shape[0], c.device, c=c)[5]
    return f


def train_field(task, see_source, lr, width, steps=STEPS, seed=0):
    set_seed(seed)
    net = FieldNet(task.d, width, BLOCKS, see_source, mup=True).to(DEVICE)
    opt = torch.optim.AdamW(net.param_groups(lr))
    every = max(200, steps // 8)
    best = dict(w1=float("inf"), state=None, step=-1)
    run_t, t0 = None, time.time()
    for step in range(1, steps + 1):
        c, xs, xt, m, z0, z1 = make_batch(task, BATCH, DEVICE)
        t = torch.rand(BATCH, device=DEVICE)
        zt = slerp(z0, z1, t[:, None])
        aux = z0 if see_source else None
        pred = project_tangent(net.raw(zt, t, c, aux), zt)
        loss = F.mse_loss(pred, slerp_velocity(z0, z1, t[:, None]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        ld = loss.detach()
        run_t = ld if run_t is None else 0.95 * run_t + 0.05 * ld
        if step % every == 0 or step == steps:
            w1 = dist_report(sf_ode(net), task, n_cond=6, n_per=64, seed=99)["w1_u"]
            if w1 < best["w1"]:
                best = dict(w1=w1, step=step,
                            state={k: v.detach().clone() for k, v in net.state_dict().items()})
    if best["state"] is not None:
        net.load_state_dict(best["state"])
    return net, float(run_t), best["step"], time.time() - t0


log(model="ready")

# %% [code]
tick("refs")
rows = []
for nm, sf in [("identity", sf_identity()), ("true_posterior", sf_true_post(task0))]:
    ev = evaluate(sf, task0)
    dv = dist_report(sf, task0)
    rows.append(dict(lr=0.0, width=0, seed=-1, arm=nm, final_loss=float("nan"),
                     es_step=-1, **ev, **dv))
    log(arm=nm, fid=ev["fid"], paircos=ev["paircos"], w1=dv["w1_u"])
PAIR_TRUE = [r for r in rows if r["arm"] == "true_posterior"][0]["paircos"]
tock("refs")

# %% [code]
tick("sweep")
for lr in LRS:
    for w in WIDTHS:
        for sd in SEEDS:
            for see in [False, True]:
                net, lo, es, secs = train_field(task0, see, lr, w, seed=sd)
                sf = sf_ode(net)
                ev = evaluate(sf, task0)
                dv = dist_report(sf, task0)
                rows.append(dict(lr=lr, width=w, seed=sd, arm="c2" if see else "c1",
                                 final_loss=lo, es_step=es, **ev, **dv))
                log(lr=lr, width=w, arm="c2" if see else "c1", seed=sd, fid=ev["fid"],
                    paircos=ev["paircos"], w1=dv["w1_u"], loss=lo, es=es, secs=secs)
        save_csv("lr3_scaling.csv", rows)
        flush_reg()
save_csv("lr3_scaling.csv", rows)
tock("sweep")

# %% [code]
def cells(lr, arm, w, key="fid"):
    v = [r[key] for r in rows if r["lr"] == lr and r["arm"] == arm and r["width"] == w]
    return float(np.mean(v)), float(np.std(v)), len(v)


def gap(lr, w):
    m1, s1, n1 = cells(lr, "c1", w)
    m2, s2, n2 = cells(lr, "c2", w)
    return m1 - m2, math.sqrt(s1 ** 2 / max(n1, 1) + s2 ** 2 / max(n2, 1))


for lr in LRS:
    for w in WIDTHS:
        g, gs = gap(lr, w)
        log(lr=lr, width=w, c1=cells(lr, "c1", w)[0], c2=cells(lr, "c2", w)[0],
            gap=g, gap_se=gs,
            paircos_c1=cells(lr, "c1", w, "paircos")[0],
            paircos_c2=cells(lr, "c2", w, "paircos")[0])

allg = [(lr, w) + gap(lr, w) for lr in LRS for w in WIDTHS]
reg("lr3_gap_sign_universal", all(g > 2 * gs for _, _, g, gs in allg),
    "the marginal model beats the source-conditioned model in every cell of the "
    "learning-rate-by-width grid",
    n_cells=len(allg), min_gap_over_se=min(g / max(gs, 1e-12) for _, _, g, gs in allg))

mono = 0
for w in WIDTHS:
    gs_ = [gap(lr, w)[0] for lr in LRS]
    mono += int(all(gs_[i] < gs_[i + 1] for i in range(len(gs_) - 1)))
rho_lr = spearman([LRS.index(lr) for lr, _, _, _ in allg], [g for _, _, g, _ in allg])
reg("lr3_gap_grows_with_step_size", mono == len(WIDTHS) and rho_lr > 0.5,
    "the deficit grows monotonically with the training step size at every width, which is what "
    "an amplification mechanism near the initial time predicts and a generic optimization "
    "failure does not",
    monotone_widths=mono, total_widths=len(WIDTHS), spearman_lr_gap=rho_lr,
    gaps_by_lr=str({lr: round(float(np.mean([gap(lr, w)[0] for w in WIDTHS])), 4)
                    for lr in LRS}))

closed = 0
for lr in LRS:
    g0, s0 = gap(lr, WIDTHS[0])
    g1, s1 = gap(lr, WIDTHS[-1])
    closed += int(g1 < g0 - 2 * math.sqrt(s0 ** 2 + s1 ** 2))
reg("lr3_capacity_never_closes_gap", closed == 0,
    "at no learning rate does the widest model close the gap relative to the narrowest",
    lrs_where_closed=closed, n_lrs=len(LRS))

pc_ok = True
for lr in LRS:
    for w in WIDTHS:
        p1, sd1, _ = cells(lr, "c1", w, "paircos")
        p2, sd2, _ = cells(lr, "c2", w, "paircos")
        if not (p2 - p1 > 3 * max(sd1, sd2, 1e-12)):
            pc_ok = False
reg("lr3_diversity_collapse_universal", pc_ok,
    "the source-conditioned model collapses conditional diversity in every cell, measured as "
    "pairwise cosine among samples sharing a condition against the true-posterior reference",
    paircos_true=PAIR_TRUE,
    paircos_c1_mean=float(np.mean([r["paircos"] for r in rows if r["arm"] == "c1"])),
    paircos_c2_mean=float(np.mean([r["paircos"] for r in rows if r["arm"] == "c2"])))
flush_reg()

# %% [code]
fig, ax = plt.subplots(1, 3, figsize=(14.0, 3.8))
mk = {lr: m for lr, m in zip(LRS, ["o", "s", "^"])}
for lr in LRS:
    ax[0].errorbar(WIDTHS, [cells(lr, "c1", w)[0] for w in WIDTHS],
                   yerr=[cells(lr, "c1", w)[1] for w in WIDTHS],
                   fmt=mk[lr] + "-", capsize=3, label=f"c1 lr={lr}")
    ax[0].errorbar(WIDTHS, [cells(lr, "c2", w)[0] for w in WIDTHS],
                   yerr=[cells(lr, "c2", w)[1] for w in WIDTHS],
                   fmt=mk[lr] + "--", capsize=3, label=f"c2 lr={lr}")
    ax[1].errorbar(WIDTHS, [gap(lr, w)[0] for w in WIDTHS],
                   yerr=[gap(lr, w)[1] for w in WIDTHS],
                   fmt=mk[lr] + "-", capsize=3, label=f"lr={lr}")
    ax[2].plot(WIDTHS, [cells(lr, "c2", w, "paircos")[0] for w in WIDTHS],
               mk[lr] + "--", label=f"c2 lr={lr}")
    ax[2].plot(WIDTHS, [cells(lr, "c1", w, "paircos")[0] for w in WIDTHS],
               mk[lr] + "-", label=f"c1 lr={lr}")
ax[1].axhline(0, c="k", lw=0.8)
ax[2].axhline(PAIR_TRUE, ls=":", c="k", lw=1.2, label="true posterior")
ax[0].set_ylabel("fid")
ax[1].set_ylabel("c1 - c2")
ax[2].set_ylabel("pairwise cosine")
for a in ax:
    a.set_xscale("log", base=2)
    a.set_xlabel("width")
    a.legend(fontsize=6)
save_fig("L1_lr3")
flush_reg()
log(done=1, out=OUT)
