# %% [code]
# Is the source-conditioning gap a property of the problem, or of predicting the velocity?
#
# M2 regresses the velocity. Given (z_t, z0) the paired target is exactly recoverable, and that
# recovery has a 1/t Jacobian. If the gap is an artefact of asking the network to represent a
# quantity whose implicit inversion is ill-conditioned at small t, then predicting the endpoint
# z1 directly and deriving the velocity from it should remove the gap. If the gap survives, the
# degradation is structural and the velocity parameterisation is not the cause.
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

FAST = os.environ.get("PARAM_FAST", "0") == "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "/kaggle/working/param" if os.path.isdir("/kaggle/working") else "runs/param"
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
os.makedirs(os.path.join(OUT, "results"), exist_ok=True)

D_OP, A_OP, BETA_OP, J_OP = 256, 0.37, 0.8, 0.15
STEPS = 300 if FAST else 3000
BATCH = 128 if FAST else 256
BASE_WIDTH = 256
WIDTH = 256 if FAST else 512
BLOCKS = 3
ODE_K = 8 if FAST else 32
EVAL_N = 128 if FAST else 768
KDIV = 4 if FAST else 6
EPS = 1e-6
TAIL = 1e-4                      # floor on (1-t) when converting an endpoint to a velocity

PARAMS = ["v", "x1"]
ARMS = [False, True]             # see_source: False = M1, True = M2
LRS = [1e-3] if FAST else [1e-4, 1e-3]
SEEDS = [0, 1] if FAST else [0, 1, 2, 3, 4]

DIST_NCOND = 4 if FAST else 16
DIST_NPER = 32 if FAST else 128
DIST_KNN = 5

REG, TIMES = [], {}


def log(**kw):
    print(" ".join(f"{k}={v:.5g}" if isinstance(v, float) else f"{k}={v}"
                   for k, v in kw.items()), flush=True)


def reg(name, ok, predicts, **kw):
    REG.append(dict(tag="PASS" if ok else "WARN", name=name, predicts=predicts,
                    detail={k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                            for k, v in kw.items()}))
    log(reg=REG[-1]["tag"], name=name, **kw)


def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)


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
    plt.savefig(os.path.join(OUT, "figures", name + ".png"), dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(OUT, "figures", name + ".pdf"), bbox_inches="tight")
    plt.close()


def flush_reg():
    with open(os.path.join(OUT, "results", "register.json"), "w") as f:
        json.dump(dict(register=REG, times_min=TIMES), f, indent=1)


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
    nu, acc = d / 2.0, 0.0
    for k in range(int(max(500, 2.2 * kappa + 100)), 0, -1):
        acc = 1.0 / (2.0 * (nu + k) / kappa + acc)
    return 1.0 / (2.0 * nu / kappa + acc)


_KC = {}


def kappa_from_A(d, A):
    key = (d, round(A, 6))
    if key in _KC:
        return _KC[key]
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


_BC = {}


@torch.no_grad()
def sample_vmf(mu, kappa, over=4):
    B, d = mu.shape
    key = (d, str(mu.device))
    if key not in _BC:
        a = torch.tensor(0.5 * (d - 1), device=mu.device)
        _BC[key] = torch.distributions.Beta(a, a)
    bd = _BC[key]
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
    xi = rand_tangent_dir(c)
    s = math.sqrt(max(0.0, 1 - task.beta ** 2))
    m = normalize(s * c + task.beta * xi)
    z1 = sample_vmf(m, task.kappa)
    phi = torch.full((B, 1), float(task.jit), device=device)
    z0 = normalize(torch.cos(phi) * c + torch.sin(phi) * xi)
    return c, xi, xi, m, z0, z1


task0 = Task().resolve()

# %% [code]
def w1_1d(a, b):
    n = min(a.numel(), b.numel())
    return float((a.flatten()[:n].sort().values - b.flatten()[:n].sort().values).abs().mean())


@torch.no_grad()
def dist_report(sample_fn, task, n_cond=DIST_NCOND, n_per=DIST_NPER, seed=1234):
    set_seed(seed)
    cb = random_sphere((n_cond, task.d), DEVICE)
    c = cb.repeat_interleave(n_per, 0)
    _, xs, xt, m, z0, _ = make_batch(task, c.shape[0], DEVICE, c=c)
    xf = sample_fn(c, xs, xt, m, z0)
    _, _, _, _, _, ra = make_batch(task, c.shape[0], DEVICE, c=c)
    acc = []
    for i in range(n_cond):
        sl = slice(i * n_per, (i + 1) * n_per)
        ci = cb[i:i + 1]
        acc.append(w1_1d((xf[sl] * ci).sum(-1), (ra[sl] * ci).sum(-1)))
    return float(np.mean(acc))


@torch.no_grad()
def evaluate(sample_fn, task, n=EVAL_N, k_div=KDIV, seed=4242):
    set_seed(seed)                      # seeded, unlike the original routine
    cb = random_sphere((n, task.d), DEVICE)
    c = cb.repeat_interleave(k_div, 0)
    cc, xs, xt, m, z0, z1 = make_batch(task, c.shape[0], DEVICE, c=c)
    xf = sample_fn(cc, xs, xt, m, z0)
    g = normalize(xf).view(n, k_div, -1)
    pw = torch.einsum("nkd,nld->nkl", g, g)
    off = (pw.sum((1, 2)) - k_div) / (k_div * (k_div - 1))
    return dict(fid=float((xf * z1).sum(-1).mean()), paircos=float(off.mean()))


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

    def param_groups(self, base_lr):
        inp = list(self.inp.parameters()) + list(self.tproj.parameters())
        outp = list(self.out.parameters())
        ids = {id(p) for p in inp + outp}
        hid = [p for p in self.parameters() if id(p) not in ids]
        return [dict(params=inp, lr=base_lr),
                dict(params=hid, lr=base_lr / self.mult),
                dict(params=outp, lr=base_lr / self.mult)]

    def raw(self, z, t, c, aux=None):
        parts = [z, c]
        if self.see_source:
            parts.append(aux if aux is not None else torch.zeros_like(z))
        h = self.inp(torch.cat(parts, -1))
        te = self.tproj(self.temb(t))
        for b in self.blocks:
            h = b(h, te)
        return self.out(h)


# velocity implied by a predicted endpoint: travel the geodesic from z to zhat1 so as to
# arrive at t = 1. The 1/(1-t) factor sits at the terminal end, not the initial one.
def endpoint_from_raw(z, raw):
    # residual head: raw = 0 predicts "the endpoint is where we already are"
    return normalize(z + raw)


def vel_from_endpoint(z, zhat1, tq, tail):
    # travel the geodesic toward the predicted endpoint so as to arrive at t = 1; the
    # remaining time is floored at half a step so the last Heun evaluation stays finite
    return slerp_velocity(z, zhat1, torch.zeros_like(z[:, :1])) / max(1.0 - tq, tail)


def make_vel(net, c, aux, param, tail=TAIL):
    def vel(z, tq):
        t = torch.full((z.shape[0],), float(tq), device=z.device)
        raw = net.raw(z, t, c, aux)
        if param == "v":
            return project_tangent(raw, z)
        return project_tangent(
            vel_from_endpoint(z, endpoint_from_raw(z, raw), float(tq), tail), z)
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


def sf(net, param, steps=ODE_K):
    def f(c, xs, xt, m, z0):
        aux = z0 if net.see_source else None
        return integrate(make_vel(net, c, aux, param, tail=0.5 / steps), z0, steps)
    return f


def train(task, see_source, param, lr, width, steps=STEPS, seed=0):
    set_seed(seed)
    net = FieldNet(task.d, width, BLOCKS, see_source).to(DEVICE)
    opt = torch.optim.AdamW(net.param_groups(lr))
    run, t0 = None, time.time()
    for _ in range(steps):
        c, xs, xt, m, z0, z1 = make_batch(task, BATCH, DEVICE)
        t = torch.rand(BATCH, device=DEVICE)
        zt = slerp(z0, z1, t[:, None])
        aux = z0 if see_source else None
        raw = net.raw(zt, t, c, aux)
        if param == "v":
            pred = project_tangent(raw, zt)
            loss = F.mse_loss(pred, slerp_velocity(z0, z1, t[:, None]))
        else:
            loss = F.mse_loss(endpoint_from_raw(zt, raw), z1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        ld = loss.detach()
        run = ld if run is None else 0.95 * run + 0.05 * ld
    return net, float(run), time.time() - t0


log(device=DEVICE, fast=FAST, trainings=len(PARAMS) * len(ARMS) * len(LRS) * len(SEEDS),
    width=WIDTH, gpu=torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu")

# %% [code]
TIMES["sweep"] = time.time()
rows = []
for param in PARAMS:
    for lr in LRS:
        for sd in SEEDS:
            for see in ARMS:
                net, lo, secs = train(task0, see, param, lr, WIDTH, seed=sd)
                s = sf(net, param)
                ev = evaluate(s, task0)
                w1 = dist_report(s, task0)
                rows.append(dict(param=param, lr=lr, seed=sd, arm="c2" if see else "c1",
                                 final_loss=lo, fid=ev["fid"], paircos=ev["paircos"], w1_u=w1))
                log(param=param, lr=lr, seed=sd, arm=rows[-1]["arm"], fid=ev["fid"],
                    paircos=ev["paircos"], w1=w1, loss=lo, secs=secs)
        save_csv("param_control.csv", rows)
TIMES["sweep"] = (time.time() - TIMES["sweep"]) / 60.0
save_csv("param_control.csv", rows)
log(stage="sweep", minutes=TIMES["sweep"])

# %% [code]
def cell(param, lr, arm, key="fid"):
    v = [r[key] for r in rows if r["param"] == param and r["lr"] == lr and r["arm"] == arm]
    return (float(np.mean(v)), float(np.std(v)) / math.sqrt(max(len(v), 1)), len(v))


def gap(param, lr):
    m1, s1, n1 = cell(param, lr, "c1")
    m2, s2, n2 = cell(param, lr, "c2")
    return m1 - m2, math.sqrt(s1 ** 2 + s2 ** 2)


print("\n" + "=" * 74)
print(f"{'param':>6} {'lr':>8} {'M1 fid':>9} {'M2 fid':>9} {'gap':>9} {'gap/se':>8} "
      f"{'M1 pair':>9} {'M2 pair':>9}")
for param in PARAMS:
    for lr in LRS:
        g, se = gap(param, lr)
        print(f"{param:>6} {lr:>8g} {cell(param,lr,'c1')[0]:>9.4f} {cell(param,lr,'c2')[0]:>9.4f} "
              f"{g:>+9.4f} {g/max(se,1e-12):>8.1f} "
              f"{cell(param,lr,'c1','paircos')[0]:>9.4f} {cell(param,lr,'c2','paircos')[0]:>9.4f}")
print("=" * 74 + "\n")

for lr in LRS:
    gv, sev = gap("v", lr)
    gx, sex = gap("x1", lr)
    reg(f"param_gap_survives_endpoint_prediction_lr{lr:g}",
        gx > 2 * max(sex, 1e-9),
        "if the source-conditioning gap is an artefact of regressing the velocity, predicting "
        "the endpoint directly removes it; a gap that survives endpoint prediction is a "
        "property of the problem rather than of the parameterisation",
        lr=lr, gap_velocity=gv, se_velocity=sev, gap_endpoint=gx, se_endpoint=sex,
        ratio_endpoint_over_velocity=gx / gv if abs(gv) > 1e-12 else float("nan"))

gv_all = float(np.mean([gap("v", lr)[0] for lr in LRS]))
gx_all = float(np.mean([gap("x1", lr)[0] for lr in LRS]))
reg("param_gap_not_explained_by_parameterisation", gx_all > 0.5 * gv_all,
    "averaged over learning rates, the gap under endpoint prediction retains at least half its "
    "size under velocity prediction",
    mean_gap_velocity=gv_all, mean_gap_endpoint=gx_all,
    retained_fraction=gx_all / gv_all if abs(gv_all) > 1e-12 else float("nan"))

for lr in LRS:
    l1 = cell("x1", lr, "c1", "final_loss")[0]
    l2 = cell("x1", lr, "c2", "final_loss")[0]
    reg(f"param_endpoint_loss_still_lower_for_M2_lr{lr:g}", l2 < l1,
        "under endpoint prediction the source-conditioned arm still attains the lower training "
        "loss, so the regression problem remains easier even where the velocity target is not "
        "the object being fit",
        lr=lr, loss_M1=l1, loss_M2=l2)
flush_reg()

# %% [code]
fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.9))
x = np.arange(len(LRS))
w = 0.34
for k, (param, col) in enumerate(zip(PARAMS, ["#2b6cb0", "#b0421f"])):
    g = [gap(param, lr)[0] for lr in LRS]
    se = [gap(param, lr)[1] for lr in LRS]
    ax[0].bar(x + (k - 0.5) * w, g, w, yerr=se, capsize=4, color=col,
              label="velocity" if param == "v" else "endpoint")
    ax[1].plot(x, [cell(param, lr, "c2", "paircos")[0] for lr in LRS], "o--", color=col,
               label=f"$\\mathcal{{M}}_2$, {'velocity' if param=='v' else 'endpoint'}")
    ax[1].plot(x, [cell(param, lr, "c1", "paircos")[0] for lr in LRS], "o-", color=col,
               label=f"$\\mathcal{{M}}_1$, {'velocity' if param=='v' else 'endpoint'}")
ax[0].axhline(0, c="k", lw=0.9)
ax[0].set_xticks(x)
ax[0].set_xticklabels([f"$\\eta={lr:g}$" for lr in LRS])
ax[0].set_ylabel("$\\mathcal{M}_1-\\mathcal{M}_2$ terminal fidelity")
ax[0].set_title("Does the gap survive endpoint prediction?", fontsize=10)
ax[0].legend(fontsize=8, title="parameterisation", title_fontsize=8)
ax[1].set_xticks(x)
ax[1].set_xticklabels([f"$\\eta={lr:g}$" for lr in LRS])
ax[1].set_ylabel("pairwise cosine (lower = more variety)")
ax[1].set_title("Conditional diversity", fontsize=10)
ax[1].legend(fontsize=7)
fig.tight_layout()
save_fig("P1_param_control")
log(passed=sum(1 for g in REG if g["tag"] == "PASS"), total=len(REG), out=OUT)
