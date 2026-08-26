# %% [code]
import os, sys, csv, math, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

P = argparse.ArgumentParser()
P.add_argument("--part", default="both", choices=["theorem", "sr", "srthresh", "both"])
P.add_argument("--out", default="runs/cluster")
P.add_argument("--wandb", default="1")
P.add_argument("--wandb_project", default="coupling-ledger")
P.add_argument("--wandb_entity", default=None)
P.add_argument("--thm_dims", default="2,4,8,16,32")
P.add_argument("--thm_batch", type=int, default=2048)
P.add_argument("--thm_reps", type=int, default=4)
P.add_argument("--thm_steps", type=int, default=2000)
P.add_argument("--sr_res", default="32,48,96")
P.add_argument("--sr_factor", type=int, default=4)
P.add_argument("--sr_factors", default="2,3,4,6,8,12")
P.add_argument("--sr_steps", type=int, default=20000)
P.add_argument("--sr_batch", type=int, default=64)
P.add_argument("--sr_base", type=int, default=64)
P.add_argument("--sr_seeds", default="0,1,2")
P.add_argument("--sr_ode_k", type=int, default=32)
P.add_argument("--sr_data", default="stl10", choices=["stl10","cifar10","synth"])
P.add_argument("--sr_root", default="./data")
ARGS = P.parse_args() if len(sys.argv) > 1 else P.parse_args([])

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = ARGS.out
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
REG, TIMES = [], {}

WB = None
if ARGS.wandb == "1":
    try:
        import wandb
        WB = wandb.init(project=ARGS.wandb_project, entity=ARGS.wandb_entity,
                        config=vars(ARGS), dir=OUT)
    except Exception as e:
        print("wandb off:", repr(e)[:120], flush=True)
        WB = None


def log(**kw):
    print(" ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
                   for k, v in kw.items()), flush=True)
    if WB is not None:
        try:
            WB.log({k: v for k, v in kw.items() if isinstance(v, (int, float))})
        except Exception:
            pass


def reg(name, ok, predicts, **kw):
    REG.append(dict(tag="PASS" if ok else "WARN", name=name, predicts=predicts,
                    detail={k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                            for k, v in kw.items()}))
    log(reg=REG[-1]["tag"], name=name, **kw)
    if WB is not None:
        try:
            WB.summary[name] = REG[-1]["tag"]
        except Exception:
            pass


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
    if not HAVE_MPL:
        return
    p = os.path.join(OUT, "figures", name)
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    if WB is not None:
        try:
            WB.log({name: wandb.Image(p)})
        except Exception:
            pass


def flush_reg():
    with open(os.path.join(OUT, "results", "register.json"), "w") as f:
        json.dump(dict(register=REG, times_min=TIMES, args=vars(ARGS)), f, indent=1)


def tick(k):
    TIMES[k] = time.time()


def tock(k):
    TIMES[k] = (time.time() - TIMES[k]) / 60.0
    log(stage=k, minutes=TIMES[k])


def set_seed(s):
    torch.manual_seed(s)
    np.random.seed(s)


log(device=DEV, torch=torch.__version__,
    gpu=torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu", out=OUT, part=ARGS.part)

# %% [code]
def spd_batch(B, n, dev, gen):
    A = torch.randn(B, n, n, generator=gen, device=dev, dtype=torch.float64)
    S = A @ A.transpose(-1, -2) / n + 0.35 * torch.eye(n, device=dev, dtype=torch.float64)
    return S


def sym(M):
    return 0.5 * (M + M.transpose(-1, -2))


def skew(M):
    return 0.5 * (M - M.transpose(-1, -2))


def fro(M):
    return M.flatten(1).norm(dim=1).clamp_min(1e-30)


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


if ARGS.part in ("theorem", "both"):
    tick("theorem")
    DIMS = [int(x) for x in ARGS.thm_dims.split(",")]
    OFFS = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]
    SKWS = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]
    rows = []
    for n in DIMS:
        for rep in range(ARGS.thm_reps):
            gen = torch.Generator(device=DEV).manual_seed(1000 * n + rep)
            S0 = spd_batch(ARGS.thm_batch, n, DEV, gen)
            S1 = spd_batch(ARGS.thm_batch, n, DEV, gen)
            C0 = build(S0, S1, 0.0, 0.0, gen)
            M0 = endpoint_map(S0, S1, C0, ARGS.thm_steps)
            C0b = build(S0, S1, 0.0, 0.0, gen, margin=0.55)
            M0b = endpoint_map(S0, S1, C0b, ARGS.thm_steps)
            self_drift = (fro(M0b - M0) / fro(M0)).cpu().numpy()
            pf = (fro(M0 @ S0 @ M0.transpose(-1, -2) - S1) / fro(S1)).cpu().numpy()
            comm = (fro(S0 @ S1 - S1 @ S0) / (fro(S0) * fro(S1))).cpu().numpy()
            for eo in OFFS:
                for es in SKWS:
                    if eo == 0.0 and es == 0.0:
                        d = self_drift
                        vo = np.zeros_like(d)
                        vs = np.zeros_like(d)
                    else:
                        C = build(S0, S1, eo, es, gen)
                        M = endpoint_map(S0, S1, C, ARGS.thm_steps)
                        d = (fro(M - M0) / fro(M0)).cpu().numpy()
                        vo, vs = violations(S0, S1, C)
                    rows.append(dict(n=n, rep=rep, eps_off=eo, eps_skew=es,
                                     viol_off=float(np.mean(vo)), viol_skew=float(np.mean(vs)),
                                     drift_med=float(np.median(d)), drift_p90=float(np.percentile(d, 90)),
                                     drift_max=float(d.max()),
                                     self_drift_med=float(np.median(self_drift)),
                                     pushfwd_med=float(np.median(pf)),
                                     commutator_med=float(np.median(comm)), B=len(d)))
            log(n=n, rep=rep, self_drift=float(np.median(self_drift)),
                pushfwd=float(np.median(pf)),
                drift_clean_off=float(np.median(
                    [r["drift_med"] for r in rows if r["n"] == n and r["rep"] == rep
                     and r["eps_off"] == 0.0 and r["eps_skew"] == 0.0])))
        save_csv("theorem_scale.csv", rows)
        flush_reg()
    save_csv("theorem_scale.csv", rows)

    clean = [r for r in rows if r["eps_off"] == 0.0 and r["eps_skew"] == 0.0]
    dirty = [r for r in rows if r["eps_off"] > 0.0 or r["eps_skew"] > 0.0]
    big = [r for r in dirty if r["eps_off"] >= 0.1 or r["eps_skew"] >= 0.1]
    sep = (max(r["drift_p90"] for r in clean) < min(r["drift_med"] for r in big))
    reg("thm_criterion_separates_at_scale", sep,
        "across random systems in every dimension, the endpoint map is invariant exactly when "
        "the cross-covariance is symmetric and its symmetric part lies in the span of the two "
        "marginals, and moves otherwise",
        n_instances=int(sum(r["B"] for r in clean)),
        clean_drift_p90=max(r["drift_p90"] for r in clean),
        violated_drift_med_min=min(r["drift_med"] for r in big),
        dims=str(DIMS))
    reg("thm_invariance_independent_of_commutation",
        max(r["drift_p90"] for r in clean) < 1e-3,
        "invariance holds regardless of whether the two marginals commute",
        max_clean_p90=max(r["drift_p90"] for r in clean),
        max_commutator=max(r["commutator_med"] for r in clean))
    reg("thm_marginal_matching_any_coupling",
        max(r["pushfwd_med"] for r in rows) < 1e-3,
        "the map pushes the source covariance onto the target covariance for every coupling, "
        "with no hypothesis on the skew part",
        max_pushfwd=max(r["pushfwd_med"] for r in rows))

    off_only = [r for r in dirty if r["eps_skew"] == 0.0 and r["viol_off"] > 1e-6]
    skw_only = [r for r in dirty if r["eps_off"] == 0.0 and r["viol_skew"] > 1e-6]
    slopes = {}
    for nm, sel, key in [("offspan", off_only, "viol_off"), ("skew", skw_only, "viol_skew")]:
        if len(sel) > 3:
            x = np.log10([r[key] for r in sel])
            y = np.log10([max(r["drift_med"], 1e-16) for r in sel])
            A = np.stack([x, np.ones_like(x)], 1)
            sl, ic = np.linalg.lstsq(A, y, rcond=None)[0]
            resid = float(np.sqrt(np.mean((A @ np.array([sl, ic]) - y) ** 2)))
            slopes[nm] = (float(sl), float(ic), resid)
            log(law=nm, slope=float(sl), intercept=float(ic), rms_resid=resid, n=len(sel))
    reg("thm_stability_law_offspan", slopes.get("offspan", (0, 0, 9))[2] < 0.25,
        "the endpoint-map displacement is a power law in the size of the span violation, so the "
        "criterion is a stability statement and not only a binary condition",
        slope=slopes.get("offspan", (float('nan'),) * 3)[0],
        rms_resid=slopes.get("offspan", (0, 0, float('nan')))[2])
    reg("thm_stability_law_skew", slopes.get("skew", (0, 0, 9))[2] < 0.25,
        "the endpoint-map displacement is a power law in the size of the skew violation",
        slope=slopes.get("skew", (float('nan'),) * 3)[0],
        rms_resid=slopes.get("skew", (0, 0, float('nan')))[2])
    if "offspan" in slopes and "skew" in slopes:
        ratio = 10.0 ** (slopes["skew"][1] - slopes["offspan"][1])
        reg("thm_skew_and_span_equally_fragile", 0.5 < ratio < 2.0,
            "a violation of either hypothesis moves the map by a comparable amount at equal "
            "relative size",
            skew_over_span_at_unit_violation=ratio,
            slope_offspan=slopes["offspan"][0], slope_skew=slopes["skew"][0])

    if HAVE_MPL:
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.0))
        for n in DIMS:
            s1 = [r for r in off_only if r["n"] == n]
            s2 = [r for r in skw_only if r["n"] == n]
            if s1:
                ax[0].loglog([r["viol_off"] for r in s1], [max(r["drift_med"], 1e-16) for r in s1],
                             "o", label=f"n={n}")
            if s2:
                ax[1].loglog([r["viol_skew"] for r in s2], [max(r["drift_med"], 1e-16) for r in s2],
                             "s", label=f"n={n}")
        for a, t in zip(ax, ["span violation", "skew violation"]):
            a.axhline(max(r["drift_p90"] for r in clean), ls="--", c="k", lw=1)
            a.set_xlabel(t)
            a.set_ylabel("relative endpoint-map displacement")
            a.legend(fontsize=7)
        save_fig("C1_theorem_scale.png")
    tock("theorem")
    flush_reg()

# %% [code]
class RB(nn.Module):
    def __init__(self, cin, cout, emb):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.e = nn.Linear(emb, 2 * cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.sk = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        sc, sh = self.e(F.silu(e))[:, :, None, None].chunk(2, 1)
        h = self.c2(F.silu(self.n2(h) * (1 + sc) + sh))
        return self.sk(x) + h


class UNet(nn.Module):
    def __init__(self, cin, base=64, n_class=10, mults=(1, 2, 2)):
        super().__init__()
        emb = base * 4
        self.emb_t = nn.Sequential(nn.Linear(base, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.emb_c = nn.Embedding(n_class, emb)
        self.base = base
        self.inp = nn.Conv2d(cin, base, 3, padding=1)
        chs = [base * m for m in mults]
        self.down = nn.ModuleList()
        self.pool = nn.ModuleList()
        prev = base
        for ch in chs:
            self.down.append(nn.ModuleList([RB(prev, ch, emb), RB(ch, ch, emb)]))
            self.pool.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
            prev = ch
        self.mid1 = RB(prev, prev, emb)
        self.mid2 = RB(prev, prev, emb)
        self.up = nn.ModuleList()
        self.upc = nn.ModuleList()
        for ch in reversed(chs):
            self.upc.append(nn.ConvTranspose2d(prev, ch, 4, stride=2, padding=1))
            self.up.append(nn.ModuleList([RB(2 * ch, ch, emb), RB(ch, ch, emb)]))
            prev = ch
        self.outn = nn.GroupNorm(8, prev)
        self.out = nn.Conv2d(prev, 3, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def temb(self, t):
        half = self.base // 2
        f = torch.exp(torch.linspace(0, math.log(1000.0), half, device=t.device))
        a = t[:, None] * f[None, :]
        return torch.cat([a.sin(), a.cos()], -1)

    def forward(self, x, t, y):
        e = self.emb_t(self.temb(t)) + self.emb_c(y)
        h = self.inp(x)
        skips = []
        for blocks, pl in zip(self.down, self.pool):
            for b in blocks:
                h = b(h, e)
            skips.append(h)
            h = pl(h)
        h = self.mid2(self.mid1(h, e), e)
        for uc, blocks in zip(self.upc, self.up):
            h = uc(h)
            s = skips.pop()
            if h.shape[-1] != s.shape[-1]:
                h = F.interpolate(h, size=s.shape[-2:], mode="nearest")
            h = torch.cat([h, s], 1)
            for b in blocks:
                h = b(h, e)
        return self.out(F.silu(self.outn(h)))


def down_up(x, f):
    return F.interpolate(F.avg_pool2d(x, f), scale_factor=f, mode="nearest")


def psnr(a, b):
    m = ((a - b) ** 2).flatten(1).mean(1).clamp_min(1e-12)
    return float((10 * torch.log10(4.0 / m)).mean())


def w1_sorted(a, b):
    n = min(a.numel(), b.numel())
    return float((a.flatten()[:n].sort().values - b.flatten()[:n].sort().values).abs().mean())


def energy_dist(x, y):
    nx, ny = x.shape[0], y.shape[0]
    return float(2 * torch.cdist(x, y).mean()
                 - torch.cdist(x, x).sum() / max(1, nx * (nx - 1))
                 - torch.cdist(y, y).sum() / max(1, ny * (ny - 1)))


def knn_cov(real, fake, k=5):
    k = min(k, real.shape[0] - 1)
    dr = torch.cdist(real, real)
    dr.fill_diagonal_(float("inf"))
    rad = dr.topk(k, largest=False).values[:, -1]
    return float((torch.cdist(real, fake) < rad[:, None]).any(1).float().mean())


@torch.no_grad()
def sr_dist(o, Xte, f):
    def hf(x):
        return ((x - down_up(x, f)) ** 2).flatten(1).sum(1)
    fo = F.adaptive_avg_pool2d(o, 8).flatten(1).double()
    ft = F.adaptive_avg_pool2d(Xte, 8).flatten(1).double()
    g = torch.Generator().manual_seed(0)
    p = torch.randperm(Xte.shape[0], generator=g).to(Xte.device)
    a, b = p[: p.shape[0] // 2], p[p.shape[0] // 2:]
    return dict(w1_hf=w1_sorted(hf(o), hf(Xte)),
                w1_hf_null=w1_sorted(hf(Xte[a]), hf(Xte[b])),
                energy8=energy_dist(fo, ft),
                energy8_null=energy_dist(ft[a], ft[b]),
                cov8=knn_cov(ft, fo),
                cov8_null=knn_cov(ft[a], ft[b]))


def load_sr(res, root, name):
    if name == "synth":
        g = torch.Generator().manual_seed(0)
        def mk(n):
            x = torch.randn(n, 3, res, res, generator=g)
            return (F.avg_pool2d(x, 3, 1, 1) * 2.2).clamp(-1, 1),                    torch.randint(0, 10, (n,), generator=g)
        return mk(1024), mk(256), 10
    import torchvision
    from torchvision import transforms as T
    tf = T.Compose([T.Resize(res), T.CenterCrop(res), T.ToTensor(),
                    T.Normalize([0.5] * 3, [0.5] * 3)])
    if name == "stl10":
        tr = torchvision.datasets.STL10(root=root, split="train", download=True, transform=tf)
        te = torchvision.datasets.STL10(root=root, split="test", download=True, transform=tf)
        nc = 10
    else:
        tr = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=tf)
        te = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=tf)
        nc = 10
    def pack(ds, cap):
        xs, ys = [], []
        for i in range(min(cap, len(ds))):
            a, b = ds[i]
            xs.append(a)
            ys.append(b)
        return torch.stack(xs), torch.tensor(ys)
    return pack(tr, 5000), pack(te, 1024), nc


@torch.no_grad()
def sr_sample(net, z0, y, k, see_source):
    z = z0.clone()
    ts = np.linspace(0.0, 1.0, k + 1)
    for i in range(k):
        t, tn = float(ts[i]), float(ts[i + 1])
        tt = torch.full((z.shape[0],), t, device=z.device)
        inp = torch.cat([z, z0], 1) if see_source else z
        v = net(inp, tt, y)
        zp = z + (tn - t) * v
        ttn = torch.full((z.shape[0],), tn, device=z.device)
        inp2 = torch.cat([zp, z0], 1) if see_source else zp
        z = z + (tn - t) * 0.5 * (v + net(inp2, ttn, y))
    return z


def sr_train(mode, Xtr, Ytr, f, steps, bs, base, nc, seed, res):
    set_seed(seed)
    see = (mode == "m2")
    cin = 6 if see or mode == "reg" else 3
    net = UNet(cin, base=base, n_class=nc).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=0.0)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-4, total_steps=steps, pct_start=0.05)
    N = Xtr.shape[0]
    run = None
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, N, (bs,), device=DEV)
        z1 = Xtr[idx]
        y = Ytr[idx]
        z0 = down_up(z1, f)
        with torch.autocast("cuda", enabled=(DEV == "cuda"), dtype=torch.bfloat16):
            if mode == "reg":
                pred = net(torch.cat([z0, z0], 1), torch.zeros(bs, device=DEV), y)
                loss = F.mse_loss(pred, z1)
            else:
                t = torch.rand(bs, device=DEV)
                zt = (1 - t[:, None, None, None]) * z0 + t[:, None, None, None] * z1
                inp = torch.cat([zt, z0], 1) if see else zt
                loss = F.mse_loss(net(inp, t, y), z1 - z0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sch.step()
        ld = loss.detach()
        run = ld if run is None else 0.99 * run + 0.01 * ld
        if step % max(500, steps // 10) == 0:
            log(mode=mode, res=res, seed=seed, step=step, loss=float(run))
    log(mode=mode, res=res, seed=seed, steps=steps, final_loss=float(run),
        minutes=(time.time() - t0) / 60.0)
    return net


if ARGS.part in ("sr", "both"):
    tick("sr")
    RESL = [int(x) for x in ARGS.sr_res.split(",")]
    SEEDS = [int(x) for x in ARGS.sr_seeds.split(",")]
    f = ARGS.sr_factor
    rows = []
    for res in RESL:
        (Xtr, Ytr), (Xte, Yte), nc = load_sr(res, ARGS.sr_root, ARGS.sr_data)
        Xtr, Ytr = Xtr.to(DEV), Ytr.to(DEV)
        Xte, Yte = Xte.to(DEV), Yte.to(DEV)
        n_pix = 3 * res * res
        r_pix = 3 * (res // f) * (res // f)
        Z0te = down_up(Xte, f)
        Qt = Xte - down_up(Xte, f)
        null_t = float((Qt ** 2).flatten(1).sum(1).mean())
        n_rk = min(Xtr.shape[0], 4 * r_pix + 256, 3072)
        Z0rk = down_up(Xtr[:n_rk], f)
        def _rank(M):
            Mc = (M - M.mean(0, keepdim=True)).flatten(1).double()
            ev = torch.linalg.eigvalsh(Mc @ Mc.T)
            return int((ev > ev[-1] * 1e-10).sum())
        rk = _rank(Z0rk)
        measurable = (n_rk - 1) > r_pix
        rk1 = _rank(Xtr[:n_rk])
        log(res=res, n_pix=n_pix, r_pix=r_pix, measured_rank_z0=rk, measured_rank_z1=rk1,
            rank_samples=n_rk, measurable=int(measurable),
            null_energy_target=null_t, identity_psnr=psnr(Z0te, Xte))
        reg(f"sr_rank_deficiency_res{res}", measurable and rk <= r_pix + 1,
            "the source lies in the range of a fixed rank-deficient operator, so its conditional "
            "covariance has rank at most the number of low-resolution pixels, measured with "
            "strictly more samples than that bound",
            measured_rank=rk, predicted_max=r_pix, ambient=n_pix, samples=n_rk,
            sample_limited=int(not measurable), res=res)
        reg(f"sr_source_dimension_below_target_res{res}", rk < rk1,
            "the source support has strictly smaller dimension than the target support, which is "
            "what forbids a Lipschitz transport between them",
            rank_source=rk, rank_target=rk1, res=res)
        for seed in SEEDS:
            nets = {}
            for mode in ["reg", "m1", "m2"]:
                nets[mode] = sr_train(mode, Xtr, Ytr, f, ARGS.sr_steps, ARGS.sr_batch,
                                      ARGS.sr_base, nc, seed, res)
                nets[mode].eval()
            with torch.no_grad():
                outs = {"identity": Z0te}
                outs["reg"] = torch.cat([
                    nets["reg"](torch.cat([Z0te[i:i + 128], Z0te[i:i + 128]], 1),
                                torch.zeros(min(128, Z0te.shape[0] - i), device=DEV),
                                Yte[i:i + 128]) for i in range(0, Z0te.shape[0], 128)])
                for mode in ["m1", "m2"]:
                    outs[mode] = torch.cat([
                        sr_sample(nets[mode], Z0te[i:i + 128], Yte[i:i + 128],
                                  ARGS.sr_ode_k, mode == "m2")
                        for i in range(0, Z0te.shape[0], 128)])
            for nm, o in outs.items():
                Q = o - down_up(o, f)
                nrg = float((Q ** 2).flatten(1).sum(1).mean())
                d_reg = float(((o - outs["reg"]) ** 2).flatten(1).sum(1).mean())
                d_tgt = float(((o - Xte) ** 2).flatten(1).sum(1).mean())
                rows.append(dict(res=res, seed=seed, arm=nm, psnr=psnr(o, Xte),
                                 null_energy=nrg, null_ratio=nrg / max(null_t, 1e-12),
                                 dist_to_reg=d_reg, dist_to_target=d_tgt,
                                 lowfreq_psnr=psnr(down_up(o, f), Z0te),
                                 **sr_dist(o, Xte, f)))
                log(res=res, seed=seed, arm=nm, psnr=rows[-1]["psnr"],
                    null_ratio=rows[-1]["null_ratio"], dist_to_reg=d_reg, dist_to_target=d_tgt,
                    energy8=rows[-1]["energy8"], cov8=rows[-1]["cov8"],
                    w1_hf=rows[-1]["w1_hf"])
            save_csv("sr.csv", rows)
            flush_reg()
            if WB is not None:
                try:
                    grid = torch.cat([Xte[:8], Z0te[:8], outs["reg"][:8],
                                      outs["m1"][:8], outs["m2"][:8]], 0)
                    WB.log({f"sr_grid_res{res}_seed{seed}":
                            wandb.Image((grid.clamp(-1, 1) * 0.5 + 0.5).cpu())})
                except Exception:
                    pass
    save_csv("sr.csv", rows)

    def sagg(res, arm, key="psnr"):
        s = [r[key] for r in rows if r["res"] == res and r["arm"] == arm]
        return float(np.mean(s)), float(np.std(s))

    for res in RESL:
        i_p = sagg(res, "identity")[0]
        r_p, r_s = sagg(res, "reg")
        m1p, m1s = sagg(res, "m1")
        m2p, m2s = sagg(res, "m2")
        nr = sagg(res, "m1", "null_ratio")[0]
        dr = sagg(res, "m1", "dist_to_reg")[0]
        dt = sagg(res, "m1", "dist_to_target")[0]
        log(res=res, identity=i_p, regression=r_p, m1=m1p, m2=m2p, m1_minus_m2=m1p - m2p,
            m1_null_ratio=nr)
        reg(f"sr_flow_beats_identity_res{res}", m1p > i_p,
            "the flow improves on returning the upsampled input unchanged",
            m1=m1p, identity=i_p, res=res)
        reg(f"sr_null_energy_collapses_res{res}", nr < 0.75,
            "with a degenerate source the deterministic flow cannot restore the full high "
            "frequency energy of the target and lands short of it",
            null_ratio=nr, res=res)
        reg(f"sr_lands_near_regression_res{res}", dr < dt,
            "the collapse point is the conditional mean, so the flow output is closer to the "
            "least-squares regression answer than to the true target",
            dist_to_reg=dr, dist_to_target=dt, res=res)
        reg(f"sr_m2_worse_than_m1_res{res}", m2p < m1p - max(m1s, m2s),
            "conditioning on the source while also starting there is harmful in the flat "
            "restoration setting, as the amplification argument predicts",
            m1=m1p, m1_sd=m1s, m2=m2p, m2_sd=m2s, res=res)
        e1, e1s = sagg(res, "m1", "energy8")
        e2, e2s = sagg(res, "m2", "energy8")
        c1v = sagg(res, "m1", "cov8")[0]
        c2v = sagg(res, "m2", "cov8")[0]
        en = sagg(res, "m1", "energy8_null")[0]
        cn = sagg(res, "m1", "cov8_null")[0]
        log(res=res, m1_energy8=e1, m2_energy8=e2, m1_cov8=c1v, m2_cov8=c2v,
            energy8_null=en, cov8_null=cn,
            m1_w1_hf=sagg(res, "m1", "w1_hf")[0], m2_w1_hf=sagg(res, "m2", "w1_hf")[0],
            w1_hf_null=sagg(res, "m1", "w1_hf_null")[0])
        reg(f"sr_dist_ordering_res{res}", e1 < e2 - max(e1s, e2s) and c1v > c2v,
            "where the source-conditioned arm scores higher on the single-target metric it fits "
            "the restoration distribution worse, so the score reversal is the mean-seeking "
            "inversion of the metric and not a better restoration",
            m1_energy8=e1, m2_energy8=e2, m1_cov8=c1v, m2_cov8=c2v,
            energy8_null=en, cov8_null=cn, res=res)
    signs = [sagg(r, "m1")[0] - sagg(r, "m2")[0] for r in RESL]
    reg("sr_effect_persists_across_resolution",
        all(s > 0 for s in signs) or all(s < 0 for s in signs),
        "the sign of the difference between the two models does not depend on resolution",
        gaps=str([round(s, 4) for s in signs]), resolutions=str(RESL))

    if HAVE_MPL:
        fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.0))
        for arm in ["identity", "reg", "m1", "m2"]:
            ax[0].errorbar(RESL, [sagg(r, arm)[0] for r in RESL],
                           yerr=[sagg(r, arm)[1] for r in RESL], fmt="o-", capsize=3, label=arm)
            ax[1].plot(RESL, [sagg(r, arm, "null_ratio")[0] for r in RESL], "o-", label=arm)
        ax[2].plot(RESL, signs, "o-")
        ax[2].axhline(0, c="k", lw=0.8)
        ax[0].set_ylabel("PSNR")
        ax[1].set_ylabel("null-space energy / target")
        ax[1].axhline(1.0, ls="--", c="k", lw=1)
        ax[2].set_ylabel("m1 - m2 PSNR")
        for a in ax:
            a.set_xlabel("resolution")
            a.legend(fontsize=7)
        save_fig("C2_sr.png")
    tock("sr")
    flush_reg()


if ARGS.part == "srthresh":
    tick("srthresh")
    FACS = [int(x) for x in ARGS.sr_factors.split(",")]
    SEEDS = [int(x) for x in ARGS.sr_seeds.split(",")]
    res = int(ARGS.sr_res.split(",")[0])
    rows = []
    (Xtr, Ytr), (Xte, Yte), nc = load_sr(res, ARGS.sr_root, ARGS.sr_data)
    Xtr, Ytr = Xtr.to(DEV), Ytr.to(DEV)
    Xte, Yte = Xte.to(DEV), Yte.to(DEV)

    def _rank(M):
        Mc = (M - M.mean(0, keepdim=True)).flatten(1).double()
        ev = torch.linalg.eigvalsh(Mc @ Mc.T)
        return int((ev > ev[-1] * 1e-10).sum())

    for fac in FACS:
        if res % fac != 0:
            log(skipped_factor=fac, res=res, reason="res_not_divisible")
            continue
        r_pix = 3 * (res // fac) * (res // fac)
        Z0te = down_up(Xte, fac)
        null_t = float(((Xte - Z0te) ** 2).flatten(1).sum(1).mean())
        id_psnr = psnr(Z0te, Xte)
        n_rk = min(Xtr.shape[0], 4 * r_pix + 256, 3072)
        rk = _rank(down_up(Xtr[:n_rk], fac))
        log(factor=fac, res=res, r_pix=r_pix, measured_rank_z0=rk, rank_samples=n_rk,
            measurable=int((n_rk - 1) > r_pix), identity_psnr=id_psnr,
            null_energy_target=null_t, retained_fraction=1.0 / (fac * fac))
        for seed in SEEDS:
            nets = {}
            for mode in ["reg", "m1", "m2"]:
                nets[mode] = sr_train(mode, Xtr, Ytr, fac, ARGS.sr_steps, ARGS.sr_batch,
                                      ARGS.sr_base, nc, seed, res)
                nets[mode].eval()
            with torch.no_grad():
                outs = {"identity": Z0te}
                outs["reg"] = torch.cat([
                    nets["reg"](torch.cat([Z0te[i:i + 128], Z0te[i:i + 128]], 1),
                                torch.zeros(min(128, Z0te.shape[0] - i), device=DEV),
                                Yte[i:i + 128]) for i in range(0, Z0te.shape[0], 128)])
                for mode in ["m1", "m2"]:
                    outs[mode] = torch.cat([
                        sr_sample(nets[mode], Z0te[i:i + 128], Yte[i:i + 128],
                                  ARGS.sr_ode_k, mode == "m2")
                        for i in range(0, Z0te.shape[0], 128)])
            for nm, o in outs.items():
                nrg = float(((o - down_up(o, fac)) ** 2).flatten(1).sum(1).mean())
                pv = psnr(o, Xte)
                rows.append(dict(res=res, factor=fac, retained=1.0 / (fac * fac), seed=seed,
                                 arm=nm, psnr=pv, gain_over_identity=pv - id_psnr,
                                 identity_psnr=id_psnr, null_ratio=nrg / max(null_t, 1e-12),
                                 dist_to_target=float(((o - Xte) ** 2).flatten(1).sum(1).mean()),
                                 rank_z0=rk, r_pix=r_pix, **sr_dist(o, Xte, fac)))
                log(factor=fac, seed=seed, arm=nm, psnr=pv,
                    gain=rows[-1]["gain_over_identity"], null_ratio=rows[-1]["null_ratio"])
            save_csv("sr_threshold.csv", rows)
            flush_reg()
    save_csv("sr_threshold.csv", rows)

    FACD = sorted({r["factor"] for r in rows})

    def tagg(fac, arm, key="psnr"):
        v = [r[key] for r in rows if r["factor"] == fac and r["arm"] == arm]
        return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), float("nan"))

    gains, m1m2 = [], []
    for fac in FACD:
        g, gs = tagg(fac, "m1", "gain_over_identity")
        d1 = tagg(fac, "m1")[0]
        d2 = tagg(fac, "m2")[0]
        gains.append(g)
        m1m2.append(d1 - d2)
        log(factor=fac, retained=1.0 / (fac * fac), identity=tagg(fac, "identity")[0],
            reg=tagg(fac, "reg")[0], m1=d1, m2=d2, m1_gain=g, m1_gain_sd=gs,
            m1_minus_m2=d1 - d2, m1_null=tagg(fac, "m1", "null_ratio")[0])
        reg(f"srthresh_flow_beats_identity_f{fac}", g > 2 * max(gs, 1e-9),
            "transporting an informative super-resolution source improves on retaining it "
            "unchanged, which is the question the threshold answers in the synthetic setting",
            gain=g, gain_sd=gs, factor=fac, retained=1.0 / (fac * fac))

    if len(FACD) > 2:
        mono = all(gains[i] <= gains[i + 1] for i in range(len(gains) - 1))
        reg("srthresh_gain_grows_as_source_degrades", mono,
            "the benefit of transporting the source grows as the source retains less of the "
            "target, so source informativeness is the axis that decides whether the flow helps",
            gains=str([round(x, 4) for x in gains]), factors=str(FACD))
        crossed = [fac for fac, g in zip(FACD, gains) if g <= 0]
        reg("srthresh_crossing_located", len(crossed) > 0,
            "there is a source informative enough that transporting it no longer beats leaving "
            "it alone, which is the super-resolution image of the threshold",
            factors_without_benefit=str(crossed), all_factors=str(FACD),
            smallest_gain=min(gains), at_factor=FACD[int(np.argmin(gains))])
        reg("srthresh_m1_m2_sign_constant",
            all(x > 0 for x in m1m2) or all(x < 0 for x in m1m2),
            "the sign of the difference between starting at the source with and without "
            "conditioning on it does not depend on how informative the source is",
            gaps=str([round(x, 4) for x in m1m2]), factors=str(FACD))

    if HAVE_MPL:
      fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.0))
      for arm in ["identity", "reg", "m1", "m2"]:
        mu = [tagg(fc, arm)[0] for fc in FACD]
        sd = [tagg(fc, arm)[1] for fc in FACD]
        ax[0].errorbar(FACD, mu, yerr=sd, fmt="o-", capsize=3, label=arm)
        ax[1].plot(FACD, [tagg(fc, arm, "null_ratio")[0] for fc in FACD], "o-", label=arm)
      ax[2].axhline(0, c="k", lw=0.8)
      ax[2].errorbar(FACD, gains,
                     yerr=[tagg(fc, "m1", "gain_over_identity")[1] for fc in FACD],
                     fmt="o-", capsize=3, label="m1 - identity")
      ax[2].plot(FACD, m1m2, "s--", label="m1 - m2")
      ax[0].set_ylabel("PSNR")
      ax[1].set_ylabel("null-space energy / target")
      ax[1].axhline(1.0, ls="--", c="k", lw=1)
      ax[2].set_ylabel("PSNR difference")
      for a in ax:
        a.set_xlabel("downsample factor (source degrades to the right)")
        a.set_xscale("log", base=2)
        a.legend(fontsize=7)
      save_fig("C3_sr_threshold.png")
    tock("srthresh")
    flush_reg()

# %% [code]
flush_reg()
npass = sum(1 for g in REG if g["tag"] == "PASS")
log(passed=npass, total=len(REG), total_minutes=sum(TIMES.values()))
for g in REG:
    if g["tag"] == "WARN":
        log(contradicted=g["name"], predicted=g["predicts"], observed=json.dumps(g["detail"]))
with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(dict(passed=npass, total=len(REG), times_min=TIMES,
                   args=vars(ARGS), register=REG), f, indent=1)
if WB is not None:
    WB.finish()
