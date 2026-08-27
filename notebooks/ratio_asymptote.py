# %% [code]
import os, csv, json, math, time
import numpy as np
import torch

torch.set_default_dtype(torch.float64)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "/kaggle/working/ratio_asym" if os.path.isdir("/kaggle/working") else "runs/ratio_asym"
os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
REG = []


def log(**kw):
    print(" ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
                   for k, v in kw.items()), flush=True)


def reg(name, ok, predicts, **kw):
    REG.append(dict(tag="PASS" if ok else "WARN", name=name, predicts=predicts,
                    detail={k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                            for k, v in kw.items()}))
    log(reg=REG[-1]["tag"], name=name, **kw)


def save_csv(name, rows):
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


def sym(M):
    return 0.5 * (M + M.transpose(-1, -2))


def skew(M):
    return 0.5 * (M - M.transpose(-1, -2))


def fro(M):
    return M.flatten(1).norm(dim=1).clamp_min(1e-30)


def spd_batch(B, n, dev, gen):
    A = torch.randn(B, n, n, generator=gen, device=dev, dtype=torch.float64)
    return A @ A.transpose(-1, -2) / n + 0.35 * torch.eye(n, device=dev, dtype=torch.float64)


def msqrt_inv(S):
    w, Q = torch.linalg.eigh(S)
    return Q @ torch.diag_embed(w.clamp_min(1e-12).rsqrt()) @ Q.transpose(-1, -2)


def project_offspan(R, S0, S1):
    B0 = S0 / fro(S0)[:, None, None]
    p1 = S1 - (S1 * B0).flatten(1).sum(1)[:, None, None] * B0
    B1 = p1 / fro(p1)[:, None, None]
    R = R - (R * B0).flatten(1).sum(1)[:, None, None] * B0
    return R - (R * B1).flatten(1).sum(1)[:, None, None] * B1


def rescale_psd(S0, S1, C, margin=0.92):
    Ki = msqrt_inv(S0) @ C @ msqrt_inv(S1)
    smax = torch.linalg.matrix_norm(Ki, ord=2).clamp_min(1e-30)
    return C * torch.clamp(margin / smax, max=1.0)[:, None, None]


def endpoint_map(S0, S1, C, steps):
    B, n, _ = S0.shape
    M = torch.eye(n, device=S0.device, dtype=S0.dtype).expand(B, n, n).clone()
    S = C + C.transpose(-1, -2)
    Ct = C.transpose(-1, -2)
    ts = np.linspace(0.0, 1.0, steps + 1)

    def Kof(t):
        V = (1 - t) ** 2 * S0 + t * (1 - t) * S + t ** 2 * S1
        G = (1 - t) * (Ct - S0) + t * (S1 - C)
        return torch.linalg.solve(V.transpose(-1, -2), G.transpose(-1, -2)).transpose(-1, -2)

    Ka = Kof(0.0)
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
    R = project_offspan(sym(torch.randn(B, n, n, generator=gen, device=dev, dtype=dt)), S0, S1)
    R = R / fro(R)[:, None, None] * fro(S_clean)[:, None, None]
    W = skew(torch.randn(B, n, n, generator=gen, device=dev, dtype=dt))
    W = W / fro(W)[:, None, None] * fro(S_clean)[:, None, None]
    C = 0.5 * (S_clean + eps_off * R) + eps_skew * W
    return rescale_psd(S0, S1, C, margin)


def violations(S0, S1, C):
    S = C + C.transpose(-1, -2)
    Sp = project_offspan(S, S0, S1)
    return (fro(Sp) / fro(S)).cpu().numpy(), (fro(skew(C)) / fro(C)).cpu().numpy()


def fit(x, y):
    lx, ly = np.log10(x), np.log10(y)
    Amat = np.stack([lx, np.ones_like(lx)], 1)
    sl, ic = np.linalg.lstsq(Amat, ly, rcond=None)[0]
    resid = float(np.sqrt(np.mean((Amat @ np.array([sl, ic]) - ly) ** 2)))
    return float(sl), float(ic), resid


# %% [code]
# The claim under test: as the two marginals become equal the span response vanishes
# linearly in the separation while the skew response stays O(1), so the ratio scales
# as 1/delta and the log-log slope tends to -1. The published sweep covers
# delta in [0.05, 1] and measures -0.756; this extends it downward by a decade.
n, B, STEPS, EPSV = 16, 64, 1500, 0.05
DELTAS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]

gen0 = torch.Generator(device=DEVICE)
gen0.manual_seed(0)
S0 = spd_batch(B, n, DEVICE, gen0)
S1full = spd_batch(B, n, DEVICE, gen0)

rows = []
t0 = time.time()
for d in DELTAS:
    S1 = S0 + d * (S1full - S0)
    resp = {}
    for fam, eo, es in [("off", EPSV, 0.0), ("skw", 0.0, EPSV)]:
        g = torch.Generator(device=DEVICE)
        g.manual_seed(1234)
        C_cl = build(S0, S1, 0.0, 0.0, g)
        M0 = endpoint_map(S0, S1, C_cl, STEPS)
        g = torch.Generator(device=DEVICE)
        g.manual_seed(1234)
        C = build(S0, S1, eo, es, g)
        M = endpoint_map(S0, S1, C, STEPS)
        drift = (fro(M - M0) / fro(M0)).cpu().numpy()
        vo, vs = violations(S0, S1, C)
        viol = vo if fam == "off" else vs
        resp[fam] = float(np.median(drift / np.clip(viol, 1e-12, None)))
    # how far apart the marginals actually are, in the same relative Frobenius units
    sep = float(np.median((fro(S1 - S0) / fro(S0)).cpu().numpy()))
    rows.append(dict(delta=d, marginal_separation=sep, resp_span=resp["off"],
                     resp_skew=resp["skw"], ratio=resp["skw"] / resp["off"]))
    log(delta=d, separation=sep, span=resp["off"], skew=resp["skw"],
        ratio=resp["skw"] / resp["off"])
save_csv("asymptote.csv", rows)
log(minutes=(time.time() - t0) / 60.0)

# %% [code]
dl = np.array([r["delta"] for r in rows])
span = np.array([r["resp_span"] for r in rows])
skw = np.array([r["resp_skew"] for r in rows])
rat = np.array([r["ratio"] for r in rows])

pub = dl >= 0.05          # the window reported in the note
tail = dl <= 0.05         # the decade added here

for nm, mask in [("published_window", pub), ("small_delta_tail", tail), ("all", dl > 0)]:
    s_span = fit(dl[mask], span[mask])
    s_skew = fit(dl[mask], skw[mask])
    s_rat = fit(dl[mask], rat[mask])
    log(window=nm, n=int(mask.sum()), slope_span=s_span[0], slope_skew=s_skew[0],
        slope_ratio=s_rat[0], resid_ratio=s_rat[2],
        identity_check=s_skew[0] - s_span[0] - s_rat[0])

s_span_p, s_skew_p, s_rat_p = fit(dl[pub], span[pub]), fit(dl[pub], skw[pub]), fit(dl[pub], rat[pub])
s_span_t, s_skew_t, s_rat_t = fit(dl[tail], span[tail]), fit(dl[tail], skw[tail]), fit(dl[tail], rat[tail])

reg("asym_ratio_slope_is_the_difference_of_the_two",
    abs((s_skew_p[0] - s_span_p[0]) - s_rat_p[0]) < 1e-9,
    "the ratio slope is exactly the skew slope minus the span slope, so a ratio slope milder "
    "than -1 must come from the span response being sublinear, not from the skew constant",
    slope_span=s_span_p[0], slope_skew=s_skew_p[0], slope_ratio=s_rat_p[0])

reg("asym_published_window_span_is_sublinear", 0.4 < s_span_p[0] < 0.9,
    "over the window reported in the note the span response scales as delta to a power well "
    "below one, which is why the measured ratio slope is -0.76 and not -1",
    span_exponent=s_span_p[0])

reg("asym_span_approaches_linear_as_delta_shrinks", s_span_t[0] > s_span_p[0] + 0.1,
    "extending the sweep by a decade below the published window, the span exponent moves "
    "toward one, confirming that the milder measured slope is a pre-asymptotic effect and not "
    "a contradiction of the first-order prediction",
    span_exponent_published=s_span_p[0], span_exponent_tail=s_span_t[0])

reg("asym_skew_response_is_order_one", abs(s_skew_t[0]) < 0.25,
    "the skew response tends to a finite non-zero limit as the marginals merge, as the "
    "homogeneous-marginal proposition requires",
    skew_exponent_tail=s_skew_t[0], skew_at_smallest_delta=float(skw[0]))

reg("asym_ratio_slope_approaches_minus_one", s_rat_t[0] < s_rat_p[0] - 0.1,
    "the ratio slope steepens toward -1 as the marginals merge, so -1 is the asymptotic "
    "statement and -0.756 is the value over the tested window",
    slope_published=s_rat_p[0], slope_tail=s_rat_t[0])

flush_reg()
log(passed=sum(1 for g in REG if g["tag"] == "PASS"), total=len(REG), out=OUT)
