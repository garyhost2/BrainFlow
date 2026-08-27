# %% [code]
import os, csv, math, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D_OP, A_OP, BETA_OP, J_OP = 256, 0.37, 0.8, 0.15
CAND = ["/kaggle/input", "/kaggle/working", "runs", "."]
OUT = "/kaggle/working/figpack" if os.path.isdir("/kaggle/working") else "runs/figpack"
os.makedirs(OUT, exist_ok=True)
ANSWERS = {}


def find(name):
    hits = []
    for c in CAND:
        if os.path.isdir(c):
            hits += glob.glob(os.path.join(c, "**", name), recursive=True)
    hits = [h for h in hits if os.path.getsize(h) > 0 and os.path.abspath(OUT) not in os.path.abspath(h)]
    return sorted(hits, key=os.path.getmtime)[-1] if hits else None


def load(name):
    p = find(name)
    if p is None:
        print(f"MISSING {name} - the block that needs it is skipped", flush=True)
        return None
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        for k, v in list(r.items()):
            if v is None or v == "":
                r[k] = float("nan")
                continue
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    print(f"loaded {name} <- {p} rows={len(rows)}", flush=True)
    return rows


def save(name):
    plt.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(OUT, name + ".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"wrote {name}.pdf {name}.png -> {OUT}", flush=True)


def spearman(a, b):
    def rk(v):
        return np.argsort(np.argsort(np.asarray(v, float))).astype(float)
    ra, rb = rk(a), rk(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = math.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else 0.0


def ladder_src(A, beta, j):
    s = math.sqrt(max(0.0, 1 - beta * beta))
    return A * (s * math.cos(j) + beta * math.sin(j))


def law_markov(A, beta):
    s2 = max(0.0, 1 - beta * beta)
    return A * (A * s2 + beta * math.sqrt(max(0.0, 1 - A * A * s2)))


def beta_star(A, j):
    if A >= math.cos(j):
        return float("nan")
    f = lambda b: law_markov(A, b) - ladder_src(A, b, j)
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


def crossing(xs, ys):
    pts = [(x, y) for x, y in zip(xs, ys)
           if isinstance(x, (int, float)) and isinstance(y, (int, float))
           and np.isfinite(x) and np.isfinite(y)]
    for i in range(1, len(pts)):
        (x0, y0), (x1, y1) = pts[i - 1], pts[i]
        if y0 <= 0 < y1:
            return x0 + (x1 - x0) * (-y0) / (y1 - y0)
    return float("nan")


bs = load("beta_star.csv")
lr2 = load("lr2_scaling.csv") or []
lr3 = load("lr3_scaling.csv") or []
gd = load("guidance.csv")
fx = load("fixes.csv")
mo = load("metric_ordering.csv")

cap = [r for r in lr2 if r.get("arm") in ("c1", "c2")]
cap += [r for r in lr3 if r.get("arm") in ("c1", "c2")
        and not any(x["lr"] == r["lr"] and x["width"] == r["width"] and x["arm"] == r["arm"]
                    and x["seed"] == r["seed"] for x in cap)]
LRS = sorted({r["lr"] for r in cap})
WIDTHS = sorted({int(r["width"]) for r in cap})


def cagg(lr, arm, w, key="fid"):
    v = [r[key] for r in cap if r["lr"] == lr and r["arm"] == arm and r["width"] == w]
    return (float(np.mean(v)), float(np.std(v)) / math.sqrt(max(len(v), 1)), len(v)) if v \
        else (float("nan"), float("nan"), 0)


# %% [code]
if bs and gd and cap:
    fig, ax = plt.subplots(1, 3, figsize=(13.2, 3.5))

    op = sorted([r for r in bs if abs(r["A"] - A_OP) < 1e-9 and abs(r["j"] - J_OP) < 1e-9],
                key=lambda r: r["beta"])
    bgrid = [r["beta"] for r in op]
    bfine = np.linspace(0, 0.95, 200)
    bst = beta_star(A_OP, J_OP)
    emp = crossing(bgrid, [r["delta"] for r in op])
    ANSWERS["operating_point_threshold"] = dict(analytic=bst, empirical=emp)
    lams = sorted({r["lam"] for r in gd})
    mus3 = [float(np.mean([r["dfid"] for r in gd if r["lam"] == l and r["w"] == 3.0]))
            for l in lams]
    lam0 = crossing(lams, [-m for m in mus3])
    ANSWERS["guidance_zero_crossing_lambda_w3"] = lam0

    # each panel is drawn by its own function so the three can be emitted standalone as
    # well as composed; fs scales the text when a panel is used on its own
    def panel_a(a, fs=0):
        a.plot(bfine, [law_markov(A_OP, b) for b in bfine], "-", lw=1.5,
               label="population landing (analytic)")
        a.plot(bfine, [ladder_src(A_OP, b, J_OP) for b in bfine], "--", lw=1.5,
               label="do nothing (analytic)")
        a.plot(bgrid, [r["flow"] for r in op], "o", ms=5,
               label="population field (measured)")
        a.plot(bgrid, [r["identity"] for r in op], "s", ms=5, label="do nothing (measured)")
        a.axvline(bst, c="k", lw=0.8, ls=":")
        a.annotate(f"$\\beta^\\star={bst:.3f}$", xy=(bst, law_markov(A_OP, bst)),
                   xytext=(bst - 0.30, 0.335), fontsize=8.5 + fs,
                   arrowprops=dict(arrowstyle="-", lw=0.7, color="k", shrinkA=0, shrinkB=4))
        a.set_xlabel("$\\beta$", fontsize=10 + fs)
        a.set_ylabel("expected target alignment", fontsize=10 + fs)
        a.legend(fontsize=6.5 + fs)

    def panel_b(a, fs=0):
        for lr, m in zip(LRS, ["o", "s", "^"]):
            gaps, ses = [], []
            for w in WIDTHS:
                m1, s1, _ = cagg(lr, "c1", w)
                m2, s2, _ = cagg(lr, "c2", w)
                gaps.append(m1 - m2)
                ses.append(math.sqrt(s1 ** 2 + s2 ** 2))
            a.errorbar(WIDTHS, gaps, yerr=ses, fmt=m + "-", capsize=3, label=f"$\\eta$={lr:g}")
        a.axhline(0, c="k", lw=0.8)
        a.set_xscale("log", base=2)
        a.set_xlabel("width", fontsize=10 + fs)
        a.set_ylabel("marginal $-$ source-conditioned", fontsize=10 + fs)
        a.legend(fontsize=7 + fs)

    def panel_c(a, fs=0):
        for w in [2.0, 3.0]:
            mus = [float(np.mean([r["dfid"] for r in gd if r["lam"] == l and r["w"] == w]))
                   for l in lams]
            sds = [float(np.std([r["dfid"] for r in gd if r["lam"] == l and r["w"] == w]))
                   for l in lams]
            a.errorbar(lams, mus, yerr=sds, fmt="o-", capsize=3, label=f"w={w:g}")
        a.axvline(lam0, c="k", lw=0.8, ls=":")
        a.annotate(f"$\\lambda_0$={lam0:.2f}", (lam0, 0.02), fontsize=8 + fs)
        a.axhline(0, c="k", lw=0.8)
        a.set_xlabel("source informativeness $\\lambda$", fontsize=10 + fs)
        a.set_ylabel("guidance gain in target cosine", fontsize=10 + fs)
        a.legend(fontsize=7 + fs)

    for fn, aa in zip((panel_a, panel_b, panel_c), ax):
        fn(aa)
    save("fig1_main")

    for nm, fn in [("fig1a_threshold", panel_a), ("fig1b_capacity", panel_b),
                   ("fig1c_guidance", panel_c)]:
        fg, aa = plt.subplots(figsize=(4.8, 3.7))
        fn(aa, fs=1)
        fg.tight_layout()
        save(nm)

# %% [code]
if bs:
    A_G = sorted({r["A"] for r in bs})
    J_G = sorted({r["j"] for r in bs})
    tab = []
    for A in A_G:
        for j in J_G:
            sel = sorted([r for r in bs if r["A"] == A and r["j"] == j],
                         key=lambda r: r["beta"])
            if not sel:
                continue
            pred = beta_star(A, j)
            meas = crossing([r["beta"] for r in sel], [r["delta"] for r in sel])
            tab.append(dict(A=A, j=j, pred=pred, meas=meas,
                            err=abs(pred - meas) if np.isfinite(pred) and np.isfinite(meas)
                            else float("nan")))
    both = [t for t in tab if np.isfinite(t["err"]) and t["j"] > 0]
    ANSWERS["threshold_grid"] = tab
    ANSWERS["threshold_max_abs_err"] = max(t["err"] for t in both)
    ANSWERS["threshold_cells_compared"] = len(both)
    print("\n(A, j): predicted vs measured threshold")
    for t in tab:
        print(f"  A={t['A']:.2f} j={t['j']:.2f}  pred={t['pred']:.4f}  "
              f"meas={t['meas']:.4f}  err={t['err']:.4f}")
    print(f"max abs err over {len(both)} cells with j>0: "
          f"{ANSWERS['threshold_max_abs_err']:.4f}")

    sgn = [t["meas"] - t["pred"] for t in both]
    ANSWERS["threshold_mean_signed_error"] = float(np.mean(sgn))
    ANSWERS["threshold_cells_measured_above"] = int(sum(1 for s in sgn if s > 0))
    print(f"measured above analytic in {ANSWERS['threshold_cells_measured_above']}"
          f"/{len(sgn)} cells; mean signed error {np.mean(sgn):+.4f}")

    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    for j in J_G:
        Afine = np.linspace(min(A_G), max(A_G), 200)
        ax.plot(Afine, [beta_star(A, j) for A in Afine], "-", lw=1.2)
        # only cells where the criterion says a threshold exists; the interpolator
        # also returns a crossing where A >= cos j, which Prop 6.3 forbids
        pts = [t for t in tab if t["j"] == j and np.isfinite(t["meas"])
               and np.isfinite(t["pred"])]
        ax.plot([t["A"] for t in pts], [t["meas"] for t in pts], "o", ms=4, label=f"j={j:g}")
    ax.set_xlabel("A")
    ax.set_ylabel("$\\beta^\\star$")
    ax.legend(fontsize=7, title="measured", title_fontsize=7)
    save("fig_threshold_map")

# %% [code]
if cap and mo:
    truth = [r for r in mo if r.get("name") == "true_posterior"]
    pair_true = truth[0]["paircos"] if truth else float("nan")
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    for lr, m in zip(LRS, ["o", "s", "^"]):
        ax.plot(WIDTHS, [cagg(lr, "c2", w, "paircos")[0] for w in WIDTHS], m + "--",
                label=f"source-conditioned $\\eta$={lr:g}")
        ax.plot(WIDTHS, [cagg(lr, "c1", w, "paircos")[0] for w in WIDTHS], m + "-",
                label=f"marginal $\\eta$={lr:g}")
    ax.axhline(pair_true, ls=":", c="k", lw=1.3, label="true posterior")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("width")
    ax.set_ylabel("pairwise cosine among samples sharing a condition")
    ax.legend(fontsize=6.5)
    save("fig_diversity")
    ANSWERS["paircos_true_posterior"] = pair_true
    ANSWERS["paircos_by_cell"] = [dict(lr=lr, width=w,
                                       c1=cagg(lr, "c1", w, "paircos")[0],
                                       c2=cagg(lr, "c2", w, "paircos")[0])
                                  for lr in LRS for w in WIDTHS]

# %% [code]
if mo:
    print("\nreference ladder (single-target cosine vs pairwise cosine vs polar W1)")
    for r in [x for x in mo if x.get("family") == "estimator"]:
        print(f"  {r['name']:>16}  fid={r['fid']:.4f}  paircos={r['paircos']:.4f}  "
              f"w1={r['w1_u']:.4f}")
    est = [x for x in mo if x.get("family") == "estimator"]
    ANSWERS["reference_ladder"] = [dict(name=r["name"], fid=r["fid"], paircos=r["paircos"],
                                        w1_u=r["w1_u"]) for r in est]
    stp = sorted([x for x in mo if x.get("family") == "steps" and x.get("churn") == 0.0],
                 key=lambda r: r["knob"])
    print("\nintegration-step sweep (Heun; K vs cosine vs pairwise cosine)")
    for r in stp:
        print(f"  K={int(r['knob']):>3}  fid={r['fid']:.4f}  paircos={r['paircos']:.4f}  "
              f"w1={r['w1_u']:.4f}")
    ANSWERS["step_sweep"] = [dict(K=int(r["knob"]), fid=r["fid"], paircos=r["paircos"],
                                  w1_u=r["w1_u"]) for r in stp]

if fx:
    tr = sorted([r for r in fx if r.get("remedy") == "truncation"], key=lambda r: r["knob"])
    print("\ntruncation sweep (eps, amplification integral with eps floored at 1e-3, fid)")
    for r in tr:
        print(f"  eps={r['knob']:.2f}  amp_int={r['amp_int']:.3f}  fid={r['fid']:.4f}  "
              f"w1={r['w1_u']:.4f}  paircos={r['paircos']:.4f}")
    rho = spearman([r["amp_int"] for r in tr], [-r["fid"] for r in tr])
    print(f"spearman(amp_int, -fid) = {rho:.3f}")
    ANSWERS["truncation_sweep"] = [dict(eps=r["knob"], amp_int=r["amp_int"], fid=r["fid"],
                                        w1_u=r["w1_u"], paircos=r["paircos"]) for r in tr]
    ANSWERS["truncation_spearman"] = rho

    ax_ = sorted([r for r in fx if r.get("remedy") == "aux_noise"], key=lambda r: r["knob"])
    print("\nauxiliary-noise remedy (sigma, fid, w1, paircos) - the health check")
    for r in ax_:
        print(f"  sigma={r['knob']:.2f}  fid={r['fid']:.4f}  w1={r['w1_u']:.4f}  "
              f"paircos={r['paircos']:.4f}")
    ANSWERS["aux_noise_sweep"] = [dict(sigma=r["knob"], fid=r["fid"], w1_u=r["w1_u"],
                                       paircos=r["paircos"]) for r in ax_]

if gd:
    ANSWERS["guidance_sweep"] = [dict(lam=r["lam"], w=r["w"], seed=r.get("seed"),
                                      dfid=r["dfid"], fid=r["fid"]) for r in gd]

# %% [code]
if ANSWERS:
    with open(os.path.join(OUT, "answers.json"), "w") as f:
        json.dump(ANSWERS, f, indent=1, default=float)
    print(f"\nanswers.json + figures -> {OUT}")
    print("send: fig1_main.pdf, fig_threshold_map.pdf, fig_diversity.pdf, answers.json")
else:
    print("\nnothing was produced: none of the input csvs were found, so no answers.json "
          "is written. point this at a directory containing the ledger6 results.")
