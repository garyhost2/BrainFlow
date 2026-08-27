# %% [code]
import os, csv, glob, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CAND = ["/kaggle/input", "/kaggle/working", "runs", "."]
OUT = "/kaggle/working/figpack" if os.path.isdir("/kaggle/working") else "runs/figpack"
os.makedirs(OUT, exist_ok=True)


def find_all(name):
    hits = []
    for c in CAND:
        if os.path.isdir(c):
            hits += glob.glob(os.path.join(c, "**", name), recursive=True)
    hits = [h for h in hits if os.path.getsize(h) > 0
            and os.path.abspath(OUT) not in os.path.abspath(h)]
    seen, uniq = set(), []
    for h in hits:
        rp = os.path.realpath(h)
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(h)
    return sorted(uniq, key=os.path.getmtime)


def load_all(name):
    rows = []
    for p in find_all(name):
        for r in csv.DictReader(open(p)):
            for k, v in list(r.items()):
                if v is None or v == "":
                    r[k] = float("nan")
                    continue
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    pass
            rows.append(r)
    print(f"{name}: {len(rows)} rows from {len(find_all(name))} file(s)", flush=True)
    return rows


def load_json(name):
    hits = find_all(name)
    if not hits:
        print(f"MISSING {name}", flush=True)
        return None
    return json.load(open(hits[-1]))


def save(name):
    plt.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(OUT, name + ".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"wrote {name}.pdf / .png", flush=True)


def crossing(xs, ys):
    pts = [(x, y) for x, y in zip(xs, ys)
           if isinstance(x, (int, float)) and isinstance(y, (int, float))
           and np.isfinite(x) and np.isfinite(y)]
    for i in range(1, len(pts)):
        (x0, y0), (x1, y1) = pts[i - 1], pts[i]
        if y0 <= 0 < y1:
            return x0 + (x1 - x0) * (-y0) / (y1 - y0)
    return float("nan")


NOTE = {}

# %% [code]
os_rows = load_all("offshelf.csv")
if os_rows:
    sd = [r for r in os_rows if r.get("arm") == "sdedit"]
    FACS = sorted({int(r["factor"]) for r in sd})
    STR = sorted({r["strength"] for r in sd})

    def agg(f, s):
        v = [r["gain"] for r in sd if int(r["factor"]) == f and r["strength"] == s]
        return (float(np.mean(v)), float(np.std(v)) / math.sqrt(max(len(v), 1))) if v \
            else (float("nan"), float("nan"))

    dn = {int(r["factor"]): r["align"] for r in os_rows if r.get("arm") == "do_nothing"}
    fig, ax = plt.subplots(1, 2, figsize=(10.6, 3.9))
    cols = plt.cm.viridis(np.linspace(0.12, 0.88, len(FACS)))
    for f, c in zip(FACS, cols):
        mu = [agg(f, s)[0] for s in STR]
        se = [agg(f, s)[1] for s in STR]
        ax[0].errorbar(STR, mu, yerr=se, fmt="o-", capsize=3, color=c,
                       label=f"${f}\\times$  (source {dn.get(f, float('nan')):.2f})")
    ax[0].axhline(0, c="k", lw=0.9)
    ax[0].set_xlabel("transport strength applied")
    ax[0].set_ylabel("CLIP alignment gain over doing nothing")
    ax[0].legend(fontsize=7, title="degradation", title_fontsize=7)

    best = [max(agg(f, s)[0] for s in STR) for f in FACS]
    al = [dn.get(f, float("nan")) for f in FACS]
    order = np.argsort(al)
    al_s = [al[i] for i in order]
    best_s = [best[i] for i in order]
    ax[1].plot(al_s, best_s, "o-", color="#2b6cb0")
    ax[1].axhline(0, c="k", lw=0.9)
    xc = crossing(al_s, [-b for b in best_s])
    if np.isfinite(xc):
        ax[1].axvline(xc, ls=":", c="k", lw=1.1)
        ax[1].annotate(f"transport stops paying\nabove {xc:.2f}", (xc, 0),
                       xytext=(xc + 0.03, max(best_s) * 0.45), fontsize=7.5)
        NOTE["offshelf_alignment_crossing"] = xc
    for f, a, b in zip(FACS, al, best):
        ax[1].annotate(f"${f}\\times$", (a, b), textcoords="offset points",
                       xytext=(4, -9), fontsize=7)
    ax[1].set_xlabel("how informative the source already is (CLIP alignment)")
    ax[1].set_ylabel("best achievable gain")
    save("fig_offshelf")

# %% [code]
sr_rows = [r for r in load_all("sr.csv") if r.get("arm") in ("identity", "reg", "m1", "m2")]
if sr_rows:
    RES = sorted({int(r["res"]) for r in sr_rows})

    def sagg(res, arm, key):
        v = [r[key] for r in sr_rows if int(r["res"]) == res and r["arm"] == arm
             and isinstance(r.get(key), (int, float)) and np.isfinite(r[key])]
        return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"), float("nan"))

    keys = [("psnr", "PSNR $\\uparrow$", False),
            ("energy_r", "energy distance $\\downarrow$", True),
            ("w1_hf", "$W_1$ high-frequency $\\downarrow$", True)]
    fig, ax = plt.subplots(1, len(keys), figsize=(3.7 * len(keys), 3.5),
                           constrained_layout=True)
    w = 0.30
    xs = np.arange(len(RES))
    for a, (k, lab, lower_better) in zip(ax, keys):
        m1 = [sagg(r, "m1", k)[0] for r in RES]
        m2 = [sagg(r, "m2", k)[0] for r in RES]
        e1 = [sagg(r, "m1", k)[1] for r in RES]
        e2 = [sagg(r, "m2", k)[1] for r in RES]
        a.bar(xs - w / 2, m1, w, yerr=e1, capsize=3, label="$\\mathcal{M}_1$ marginal",
              color="#2b6cb0")
        a.bar(xs + w / 2, m2, w, yerr=e2, capsize=3, label="$\\mathcal{M}_2$ source-conditioned",
              color="#b0421f")
        if lower_better:
            nk = k + "_null"
            nulls = [sagg(r, "m1", nk)[0] for r in RES]
            if np.isfinite(nulls[0]):
                a.plot(xs, nulls, "k_", ms=26, mew=1.4)
                a.annotate("null floor", (xs[0], nulls[0]), textcoords="offset points",
                           xytext=(-6, 8), fontsize=7)
        a.set_xticks(xs)
        a.set_xticklabels([f"res {r}" for r in RES])
        a.set_title(lab, fontsize=10)
        if k == "psnr":
            lo, hi = min(m1 + m2), max(m1 + m2)
            a.set_ylim(lo - 0.35 * (hi - lo) - 0.2, hi + 0.12 * (hi - lo) + 0.1)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8.5, ncol=2, loc="lower center", frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("The source-conditioned model wins the single-target score "
                 "and loses both distributional instruments", fontsize=10)
    save("fig_sr_inversion")

# %% [code]
ans = load_json("answers.json")
fx = load_all("fixes.csv")
tr = sorted([r for r in fx if r.get("remedy") == "truncation"], key=lambda r: r["knob"]) \
    if fx else []
ax_ = sorted([r for r in fx if r.get("remedy") == "aux_noise"], key=lambda r: r["knob"]) \
    if fx else []
if not tr and ans:
    tr = [dict(knob=r["eps"], fid=r["fid"], w1_u=r["w1_u"]) for r in ans.get("truncation_sweep", [])]
if not ax_ and ans:
    ax_ = [dict(knob=r["sigma"], fid=r["fid"], w1_u=r["w1_u"]) for r in ans.get("aux_noise_sweep", [])]

if tr and ax_:
    lad = {r["name"]: r for r in (ans or {}).get("reference_ladder", [])}
    m1_fid = lad.get("c1_K64", {}).get("fid", float("nan"))
    idw = lad.get("identity", {}).get("w1_u", float("nan"))
    fig, ax = plt.subplots(1, 2, figsize=(10.6, 3.9), sharey=False)
    for a, rows, xlab, title in [
            (ax[0], tr, "truncation $\\varepsilon$", "Truncation: a probe, not a remedy"),
            (ax[1], ax_, "auxiliary noise $\\sigma_{\\mathrm{aux}}$", "Noising the source: a remedy")]:
        x = [r["knob"] for r in rows]
        a.plot(x, [r["fid"] for r in rows], "o-", color="#2b6cb0", label="target cosine (up = better)")
        if np.isfinite(m1_fid):
            a.axhline(m1_fid, ls="--", lw=1, c="#2b6cb0", alpha=0.55)
            a.annotate("marginal model", (x[0], m1_fid), textcoords="offset points",
                       xytext=(2, 4), fontsize=7, color="#2b6cb0")
        a.set_xlabel(xlab)
        a.set_ylabel("target cosine", color="#2b6cb0")
        a.tick_params(axis="y", labelcolor="#2b6cb0")
        a.set_title(title, fontsize=9.5)
        b = a.twinx()
        b.plot(x, [r["w1_u"] for r in rows], "s--", color="#b0421f",
               label="distributional error (down = better)")
        if np.isfinite(idw):
            b.axhline(idw, ls=":", lw=1, c="#b0421f", alpha=0.6)
            b.annotate("doing nothing", (x[-1], idw), textcoords="offset points",
                       xytext=(-52, 3), fontsize=7, color="#b0421f")
        b.set_ylabel("polar $W_1$", color="#b0421f")
        b.tick_params(axis="y", labelcolor="#b0421f")
        lo = min(min(r["w1_u"] for r in tr), min(r["w1_u"] for r in ax_))
        hi = max(max(r["w1_u"] for r in tr), max(r["w1_u"] for r in ax_),
                 idw if np.isfinite(idw) else 0)
        b.set_ylim(lo - 0.03, hi * 1.08)
    save("fig_remedy")

# %% [code]
lr3 = load_all("lr3_scaling.csv")
if lr3:
    cells = [r for r in lr3 if r.get("arm") in ("c1", "c2")]
    LRS = sorted({r["lr"] for r in cells})
    WID = sorted({int(r["width"]) for r in cells})

    def cagg(lr, arm, w, key="fid"):
        v = [r[key] for r in cells if r["lr"] == lr and r["arm"] == arm and int(r["width"]) == w]
        return (float(np.mean(v)), float(np.std(v)) / math.sqrt(max(len(v), 1))) if v \
            else (float("nan"), float("nan"))

    pt = (load_json("answers.json") or {}).get("paircos_true_posterior", float("nan"))
    fig, ax = plt.subplots(1, 2, figsize=(10.6, 3.9))
    mk = dict(zip(LRS, ["o", "s", "^", "v"]))
    for lr in LRS:
        g = [cagg(lr, "c1", w)[0] - cagg(lr, "c2", w)[0] for w in WID]
        se = [math.sqrt(cagg(lr, "c1", w)[1] ** 2 + cagg(lr, "c2", w)[1] ** 2) for w in WID]
        ax[0].errorbar(WID, g, yerr=se, fmt=mk[lr] + "-", capsize=3, label=f"$\\eta={lr:g}$")
        ax[1].plot(WID, [cagg(lr, "c2", w, "paircos")[0] for w in WID], mk[lr] + "--",
                   color="#b0421f", alpha=0.85)
        ax[1].plot(WID, [cagg(lr, "c1", w, "paircos")[0] for w in WID], mk[lr] + "-",
                   color="#2b6cb0", alpha=0.85)
    ax[0].axhline(0, c="k", lw=0.9)
    ax[0].set_ylabel("marginal $-$ source-conditioned")
    ax[0].legend(fontsize=7.5)
    if np.isfinite(pt):
        ax[1].axhline(pt, ls=":", c="k", lw=1.2)
        ax[1].annotate("true posterior", (WID[0], pt), textcoords="offset points",
                       xytext=(2, 5), fontsize=7)
    ax[1].set_ylabel("pairwise cosine (lower = more variety)")
    ax[1].annotate("source-conditioned", (WID[-1], cagg(LRS[-1], "c2", WID[-1], "paircos")[0]),
                   textcoords="offset points", xytext=(-64, 6), fontsize=7, color="#b0421f")
    ax[1].annotate("marginal", (WID[-1], cagg(LRS[-1], "c1", WID[-1], "paircos")[0]),
                   textcoords="offset points", xytext=(-34, 6), fontsize=7, color="#2b6cb0")
    for a in ax:
        a.set_xscale("log", base=2)
        a.set_xlabel("width")
    save("fig_capacity")

# %% [code]
if NOTE:
    with open(os.path.join(OUT, "cluster_notes.json"), "w") as f:
        json.dump(NOTE, f, indent=1)
print(f"\nfigures written to {OUT}")
