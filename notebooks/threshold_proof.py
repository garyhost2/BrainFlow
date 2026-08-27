# %% [code]
import os, csv, json, math
import numpy as np

OUT = "/kaggle/working/threshold" if os.path.isdir("/kaggle/working") else "runs/threshold"
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
        json.dump(dict(register=REG), f, indent=1)


# the two reference levels, exactly as the sweep computes them
def src(A, b, j):
    s = math.sqrt(max(0.0, 1 - b * b))
    return A * (s * math.cos(j) + b * math.sin(j))


def law(A, b):
    s2 = max(0.0, 1 - b * b)
    return A * (A * s2 + b * math.sqrt(max(0.0, 1 - A * A * s2)))


def Delta(A, b, j):
    return law(A, b) - src(A, b, j)


def psi(A, phi):
    return math.acos(min(1.0, max(-1.0, A * math.cos(phi))))


A_GRID = [0.05, 0.1, 0.15, 0.25, 0.37, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
J_GRID = [0.02, 0.05, 0.15, 0.3, 0.5, 0.8, 1.0, 1.2, 1.4, 1.5]

# %% [code]
# Lemma 1.  With beta = sin(phi) and psi(phi) = arccos(A cos phi),
#           Delta / A = cos(psi - phi) - cos(phi - j).
err = 0.0
for A in A_GRID:
    for j in J_GRID:
        for phi in np.linspace(1e-6, math.pi / 2, 400):
            lhs = Delta(A, math.sin(float(phi)), j) / A
            rhs = math.cos(psi(A, float(phi)) - float(phi)) - math.cos(float(phi) - j)
            err = max(err, abs(lhs - rhs))
reg("thr_angle_identity", err < 1e-12,
    "under beta = sin(phi) both reference levels are cosines of angle differences, so the "
    "gap is cos(psi - phi) - cos(phi - j) with psi = arccos(A cos phi)",
    max_abs_err=err, n_points=len(A_GRID) * len(J_GRID) * 400)

# Lemma 2.  psi(phi) >= phi, with equality only at phi = pi/2.
worst = 1e9
for A in A_GRID:
    for phi in np.linspace(0.0, math.pi / 2, 2000):
        worst = min(worst, psi(A, float(phi)) - float(phi))
reg("thr_psi_dominates_phi", worst >= -1e-12,
    "since A < 1 the auxiliary angle psi = arccos(A cos phi) is never below phi, so the "
    "landing cosine is cos(psi - phi) with a non-negative argument",
    min_psi_minus_phi=worst)

# Lemma 3.  psi'(phi) = A sin phi / sin psi <= A, hence h(phi) = 2 phi - j - psi(phi)
#           is strictly increasing with slope at least 2 - A.
worst_d, worst_slope = 0.0, 1e9
for A in A_GRID:
    for phi in np.linspace(1e-4, math.pi / 2 - 1e-4, 2000):
        p = float(phi)
        h = 1e-7
        num = (psi(A, p + h) - psi(A, p - h)) / (2 * h)
        cf = A * math.sin(p) / math.sin(psi(A, p))
        worst_d = max(worst_d, abs(num - cf))
        worst_slope = min(worst_slope, 2.0 - cf)
reg("thr_psi_derivative_bounded_by_A", worst_d < 1e-6 and worst_slope > 1.0,
    "psi' = A sin phi / sin psi is bounded above by A because psi >= phi, so h = 2 phi - j - psi "
    "has derivative at least 2 - A > 1 and is strictly increasing",
    max_derivative_err=worst_d, min_slope_of_h=worst_slope)

# %% [code]
# Proposition.  Three exact sign facts.
#   Delta(0)/A     = A - cos j
#   Delta(sin j)/A = cos(psi(j) - j) - 1  < 0  for every A < 1
#   Delta(1)/A     = 1 - sin j            > 0
e0 = em = e1 = 0.0
mid_max = -1e9
for A in A_GRID:
    for j in J_GRID:
        e0 = max(e0, abs(Delta(A, 0.0, j) / A - (A - math.cos(j))))
        em = max(em, abs(Delta(A, math.sin(j), j) / A - (math.cos(psi(A, j) - j) - 1)))
        e1 = max(e1, abs(Delta(A, 1.0, j) / A - (1 - math.sin(j))))
        mid_max = max(mid_max, Delta(A, math.sin(j), j))
reg("thr_endpoint_and_midpoint_values", max(e0, em, e1) < 1e-12,
    "the gap takes the stated closed forms at beta = 0, beta = sin j and beta = 1",
    err_at_0=e0, err_at_sin_j=em, err_at_1=e1)
reg("thr_gap_is_negative_at_sin_j", mid_max < 0,
    "at beta = sin j the comparison term is cos(0) = 1, its maximum, while the landing term is "
    "strictly below 1, so the source strictly beats the flow there for every A < 1 and j > 0, "
    "irrespective of whether A < cos j",
    max_over_grid=mid_max)

# %% [code]
# Theorem.  Exactly one root above sin j, always; and exactly one below sin j iff A > cos j.
def roots(A, j, n=400001):
    bs = np.linspace(0.0, 1.0, n)
    d = np.array([Delta(A, float(b), j) for b in bs])
    return [0.5 * (bs[i - 1] + bs[i]) for i in range(1, n) if d[i - 1] * d[i] < 0]


rows, bad_upper, bad_lower = [], 0, 0
for A in A_GRID:
    for j in J_GRID:
        r = roots(A, j)
        up = [x for x in r if x > math.sin(j)]
        lo = [x for x in r if x < math.sin(j)]
        lt = A < math.cos(j)
        if len(up) != 1:
            bad_upper += 1
        if len(lo) != (0 if lt else 1):
            bad_lower += 1
        rows.append(dict(A=A, j=j, cos_j=math.cos(j), A_lt_cos_j=int(lt), sin_j=math.sin(j),
                         n_roots=len(r), n_above=len(up), n_below=len(lo),
                         upper_root=up[0] if up else float("nan"),
                         lower_root=lo[0] if lo else float("nan")))
save_csv("root_structure.csv", rows)
reg("thr_upper_root_always_exists_and_is_unique", bad_upper == 0,
    "for every tested (A, j) the gap has exactly one root above beta = sin j: it is negative "
    "there and positive at beta = 1, and h is strictly increasing so the crossing is unique. "
    "a threshold therefore exists unconditionally, and A < cos j is not required for it",
    cells=len(rows), exceptions=bad_upper)
reg("thr_A_lt_cos_j_characterises_uniqueness", bad_lower == 0,
    "below beta = sin j the gap is positive exactly when A cos phi > cos j, which is strictly "
    "decreasing in phi; so there is one further root there iff A > cos j. the condition "
    "A < cos j therefore characterises uniqueness of the transition, not its existence",
    cells=len(rows), exceptions=bad_lower)

# the three cells the sweep measured but the criterion declared empty
CHECK = [(0.70, 0.80, 0.8286), (0.90, 0.50, 0.5600), (0.90, 0.80, 0.7586)]
devs = []
for A, j, meas in CHECK:
    up = [x for x in roots(A, j) if x > math.sin(j)][0]
    devs.append(abs(up - meas))
    log(A=A, j=j, analytic_upper_root=up, measured=meas, dev=abs(up - meas))
reg("thr_declared_empty_cells_match_the_upper_root", max(devs) < 0.03,
    "the crossings the sweep found in the three cells with A >= cos j are not artefacts: they "
    "agree with the analytic upper root to the same accuracy as the cells already reported",
    max_dev=max(devs), n=len(CHECK))

flush_reg()
npass = sum(1 for g in REG if g["tag"] == "PASS")
log(passed=npass, total=len(REG), out=OUT)
