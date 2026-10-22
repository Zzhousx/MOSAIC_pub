"""Numerical routines for MOSAIC, its controlled study and protocol-safe retrain caching."""
import glob, itertools, json, math, os
import numpy as np


# ------------------------------------------------------------------------------------------ metrics
def auc_np(y, s):
    """Tie-aware ROC AUC (rank statistic). y in {0,1}, s = scores (any monotone transform, e.g. logits)."""
    y = np.asarray(y); s = np.asarray(s, dtype=np.float64)
    order = np.argsort(s, kind="mergesort"); s_sorted = s[order]
    _, inv, counts = np.unique(s_sorted, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts); start = cum - counts
    ranks = np.empty(len(s)); ranks[order] = ((start + cum - 1) / 2.0 + 1.0)[inv]
    npos = float(y.sum()); nneg = len(y) - npos
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def logloss_logits(y, z):
    """Mean binary cross-entropy computed stably from logits."""
    z = np.asarray(z, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


# ------------------------------------------------------------------------------------------ masks
def bits(m, F):
    return [f for f in range(F) if m >> f & 1]


def to_mask(cols):
    return int(sum(1 << int(c) for c in cols))


def mask_vec(m, F):
    return np.array([(m >> f) & 1 for f in range(F)], dtype=np.float64)


# ------------------------------------------------------------------------------------------ selection
def quad_value(X, a, b):
    """Pairwise surrogate U(S) = sum_{i in S} a_i + sum_{i<j in S} b_ij for rows of the 0/1 matrix X."""
    return X @ a + 0.5 * np.einsum("ij,jk,ik->i", X, b, X)


def select_subsets(a, b, F, k, top=3, restarts=64, rng=None, exact_limit=2_000_000):
    """Best size-k subsets (as bit masks, best first) of the pairwise surrogate.
    Exact enumeration when C(F,k) <= exact_limit, otherwise best-improvement swap local search from a greedy
    start plus (restarts-1) random starts. The swap gain uses
        U(S - i + j) - U(S) = a_j + s_j - b_ij - a_i - s_i,   s = b @ x   (b symmetric, zero diagonal)."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    b = 0.5 * (b + b.T); np.fill_diagonal(b, 0.0)
    if k >= F: return [(1 << F) - 1]
    if math.comb(F, k) <= exact_limit:
        combos = np.array(list(itertools.combinations(range(F), k))); best = []
        for ch in np.array_split(combos, max(1, len(combos) // 50000)):
            X = np.zeros((len(ch), F)); np.put_along_axis(X, ch, 1, axis=1); v = quad_value(X, a, b)
            for i in np.argsort(-v)[:top]: best.append((float(v[i]), to_mask(ch[i])))
        return [m for _, m in sorted(best, reverse=True)[:top]]
    rng = rng if rng is not None else np.random.default_rng(0)
    found = {}
    for r in range(restarts):
        x = np.zeros(F, dtype=bool)
        if r == 0:                                  # greedy start
            for _ in range(k):
                gain = np.where(x, -np.inf, a + b @ x); x[int(np.argmax(gain))] = True
        else:
            x[rng.permutation(F)[:k]] = True
        for _ in range(10 * F * F):                 # best-improvement swaps (always terminates; bound is a safeguard)
            s = b @ x; out_, in_ = np.where(x)[0], np.where(~x)[0]
            D = (a[in_] + s[in_])[None, :] - b[np.ix_(out_, in_)] - (a[out_] + s[out_])[:, None]
            i, j = np.unravel_index(int(np.argmax(D)), D.shape)
            if D[i, j] <= 1e-12: break
            x[out_[i]] = False; x[in_[j]] = True
        m = to_mask(np.where(x)[0]); found[m] = float(quad_value(x[None].astype(float), a, b)[0])
    return [m for m, _ in sorted(found.items(), key=lambda kv: -kv[1])[:top]]


# ------------------------------------------------------------------------------------------ surrogates
KERNEL_SIGMA = 0.25


def _est_arrays(est):
    """est = {str(p): {"g": [...], "H": [[...]]}} (as stored in run outputs) -> sorted densities, {p: (g, H)}."""
    E = {float(p): (np.asarray(v["g"], dtype=np.float64), np.asarray(v["H"], dtype=np.float64)) for p, v in est.items()}
    return sorted(E), E


def budget_surrogate(est, F, k, sigma=KERNEL_SIGMA):
    """Pairwise surrogate U_k(S) = sum_{i in S} a_i + sum_{i<j in S} b_ij for size-k subsets.
    sigma > 0 (default): BUDGET-LOCAL second-order expansion of the multilinear extension around density k/F:
        b = Gaussian-kernel (width sigma in density) weighted mean of H(p) over the estimated densities.
    sigma None (density-averaged control): b = plain mean of H(p) over all estimated densities.
    In both cases a = g(p*) - (b 1) k/F with p* the estimated density nearest to k/F.
    Budget locality: on Avazu the estimated interactions shrink ~8x from p=0.18 to p=0.73 (strong higher-order structure),
    so a density-averaged b under-penalises redundancy at small budgets and over-penalises it at large ones."""
    ps, E = _est_arrays(est); pk = k / F
    if sigma is None: b = np.mean([E[p][1] for p in ps], 0)
    else:
        w = np.array([np.exp(-0.5 * ((p - pk) / sigma) ** 2) for p in ps]); w /= w.sum()
        b = sum(wi * E[p][1] for wi, p in zip(w, ps))
    b = 0.5 * (b + b.T); np.fill_diagonal(b, 0.0)
    g = E[min(ps, key=lambda p: abs(p - pk))][0]
    return g - b.sum(1) * pk, b


# ------------------------------------------------------------------------------------------ controlled-study ladder
RUNG_AGNOSTIC = "Rank by budget-agnostic importance (p=0.5)"   # first-order mask moment at p=0.5, top-k
RUNG_LOCAL = "Rank by budget-local importance"                  # first-order mask moment at p ~ k/F, top-k
RUNG_V1 = "Density-averaged interactions"                                  # pairwise surrogate, density-averaged b, no verification
RUNG_PAIR = "MOSAIC w/o verification"                           # pairwise surrogate, budget-local b, no verification
RUNG_FULL = "MOSAIC"                                            # + shared-model verification of the top-3
LADDER = [RUNG_AGNOSTIC, RUNG_LOCAL, RUNG_V1, RUNG_PAIR, RUNG_FULL]


def first_order_subset(est, F, k, density=None):
    """Rank-then-cut on the shared model: the k fields with the largest first-order mask moment g(p).
    density None = budget-local (the estimated density nearest to k/F, the same p* as budget_surrogate);
    otherwise the estimated density nearest to `density` (0.5 = budget-agnostic, plain field-dropout importance)."""
    ps, E = _est_arrays(est); target = k / F if density is None else density
    p = min(ps, key=lambda q: abs(q - target))
    return to_mask(np.argsort(-E[p][0], kind="stable")[:k])


def ladder_plan(run):
    """Subsets of every ladder rung for a finished v2 `run` output (dict as stored by mosaic_gpu.py run).
    The two first-order rungs are computed from the stored moments; the three MOSAIC rungs are read from the run.
    Consistency check: the budget-local pairwise selection is re-derived from the stored moments with the same random
    sequence as the run. Returns ({rung: {k: mask}}, budgets re-derived identically, number of budgets)."""
    fields = list(run["fields"]); F = len(fields); idx = {f: i for i, f in enumerate(fields)}
    budgets = [int(k) for k in run["budgets"]]; est = run["est"]
    plan = {RUNG_AGNOSTIC: {k: first_order_subset(est, F, k, density=0.5) for k in budgets},
            RUNG_LOCAL: {k: first_order_subset(est, F, k) for k in budgets}}
    for name in (RUNG_V1, RUNG_PAIR, RUNG_FULL):
        plan[name] = {k: to_mask([idx[f] for f in run["evals"][name]["subsets"][str(k)]]) for k in budgets}
    rng_loc = np.random.default_rng(10_000 + int(run["mseed"]))          # as in mosaic_gpu.cmd_run
    loc = {k: select_subsets(*budget_surrogate(est, F, k), F, k, top=3, rng=rng_loc) for k in budgets}
    same = sum(loc[k][0] == plan[RUNG_PAIR][k] for k in budgets)
    return plan, int(same), len(budgets)


# ------------------------------------------------------------------------------------------ retrain cache
class RetrainCache:
    """JSON cache of retrain results, keyed "mask|seed|full" (final evaluation retrains on all training rows).
    The file stores the protocol settings under
    "__config__"; reusing a file with different settings is refused (use a new --out name).
    Entries of other *.cache.json files in the same folder (and in `extra_globs`) with an IDENTICAL config are
    imported read-only: same subset + same seed + same protocol
    = the same experiment, so it is not repeated."""

    def __init__(self, path, config, extra_globs=()):
        self.path, self.config, self.data, self.imported = path, config, {}, 0
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if os.path.exists(path):
            J = json.load(open(path))
            if J.get("__config__") != config:
                raise SystemExit(f"{path} was created with different settings:\n  {J.get('__config__')}\n"
                                 f"now:\n  {config}\nUse a new --out name (never change settings for an existing output).")
            self.data = {k: v for k, v in J.items() if k != "__config__"}
        folder = os.path.dirname(os.path.abspath(path))
        files = set(glob.glob(os.path.join(folder, "*.cache.json")))
        for g in extra_globs: files |= set(glob.glob(g))
        for p in sorted(files):
            if os.path.abspath(p) == os.path.abspath(path): continue
            try: J = json.load(open(p))
            except Exception: continue
            if J.get("__config__") != config: continue
            for k, v in J.items():
                if k != "__config__" and k not in self.data: self.data[k] = v; self.imported += 1

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        self.data[key] = value
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f: json.dump({"__config__": self.config, **self.data}, f)
        os.replace(tmp, self.path)

    def __len__(self):
        return len(self.data)
