"""Selection routines of MOSAIC used by make_figures.py: the budget-local pairwise interaction model, the search over
field sets, and the names of the selection rules of the controlled study (Table 4)."""
import itertools, math
import numpy as np


def to_mask(cols):
    return int(sum(1 << int(c) for c in cols))


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


KERNEL_SIGMA = 0.25


def _est_arrays(est):
    """est = {str(p): {"g": [...], "H": [[...]]}} (as stored in run outputs) -> sorted densities, {p: (g, H)}."""
    E = {float(p): (np.asarray(v["g"], dtype=np.float64), np.asarray(v["H"], dtype=np.float64)) for p, v in est.items()}
    return sorted(E), E


def budget_surrogate(est, F, k, sigma=KERNEL_SIGMA):
    """Pairwise surrogate U_k(S) = sum_{i in S} a_i + sum_{i<j in S} b_ij for size-k subsets.
    sigma > 0 (v2, default): BUDGET-LOCAL second-order expansion of the multilinear extension around density k/F:
        b = Gaussian-kernel (width sigma in density) weighted mean of H(p) over the estimated densities.
    sigma None: b = plain mean of H(p) over all estimated densities (the averaged rule of Table 4).
    In both cases a = g(p*) - (b 1) k/F with p* the estimated density nearest to k/F.
    Why v2: on Avazu the estimated interactions shrink ~8x from p=0.18 to p=0.73 (strong higher-order structure),
    so a density-averaged b under-penalises redundancy at small budgets and over-penalises it at large ones."""
    ps, E = _est_arrays(est); pk = k / F
    if sigma is None: b = np.mean([E[p][1] for p in ps], 0)
    else:
        w = np.array([np.exp(-0.5 * ((p - pk) / sigma) ** 2) for p in ps]); w /= w.sum()
        b = sum(wi * E[p][1] for wi, p in zip(w, ps))
    b = 0.5 * (b + b.T); np.fill_diagonal(b, 0.0)
    g = E[min(ps, key=lambda p: abs(p - pk))][0]
    return g - b.sum(1) * pk, b


# selection rules of the controlled study (Table 4), as named in the result files
RUNG_AGNOSTIC = "Rank by budget-agnostic importance (p=0.5)"   # first-order mask moment at p=0.5, top-k
RUNG_LOCAL = "Rank by budget-local importance"                  # first-order mask moment at p ~ k/F, top-k
RUNG_V1 = "Density-averaged interactions"                                  # pairwise surrogate, density-averaged b, no verification
RUNG_PAIR = "MOSAIC w/o verification"                           # pairwise surrogate, budget-local b, no verification
RUNG_FULL = "MOSAIC"                                            # + shared-model verification of the top-3
LADDER = [RUNG_AGNOSTIC, RUNG_LOCAL, RUNG_V1, RUNG_PAIR, RUNG_FULL]
