"""Construct fixed noisy copies with unchanged row order and train/validation/test splits."""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--npz"); ap.add_argument("--cache"); ap.add_argument("--rows", default=None, help="train,val,test subsample (cache only)")
ap.add_argument("--factor", type=int, required=True, help="total columns per original field (1 = no copies)")
ap.add_argument("--rate", type=float, default=0.1); ap.add_argument("--seed", type=int, default=2031)
ap.add_argument("--out", required=True)
a = ap.parse_args()
KEYS = [("Xtr", "ytr"), ("Xva", "yva"), ("Xte", "yte")]
if a.npz:
    z = np.load(a.npz); parts = [(np.asarray(z[X]), np.asarray(z[y])) for X, y in KEYS]; fields = [str(f) for f in z["fields"]]
else:
    M = json.load(open(os.path.join(a.cache, "meta.json"))); fields = [str(f) for f in M["fields"]]
    rows = [int(x) for x in a.rows.split(",")] if a.rows else None; parts = []
    for i, (X, y) in enumerate(KEYS):
        Xm = np.load(os.path.join(a.cache, X + ".npy"), mmap_mode="r"); ym = np.load(os.path.join(a.cache, y + ".npy"), mmap_mode="r")
        if rows and rows[i] < len(ym):
            sel = np.sort(np.random.default_rng(2032 + i).choice(len(ym), rows[i], replace=False)); parts.append((np.asarray(Xm[sel]), np.asarray(ym[sel])))
        else: parts.append((np.asarray(Xm), np.asarray(ym)))
n = [len(p[1]) for p in parts]; N = sum(n); F0 = len(fields); F = F0 * a.factor
Xall = np.empty((N, F), dtype=np.int32); names, card = [], []
rng = np.random.default_rng(a.seed)
for j in range(F0):
    col = np.concatenate([p[0][:, j] for p in parts]).astype(np.int64)
    for c in range(a.factor):
        v = col.copy()
        if c > 0:
            r = rng.random(N) < a.rate; v[r] = rng.choice(col, size=int(r.sum()))
        u, inv = np.unique(v, return_inverse=True)
        Xall[:, j * a.factor + c] = inv.astype(np.int32); card.append(len(u))
        names.append(fields[j] if c == 0 else f"{fields[j]}_copy{c}")
    print(f"field {fields[j]}: card {card[-a.factor]}" + (f", copies {card[-a.factor + 1:]}" if a.factor > 1 else ""), flush=True)
out = {}; o = 0
for (X, y), m, p in zip(KEYS, n, parts):
    out[X] = Xall[o:o + m]; out[y] = p[1].astype(np.float32); o += m
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
np.savez(a.out, **out, card=np.array(card, dtype=np.int64), fields=np.array(names))
json.dump(dict(source=a.npz or a.cache, rows=n, factor=a.factor, rate=a.rate, seed=a.seed, fields=names, card=card,
               copy_of={nm: nm.split("_copy")[0] for nm in names if "_copy" in nm}),
          open(a.out[:-4] + ".meta.json", "w"), indent=1)
print(f"wrote {a.out}: {F} fields ({F0} originals x {a.factor}), rows {n}")
