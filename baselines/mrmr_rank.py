"""mRMR using training-only Miller-Madow mutual information and the MID selection criterion. Uses all rows when the training set is smaller than the2M-row cap."""
import argparse, json, os, time
import numpy as np


def load_train(C, ds):
    D = C["datasets"][ds]; src = D["source"]
    if "npz" in src:
        z = np.load(src["npz"]); X, y, fields = z["Xtr"], z["ytr"], [str(f) for f in z["fields"]]
    else:
        cdir = os.path.join(src["data_path"], src["erase_dataset"], "mosaic_cache")
        fields = [str(f) for f in json.load(open(os.path.join(cdir, "meta.json")))["fields"]]
        X = np.load(os.path.join(cdir, "Xtr.npy"), mmap_mode="r"); y = np.load(os.path.join(cdir, "ytr.npy"), mmap_mode="r")
    mtr = D.get("max_train_rows")
    if mtr and len(y) > mtr:                       # identical to mosaic_gpu.load_data: the same fixed training subsample
        sel = np.sort(np.random.default_rng(2027).choice(len(y), mtr, replace=False)); return X, y, fields, sel
    return X, y, fields, np.arange(len(y))


def recode(col, min_count):
    """Map a column to 0..c-1, merging values seen < min_count times into one value. Returns (codes int64, c)."""
    u, inv, cnt = np.unique(col, return_inverse=True, return_counts=True)
    keep = cnt >= min_count
    newid = np.where(keep, np.cumsum(keep) - 1, keep.sum())   # kept values -> 0..K-1, rare values -> K
    codes = newid[inv].astype(np.int64)
    return codes, int(keep.sum() + (1 if (~keep).any() else 0))


def entropy_of(keys):
    """Miller-Madow corrected entropy (nats) of the empirical distribution of keys."""
    _, cnt = np.unique(keys, return_counts=True); n = cnt.sum(); p = cnt / n
    return float(-(p * np.log(p)).sum() + (len(cnt) - 1) / (2 * n))


def mrmr(codes, cards, y):
    F = len(codes); H = [entropy_of(c) for c in codes]; Hy = entropy_of(y)
    rel = np.array([H[i] + Hy - entropy_of(codes[i] * 2 + y) for i in range(F)])
    red_sum = np.zeros(F); order = [int(np.argmax(rel))]; left = [i for i in range(F) if i != order[0]]
    while left:
        s = order[-1]
        for i in left:                              # add I(i; last selected) to the running redundancy sum
            red_sum[i] += H[i] + H[s] - entropy_of(codes[i] * cards[s] + codes[s])
        score = [rel[i] - red_sum[i] / len(order) for i in left]
        j = left[int(np.argmax(score))]; order.append(j); left.remove(j)
    return order, rel


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="plan.json"); ap.add_argument("--ds", required=True)
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=2_000_000); ap.add_argument("--min_count", type=int, default=10)
    a = ap.parse_args(); t0 = time.time()
    C = json.load(open(a.config)); X, y, fields, pool = load_train(C, a.ds)
    rng = np.random.default_rng(a.seed)
    sel = np.sort(rng.choice(pool, a.rows, replace=False)) if len(pool) > a.rows else pool
    Xs = np.asarray(X[sel]); ys = np.asarray(y[sel]).astype(np.int64)
    codes, cards = zip(*[recode(Xs[:, i], a.min_count) for i in range(len(fields))])
    order, rel = mrmr(list(codes), list(cards), ys)
    ranking = [fields[i] for i in order]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f: f.write(",".join(ranking) + "\n")
    json.dump(dict(method="mRMR (MID, Miller-Madow MI)", ds=a.ds, seed=a.seed, rows=int(len(sel)), min_count=a.min_count,
                   ranking=ranking, relevance_nats={fields[i]: float(rel[i]) for i in range(len(fields))},
                   recoded_cardinality={fields[i]: int(cards[i]) for i in range(len(fields))},
                   seconds=time.time() - t0), open(a.out[:-4] + ".json", "w"), indent=1)
    print(f"[mrmr] {a.ds} seed {a.seed}: {len(sel)} rows, {time.time()-t0:.0f}s\n[mrmr] ranking: {','.join(ranking)}")
