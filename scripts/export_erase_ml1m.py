"""Write a MovieLens .npz (from ../cpu/prep_ml1m9.py or prep_ml1m20.py) as ERASE's movielens-1m CSV so that ERASE's
methods (fs_run.py --dataset movielens-1m) see EXACTLY our fields and our train/val/test split.

ERASE's reader shuffles with df.sample(frac=1, random_state=43) and splits 70/20/10. Our .npz rows are already in
that shuffled order, so the CSV is written in the inverse order; ERASE's shuffle then reproduces our split.
The label is written as rating 5 (positive) / 1 (negative) because ERASE recomputes label = rating > 3.
Feature columns hold our integer codes (ERASE's LabelEncoder maps them to themselves).
The script re-reads the CSV exactly as ERASE does and asserts equality with the .npz.

Usage: python gpu/make_erase_ml1m.py --npz data/ml1m20.npz --out erase_data_ml1m20
       -> erase_data_ml1m20/movielens-1m/ml-1m.csv ; then run ERASE with --data_path <abs path>/erase_data_ml1m20/"""
import argparse, os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

ap = argparse.ArgumentParser(); ap.add_argument("--npz", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
z = np.load(a.npz); fields = [str(f) for f in z["fields"]]
X = np.vstack([z["Xtr"], z["Xva"], z["Xte"]]); y = np.concatenate([z["ytr"], z["yva"], z["yte"]])
n = len(X); perm = np.random.RandomState(43).permutation(n)        # = the order produced by df.sample(frac=1, random_state=43)
Xo = np.empty_like(X); Xo[perm] = X; yo = np.empty_like(y); yo[perm] = y
df = pd.DataFrame(Xo, columns=fields); df["rating"] = np.where(yo > 0.5, 5, 1)
os.makedirs(os.path.join(a.out, "movielens-1m"), exist_ok=True)
path = os.path.join(a.out, "movielens-1m", "ml-1m.csv"); df.to_csv(path, index=False)

# verification: ERASE's read_movielens1m_ml, line by line
d2 = pd.read_csv(path)
d2["rating"] = d2["rating"].apply(lambda x: 1 if x > 3 else 0)
d2 = d2.sample(frac=1, random_state=43)
tr, va = int(d2.shape[0] * 0.7), int(d2.shape[0] * 0.9)
feats = [f for f in d2.columns if f not in ["rating"]]
for f in feats: d2[f] = LabelEncoder().fit_transform(d2[f])
assert feats == fields, (feats, fields)
assert [int(d2[c].max()) + 1 for c in feats] == [int(c) for c in z["card"]], "cardinalities differ"
for (A, B) in [(d2[feats].values[:tr], z["Xtr"]), (d2[feats].values[tr:va], z["Xva"]), (d2[feats].values[va:], z["Xte"]),
               (d2["rating"].values[:tr], z["ytr"]), (d2["rating"].values[tr:va], z["yva"]), (d2["rating"].values[va:], z["yte"])]:
    assert A.shape[0] == B.shape[0] and np.array_equal(A.astype(np.int64), B.astype(np.int64)), "split mismatch"
print(f"wrote {path}: {n} rows, {len(fields)} fields; verified identical to {a.npz} under ERASE's reader")
