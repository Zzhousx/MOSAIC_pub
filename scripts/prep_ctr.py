"""Low-memory replacement for ERASE's data/preprocess.py (same output format, same encoding rules).
ERASE's script converts the whole table to Python strings at once (~50-60 GB RAM for Avazu, more for Criteo).
This version streams the input twice in chunks (peak RAM ~3-8 GB) and writes the CSV that ERASE's reader expects.

Encoding (identical to ERASE): every column, including the label, is label-encoded on its string value
(sorted order, like sklearn's LabelEncoder). Avazu: column 'id' dropped, all other columns kept in file order.
Criteo: columns named '0'..'39' ('0' = label, '1'..'13' dense, '14'..'39' categorical); missing categorical ->
'-999'; missing dense -> -999; dense x -> str(int(log(x)^2)) if x > 2 else str(int(x) - 2).
Row order = concatenation of the input files in the order given (ERASE splits 70/20/10 in that order).

Usage
  python scripts/prep_ctr.py avazu --inputs train.csv --out third_party/ERASE/data/avazu/preprocessed_avazu.csv
  python scripts/prep_ctr.py criteo --inputs train.txt --out third_party/ERASE/data/criteo/preprocessed_criteo.csv"""
import argparse, os, time
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("dataset", choices=["avazu", "criteo"])
ap.add_argument("--inputs", nargs="+", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--format", default="raw", choices=["raw", "x4"], help="criteo only: raw Kaggle train.txt or reczoo Criteo_x4 csv")
ap.add_argument("--chunksize", type=int, default=2_000_000)
a = ap.parse_args(); T0 = time.time()
DENSE = [str(i) for i in range(1, 14)]; SPARSE = [str(i) for i in range(14, 40)]


def dense_transform(x):
    x = x.astype(np.float64).fillna(-999.0).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        big = np.trunc(np.log(np.where(x > 2, x, 3.0)) ** 2)
    out = np.where(x > 2, big, np.trunc(x) - 2).astype(np.int64)
    return pd.Series(out.astype(str))


def chunks():
    for path in a.inputs:
        if a.dataset == "avazu":
            it = pd.read_csv(path, dtype=str, keep_default_na=False, chunksize=a.chunksize)
            for c in it: yield c.drop(columns=["id"]) if "id" in c.columns else c
        elif a.format == "raw":
            dt = {0: np.int64, **{i: np.float64 for i in range(1, 14)}, **{i: str for i in range(14, 40)}}
            for c in pd.read_csv(path, sep="\t", header=None, dtype=dt, chunksize=a.chunksize):
                c.columns = [str(i) for i in range(40)]; yield fix_criteo(c)
        else:
            for c in pd.read_csv(path, chunksize=a.chunksize, dtype=str, keep_default_na=True):
                assert len(c.columns) == 40, f"expected 40 columns (Label, I1-I13, C1-C26), got {list(c.columns)}"
                c.columns = [str(i) for i in range(40)]
                c["0"] = c["0"].astype(float).astype(np.int64)
                for col in DENSE: c[col] = pd.to_numeric(c[col], errors="raise")
                yield fix_criteo(c)


def fix_criteo(c):
    out = pd.DataFrame({"0": c["0"].astype(np.int64).astype(str).to_numpy()})
    for col in DENSE: out[col] = dense_transform(c[col]).to_numpy()
    for col in SPARSE: out[col] = c[col].fillna("-999").astype(str).to_numpy()
    return out


# pass 1: vocabularies
uniq, cols, n = {}, None, 0
for c in chunks():
    if cols is None: cols = list(c.columns); uniq = {k: set() for k in cols}
    for k in cols: uniq[k].update(pd.unique(c[k].to_numpy()))
    n += len(c); print(f"[pass 1] {n:,} rows, {(time.time()-T0)/60:.1f} min", flush=True)
index = {k: pd.Index(sorted(uniq[k])) for k in cols}; del uniq
print("vocabulary sizes:", {k: len(v) for k, v in index.items()}, flush=True)
# pass 2: encode and write
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); tmp = a.out + ".partial"; first, m = True, 0
for c in chunks():
    enc = {}
    for k in cols:
        codes = index[k].get_indexer(c[k].to_numpy())
        assert (codes >= 0).all(), f"unseen value in column {k}"
        enc[k] = codes.astype(np.int32)
    pd.DataFrame(enc, columns=cols).to_csv(tmp, mode="w" if first else "a", header=first, index=False)
    first = False; m += len(c); print(f"[pass 2] {m:,}/{n:,} rows written, {(time.time()-T0)/60:.1f} min", flush=True)
os.replace(tmp, a.out); print(f"done: {a.out} ({m:,} rows, {len(cols)} columns) in {(time.time()-T0)/60:.1f} min")
