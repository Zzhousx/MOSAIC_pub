"""MovieLens-1M with 20 fields: ERASE's 9 fields + 11 derived fields with realistic redundancy/synergy.
Same shuffle/split as ERASE (random_state=43, 70/20/10). Count-based fields use the TRAIN split only."""
import sys, re, numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder
src, out = sys.argv[1], sys.argv[2]
rd = lambda f, cols: pd.read_csv(f"{src}/{f}", sep="::", engine="python", names=cols, encoding="latin-1")
r = rd("ratings.dat", ["user_id", "movie_id", "rating", "timestamp"])
u = rd("users.dat", ["user_id", "gender", "age", "occupation", "zip"])
m = rd("movies.dat", ["movie_id", "title", "genres"])
df = r.merge(m, on="movie_id").merge(u, on="user_id")
base = ["user_id", "movie_id", "timestamp", "title", "genres", "gender", "age", "occupation", "zip"]
df = df[base + ["rating"]].copy()
df["rating"] = (df["rating"] > 3).astype(np.int8)
df = df.sample(frac=1, random_state=43).reset_index(drop=True)
n = len(df); a, b = int(n * 0.7), int(n * 0.9)
yr = df["title"].str.extract(r"\((\d{4})\)\s*$")[0].fillna("0").astype(int)
df["year"] = yr
df["decade"] = (yr // 10) * 10
df["first_genre"] = df["genres"].str.split("|").str[0]
df["n_genres"] = df["genres"].str.count(r"\|") + 1
z = df["zip"].astype(str)
df["zip3"] = z.str[:3]
df["zip1"] = z.str[:1]
ts = pd.to_datetime(df["timestamp"], unit="s")
df["hour"] = ts.dt.hour
df["weekday"] = ts.dt.weekday
tr = df.iloc[:a]
ucnt = tr["user_id"].value_counts(); mcnt = tr["movie_id"].value_counts()
df["user_activity"] = np.floor(np.log2(df["user_id"].map(ucnt).fillna(0) + 1)).astype(int)
df["item_popularity"] = np.floor(np.log2(df["movie_id"].map(mcnt).fillna(0) + 1)).astype(int)
df["age_gender"] = df["age"].astype(str) + "_" + df["gender"]
fields = base + ["year", "decade", "first_genre", "n_genres", "zip3", "zip1", "hour", "weekday",
                 "user_activity", "item_popularity", "age_gender"]
for f in fields: df[f] = LabelEncoder().fit_transform(df[f].astype(str))
X = df[fields].values.astype(np.int64); y = df["rating"].values.astype(np.float32)
card = np.array([df[f].max() + 1 for f in fields])
np.savez_compressed(out, Xtr=X[:a], ytr=y[:a], Xva=X[a:b], yva=y[a:b], Xte=X[b:], yte=y[b:], card=card, fields=np.array(fields))
print(len(fields), "fields; cardinalities:", dict(zip(fields, card.tolist())))
