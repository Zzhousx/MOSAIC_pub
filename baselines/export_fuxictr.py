"""Export the exact MOSAIC data split and encoding to author FairFS/FuxiCTR CSV/configuration."""
import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--npz"); ap.add_argument("--cache", help="mosaic_cache folder with meta.json and X*/y*.npy (Avazu, Criteo)")
ap.add_argument("--name", required=True); ap.add_argument("--fuxictr", required=True, help="FuxiCTR repository root")
ap.add_argument("--seeds", default="2027,2028,2029")
ap.add_argument("--min_count", type=int, default=1, help="FuxiCTR min_categr_count (1 = keep our vocabulary exactly)")
ap.add_argument("--emb_dim", type=int, default=32, help="FairFS DeepFM template value; lower only if out of GPU memory")
ap.add_argument("--batch_size", type=int, default=10000, help="FairFS DeepFM template value")
ap.add_argument("--chunk", type=int, default=2_000_000)
ap.add_argument("--model", default="DCN", choices=["DCN"], help="FairFS model whose config receives the entries")
ap.add_argument("--config_only", action="store_true", help="data already exported: only append the config entries")
ap.add_argument("--epochs", type=int, default=None, help="default: the FairFS template value (DeepFM 100, DCN 10)")
a = ap.parse_args()

if a.npz:
    z = np.load(a.npz); D = {k: z[k] for k in ["Xtr", "ytr", "Xva", "yva", "Xte", "yte"]}; fields = [str(f) for f in z["fields"]]
else:
    M = json.load(open(os.path.join(a.cache, "meta.json"))); fields = [str(f) for f in M["fields"]]
    D = {k: np.load(os.path.join(a.cache, k + ".npy"), mmap_mode="r") for k in ["Xtr", "ytr", "Xva", "yva", "Xte", "yte"]}
cols = [f"f_{f}" if f[:1].isdigit() else f for f in fields]       # FuxiCTR/YAML-safe names; mapped back in ext_eval
ds_id = f"mosaic_{a.name}"; out = os.path.join(a.fuxictr, "data", ds_id); os.makedirs(out, exist_ok=True)
for split, X, y in ([] if a.config_only else [("train", "Xtr", "ytr"), ("valid", "Xva", "yva"), ("test", "Xte", "yte")]):
    path = os.path.join(out, f"{split}.csv")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(",".join(["label"] + cols) + "\n")
        n = len(D[y])
        for i in range(0, n, a.chunk):
            Xi = np.asarray(D[X][i:i + a.chunk]); yi = np.asarray(D[y][i:i + a.chunk]).astype(int)
            block = np.column_stack([yi, Xi]).astype(np.int64)
            np.savetxt(fh, block, fmt="%d", delimiter=",")
    print(f"wrote {path}: {len(D[y])} rows, {len(fields)} fields")

cfg = os.path.join(a.fuxictr, "model_zoo", a.model, f"{a.model}_torch", "config")
if not os.path.isdir(cfg): raise SystemExit(f"{cfg} not found: copy FairFS's DCN folder into FuxiCTR/model_zoo first")
dsy = (f"\n{ds_id}:\n    data_root: ../../../data/\n    data_format: csv\n"
       f"    train_data: ../../../data/{ds_id}/train.csv\n    valid_data: ../../../data/{ds_id}/valid.csv\n"
       f"    test_data: ../../../data/{ds_id}/test.csv\n    min_categr_count: {a.min_count}\n"
       f"    feature_cols:\n        - {{active: true, dtype: str, type: categorical, name: [{', '.join(cols)}]}}\n"
       f"    label_col: {{name: label, dtype: float}}\n")
with open(os.path.join(cfg, "dataset_config.yaml"), "a", encoding="utf-8") as fh: fh.write(dsy)
for s in a.seeds.split(","):
    if a.model == "DCN":     # FairFS's DCN_criteo_pre / DCN_avazu_pre template; only dataset_id, seed, embedding_dim differ
        my = (f"\nFairFS_DCN_{a.name}_s{s}:\n    model: DCN\n    dataset_id: {ds_id}\n    loss: 'binary_crossentropy'\n"
              f"    metrics: ['logloss', 'AUC']\n    task: binary_classification\n    optimizer: adam\n"
              f"    learning_rate: 1.0e-3\n    embedding_regularizer: 0\n    net_regularizer: 0\n"
              f"    batch_size: {a.batch_size}\n    embedding_dim: {a.emb_dim}\n    dnn_hidden_units: [1024, 512, 256]\n"
              f"    dnn_activations: relu\n    num_cross_layers: 3\n    net_dropout: 0\n    batch_norm: False\n"
              f"    epochs: {a.epochs or 10}\n    shuffle: True\n    seed: {s}\n    monitor: {{'AUC': 1, 'logloss': -1}}\n"
              f"    monitor_mode: 'max'\n")
        with open(os.path.join(cfg, "model_config.yaml"), "a", encoding="utf-8") as fh: fh.write(my)
        continue
if not a.config_only: json.dump(dict(fields=fields, columns=cols), open(os.path.join(out, "field_names.json"), "w"))
print(f"appended dataset {ds_id} and models FairFS_{a.model}_{a.name}_s{{{a.seeds}}} to {cfg}")
