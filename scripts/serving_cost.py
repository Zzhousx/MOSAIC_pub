"""Exact vocabulary memory and random-weight forward latency for the paper field sets."""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mosaic"))
from mosaic_gpu import read_source, CTRModel


@torch.no_grad()
def time_forward(model, card, batch, device, iters, warmup=10):
    gen = torch.Generator().manual_seed(0)
    X = torch.stack([torch.randint(0, int(c), (batch,), generator=gen) for c in card], 1).to(device)
    model.eval()
    for _ in range(warmup): model(X)
    if device.startswith("cuda"): torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters): model(X)
    if device.startswith("cuda"): torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3          # ms per batch


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz"); ap.add_argument("--erase_root"); ap.add_argument("--dataset"); ap.add_argument("--data_path", default="data/")
    ap.add_argument("--budgets", required=True); ap.add_argument("--dim", type=int, default=8)
    ap.add_argument("--results", default="results"); ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True, help="dataset key used in result file names (ml1m20, avazu, criteo)")
    ap.add_argument("--gpu_iters", type=int, default=200); ap.add_argument("--cpu_iters", type=int, default=200)
    a, _ = ap.parse_known_args()                            # other dataset args from run_plan are ignored
    if a.npz is None: a.batch_size = 4096                   # read_source needs it only for ERASE's reader (first read)
    d, card, fields = read_source(a); card = np.asarray(card, dtype=np.int64); F = len(fields); idx = {f: i for i, f in enumerate(fields)}
    run = json.load(open(os.path.join(a.results, f"{a.name}_deepfm_s0_v2.json")))
    sets = {int(k): v for k, v in run["evals"]["MOSAIC"]["subsets"].items()}; sets[F] = list(fields)
    gpu = "cuda" if torch.cuda.is_available() else None
    out = dict(kind="serving_cost", ds=a.name, dim=a.dim, fields=fields, card=card.tolist(),
               bytes_per_value=(a.dim + 1) * 4, gpu=torch.cuda.get_device_name(0) if gpu else None,
               cpu_threads=1, latency={})
    for k in sorted(sets):
        cols = [idx[f] for f in sets[k]]; c = card[cols]; row = dict(fields=sets[k], emb_mb=float(c.sum() * (a.dim + 1) * 4 / 1e6))
        for kind in ["deepfm", "dcnv2"]:
            torch.set_num_threads(1)
            m = CTRModel(c, kind=kind, dim=a.dim)
            row[f"{kind}_cpu_b1_ms"] = time_forward(m, c, 1, "cpu", a.cpu_iters)
            row[f"{kind}_cpu_b256_ms"] = time_forward(m, c, 256, "cpu", max(20, a.cpu_iters // 4))
            if gpu:
                m = m.to(gpu); row[f"{kind}_gpu_b4096_ms"] = time_forward(m, c, 4096, gpu, a.gpu_iters); del m; torch.cuda.empty_cache()
        out["latency"][str(k)] = row
        print(f"[serving] {a.name} k={k:3d}: emb {row['emb_mb']:9.1f} MB | " +
              " | ".join(f"{key} {v:.3f}" for key, v in row.items() if key.endswith("_ms")), flush=True)
    json.dump(out, open(a.out, "w"), indent=1); print(f"wrote {a.out}")
