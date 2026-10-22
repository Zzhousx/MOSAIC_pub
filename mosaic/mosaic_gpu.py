"""MOSAIC: publication implementation for field selection, controls and paper experiments."""
import argparse,json,math,os,sys,time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from mosaic_lib import (auc_np,logloss_logits,bits,to_mask,select_subsets,RetrainCache,budget_surrogate,KERNEL_SIGMA,first_order_subset,ladder_plan,LADDER,RUNG_AGNOSTIC,RUNG_LOCAL,RUNG_V1,RUNG_PAIR,RUNG_FULL)
T0=time.time()
ARRS=['Xtr','ytr','Xva','yva','Xte','yte']
def log(msg):
    print(f"[{(time.time() - T0) / 60:7.1f} min] {msg}", flush=True)

def save_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f: json.dump(obj, f, indent=1)
    os.replace(tmp, path)

def read_source(args):
    if args.npz:
        z = np.load(args.npz)
        return {k: z[k] for k in ARRS}, [int(c) for c in z["card"]], [str(f) for f in z["fields"]]
    cdir = os.path.join(args.data_path, args.dataset, "mosaic_cache"); meta = os.path.join(cdir, "meta.json")
    if os.path.exists(meta):
        M = json.load(open(meta)); log(f"loading cached arrays from {cdir}")
        return {k: np.load(os.path.join(cdir, k + ".npy")) for k in ARRS}, M["card"], M["fields"]
    sys.path.insert(0, os.path.abspath(args.erase_root))
    from utils.datasets import read_dataset                    # ERASE
    log("reading with ERASE's reader (first time only; arrays are cached afterwards)")
    feats, uniq, (trx, tr_y, vax, va_y, tex, te_y) = read_dataset(
        args.dataset, args.data_path, args.batch_size, True, 0, machine_learning_method=True)
    d = dict(Xtr=trx.to_numpy(dtype=np.int32), ytr=tr_y.to_numpy(dtype=np.float32),
             Xva=vax.to_numpy(dtype=np.int32), yva=va_y.to_numpy(dtype=np.float32),
             Xte=tex.to_numpy(dtype=np.int32), yte=te_y.to_numpy(dtype=np.float32))
    card, fields = [int(u) for u in uniq], [str(f) for f in feats]
    os.makedirs(cdir, exist_ok=True)
    for k in ARRS: np.save(os.path.join(cdir, k + ".npy"), d[k])
    save_json(dict(card=card, fields=fields), meta)
    return d, card, fields

def load_data(args):
    d, card, fields = read_source(args)
    for k in ["Xtr", "Xva", "Xte"]: d[k] = np.ascontiguousarray(d[k], dtype=np.int32)
    for k in ["ytr", "yva", "yte"]: d[k] = np.ascontiguousarray(d[k], dtype=np.float32)
    if args.smoke:
        for k, n in [("Xtr", 50000), ("ytr", 50000), ("Xva", 20000), ("yva", 20000), ("Xte", 20000), ("yte", 20000)]:
            d[k] = d[k][:n]
    if args.max_train_rows and len(d["Xtr"]) > args.max_train_rows:      # one fixed subsample for EVERYTHING
        sel = np.sort(np.random.default_rng(2027).choice(len(d["Xtr"]), args.max_train_rows, replace=False))
        d["Xtr"], d["ytr"] = d["Xtr"][sel], d["ytr"][sel]
    if args.max_eval_rows:
        for X, y in [("Xva", "yva"), ("Xte", "yte")]:
            if len(d[X]) > args.max_eval_rows:
                sel = np.sort(np.random.default_rng(2028).choice(len(d[X]), args.max_eval_rows, replace=False))
                d[X], d[y] = d[X][sel], d[y][sel]
    card = np.array(card, dtype=np.int64); F = len(card)
    for k in ["Xtr", "Xva", "Xte"]:
        assert d[k].shape[1] == F, (k, d[k].shape, F)
        assert d[k].min() >= 0 and (d[k].max(0) < card).all(), f"{k}: feature ids outside [0, card)"
    for k in ["ytr", "yva", "yte"]:
        assert set(np.unique(d[k]).tolist()) <= {0.0, 1.0}, f"{k}: labels must be 0/1"
    d["card"], d["fields"] = card, fields
    on_gpu = False
    if str(args.device).startswith("cuda"):
        total = torch.cuda.get_device_properties(torch.device(args.device)).total_memory
        model_bytes = (int(card.sum()) + F) * (args.dim + 1) * 4 * 5      # params, grads, 2 Adam states, margin
        data_bytes = sum(d[k].nbytes for k in ARRS)
        on_gpu = {"auto": model_bytes + data_bytes + 2.5e9 < total, "1": True, "0": False}[args.data_on_gpu]
    dev = args.device if on_gpu else "cpu"
    for k in ARRS: d[k + "_t"] = torch.from_numpy(d[k]).to(dev)
    log(f"[data] train {len(d['Xtr'])} val {len(d['Xva'])} test {len(d['Xte'])} | {F} fields | "
        f"{int(card.sum())} feature values | pos rate {d['ytr'].mean():.4f} | data tensors on {dev}")
    return d

class CTRModel(nn.Module):
    """Embedding (+ optional learned per-field 'absent' token) -> FM + DNN (deepfm) or Cross-v2 + DNN (dcnv2)."""
    def __init__(self, card, kind="deepfm", dim=8, absent=False, hidden=(400, 400, 400), cross_layers=3, sparse=False):
        super().__init__()
        card = torch.as_tensor(np.asarray(card), dtype=torch.long); self.F = len(card); self.kind = kind
        sizes = card + (1 if absent else 0)
        self.register_buffer("offsets", torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(sizes, 0)[:-1]]))
        self.register_buffer("absent_idx", self.offsets + card)            # used only when absent=True
        V = int(sizes.sum())
        self.emb = nn.Embedding(V, dim, sparse=sparse); nn.init.normal_(self.emb.weight, 0, 0.01)
        self.lin = nn.Embedding(V, 1, sparse=sparse); nn.init.zeros_(self.lin.weight); self.bias = nn.Parameter(torch.zeros(1))
        din = self.F * dim; layers, h = [], din
        for u in hidden: layers += [nn.Linear(h, u), nn.ReLU()]; h = u
        self.dnn = nn.Sequential(*layers); self.dnn_out = nn.Linear(h, 1)
        if kind == "dcnv2":
            self.cross = nn.ModuleList([nn.Linear(din, din) for _ in range(cross_layers)])
            self.cross_out = nn.Linear(din, 1)

    def forward(self, x, mask=None):
        """x: (B, F) long feature ids; mask: (B, F) bool, False = field absent. Returns logits (B,)."""
        idx = x + self.offsets
        if mask is not None: idx = torch.where(mask, idx, self.absent_idx.expand_as(idx))
        e = self.emb(idx); B = e.shape[0]; h0 = e.reshape(B, -1)
        out = self.bias + self.lin(idx).sum(dim=(1, 2))
        if self.kind == "deepfm":
            s = e.sum(1); out = out + 0.5 * (s.pow(2) - e.pow(2).sum(1)).sum(1)
        else:
            xl = h0
            for layer in self.cross: xl = h0 * layer(xl) + xl
            out = out + self.cross_out(xl).squeeze(1)
        return out + self.dnn_out(self.dnn(h0)).squeeze(1)

def make(card, args, absent=False):
    return CTRModel(card, kind=args.model, dim=args.dim, absent=absent, sparse=args.sparse_emb).to(args.device)

class Optim:
    """Adam on all parameters, or (--sparse_emb) SparseAdam ("lazy Adam") on the embedding tables + Adam on the rest.
    Sparse updates make a step cost independent of the vocabulary size (Criteo has ~34M feature values)."""
    def __init__(self, model, args):
        if not args.sparse_emb: self.opts = [torch.optim.Adam(model.parameters(), lr=args.lr)]; return
        sp = [model.emb.weight, model.lin.weight]; ids = {id(p) for p in sp}
        self.opts = [torch.optim.SparseAdam(sp, lr=args.lr),
                     torch.optim.Adam([p for p in model.parameters() if id(p) not in ids], lr=args.lr)]

    def step(self, loss):
        for o in self.opts: o.zero_grad(set_to_none=True)
        loss.backward()
        for o in self.opts: o.step()

@torch.no_grad()
def logits(model, X, args, cols=None, mrow=None):
    """Logits (float64 numpy) for tensor X (any device); cols = field subset of X; mrow = (F,) 0/1 field mask."""
    model.eval(); out = []
    cols_t = None if cols is None else torch.as_tensor(cols, dtype=torch.long, device=X.device)
    mask = None if mrow is None else torch.as_tensor(np.asarray(mrow, dtype=bool), device=args.device)
    for i in range(0, len(X), args.chunk):
        xb = X[i:i + args.chunk]
        if cols_t is not None: xb = xb.index_select(1, cols_t)
        xb = xb.to(args.device, non_blocking=True).long()
        out.append(model(xb, None if mask is None else mask.expand(xb.shape[0], -1)).double().cpu())
    return torch.cat(out).numpy()

def batch_indices(n, bs, gen, device, pool=None):
    perm = torch.randperm(n, generator=gen)
    if pool is not None: perm = pool[perm]
    perm = perm.to(device)
    for i in range(0, n, bs): yield perm[i:i + bs]

def snapshot(model):
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}

def train_subset(d, cols, args, seed, search=False):
    """Fresh model on field subset `cols` (Adam, early stopping on validation logloss, patience 1).
    search=True: cheap search retrain (--search_rows fixed random rows, --search_epochs epochs).
    Returns (model restored to its best epoch, validation logits of that epoch)."""
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    model = make(d["card"][cols], args); opt = Optim(model, args)
    X, y = d["Xtr_t"], d["ytr_t"]; pool = d.get("search_pool") if search else None
    n = len(X) if pool is None else len(pool); cols_t = torch.as_tensor(cols, dtype=torch.long, device=X.device)
    best, best_state, best_vz = math.inf, None, None
    for ep in range(args.search_epochs if search else args.epochs):
        model.train()
        for b in batch_indices(n, args.batch_size, gen, X.device, pool):
            xb = X.index_select(0, b).index_select(1, cols_t).to(args.device, non_blocking=True).long()
            yb = y.index_select(0, b).to(args.device, non_blocking=True)
            opt.step(Fn.binary_cross_entropy_with_logits(model(xb), yb))
        vz = logits(model, d["Xva_t"], args, cols=cols); ll = logloss_logits(d["yva"], vz)
        if not np.isfinite(ll):
            log(f"WARNING: non-finite validation loss (fields {cols}); keeping the best earlier epoch"); break
        if ll < best - 1e-5: best, best_state, best_vz = ll, snapshot(model), vz
        else: break
    if best_state is not None: model.load_state_dict(best_state)
    else: best_vz = logits(model, d["Xva_t"], args, cols=cols)
    return model, best_vz

@torch.no_grad()
def masked_loss(model, X, y, mask, args):
    model.eval(); tot = 0.0
    for i in range(0, len(X), args.chunk):
        tot += float(Fn.binary_cross_entropy_with_logits(model(X[i:i + args.chunk], mask[i:i + args.chunk]),
                                                         y[i:i + args.chunk], reduction="sum"))
    return tot / len(X)

def train_shared(d, args):
    """Shared model trained with per-sample Bernoulli field masks; the inclusion probability of each batch is drawn
    from the density grid. Epoch rule (as on CPU): train --shared_min_epochs epochs, then continue while the masked
    validation loss (fixed p=0.5 masks) improves; on the first non-improving epoch restore the previous state."""
    seed = args.seed + 5000; torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    model = make(d["card"], args, absent=True); opt = Optim(model, args)
    X, y = d["Xtr_t"], d["ytr_t"]; Xs, ys = d["est_X"], d["est_y"]
    vmask = (torch.rand(Xs.shape, generator=torch.Generator().manual_seed(12345)) < 0.5).to(args.device)
    best, used = math.inf, 0
    for ep in range(args.shared_max_epochs):
        prev = snapshot(model); model.train()
        for b in batch_indices(len(X), args.batch_size, gen, X.device):
            xb = X.index_select(0, b).to(args.device, non_blocking=True).long()
            yb = y.index_select(0, b).to(args.device, non_blocking=True)
            p = args.grid[int(torch.randint(len(args.grid), (1,), generator=gen))]
            mask = torch.rand(xb.shape, device=args.device) < p
            opt.step(Fn.binary_cross_entropy_with_logits(model(xb, mask), yb))
        used += 1; vl = masked_loss(model, Xs, ys, vmask, args)
        log(f"[shared] epoch {ep + 1}: masked validation logloss {vl:.5f}")
        if ep + 1 <= args.shared_min_epochs or vl < best - 1e-5: best = min(best, vl)
        else:
            model.load_state_dict(prev); used -= 1; log(f"[shared] stop; keeping epoch {used}"); break
    return model, used

@torch.no_grad()
def mask_moments(model, d, args):
    """Discrete-Stein mask moments with a paired control variate (forward passes only):
    g_i = -E[(l - l') z_i], H_ij = -E[(l - l') z_i z_j], z = (m - p) / (p (1 - p)); split-half estimates kept."""
    model.eval(); F = len(d["card"]); dev = args.device; Xv, yv = d["est_X"], d["est_y"]; est = {}
    for p in args.grid:
        G = torch.zeros(2, F, dtype=torch.float64, device=dev); H = torch.zeros(2, F, F, dtype=torch.float64, device=dev)
        cnt = [0, 0]
        for r in range(args.passes):
            h = r % 2
            for i in range(0, len(Xv), args.chunk):
                xb, yb = Xv[i:i + args.chunk], yv[i:i + args.chunk]
                m1 = torch.rand(xb.shape, device=dev) < p; m2 = torch.rand(xb.shape, device=dev) < p
                diff = (Fn.binary_cross_entropy_with_logits(model(xb, m1), yb, reduction="none")
                        - Fn.binary_cross_entropy_with_logits(model(xb, m2), yb, reduction="none")).double()
                z = (m1.double() - p) / (p * (1 - p))
                G[h] -= (diff[:, None] * z).sum(0); H[h] -= (z * diff[:, None]).T @ z; cnt[h] += len(xb)
        for h in range(2):
            G[h] /= max(cnt[h], 1); H[h] /= max(cnt[h], 1); H[h].fill_diagonal_(0)
        est[str(p)] = dict(g=G.mean(0).cpu().tolist(), H=H.mean(0).cpu().tolist(),
                           g_half=G.cpu().tolist(), H_half=H.cpu().tolist())
    return est

def protocol(args):
    """Settings that define a final-evaluation retrain (fingerprint of the retrain cache)."""
    return dict(data=os.path.basename(args.npz) if args.npz else args.dataset, model=args.model, dim=args.dim,
                lr=args.lr, batch_size=args.batch_size, epochs=args.epochs, sparse_emb=bool(args.sparse_emb),
                max_train_rows=args.max_train_rows,
                max_eval_rows=args.max_eval_rows, smoke=bool(args.smoke))

def make_retrainer(d, args, cache):
    F = len(d["card"])
    def retrain(m, seed, search=False):
        key = f"{m}|{seed}|" + (f"s{args.search_rows}e{args.search_epochs}" if search else "full")
        r = cache.get(key)
        if r is None:
            cols = bits(m, F); t = time.time()
            model, vz = train_subset(d, cols, args, seed=seed * 100003 + m % 100003, search=search)
            tz = logits(model, d["Xte_t"], args, cols=cols)
            r = dict(val_auc=auc_np(d["yva"], vz), test_auc=auc_np(d["yte"], tz), test_ll=logloss_logits(d["yte"], tz),
                     sec=round(time.time() - t, 2))
            cache.put(key, r); del model
            log(f"[retrain] {'search' if search else 'final '} {len(cols):2d} fields seed {seed}: val AUC "
                f"{r['val_auc']:.4f} test AUC {r['test_auc']:.4f} ({r['sec']:.0f}s; cache {len(cache)})")
        return r
    return retrain

def setup_common(args):
    d = load_data(args); F = len(d["card"])
    if getattr(args, "search_rows", None) and args.search_rows < len(d["Xtr"]):
        d["search_pool"] = torch.from_numpy(np.random.default_rng(2029).choice(len(d["Xtr"]), args.search_rows, replace=False))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    extra = [g for g in (args.import_caches or "").split(",") if g]
    cache = RetrainCache(args.out + ".cache.json", protocol(args), extra_globs=extra)
    if cache.imported: log(f"imported {cache.imported} retrains from other caches with the same protocol")
    return d, F, cache, make_retrainer(d, args, cache)

def cmd_eval_ranking(args):
    d, F, cache, retrain = setup_common(args); fields = d["fields"]
    names = [f.strip() for f in args.ranking.split(",") if f.strip()]
    missing = [f for f in names if f not in fields]
    if missing: raise SystemExit(f"ranking contains unknown fields {missing}; dataset fields are {fields}")
    order = [fields.index(f) for f in names]; budgets = [int(x) for x in args.budgets.split(",")]
    assert len(order) >= max(budgets), "ranking shorter than the largest budget"
    eval_seeds = [int(s) for s in args.eval_seeds.split(",")]; per, subsets = {}, {}
    for k in budgets:
        m = to_mask(order[:k]); vals = [retrain(m, s)["test_auc"] for s in eval_seeds]
        per[str(k)] = float(np.mean(vals)); subsets[str(k)] = [fields[f] for f in bits(m, F)]
        log(f"[eval_ranking] {args.name} k={k}: test AUC {per[str(k)]:.4f}")
    out = dict(kind="eval_ranking", name=args.name, data=args.npz or args.dataset, model=args.model, budgets=budgets,
               test_auc=per, subsets=subsets, ranking=names, protocol=protocol(args))
    save_json(out, args.out)
    log(f"[done] wrote {args.out}")

def _setup_est(d, args, F, budgets):
    """Density grid, moment/verification rows and passes exactly as in cmd_run. Returns ye (labels of those rows)."""
    args.grid = [round(x, 4) for x in sorted({min(0.9, max(0.05, k / F)) for k in budgets} | {0.5})]
    ev = np.sort(np.random.default_rng(2030 + args.seed).choice(len(d["Xva"]), min(args.est_rows, len(d["Xva"])), replace=False))
    d["est_X"] = d["Xva_t"][torch.from_numpy(ev).to(d["Xva_t"].device)].to(args.device).long()
    d["est_y"] = d["yva_t"][torch.from_numpy(ev).to(d["yva_t"].device)].to(args.device)
    if not args.passes: args.passes = max(4, math.ceil(4e6 / len(ev)))
    return d["yva"][ev], ev

def cmd_costonly(args):
    """ Wall-clock cost of every MOSAIC stage on one dataset, nothing retrained or evaluated except ONE ordinary
    training of the same backbone on all fields with the same epoch limit. Budgets 'auto' = F x {1/8,1/4,3/8,1/2,5/8}."""
    d = load_data(args); F = len(d["card"]); rng_loc = np.random.default_rng(10_000 + args.seed)
    budgets = sorted({max(2, round(F * f)) for f in (0.125, 0.25, 0.375, 0.5, 0.625)}) if args.budgets == "auto" \
        else [int(x) for x in args.budgets.split(",")]
    ye, ev = _setup_est(d, args, F, budgets)
    log(f"[costonly] {F} fields, budgets {budgets}, {len(ev)} rows x {args.passes} passes, grid {args.grid}")
    sec = {}
    if str(args.device).startswith("cuda"): torch.cuda.synchronize()
    t = time.time(); sh, used = train_shared(d, args); sec["shared_train"] = time.time() - t
    t = time.time(); est = mask_moments(sh, d, args); sec["moments"] = time.time() - t
    t = time.time(); LOC = {k: select_subsets(*budget_surrogate(est, F, k), F, k, top=3, rng=rng_loc) for k in budgets}
    sec["selection"] = time.time() - t
    t = time.time()
    for k in budgets:
        for mm in LOC[k]: auc_np(ye, logits(sh, d["est_X"], args, mrow=[(mm >> f) & 1 for f in range(F)]))
    sec["verification"] = time.time() - t
    sec["mosaic_total"] = sec["shared_train"] + sec["moments"] + sec["selection"] + sec["verification"]
    del sh
    if str(args.device).startswith("cuda"): torch.cuda.empty_cache()
    t = time.time(); _ = train_subset(d, list(range(F)), args, seed=args.seed + 7000); sec["one_full_train"] = time.time() - t
    log(f"[costonly] F={F}: shared {sec['shared_train']:.0f}s ({used} ep), moments {sec['moments']:.0f}s, selection "
        f"{sec['selection']:.1f}s, verification {sec['verification']:.1f}s | one ordinary training {sec['one_full_train']:.0f}s")
    out = dict(kind="costonly", data=args.npz or args.dataset, model=args.model, F=F, budgets=budgets, card_total=int(d["card"].sum()),
               train_rows=len(d["Xtr"]), est_rows=len(ev), passes=args.passes, grid=args.grid, shared_epochs=used,
               epochs_limit=args.epochs, seconds=sec, protocol=protocol(args),
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    save_json(out, args.out); log(f"[done] wrote {args.out}")

def cmd_sens(args):
    """ Sensitivity of MOSAIC to its own settings, all on ONE shared model and the run's protocol:
    kernel width sigma in {0.1, 0.25*, 0.5, 1.0}; number of verified candidates in {1, 3*, 5}; share of the moment rows
    in {0.25, 0.5, 1*} (same passes, so fewer samples). * = default. Verification always uses all moment rows.
    Every selected set is retrained with --eval_seeds (cached retrains of the run are reused automatically)."""
    d, F, cache, retrain = setup_common(args); fields = d["fields"]
    budgets = [int(x) for x in args.budgets.split(",")]
    ye, ev = _setup_est(d, args, F, budgets)
    sh, used = train_shared(d, args)
    memo = {}
    def sh_auc(m):
        if m not in memo: memo[m] = auc_np(ye, logits(sh, d["est_X"], args, mrow=[(m >> f) & 1 for f in range(F)]))
        return memo[m]
    def pick(est, sigma, ncand):
        rng_loc = np.random.default_rng(10_000 + args.seed)
        return {k: max(select_subsets(*budget_surrogate(est, F, k, sigma=sigma), F, k, top=ncand, rng=rng_loc), key=sh_auc)
                for k in budgets}
    t = time.time(); est_full = mask_moments(sh, d, args); mom_sec = {"rows=1.0": time.time() - t}
    V = {"default": pick(est_full, KERNEL_SIGMA, 3)}
    for sg in (0.1, 0.5, 1.0): V[f"sigma={sg}"] = pick(est_full, sg, 3)
    for c in (1, 5): V[f"candidates={c}"] = pick(est_full, KERNEL_SIGMA, c)
    X0, y0 = d["est_X"], d["est_y"]
    for fr in (0.25, 0.5):
        sel = torch.from_numpy(np.sort(np.random.default_rng(3030 + args.seed).choice(len(ev), int(fr * len(ev)), replace=False))).to(X0.device)
        d["est_X"], d["est_y"] = X0[sel], y0[sel]
        t = time.time(); est_fr = mask_moments(sh, d, args); mom_sec[f"rows={fr}"] = time.time() - t
        d["est_X"], d["est_y"] = X0, y0
        V[f"rows={fr}"] = pick(est_fr, KERNEL_SIGMA, 3)
    del sh
    if str(args.device).startswith("cuda"): torch.cuda.empty_cache()
    eval_seeds = [int(s) for s in args.eval_seeds.split(",")]; res = {}
    for name, ch in V.items():
        per = {k: [retrain(ch[k], s)["test_auc"] for s in eval_seeds] for k in budgets}
        res[name] = dict(subsets={str(k): [fields[f] for f in bits(ch[k], F)] for k in budgets},
                         test_auc={str(k): float(np.mean(v)) for k, v in per.items()})
        log(f"[sens] {name:16s} mean test AUC {np.mean(list(res[name]['test_auc'].values())):.4f}")
    same_as_run = None
    if args.run_json and os.path.exists(args.run_json):
        R = json.load(open(args.run_json))
        same_as_run = sum(sorted(R["evals"]["MOSAIC"]["subsets"][str(k)]) == sorted(res["default"]["subsets"][str(k)]) for k in budgets)
    out = dict(kind="sens", data=args.npz or args.dataset, model=args.model, mseed=args.seed, budgets=budgets, fields=fields,
               variants=res, moment_seconds=mom_sec, shared_epochs=used, passes=args.passes, est_rows=len(ev),
               default_equals_run_budgets=same_as_run, protocol=protocol(args))
    save_json(out, args.out); log(f"[done] wrote {args.out}")

def _eval_rows(d, args, F, budgets):
    args.grid = [round(x, 4) for x in sorted({min(0.9, max(0.05, k / F)) for k in budgets} | {0.5})]
    ev = np.sort(np.random.default_rng(2030 + args.seed).choice(len(d["Xva"]), min(args.est_rows, len(d["Xva"])), replace=False))
    d["est_X"] = d["Xva_t"][torch.from_numpy(ev).to(d["Xva_t"].device)].to(args.device).long()
    d["est_y"] = d["yva_t"][torch.from_numpy(ev).to(d["yva_t"].device)].to(args.device)
    return d["yva"][ev]

def _final_eval(retrain, plan, budgets, eval_seeds, fields, F):
    """Retrain every subset of plan {name: {k: mask}} with each evaluation seed; summary as in `run`."""
    out = {}
    for name, ch in plan.items():
        rr = {k: [retrain(ch[k], s) for s in eval_seeds] for k in budgets}
        out[name] = dict(subsets={str(k): [fields[f] for f in bits(ch[k], F)] for k in budgets},
                         test_auc={str(k): float(np.mean([r["test_auc"] for r in v])) for k, v in rr.items()},
                         test_auc_std={str(k): float(np.std([r["test_auc"] for r in v])) for k, v in rr.items()},
                         val_auc={str(k): float(np.mean([r["val_auc"] for r in v])) for k, v in rr.items()})
        log(f"[eval] {name:44s} mean test AUC {np.mean(list(out[name]['test_auc'].values())):.4f}")
    return out

def cmd_ladder(args):
    """Mechanism ladder of a finished run. No model is trained except the final-evaluation retrains (shared cache,
    so subsets the run already evaluated are not retrained again)."""
    R = json.load(open(args.run_json))
    if R.get("version") != "v2" or "evals" not in R: raise SystemExit(f"{args.run_json} is not a finished run")
    if R["model"] != args.model or int(R["mseed"]) != args.seed:
        raise SystemExit(f"model/seed of the run ({R['model']}, {R['mseed']}) differ from --model/--seed ({args.model}, {args.seed})")
    d, F, cache, retrain = setup_common(args); fields = d["fields"]
    if list(R["fields"]) != list(fields): raise SystemExit("the field list of the run differs from the data")
    if R["protocol"] != protocol(args):
        raise SystemExit(f"retrain protocol differs from the run:\n  run: {R['protocol']}\n  now: {protocol(args)}")
    plan, same, nb = ladder_plan(R)
    log(f"[ladder] budget-local pairwise selection re-derived from the stored moments: {same}/{nb} budgets identical to the run")
    budgets = [int(k) for k in R["budgets"]]; eval_seeds = [int(s) for s in args.eval_seeds.split(",")]
    rungs = _final_eval(retrain, {n: plan[n] for n in LADDER}, budgets, eval_seeds, fields, F)
    save_json(dict(kind="ladder", version="v2", data=args.npz or args.dataset, model=args.model, mseed=args.seed,
                   budgets=budgets, fields=fields, protocol=protocol(args), source=os.path.basename(args.run_json),
                   rederived=[same, nb], rungs=rungs, all_fields_test_auc=R.get("all_fields_test_auc")), args.out)
    log(f"[done] wrote {args.out}")

def train_shared_scheme(d, args, scheme):
    """the shared model of `run` trained under another mask distribution. Same seed, initialisation, batch order and
    epoch rule as train_shared; only the per-batch keep probability p differs:
      budget  = the MOSAIC shared model (p drawn from the budget-density grid; calls train_shared unchanged)
      half    = p = 0.5 for every batch (plain field dropout)
      uniform = p ~ Uniform(0.05, 0.95) per batch (budget-agnostic supernet sampling)"""
    if scheme == "budget": return train_shared(d, args)
    seed = args.seed + 5000; torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    gen_u = torch.Generator().manual_seed(seed + 1)
    model = make(d["card"], args, absent=True); opt = Optim(model, args)
    X, y = d["Xtr_t"], d["ytr_t"]; Xs, ys = d["est_X"], d["est_y"]
    vmask = (torch.rand(Xs.shape, generator=torch.Generator().manual_seed(12345)) < 0.5).to(args.device)
    best, used = math.inf, 0
    for ep in range(args.shared_max_epochs):
        prev = snapshot(model); model.train()
        for b in batch_indices(len(X), args.batch_size, gen, X.device):
            xb = X.index_select(0, b).to(args.device, non_blocking=True).long()
            yb = y.index_select(0, b).to(args.device, non_blocking=True)
            torch.randint(len(args.grid), (1,), generator=gen)       # keeps `gen` (batch order) in step with train_shared
            p = 0.5 if scheme == "half" else float(0.05 + 0.9 * torch.rand(1, generator=gen_u))
            mask = torch.rand(xb.shape, device=args.device) < p
            opt.step(Fn.binary_cross_entropy_with_logits(model(xb, mask), yb))
        used += 1; vl = masked_loss(model, Xs, ys, vmask, args)
        log(f"[shared {scheme}] epoch {ep + 1}: masked validation logloss {vl:.5f}")
        if ep + 1 <= args.shared_min_epochs or vl < best - 1e-5: best = min(best, vl)
        else:
            model.load_state_dict(prev); used -= 1; log(f"[shared {scheme}] stop; keeping epoch {used}"); break
    return model, used

def cmd_trainablate(args):
    """shared model (--train_masks) -> moments -> budget-local surrogate top-3 -> verification (exactly as `run`),
    plus the two no-verification rungs; every selected subset retrained with the evaluation seeds (shared cache)."""
    d, F, cache, retrain = setup_common(args); fields = d["fields"]
    budgets = [int(x) for x in args.budgets.split(",")]; assert max(budgets) < F, "budgets must be < number of fields"
    ye = _eval_rows(d, args, F, budgets)                                 # same density grid + estimation rows as `run`
    if not args.passes: args.passes = max(4, math.ceil(4e6 / len(ye)))
    log(f"[trainablate {args.train_masks}] {F} fields, budgets {budgets}, grid {args.grid}, {len(ye)} rows x {args.passes} passes")
    sec = {}; t = time.time()
    sh, shared_epochs = train_shared_scheme(d, args, args.train_masks); sec["shared_train"] = time.time() - t; t = time.time()
    est = mask_moments(sh, d, args); sec["moments"] = time.time() - t; t = time.time()
    rng_loc = np.random.default_rng(10_000 + args.seed)
    LOC = {k: select_subsets(*budget_surrogate(est, F, k), F, k, top=3, rng=rng_loc) for k in budgets}
    sec["selection"] = time.time() - t; t = time.time()
    def sh_auc(m): return auc_np(ye, logits(sh, d["est_X"], args, mrow=[(m >> f) & 1 for f in range(F)]))
    cand_auc = {k: {mm: sh_auc(mm) for mm in LOC[k]} for k in budgets}
    MOSV = {k: max(LOC[k], key=lambda mm: cand_auc[k][mm]) for k in budgets}
    sec["verification"] = time.time() - t
    # diagnostic: how well the shared model serves random field subsets of each budget size (8 fixed draws per budget)
    drng = np.random.default_rng(6060)
    rand_auc = {str(k): float(np.mean([sh_auc(to_mask(drng.choice(F, k, replace=False))) for _ in range(8)])) for k in budgets}
    full_auc = sh_auc((1 << F) - 1); sh = None
    log(f"[trainablate {args.train_masks}] shared model {shared_epochs} epochs; masked val AUC of random size-k subsets "
        + ", ".join(f"k={k}: {v:.4f}" for k, v in rand_auc.items()) + f"; all fields {full_auc:.4f}")
    plan = {RUNG_FULL: MOSV, RUNG_PAIR: {k: LOC[k][0] for k in budgets}, RUNG_LOCAL: {k: first_order_subset(est, F, k) for k in budgets}}
    eval_seeds = [int(s) for s in args.eval_seeds.split(",")]
    variants = _final_eval(retrain, plan, budgets, eval_seeds, fields, F)
    save_json(dict(kind="trainablate", version="v2", scheme=args.train_masks, data=args.npz or args.dataset, model=args.model,
                   mseed=args.seed, budgets=budgets, fields=fields, grid=args.grid, passes=args.passes, shared_epochs=shared_epochs,
                   est_rows=len(ye), protocol=protocol(args), est=est, variants=variants,
                   shared_random_subset_val_auc=rand_auc, shared_all_fields_val_auc=full_auc,
                   mosaic_candidates={str(k): [dict(fields=[fields[f] for f in bits(mm, F)], shared_val_auc=cand_auc[k][mm])
                                               for mm in LOC[k]] for k in budgets}, seconds=sec), args.out)
    log(f"[done] wrote {args.out}")

def _spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean(); den = math.sqrt(float((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")

def cmd_eval_sets(args):
    """ Retrain named field sets with --eval_seeds. --sets "name=f1,f2,...;name2=f3,..." (field names)."""
    d, F, cache, retrain = setup_common(args); fields = d["fields"]
    eval_seeds = [int(s) for s in args.eval_seeds.split(",")]; res = {}
    for item in [x for x in args.sets.split(";") if x.strip()]:
        name, fl = item.split("=", 1); names = [f.strip() for f in fl.split(",") if f.strip()]
        bad = [f for f in names if f not in fields]
        if bad: raise SystemExit(f"unknown fields {bad} in set {name}; dataset fields are {fields}")
        m = to_mask([fields.index(f) for f in names]); r = [retrain(m, s) for s in eval_seeds]
        res[name.strip()] = dict(fields=names, test_auc=float(np.mean([x["test_auc"] for x in r])),
                                 test_auc_seeds=[x["test_auc"] for x in r], val_auc=float(np.mean([x["val_auc"] for x in r])))
        log(f"[eval_sets] {name.strip():28s} {len(names):2d} fields  test AUC {res[name.strip()]['test_auc']:.4f}")
    save_json(dict(kind="eval_sets", data=args.npz or args.dataset, model=args.model, sets=res, protocol=protocol(args)), args.out)
    log(f"[done] wrote {args.out}")

def cmd_run(args):
    """MOSAIC, permutation importance and the controlled rules reported in the paper."""
    d,F,cache,retrain=setup_common(args);fields=d['fields'];rng=np.random.default_rng(args.seed)
    budgets=[int(x) for x in args.budgets.split(',')]
    assert max(budgets)<F
    ye,ev=_setup_est(d,args,F,budgets)
    sec={};t=time.time()
    sh,used=train_shared(d,args);sec['shared_train']=time.time()-t;t=time.time()
    est=mask_moments(sh,d,args);sec['moments']=time.time()-t;t=time.time()
    # The density-averaged rule is a controlled study in Table4; preserve its RNG ordering.
    averaged={k:select_subsets(*budget_surrogate(est,F,k,sigma=None),F,k,top=3,rng=rng)[0] for k in budgets}
    local_rng=np.random.default_rng(10000+args.seed)
    candidates={k:select_subsets(*budget_surrogate(est,F,k),F,k,top=3,rng=local_rng) for k in budgets}
    sec['selection']=time.time()-t;t=time.time()
    def score(mask):return auc_np(ye,logits(sh,d['est_X'],args,mrow=[(mask>>f)&1 for f in range(F)]))
    checks={k:{mask:score(mask) for mask in candidates[k]} for k in budgets}
    selected={k:max(candidates[k],key=lambda mask:checks[k][mask]) for k in budgets}
    sec['verification']=time.time()-t
    sec['mosaic_total']=sum(sec[k] for k in ('shared_train','moments','selection','verification'))
    del sh
    if str(args.device).startswith('cuda'):torch.cuda.empty_cache()
    # Permutation importance uses the same backbone, rows and random-state ordering as the paper.
    t=time.time();full,_=train_subset(d,list(range(F)),args,seed=args.seed+7000)
    sec['one_full_retrain']=time.time()-t
    reference=auc_np(ye,logits(full,d['est_X'],args));importance=[]
    for f in range(F):
        X=d['est_X'].clone();order=torch.from_numpy(rng.permutation(len(ev))).to(X.device)
        X[:,f]=d['est_X'][order,f]
        importance.append(reference-auc_np(ye,logits(full,X,args)))
    del full,X
    importance=np.asarray(importance)
    plans={RUNG_FULL:selected,RUNG_PAIR:{k:candidates[k][0] for k in budgets},RUNG_V1:averaged,
           RUNG_AGNOSTIC:{k:first_order_subset(est,F,k,density=.5) for k in budgets},
           RUNG_LOCAL:{k:first_order_subset(est,F,k) for k in budgets},
           'Permutation':{k:to_mask(np.argsort(-importance)[:k]) for k in budgets}}
    evals={}
    for name,sets in plans.items():
        per={k:[retrain(sets[k],s)['test_auc'] for s in map(int,args.eval_seeds.split(','))] for k in budgets}
        evals[name]=dict(subsets={str(k):[fields[f] for f in bits(sets[k],F)] for k in budgets},
                         test_auc={str(k):float(np.mean(v)) for k,v in per.items()},
                         test_auc_std={str(k):float(np.std(v)) for k,v in per.items()})
        log(f'[eval] {name}: mean test AUC {np.mean(list(evals[name]["test_auc"].values())):.6f}')
    allfields=[retrain((1<<F)-1,s)['test_auc'] for s in map(int,args.eval_seeds.split(','))]
    save_json(dict(kind='run',version='v2',kernel_sigma=KERNEL_SIGMA,model=args.model,data=args.npz or args.dataset,
        mseed=args.seed,budgets=budgets,evals=evals,all_fields_test_auc=float(np.mean(allfields)),fields=fields,
        grid=args.grid,passes=args.passes,shared_epochs=used,est_rows=len(ev),protocol=protocol(args),
        est=est,perm=importance.tolist(),seconds=sec,
        mosaic_candidates={str(k):[dict(fields=[fields[f] for f in bits(mask,F)],shared_val_auc=checks[k][mask])
            for mask in candidates[k]] for k in budgets}),args.out)


def cmd_agreement(args):
    """AppendixC: shared-model/retrained validation rank agreement, without unrelated search experiments."""
    d,F,cache,retrain=setup_common(args);budgets=list(map(int,args.budgets.split(',')))
    ye,_=_setup_est(d,args,F,budgets);shared,_=train_shared(d,args)
    rng=np.random.default_rng(6060+args.seed);sets={};scores={}
    for k in budgets:
        masks=set()
        while len(masks)<min(20,math.comb(F,k)):masks.add(to_mask(rng.choice(F,k,replace=False)))
        sets[k]=sorted(masks)
        scores[k]=[auc_np(ye,logits(shared,d['est_X'],args,mrow=[(mask>>f)&1 for f in range(F)])) for mask in sets[k]]
    del shared
    if str(args.device).startswith('cuda'):torch.cuda.empty_cache()
    agreement={}
    for k in budgets:
        rv=[retrain(mask,101)['val_auc'] for mask in sets[k]]
        agreement[str(k)]=dict(spearman=_spearman(np.asarray(scores[k]),np.asarray(rv)),n=len(rv),
                               shared_val_auc=scores[k],retrained_val_auc=rv)
    save_json(dict(kind='agreement',model=args.model,mseed=args.seed,budgets=budgets,agreement=agreement,
                   protocol=protocol(args)),args.out)


def build_parser():
    p=argparse.ArgumentParser(description='MOSAIC publication reproduction')
    sub=p.add_subparsers(dest='cmd',required=True)
    for name in ('run','eval_ranking','ladder','trainablate','costonly','sens','eval_sets','agreement'):
        q=sub.add_parser(name)
        q.add_argument('--npz');q.add_argument('--erase_root');q.add_argument('--dataset');q.add_argument('--data_path',default='data/')
        q.add_argument('--model',choices=['deepfm','dcnv2'],default='deepfm');q.add_argument('--dim',type=int,default=8)
        q.add_argument('--lr',type=float,default=.001);q.add_argument('--batch_size',type=int,default=1024)
        q.add_argument('--epochs',type=int,default=10);q.add_argument('--budgets',default='2,4,6,8,10,12')
        q.add_argument('--eval_seeds',default='101,102,103');q.add_argument('--seed',type=int,default=0);q.add_argument('--out',required=True)
        q.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');q.add_argument('--data_on_gpu',choices=['auto','0','1'],default='auto')
        q.add_argument('--sparse_emb',action='store_true');q.add_argument('--chunk',type=int,default=131072)
        q.add_argument('--max_train_rows',type=int);q.add_argument('--max_eval_rows',type=int)
        q.add_argument('--import_caches',default='');q.add_argument('--smoke',action='store_true')
        # Retained only as data-loader/retrainer defaults; no cheap-search experiments are exposed.
        q.set_defaults(search_rows=None,search_epochs=1)
        q.add_argument('--shared_min_epochs',type=int,default=1);q.add_argument('--shared_max_epochs',type=int,default=10)
        q.add_argument('--passes',type=int,default=0);q.add_argument('--est_rows',type=int,default=500000)
        q.add_argument('--run_json')
        if name=='eval_ranking':q.add_argument('--ranking',required=True);q.add_argument('--name',required=True)
        if name=='trainablate':q.add_argument('--train_masks',required=True,choices=['budget','half','uniform'])
        if name=='eval_sets':q.add_argument('--sets',required=True)
    return p


if __name__=='__main__':
    args=build_parser().parse_args()
    if args.npz is None and (args.erase_root is None or args.dataset is None):raise SystemExit('Provide --npz or --erase_root and --dataset')
    {'run':cmd_run,'eval_ranking':cmd_eval_ranking,'ladder':cmd_ladder,'trainablate':cmd_trainablate,
     'costonly':cmd_costonly,'sens':cmd_sens,'eval_sets':cmd_eval_sets,'agreement':cmd_agreement}[args.cmd](args)
