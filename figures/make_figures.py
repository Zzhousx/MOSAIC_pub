"""Builds every number, table and result figure of the MOSAIC paper from the result JSONs in results/.
Usage:  python make_figures.py
Outputs: figures/*.pdf|png, tables/*.tex and tex/numbers.tex (the numbers quoted in the text).
Seeds: the main comparison uses every selection seed present (DeepFM: 0-4, DCNv2: 0-2); the mechanism ladder and E4 use
the seeds for which ladder / trainablate outputs exist (0-2); their run-based rungs are restricted to the same seeds."""
import glob, json, os, sys, itertools
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "results")
sys.path.insert(0, HERE)
for _d in ("figures", "tables", "tex"): os.makedirs(os.path.join(HERE, _d), exist_ok=True)
# reference for the worst regret: best mean test AUC that any selection procedure reached at each budget
REF = json.load(open(os.path.join(D, "reference_best.json")))["settings"]
from mosaic_lib import budget_surrogate, LADDER, RUNG_AGNOSTIC, RUNG_LOCAL, RUNG_V1, RUNG_PAIR, RUNG_FULL  # noqa: E402

# (key, dataset, backbone, label, macro tag)
DS = [("ml1m20/deepfm", "ml1m20", "deepfm", "ML-1M-20", "MLA"), ("avazu/deepfm", "avazu", "deepfm", "Avazu", "AVA"),
      ("criteo/deepfm", "criteo", "deepfm", "Criteo", "CRI"), ("ml1m20/dcnv2", "ml1m20", "dcnv2", "ML-1M-20, DCNv2", "MLD"),
      ("avazu/dcnv2", "avazu", "dcnv2", "Avazu, DCNv2", "AVD"), ("criteo/dcnv2", "criteo", "dcnv2", "Criteo, DCNv2", "CRD")]
DEEPFM = [x for x in DS if x[2] == "deepfm"]; DCN = [x for x in DS if x[2] == "dcnv2"]
PUB = ["AutoField", "LPFS", "SFS", "SHARK", "FairFS", "Permutation", "mRMR"]
DEEP = [n for n in PUB if n != "mRMR"]                          # deep / model-based baselines (plotted in Figure 1)
EXT = ["AutoField", "LPFS", "SFS", "SHARK", "FairFS", "mRMR"]  # external rankings, seeds 2027-2029
SHOW = PUB + ["MOSAIC"]
ERASE_MIN = {"ml1m20": dict(AutoField=0.5, LPFS=0.3, SFS=0.4, SHARK=3.7),
             "avazu": dict(AutoField=14.1, LPFS=11.8, SFS=10.2, SHARK=16.4),
             "criteo": dict(AutoField=25.5, LPFS=22.4, SFS=22.3, SHARK=37.8)}   # search minutes of the official ERASE code
macros = {}


def J(p): return json.load(open(p))
def m(name, val): macros[name] = val
def fmt4(x): return f"{x:.4f}"
def sm(rows, ks): return np.array([np.mean([rows[s][k] for s in rows]) for k in ks])
def restrict(rows, seeds): return {s: rows[s] for s in seeds if s in rows}


def load(ds, model):
    runs = {int(J(p)["mseed"]): J(p) for p in sorted(glob.glob(os.path.join(D, f"{ds}_{model}_s[0-9]_v2.json")))}
    T = {}
    for s, R in runs.items():
        for n, e in R["evals"].items(): T.setdefault(n, {})[s] = {int(k): v for k, v in e["test_auc"].items()}
    for n in EXT:
        files = {2027: os.path.join(D, f"{ds}_{model}_{n}.json")}
        for s in (2027, 2028, 2029): files.setdefault(s, os.path.join(D, f"{ds}_{model}_{n}_seed{s}.json"))
        files.update({s: os.path.join(D, f"{ds}_{model}_{n}_seed{s}.json") for s in (2027, 2028, 2029)
                      if os.path.exists(os.path.join(D, f"{ds}_{model}_{n}_seed{s}.json"))})
        got = {-s: {int(k): v for k, v in J(p)["test_auc"].items()} for s, p in files.items() if os.path.exists(p)}
        if got: T[n] = got
    lad = {int(J(p)["mseed"]): J(p) for p in sorted(glob.glob(os.path.join(D, f"{ds}_{model}_s[0-9]_ladder.json")))}
    for s, L in lad.items():
        for n, e in L["rungs"].items():
            if n in (RUNG_AGNOSTIC, RUNG_LOCAL): T.setdefault(n, {})[s] = {int(k): v for k, v in e["test_auc"].items()}
    tm = {}
    for p in glob.glob(os.path.join(D, f"{ds}_{model}_s[0-9]_tm-*.json")):
        X = J(p); tm.setdefault(X["scheme"], {})[int(X["mseed"])] = X
        for n, e in X["variants"].items():
            T.setdefault(f"E4[{X['scheme']}] {n}", {})[int(X["mseed"])] = {int(k): v for k, v in e["test_auc"].items()}
    return runs, T, sorted(lad), tm


# ---------------------------------------------------------------------------------------------- main results
info = {}
for key, ds, model, label, tag in DS:
    runs, T, lad_seeds, tm = load(ds, model)
    R0 = runs[min(runs)]; ks = [int(k) for k in R0["budgets"]]
    best = np.array(REF[key]["best_test_auc"])
    allf = np.mean([R["all_fields_test_auc"] for R in runs.values()])
    rows = {}
    for n in SHOW:
        v = sm(T[n], ks); per_seed = [np.mean([T[n][s][k] for k in ks]) for s in sorted(T[n])]
        rows[n] = dict(aukc=float(v.mean()), sd=float(np.std(per_seed)) if len(per_seed) > 1 else None,
                       worst=float(np.max(best - v)), curve=v, seeds=len(per_seed), runs=per_seed)
    info[key] = dict(ds=ds, model=model, ks=ks, F=len(R0["fields"]), allf=allf, best=best, runs=runs, T=T, lad=lad_seeds,
                     tm=tm, label=label, tag=tag, rows=rows, nseeds=len(runs))
    pub_best = max(PUB, key=lambda n: rows[n]["aukc"]); pub_safe = min(PUB, key=lambda n: rows[n]["worst"])
    m(f"{tag}aukc", fmt4(rows["MOSAIC"]["aukc"])); m(f"{tag}bestpub", pub_best); m(f"{tag}bestpubaukc", fmt4(rows[pub_best]["aukc"]))
    m(f"{tag}gain", f"{rows['MOSAIC']['aukc'] - rows[pub_best]['aukc']:+.4f}")
    m(f"{tag}worst", f"{1e3 * rows['MOSAIC']['worst']:.1f}"); m(f"{tag}safepub", pub_safe)
    m(f"{tag}safepubworst", f"{1e3 * rows[pub_safe]['worst']:.1f}")
    m(f"{tag}ratio", f"{rows[pub_safe]['worst'] / rows['MOSAIC']['worst']:.0f}")
    m(f"{tag}allf", fmt4(allf)); m(f"{tag}F", str(len(R0["fields"]))); m(f"{tag}nseeds", str(len(runs)))
from scipy.stats import ttest_ind
sig = {}
for k, *_ in DS:
    for n in PUB:
        a, b = info[k]["rows"]["MOSAIC"]["runs"], info[k]["rows"][n]["runs"]
        p = float(ttest_ind(a, b, equal_var=False, alternative="greater").pvalue) if len(b) > 1 and np.std(b) > 0 else \
            float(ttest_ind(a, b + [b[0] + 1e-12], equal_var=False, alternative="greater").pvalue)
        sig[(k, n)] = p
m("sigmaxp", f"{max(sig.values()):.3f}")
m("signall", str(sum(p < 0.05 for p in sig.values()))); m("sigtotal", str(len(sig)))
ratios = [info[k]["rows"][min(PUB, key=lambda n: info[k]["rows"][n]["worst"])]["worst"] / info[k]["rows"]["MOSAIC"]["worst"] for k, *_ in DS]
m("maxworstmosaic", f"{max(1e3 * info[k]['rows']['MOSAIC']['worst'] for k, *_ in DS):.1f}")
m("minratio", f"{min(ratios):.0f}"); m("maxratio", f"{max(ratios):.0f}")
m("minpubworst", f"{min(1e3 * info[k]['rows'][n]['worst'] for k, *_ in DS for n in PUB):.1f}")


def rank_fmt(vals, x, hi=True, digits=4, scale=1.0):
    order = sorted(vals, reverse=hi); s = f"{x * scale:.{digits}f}"
    if x == order[0]: return rf"\textbf{{{s}}}"
    if x == order[1]: return rf"\underline{{{s}}}"
    return s


def head(key, lab):
    return "\\multicolumn{2}{c}{" + lab + " ($F{=}" + str(info[key]["F"]) + "$)}"


L = [r"\begin{table*}[t]", r"\centering",
     r"\caption{Results over the whole budget range. AUKC: test AUC averaged over budgets ($\pm$ std over runs). "
     r"Worst reg.: largest gap from the best result at any budget ($\times10^{-3}$). $^{\ast}$: \mosaic{} is significantly "
     r"better ($p<0.05$). Best in \textbf{bold}, second \underline{underlined}.}",
     r"\label{tab:main}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
     r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l" + "cc" * 3 + "}", r"\toprule"]
L.append(" & " + " & ".join(r"\multicolumn{2}{c}{" + lab + " ($F{=}" + str(info[k]["F"]) + "$)}" for k, _, _, lab, _ in DEEPFM) + r" \\")
L.append("".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(3)))
L.append("Method & " + " & ".join(["AUKC $\\uparrow$ & Worst reg. $\\downarrow$"] * 3) + r" \\")
for gname, G in [("DeepFM backbone", DEEPFM), ("DCNv2 backbone", DCN)]:
    L += [r"\midrule", rf"\multicolumn{{7}}{{l}}{{\textit{{{gname}}}}} \\", r"\midrule"]
    for n in SHOW:
        cells = []
        for k, *_ in G:
            r = info[k]["rows"][n]; A = [info[k]["rows"][x]["aukc"] for x in SHOW]; W = [info[k]["rows"][x]["worst"] for x in SHOW]
            a_ = rank_fmt(A, r["aukc"]) + (rf"\,{{\scriptsize$\pm${r['sd']:.4f}}}" if r["sd"] is not None else "")
            if n != "MOSAIC" and sig[(k, n)] < 0.05: a_ += r"$^{\ast}$"
            cells += [a_, rank_fmt(W, r["worst"], hi=False, digits=1, scale=1e3)]
        if n == "MOSAIC": L.append(r"\cmidrule(lr){1-7}")
        L.append(n + " & " + " & ".join(cells) + r" \\")
    L.append(r"\textit{All fields} & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{\textit{{{info[k]['allf']:.4f}}}}}" for k, *_ in G) + r" \\")
L += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}"]
open(os.path.join(HERE, "tables", "main.tex"), "w").write("\n".join(L) + "\n")

# ---------------------------------------------------------------------------------------------- cost (DeepFM)
def secs(k, key): return np.mean([R["seconds"][key] for R in info[k]["runs"].values()])
CDEEP = [d for d in DEEPFM if d[1] != "ml1m20"]
L = [r"\begin{table}[t]", r"\centering", r"\caption{Time of \mosaic{} for all budgets, next to one ordinary training (DeepFM).}",
     r"\label{tab:cost}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}lcc}", r"\toprule",
     " & " + " & ".join(lab for _, _, _, lab, _ in CDEEP) + r" \\", r"\midrule"]
for key_, nm in [("shared_train", "Shared masked model (s)"), ("moments", "Mask moments (s)")]:
    L.append(nm + " & " + " & ".join(f"{secs(k, key_):.0f}" for k, *_ in CDEEP) + r" \\")
L.append("Selection + verification (s) & " + " & ".join(f"{secs(k, 'selection') + secs(k, 'verification'):.1f}" for k, *_ in CDEEP) + r" \\")
L.append(r"\textbf{Total, all budgets (s)} & " + " & ".join(rf"\textbf{{{secs(k, 'mosaic_total'):.0f}}}" for k, *_ in CDEEP) + r" \\")
L += [r"\midrule", "One training, all fields (s) & " + " & ".join(f"{secs(k, 'one_full_retrain'):.0f}" for k, *_ in CDEEP) + r" \\",
      r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
open(os.path.join(HERE, "tables", "cost.tex"), "w").write("\n".join(L) + "\n")
for k, ds, _, _, tag in DEEPFM:
    m(f"{tag}cost", f"{secs(k, 'mosaic_total'):.0f}"); m(f"{tag}costx", f"{secs(k, 'mosaic_total') / secs(k, 'one_full_retrain'):.1f}")
    m(f"{tag}erasemin", f"{min(ERASE_MIN[ds].values()):.1f}--{max(ERASE_MIN[ds].values()):.1f}")

# ---------------------------------------------------------------------------------------------- interaction structure
shares = {}
for k, ds, _, _, tag in DEEPFM:
    sh, neg = [], []
    for R in info[k]["runs"].values():
        F = len(R["fields"]); pairs = list(itertools.combinations(range(F), 2))
        M = np.zeros((F, len(pairs)))
        for p, (i, j) in enumerate(pairs): M[i, p] = M[j, p] = 1
        P = M.T @ np.linalg.pinv(M @ M.T) @ M
        for kk in R["budgets"]:
            a, b = budget_surrogate(R["est"], F, kk); v = np.array([b[i, j] for i, j in pairs])
            sh.append(1 - (v @ P @ v) / (v @ v)); neg.append((v[v < 0] ** 2).sum() / (v ** 2).sum())
    shares[k] = (np.mean(sh), np.mean(neg))
    m(f"{tag}zdeg", f"{100 * np.mean(sh):.0f}"); m(f"{tag}neg", f"{100 * np.mean(neg):.0f}")
m("zdegmin", f"{100 * min(s[0] for s in shares.values()):.0f}"); m("zdegmax", f"{100 * max(s[0] for s in shares.values()):.0f}")

# ---------------------------------------------------------------------------------------------- mechanism ladder
NAMES = {RUNG_AGNOSTIC: r"Importance at $p{=}\tfrac12$, top-$k$", RUNG_LOCAL: r"Importance at $p{=}k/F$, top-$k$",
         RUNG_V1: r"+ pairwise term (density-averaged)", RUNG_PAIR: r"+ pairwise term (budget-local)",
         RUNG_FULL: r"+ verification (MOSAIC)"}
LN = {RUNG_AGNOSTIC: "impHalf", RUNG_LOCAL: "impLoc", RUNG_V1: "pairGlob", RUNG_PAIR: "pair", RUNG_FULL: "full"}
LV = {}
for k, ds, model, lab, tag in DS:
    I = info[k]
    if not I["lad"]: continue
    V = {n: sm(restrict(I["T"][n], I["lad"]), I["ks"]) for n in LADDER}
    Rg = {n: float(np.max(I["best"] - V[n])) for n in LADDER}; LV[k] = (V, Rg)
    for n, nm in LN.items(): m(f"{tag}{nm}", fmt4(V[n].mean())); m(f"{tag}{nm}Reg", f"{1e3 * Rg[n]:.1f}")
    m(f"{tag}impHalfMin", fmt4(V[RUNG_AGNOSTIC][0])); m(f"{tag}impLocMin", fmt4(V[RUNG_LOCAL][0]))
    m(f"{tag}pairGainHalf", f"{V[RUNG_PAIR].mean() - V[RUNG_AGNOSTIC].mean():+.4f}")
    m(f"{tag}pairGainLoc", f"{V[RUNG_PAIR].mean() - V[RUNG_LOCAL].mean():+.4f}")
    m(f"{tag}pairRatio", f"{min(Rg[RUNG_AGNOSTIC], Rg[RUNG_LOCAL]) / Rg[RUNG_PAIR]:.0f}")
# Table 4: worst regret per rule and setting
STEPN = {RUNG_AGNOSTIC: "Importance, one ranking", RUNG_LOCAL: "Importance, per budget",
         RUNG_V1: "+ interactions (averaged)", RUNG_PAIR: "+ interactions (budget-local)", RUNG_FULL: "+ check (MOSAIC)"}
SK = [k for k, *_ in DS if k in LV]
SL = {"ml1m20/deepfm": "ML-1M-20", "avazu/deepfm": "Avazu", "criteo/deepfm": "Criteo", "ml1m20/dcnv2": "ML-1M-20"}
L = [r"\begin{table}[t]", r"\centering",
     r"\caption{Worst regret ($\times10^{-3}$) as MOSAIC is built up step by step on one shared model (the three runs shared by all rules).}",
     r"\label{tab:steps}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
     r"\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}l" + "c" * len(SK) + "}", r"\toprule",
     "Selection rule & " + " & ".join(SL.get(k, k) for k in SK) + r" \\",
     " & " + " & ".join("DCNv2" if "dcnv2" in k else "DeepFM" for k in SK) + r" \\", r"\midrule"]
for n in LADDER:
    if n == RUNG_V1: L.append(r"\midrule")
    L.append(STEPN[n] + " & " + " & ".join(f"{1e3 * LV[k][1][n]:.1f}" for k in SK) + r" \\")
L += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
open(os.path.join(HERE, "tables", "steps.tex"), "w").write("\n".join(L) + "\n")
lr = [min(Rg[RUNG_AGNOSTIC], Rg[RUNG_LOCAL]) / Rg[RUNG_PAIR] for V, Rg in LV.values()]
m("ladRatioMin", f"{min(lr):.1f}"); m("ladRatioMax", f"{max(lr):.1f}")
m("ladImpWorstMin", f"{min(1e3 * min(Rg[RUNG_AGNOSTIC], Rg[RUNG_LOCAL]) for V, Rg in LV.values()):.1f}")
m("ladImpWorstMax", f"{max(1e3 * min(Rg[RUNG_AGNOSTIC], Rg[RUNG_LOCAL]) for V, Rg in LV.values()):.1f}")

# verification, paired over (seed x budget), all selection seeds
for k, ds, model, lab, tag in DS:
    T = info[k]["T"]; ks = info[k]["ks"]; ss = sorted(set(T[RUNG_FULL]) & set(T[RUNG_PAIR]))
    dd = np.array([T[RUNG_FULL][s][x] - T[RUNG_PAIR][s][x] for s in ss for x in ks])
    m(f"{tag}verW", str(int((dd > 1e-9).sum()))); m(f"{tag}verL", str(int((dd < -1e-9).sum()))); m(f"{tag}verN", str(len(dd)))
    m(f"{tag}verGain", f"{dd.mean():+.4f}")

# ---------------------------------------------------------------------------------------------- E4 (DeepFM)
E4 = {}
for sch, nm in [("half", r"Field dropout, $p{=}\tfrac12$"), ("uniform", r"Uniform, $p\sim U(0.05,0.95)$"),
                ("budget", r"Budget-matched (default)")]:
    cells = []
    for k, ds, model, lab, tag in DEEPFM:
        I = info[k]; common = sorted(set.intersection(*[set(I["tm"].get(x, {})) for x in ("half", "uniform", "budget")])) if I["tm"] else []
        if not common: continue
        a = sm(restrict(I["T"][f"E4[{sch}] {RUNG_FULL}"], common), I["ks"]).mean()
        b = sm(restrict(I["T"][f"E4[{sch}] {RUNG_PAIR}"], common), I["ks"]).mean()
        bw = float(np.max(I["best"] - sm(restrict(I["T"][f"E4[{sch}] {RUNG_PAIR}"], common), I["ks"])))
        E4[(k, sch)] = (a, b, bw)
if E4:
    spreads = [max(E4[(k, s)][0] for s in ("half", "uniform", "budget")) - min(E4[(k, s)][0] for s in ("half", "uniform", "budget"))
               for k, *_ in DEEPFM if (k, "budget") in E4]
    m("eSpread", f"{max(spreads):.4f}")
    for k, ds, model, lab, tag in DEEPFM:
        if (k, "budget") in E4:
            for s, nm in (("budget", "eBudget"), ("half", "eHalf"), ("uniform", "eUni")):
                m(f"{tag}{nm}", fmt4(E4[(k, s)][0])); m(f"{tag}{nm}Nv", fmt4(E4[(k, s)][1])); m(f"{tag}{nm}NvReg", f"{1e3 * E4[(k, s)][2]:.1f}")

# ---------------------------------------------------------------------------------------------- case-study numbers
I = info["ml1m20/deepfm"]; R = I["runs"][0]; F = len(R["fields"]); idx = {f: i for i, f in enumerate(R["fields"])}
a2, b2 = budget_surrogate(R["est"], F, 2)
m("MLuserzip", f"{b2[idx['user_id'], idx['zip']]:.3f}"); m("MLmovietitle", f"{b2[idx['movie_id'], idx['title']]:.3f}")
m("MLuserzipshare", f"{100 * -b2[idx['user_id'], idx['zip']] / (a2[idx['user_id']] + a2[idx['zip']]):.0f}")
m("MLlpfsktwo", fmt4(sm(I["T"]["LPFS"], [2])[0])); m("MLautoktwo", fmt4(sm(I["T"]["AutoField"], [2])[0]))
m("MLmosaicktwo", fmt4(sm(I["T"]["MOSAIC"], [2])[0])); m("MLautokeight", fmt4(sm(I["T"]["AutoField"], [8])[0]))
m("MLmosaickeight", fmt4(sm(I["T"]["MOSAIC"], [8])[0]))
m("MLsfskfour", fmt4(sm(I["T"]["SFS"], [4])[0])); m("MLmosaickfour", fmt4(sm(I["T"]["MOSAIC"], [4])[0]))
m("MLmaxpubktwo", fmt4(max(sm(I["T"][x], [2])[0] for x in PUB)))
m("MLlpfsktwoA", fmt4(I["T"]["LPFS"][-2027][2]))
m("maxpubsd", f"{max(info[k]['rows'][x]['sd'] for k, *_ in DS for x in PUB):.3f}")
m("mosaicmaxsd", f"{max(info[k]['rows']['MOSAIC']['sd'] for k, *_ in DS):.4f}")
C = info["criteo/deepfm"]
gaps = [C["best"][0] - sm(C["T"][x], [5])[0] for x in PUB]; m("CRgapmin", f"{min(gaps):.2f}"); m("CRgapmax", f"{max(gaps):.2f}")
C = info["criteo/deepfm"]; m("CRsfskfive", fmt4(sm(C["T"]["SFS"], [5])[0])); m("CRmosaickfive", fmt4(sm(C["T"]["MOSAIC"], [5])[0]))
m("CRautokfive", fmt4(sm(C["T"]["AutoField"], [5])[0]))

# ---------------------------------------------------------------------------------------------- figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.linewidth": 0.6, "pdf.fonttype": 42,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "legend.frameon": True, "legend.framealpha": 0.92,
                     "legend.edgecolor": "#e4e3df", "legend.fancybox": False})
INK2 = "#52514e"; GRID = "#e4e3df"
COL = {"MOSAIC": "#2a78d6", "AutoField": "#eb6834", "LPFS": "#1baf7a", "SFS": "#eda100", "SHARK": "#e87ba4",
       "Permutation": "#008300", "FairFS": "#6250d6", "mRMR": "#8c8a83"}
MK = {"MOSAIC": "o", "AutoField": "s", "LPFS": "^", "SFS": "D", "SHARK": "v", "Permutation": "P", "FairFS": "X", "mRMR": "*"}
NICE = {}

def style(ax):
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)

# Figure 1: AUC vs budget, 6 settings (2 x 3), legend inside two empty corners
fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.55)); axes = axes.ravel()
for ax, (k, ds, model, lab, _) in zip(axes, DEEPFM + DCN):
    I = info[k]; ks = I["ks"]; rows = I["rows"]
    for n in DEEP + ["MOSAIC"]:
        main = n == "MOSAIC"
        ax.plot(ks, rows[n]["curve"], color=COL[n], marker=MK[n], ms=3.6 if main else 2.8, lw=1.9 if main else 0.9,
                alpha=1 if main else 0.85, label=NICE.get(n, n), zorder=4 if main else 2,
                markeredgecolor="white", markeredgewidth=0.4)
    ax.axhline(I["allf"], color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=1, label="all fields")
    ax.set_title(f"{lab.split(',')[0]} ({'DeepFM' if model == 'deepfm' else 'DCNv2'})", fontsize=7.8, color="#0b0b0b", pad=3)
    ax.set_xticks(ks); style(ax)
    if model == "dcnv2": ax.set_xlabel("fields kept $k$", fontsize=7)
    lo = min(min(rows[n]["curve"]) for n in DEEP + ["MOSAIC"]); hi = max(max(rows[n]["curve"]) for n in DEEP + ["MOSAIC"])
    ax.set_ylim(lo - (0.012 if ds == "avazu" else 0.004), max(hi, I["allf"]) + 0.004)
axes[0].set_ylabel("test AUC (retrained)"); axes[3].set_ylabel("test AUC (retrained)")
h, l = axes[2].get_legend_handles_labels(); H = dict(zip(l, h))
LEG = dict(fontsize=5.6, ncol=1, handlelength=1.6, borderpad=0.35, labelspacing=0.22, handletextpad=0.4)
first = ["MOSAIC", "all fields"] + [NICE.get(n, n) for n in DEEP[:2]]
second = [NICE.get(n, n) for n in DEEP[2:]]
axes[1].legend([H[x] for x in first], first, loc="lower right", **LEG)     # legend split over two empty corners
axes[2].legend([H[x] for x in second], second, loc="lower right", **LEG)
fig.tight_layout(w_pad=0.6, h_pad=0.5)
fig.savefig(os.path.join(HERE, "figures", "curves.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(HERE, "figures", "curves.png"), dpi=200, bbox_inches="tight")

# Figure: regret map (methods x every dataset/budget cell), binned sequential colours
from matplotlib.colors import ListedColormap, BoundaryNorm
BINS = [0, 1, 3, 10, 30, 1000]; SEQ = ["#f4f2ec", "#f6d7b8", "#eda36a", "#d5622b", "#8f2d12"]
cm_r = ListedColormap(SEQ); nm_r = BoundaryNorm(BINS, cm_r.N)
order_m = ["MOSAIC"] + PUB
cols, xt, xl, grp = [], [], [], []
for k, ds, model, lab, _ in DEEPFM + DCN:
    I = info[k]
    for j, kk in enumerate(I["ks"]):
        cols.append([1e3 * (I["best"][j] - I["rows"][n]["curve"][j]) for n in order_m]); xl.append(str(kk))
    grp.append((len(cols) - len(I["ks"]), len(cols) - 1, f"{lab.split(',')[0]} ({'DeepFM' if model == 'deepfm' else 'DCNv2'})"))
    cols.append([np.nan] * len(order_m)); xl.append("")
M = np.array(cols[:-1]).T; xl = xl[:-1]
fig, ax = plt.subplots(figsize=(7.0, 1.75))
cm_r.set_bad("#ffffff")
ax.imshow(np.ma.masked_invalid(M), cmap=cm_r, norm=nm_r, aspect="auto")
ax.set_yticks(range(len(order_m))); ax.set_yticklabels([NICE.get(n, n) for n in order_m], fontsize=6.4)
ax.set_xticks([i for i, x in enumerate(xl) if x]); ax.set_xticklabels([x for x in xl if x], fontsize=5.6)
for a0, a1, name in grp: ax.text((a0 + a1) / 2, -0.85, name, ha="center", va="bottom", fontsize=6.4, color="#0b0b0b")
ax.set_xlabel("fields kept $k$", fontsize=6.6, labelpad=1)
for sp in ax.spines.values(): sp.set_visible(False)
ax.tick_params(length=0)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if not np.isnan(M[i, j]) and M[i, j] >= 30: ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=4.3, color="white")
import matplotlib.cm as mcm
cax = ax.inset_axes([1.012, 0.0, 0.012, 1.0])
cb = fig.colorbar(mcm.ScalarMappable(norm=BoundaryNorm(range(6), cm_r.N), cmap=cm_r), cax=cax, ticks=[0.5, 1.5, 2.5, 3.5, 4.5])
cb.ax.set_yticklabels(["<1", "1–3", "3–10", "10–30", ">30"], fontsize=5.4); cb.outline.set_visible(False); cb.ax.tick_params(length=0, pad=1.5)
cb.set_label("regret ($\\times10^{-3}$)", fontsize=5.8, labelpad=2)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figures", "regretmap.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "figures", "regretmap.png"), dpi=200, bbox_inches="tight")
m("cellsTotal", str(M.shape[1] - np.isnan(M[0]).sum()))
m("mosaicMaxCell", f"{np.nanmax(M[0]):.1f}")
m("cellsPubOverTen", str(min(int(np.nansum(M[i] > 10)) for i in range(1, len(order_m)))))

# Figure: which fields are kept (ML-1M-20, DeepFM): best published baseline vs MOSAIC, one run each
I = info["ml1m20/deepfm"]; R0 = I["runs"][0]; fields_ = R0["fields"]
bp = max(DEEP, key=lambda n: I["rows"][n]["aukc"])
BJ = J(os.path.join(D, f"ml1m20_deepfm_{bp}.json")); MS = R0["evals"]["MOSAIC"]["subsets"]
used = [f for f in fields_ if any(f in v for v in list(BJ["subsets"].values()) + list(MS.values()))]
fig, axes = plt.subplots(1, 2, figsize=(3.35, 2.45), sharey=True)
nk = len(I["ks"])
for ax, (name, sub, auc, col) in zip(axes, [(bp, BJ["subsets"], BJ["test_auc"], COL[bp]),
                                            ("MOSAIC", MS, R0["evals"]["MOSAIC"]["test_auc"], COL["MOSAIC"])]):
    for c, kk in enumerate(I["ks"]):
        for r, f in enumerate(used):
            on = f in sub[str(kk)]; hot = f == "timestamp"
            ax.add_patch(plt.Rectangle((c - 0.42, r - 0.4), 0.84, 0.8, color=(("#c9302c" if hot else col) if on else "#efeee9"), lw=0))
        ax.text(c, len(used) - 0.2, f"{auc[str(kk)]:.3f}", va="top", ha="center", fontsize=5.0, rotation=90, color="#0b0b0b")
    ax.set_ylim(len(used) + 1.75, -0.6); ax.set_xlim(-0.6, nk - 0.4)
    ax.set_yticks(range(len(used))); ax.set_yticklabels(used, fontsize=5.6)
    ax.set_xticks(range(nk)); ax.set_xticklabels([str(x) for x in I["ks"]], fontsize=5.8)
    ax.xaxis.tick_top()
    ax.set_title(name, fontsize=7.0, pad=9, color=col)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0, pad=1.5)
axes[0].text(-0.75, -1.05, "fields kept $k$", ha="right", va="center", fontsize=5.6, color=INK2)
axes[0].text(-0.75, len(used) + 0.75, "test AUC", ha="right", va="center", fontsize=5.6, color=INK2)
fig.tight_layout(w_pad=0.4)
fig.savefig(os.path.join(HERE, "figures", "barcode.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "figures", "barcode.png"), dpi=200, bbox_inches="tight")
m("bcBase", bp)
kt = [kk for kk in I["ks"] if "timestamp" in BJ["subsets"][str(kk)]]
m("bcTsK", str(min(kt)) if kt else "--")
m("bcBaseBefore", f"{BJ['test_auc'][str(max(x for x in I['ks'] if x < min(kt)))]:.4f}" if kt else "--")
m("bcBaseAfter", f"{BJ['test_auc'][str(min(kt))]:.4f}" if kt else "--")
m("bcMosAfter", f"{R0['evals']['MOSAIC']['test_auc'][str(min(kt))]:.4f}" if kt else "--")

# ---------------------------------------------------------------------------------------------- stress test (planted copies)
runsC, TC, _, _ = load("ml1m40c", "deepfm"); ksC = [int(k) for k in runsC[min(runsC)]["budgets"]]
bestC = np.array(REF["ml1m40c/deepfm"]["best_test_auc"])
def _pairs(sub): s_ = set(sub); return sum(1 for f in s_ if f + "_copy1" in s_)
def _subs(n):
    if n in ("MOSAIC", "Permutation"): return [R["evals"][n]["subsets"] for R in runsC.values()]
    fs = [os.path.join(D, f"ml1m40c_deepfm_{n}.json")] + [os.path.join(D, f"ml1m40c_deepfm_{n}_seed{x}.json") for x in (2028, 2029)]
    if n == "mRMR": fs = [os.path.join(D, f"ml1m40c_deepfm_mRMR_seed{x}.json") for x in (2027, 2028, 2029)]
    return [J(p_)["subsets"] for p_ in fs if os.path.exists(p_)]
STRESS = ["MOSAIC", "AutoField", "LPFS", "SFS", "SHARK", "Permutation", "mRMR"]
SC = {}
for n in STRESS:
    v = sm(TC[n], ksC); runs_ = [np.mean([TC[n][x][k] for k in ksC]) for x in TC[n]]
    SC[n] = dict(curve=v, aukc=float(v.mean()), worst=float(np.max(bestC - v)), runs=runs_,
                 pairs=[float(np.mean([_pairs(S[str(k)]) for S in _subs(n)])) for k in ksC])
bpC = max([n for n in STRESS if n != "MOSAIC"], key=lambda n: SC[n]["aukc"]); spC = min([n for n in STRESS if n != "MOSAIC"], key=lambda n: SC[n]["worst"])
m("STaukc", fmt4(SC["MOSAIC"]["aukc"])); m("STbest", bpC); m("STbestaukc", fmt4(SC[bpC]["aukc"]))
m("STworst", f"{1e3 * SC['MOSAIC']['worst']:.1f}"); m("STsafe", spC); m("STsafeworst", f"{1e3 * SC[spC]['worst']:.1f}")
m("STratio", f"{SC[spC]['worst'] / SC['MOSAIC']['worst']:.0f}"); m("STallf", fmt4(np.mean([R["all_fields_test_auc"] for R in runsC.values()])))
m("STzeroUpTo", str(max(k for j, k in enumerate(ksC) if all(SC["MOSAIC"]["pairs"][i] == 0 for i in range(j + 1)))))
m("STmosPairsMax", f"{max(SC['MOSAIC']['pairs']):.1f}")
kmax = ksC[-1]
m("STpubPairsMin", f"{min(SC[n]['pairs'][-1] for n in STRESS if n != 'MOSAIC'):.0f}"); m("STpubPairsMax", f"{max(SC[n]['pairs'][-1] for n in STRESS if n != 'MOSAIC'):.1f}")
m("STkmax", str(kmax)); m("STmlA", fmt4(info["ml1m20/deepfm"]["rows"]["MOSAIC"]["aukc"]))
fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.85))
for n in STRESS:
    main = n == "MOSAIC"; kw = dict(color=COL[n], marker=MK[n], ms=3.4 if main else 2.6, lw=1.8 if main else 0.85, zorder=4 if main else 2,
                                   markeredgecolor="white", markeredgewidth=0.3, label=NICE.get(n, n), alpha=1 if main else 0.85)
    axes[0].plot(ksC, SC[n]["pairs"], **kw); axes[1].plot(ksC, SC[n]["curve"], **kw)
axes[0].set_ylabel("original–copy pairs kept", fontsize=6.4); axes[1].set_ylabel("test AUC (retrained)", fontsize=6.4)
for ax in axes:
    ax.set_xticks(ksC); ax.tick_params(labelsize=5.6); ax.set_xlabel("fields kept $k$", fontsize=6.4); style(ax)
axes[1].set_ylim(0.72, 0.815); axes[0].set_ylim(-0.25, 4.7); axes[0].set_yticks([0, 1, 2, 3, 4])
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, columnspacing=0.8, fontsize=5.2,
           handlelength=1.6, labelspacing=0.2, handletextpad=0.3)
fig.tight_layout(w_pad=0.6, rect=(0, 0, 1, 0.89))
fig.savefig(os.path.join(HERE, "figures", "stress.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "figures", "stress.png"), dpi=200, bbox_inches="tight")

# ---------------------------------------------------------------------------------------------- cost vs fields + sensitivity
SCL = sorted([J(p_) for p_ in glob.glob(os.path.join(D, "scale_criteo_F*.json"))], key=lambda x: x["F"])
Fs = [x["F"] for x in SCL]; tot = [x["seconds"]["mosaic_total"] for x in SCL]; one = [x["seconds"]["one_full_train"] for x in SCL]
m("SCFmin", str(Fs[0])); m("SCFmax", str(Fs[-1])); m("SCtmin", f"{tot[0]:.0f}"); m("SCtmax", f"{tot[-1]:.0f}")
m("SCgrowF", f"{Fs[-1] / Fs[0]:.0f}"); m("SCgrowT", f"{tot[-1] / tot[0]:.1f}")
rat = [a_ / b_ for a_, b_ in zip(tot, one)]; m("SCratmin", f"{min(rat):.1f}"); m("SCratmax", f"{max(rat):.1f}")
SENS_V = [("sigma=0.1", r"$\sigma$=0.1"), ("sigma=0.5", r"$\sigma$=0.5"), ("sigma=1.0", r"$\sigma$=1"), ("candidates=1", "1 cand."),
          ("candidates=5", "5 cand."), ("rows=0.5", "½ rows")]
SV2 = {}
for ds in ("ml1m20", "avazu"):
    Js = [J(p_) for p_ in sorted(glob.glob(os.path.join(D, f"{ds}_deepfm_s[0-9]_sens.json")))]
    au = lambda n: float(np.mean([np.mean(list(x["variants"][n]["test_auc"].values())) for x in Js]))
    SV2[ds] = {n: au(n) - au("default") for n, _ in SENS_V}; SV2[ds]["_default"] = au("default")
m("SNmaxdev", f"{1e3 * max(abs(v) for ds in SV2 for n, v in SV2[ds].items() if n != '_default'):.1f}")
fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.55), gridspec_kw=dict(width_ratios=[1, 1.25]))
ax = axes[0]
stages = [("shared_train", "shared model", "#7fb0ea"), ("moments", "moments", "#2a78d6"), ("selection", "selection + check", "#1c5fae")]
bottom = np.zeros(len(Fs)); x = np.arange(len(Fs))
for key, lab, c in stages:
    v = np.array([s_["seconds"][key] + (s_["seconds"]["verification"] if key == "selection" else 0) for s_ in SCL])
    ax.bar(x, v, 0.6, bottom=bottom, color=c, label=lab, edgecolor="white", linewidth=0.3); bottom += v
ax.plot(x, one, color=INK2, marker="o", ms=2.5, lw=0.8, label="one training epoch")
ax.set_xticks(x); ax.set_xticklabels([str(f) for f in Fs], fontsize=5.8); ax.set_xlabel("number of fields $F$", fontsize=6.4)
ax.set_ylabel("seconds (Criteo, 2M rows)", fontsize=6.2); ax.tick_params(labelsize=5.6); style(ax)
ax.legend(loc="upper left", fontsize=4.9, handlelength=1.1, borderpad=0.3, labelspacing=0.15, handletextpad=0.3)
ax = axes[1]; yy = np.arange(len(SENS_V))
for off, ds, c, mk in [(-0.16, "ml1m20", "#2a78d6", "o"), (0.16, "avazu", "#eb6834", "s")]:
    ax.scatter([1e3 * SV2[ds][n] for n, _ in SENS_V], yy + off, color=c, marker=mk, s=10, zorder=3,
               label="ML-1M-20" if ds == "ml1m20" else "Avazu", edgecolor="white", linewidth=0.3)
ax.axvline(0, color=INK2, lw=0.6, ls=(0, (3, 2)))
ax.set_yticks(yy); ax.set_yticklabels([l for _, l in SENS_V], fontsize=5.8); ax.invert_yaxis()
ax.set_xlim(-2, 2); ax.set_xlabel("AUKC change vs default ($\\times10^{-3}$)", fontsize=6.2); ax.tick_params(labelsize=5.6)
ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
ax.legend(loc="lower right", fontsize=5.0, handletextpad=0.2, borderpad=0.3, labelspacing=0.15)
fig.tight_layout(w_pad=0.8)
fig.savefig(os.path.join(HERE, "figures", "scale_sens.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "figures", "scale_sens.png"), dpi=200, bbox_inches="tight")

# Serving cost of MOSAIC's sets (DeepFM seed-0 sets, exact vocabularies, random-weight timing)
SV = {}
for k, ds, model, lab, tag in DEEPFM:
    S = J(os.path.join(D, f"{ds}_serving.json")); F = len(S["fields"]); Lt = S["latency"]
    card = dict(zip(S["fields"], S["card"])); SV[ds] = S
    mb = lambda sub: sum(card[f] for f in sub) * S["bytes_per_value"] / 1e6
    m(f"{tag}memAll", f"{Lt[str(F)]['emb_mb']:.0f}" if Lt[str(F)]["emb_mb"] >= 10 else f"{Lt[str(F)]['emb_mb']:.1f}")
L = [r"\begin{table}[!h]", r"\centering",
     r"\caption{Serving cost of the smallest \mosaic{} set whose AUC is at least 99\% of the all-fields AUC, relative "
     r"to all fields. $\Delta$AUC: change in test AUC.}", r"\label{tab:serving}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
     r"\begin{tabular}{lcccccc}", r"\toprule",
     r" & & & & Memory & CPU & GPU \\", r"Dataset & $k$ & $\Delta$AUC & Lookups & (MB) & speed-up & speed-up \\", r"\midrule"]
for k, ds, model, lab, tag in DEEPFM:
    I = info[k]; S = SV[ds]; F = len(S["fields"]); Lt = S["latency"]
    kk = min(x for j, x in enumerate(I["ks"]) if I["rows"]["MOSAIC"]["curve"][j] >= 0.99 * I["allf"])
    j = I["ks"].index(kk); keep = I["rows"]["MOSAIC"]["curve"][j] / I["allf"]
    r, rA = Lt[str(kk)], Lt[str(F)]
    memr = r["emb_mb"] / rA["emb_mb"]; cpu = rA["dcnv2_cpu_b256_ms"] / r["dcnv2_cpu_b256_ms"]; gpu = rA["dcnv2_gpu_b4096_ms"] / r["dcnv2_gpu_b4096_ms"]
    def mbf(x): return f"{x:.1f}" if x < 10 else f"{x:.0f}"
    dl = I["rows"]["MOSAIC"]["curve"][j] - I["allf"]
    L.append(f"{lab} & {kk} & ${dl:+.4f}$ & {kk}/{F} & {mbf(r['emb_mb'])}/{mbf(rA['emb_mb'])} & "
             f"{cpu:.1f}$\\times$ & {gpu:.1f}$\\times$ \\\\")
    m(f"{tag}svK", str(kk)); m(f"{tag}svKeep", f"{100 * keep:.1f}"); m(f"{tag}svDelta", f"{dl:+.3f}"); m(f"{tag}svMem", f"{100 * memr:.0f}")
    m(f"{tag}svCpu", f"{cpu:.1f}"); m(f"{tag}svGpu", f"{gpu:.1f}")
    m(f"{tag}svMemX", f"{1 / memr:.0f}")
L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
open(os.path.join(HERE, "tables", "serving.tex"), "w").write("\n".join(L) + "\n")

# Figure: strongest redundant pairs measured by MOSAIC (share of the combined value of two fields that their overlap cancels)
PAIRK = {"ml1m20/deepfm": 2, "avazu/deepfm": 4, "criteo/deepfm": 5}
DSN = {"ml1m20/deepfm": "ML-1M-20", "avazu/deepfm": "Avazu", "criteo/deepfm": "Criteo"}
DSC = {"ml1m20/deepfm": "#2a78d6", "avazu/deepfm": "#eb6834", "criteo/deepfm": "#1baf7a"}
rows_ = []
for key, kb in PAIRK.items():
    R = info[key]["runs"][0]; F = len(R["fields"]); a_, b_ = budget_surrogate(R["est"], F, kb); fl = R["fields"]
    nm = (lambda f: f"field {f}") if key.startswith("criteo") else (lambda f: f)
    pr = sorted((b_[i, j], i, j) for i in range(F) for j in range(i + 1, F))[:4]
    pr = sorted(pr, key=lambda t: t[0] / (a_[t[1]] + a_[t[2]]))
    for v, i, j in pr:
        lab_ = f"{nm(fl[i])} – {fl[j]}" if key.startswith("criteo") else f"{fl[i]} – {fl[j]}"
        rows_.append((key, lab_.replace("field", "fields"), 100 * (-v) / (a_[i] + a_[j])))
    if "timestamp" in fl:
        t = fl.index("timestamp"); m("tsNeg", str(int((b_[t] < 0).sum()))); m("tsOther", str(F - 1))
fig, ax = plt.subplots(figsize=(3.35, 2.3))
y = 0; yt, yl = [], []
for gi, key in enumerate(PAIRK):
    G = [r for r in rows_ if r[0] == key]
    ax.text(0.6, y - 0.05, DSN[key], ha="left", va="center", fontsize=6.6, color=DSC[key])
    y += 1
    for _, lab_, v in G:
        ax.plot([0, v], [y, y], color=DSC[key], lw=1.6, solid_capstyle="butt", zorder=2)
        ax.scatter([v], [y], s=16, color=DSC[key], zorder=3, edgecolor="white", linewidth=0.4)
        ax.text(v + 0.8, y, f"{v:.0f}%", va="center", fontsize=5.6, color="#0b0b0b")
        yt.append(y); yl.append(lab_); y += 1
    y += 0.3
ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=5.8); ax.invert_yaxis()
ax.set_xlim(0, 40); ax.set_xlabel("value of the two fields lost to overlap (%)", fontsize=6.4)
ax.tick_params(axis="x", labelsize=5.6); ax.tick_params(axis="y", length=0)
for s_ in ("top", "right", "left"): ax.spines[s_].set_visible(False)
ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figures", "pairs.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(HERE, "figures", "pairs.png"), dpi=200, bbox_inches="tight")

# timestamp ablation and shared-model vs retrained agreement (ML-1M-20, DeepFM)
TS = J(os.path.join(D, "ml1m20_deepfm_sets_timestamp.json"))["sets"]
for nm_, key_ in [("tsLsix", "L6"), ("tsLeight", "L8"), ("tsLfg", "L6+first_genre"), ("tsLts", "L6+timestamp"),
                  ("tsMeight", "M8"), ("tsMts", "M8_hour_to_timestamp")]:
    m(nm_, fmt4(TS[key_]["test_auc"]))
AG = [v["spearman"] for p_ in sorted(glob.glob(os.path.join(D, "ml1m20_deepfm_s[0-9]_agreement.json")))
      for v in J(p_)["agreement"].values()]
m("agMin", f"{min(AG):.2f}"); m("agMax", f"{max(AG):.2f}"); m("agMean", f"{np.mean(AG):.2f}"); m("agN", str(len(AG)))

with open(os.path.join(HERE, "tex", "numbers.tex"), "w") as f:
    f.write("% auto-generated by make_tables.py - do not edit\n")
    for k, v in sorted(macros.items()): f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print("macros:", len(macros))
