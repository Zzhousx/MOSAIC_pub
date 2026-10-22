# MOSAIC: figures, tables and reported numbers

These scripts rebuild every result figure, every result table and every number quoted in the paper from the result
files in `results/`. No GPU is needed.

## Requirements
Python 3.9+ with `numpy`, `scipy` and `matplotlib`.

## Usage
```
python make_figures.py   # result figures, tables and tex/numbers.tex
python make_toy.py       # Figure 2 (exact construction of Theorem 3.3, no data needed)
```
Outputs go to `figures/` (PDF and PNG), `tables/` (LaTeX) and `tex/numbers.tex` (LaTeX macros with every number used in
the text).

## Map to the paper
| Paper | File | Script |
|---|---|---|
| Figure 2 | `figures/toy.pdf` | `make_toy.py` |
| Figure 3 | `figures/curves.pdf` | `make_figures.py` |
| Figure 4 | `figures/regretmap.pdf` | `make_figures.py` |
| Figure 5 | `figures/barcode.pdf` | `make_figures.py` |
| Figure 6 | `figures/stress.pdf` | `make_figures.py` |
| Figure 7 | `figures/scale_sens.pdf` | `make_figures.py` |
| Figure 8 | `figures/pairs.pdf` | `make_figures.py` |
| Table 3 | `tables/main.tex` | `make_figures.py` |
| Table 4 | `tables/steps.tex` | `make_figures.py` |
| Table 5 | `tables/cost.tex` | `make_figures.py` |
| Table 6 | `tables/serving.tex` | `make_figures.py` |

Figure 1 (the framework diagram) and Tables 1 and 2 are written directly in the paper source.

## Result files (`results/`)
| Pattern | Content |
|---|---|
| `<dataset>_<backbone>_s<seed>_v2.json` | One selection run of MOSAIC: test AUC per budget, the selected fields, the mask moments and the timings, plus the controlled-study rules and permutation importance computed in the same run |
| `<dataset>_<backbone>_<method>[_seed<s>].json` | A baseline ranking (AutoField, LPFS, SFS, SHARK, FairFS, mRMR) and the test AUC of its retrained top-k sets |
| `*_ladder.json` | The controlled study (Table 4) |
| `*_tm-*.json` | Training schemes of the shared model (Appendix C) |
| `*_sens.json` | Sensitivity to the settings of MOSAIC (Figure 7) |
| `scale_criteo_F*.json` | Cost as the number of fields grows (Figure 7) |
| `*_serving.json` | Vocabulary sizes and serving latency (Table 6) |
| `ml1m40c*` | Stress test with planted noisy copies (Figure 6) |
| `ml1m20_deepfm_s<seed>_agreement.json` | Shared model vs. retrained models on random field sets (Appendix C) |
| `reference_best.json` | Best mean test AUC reached at each budget, the reference for the worst regret (Section 5.1) |
| `ml1m20_deepfm_sets_timestamp.json` | The `timestamp` ablation (Section 5.2) |
| `ml1m20_fairfs_normk_selection.json` | Choice of the FairFS regularisation strength on ML-1M-20 (Appendix B) |

Every selected field set was retrained from scratch with the seeds 101, 102 and 103. All reported test AUCs are means
over these seeds.

`mosaic_lib.py` holds the selection routines of MOSAIC that the scripts reuse: the budget-local interaction model, the
search over field sets and the names of the controlled-study rules.
