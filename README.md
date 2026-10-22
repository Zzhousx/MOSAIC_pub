# MOSAIC

Code for **MOSAIC: Measuring Field Interactions with One Masked Model for Feature Selection in Web Recommendation**.

MOSAIC trains one field-masked model, estimates field and pair effects, builds a budget-local interaction model, and checks candidate sets on the frozen shared model. Final selected sets are retrained with seeds 101, 102 and 103.

## Contents

- `mosaic/`: DeepFM/DCNv2, mask moments, selection and the paper's controlled experiments.
- `configs/paper.json`: dataset settings, budgets and seeds.
- `scripts/`: data preparation and experiment entry points.
- `baselines/`, `third_party/`: baseline adapters and pinned upstream sources.
- `figures/`: the supplied figure/table scripts and the final data used in the paper.
- `tests/`: CPU correctness and publication-data checks.

This release contains the experiments reported in the paper. The data under `figures/results/` are the final plotting inputs. `reference_best.json` contains the paper's fixed per-budget reference AUCs for the regret metric; these values are not re-estimated from a new run.

## Install and reproduce figures

Use Python 3.10–3.12. Install a CUDA-enabled PyTorch build appropriate for your machine when running experiments.

```bash
pip install -r requirements.txt
python figures/make_figures.py
python figures/make_toy.py
python -m unittest discover -s tests -v
```

Figures need no GPU. Outputs are written under `figures/figures/`, `figures/tables/` and `figures/tex/`.
These commands rebuild the submitted figures from the bundled inputs. To plot a complete set of new experiment outputs instead, use:

```bash
python scripts/reproduce.py figures --results-dir results --output-dir reproduced_figures --reference figures/results/reference_best.json
```

This reads new results and measured search times from `results/`; it does not mix them with the bundled results. The reference AUCs remain the explicitly supplied fixed paper reference. Hardware-dependent timings will differ. The bundled search-time summaries are stored in `figures/results/search_times.json`; new baseline runs record their own times in `results/search_times.json`.

## Prepare data

Download MovieLens-1M, Avazu and Criteo from their original public sources; raw data are not distributed here. The CSV encoding and 70/20/10 splits follow ERASE.

```bash
python scripts/prep_ml1m20.py /path/to/ml-1m data/ml1m20.npz
python scripts/export_erase_ml1m.py --npz data/ml1m20.npz --out erase_data_ml1m20
python scripts/prep_ctr.py avazu --inputs /path/to/train.csv --out third_party/ERASE/data/avazu/preprocessed_avazu.csv
python scripts/prep_ctr.py criteo --inputs /path/to/train.txt --out third_party/ERASE/data/criteo/preprocessed_criteo.csv
```

## Run experiments

Run commands from the repository root, with one invocation active at a time. Commands run serially and write to `results/`. Existing result files are retained and skipped; use a clean output directory for a fresh reproduction. Add `--dry` to inspect the commands without running them.

FairFS requires a separate environment. First run `python baselines/setup_fairfs.py`, install `requirements-fairfs.txt` in that environment, and set `FAIRFS_PYTHON` to its Python executable. Run the main experiments before the MovieLens validation-only selection, then the baselines:

```bash
python scripts/reproduce.py main
python baselines/fairfs.py --dataset ml1m20 --select-norm
python scripts/reproduce.py baselines
python scripts/reproduce.py controls
python scripts/reproduce.py mask-training
python scripts/reproduce.py sensitivity
python scripts/reproduce.py copies
python scripts/reproduce.py main --dataset ml1m40c
python scripts/reproduce.py baselines --dataset ml1m40c
python scripts/reproduce.py scale
python scripts/reproduce.py serving
python scripts/reproduce.py agreement
python scripts/reproduce.py timestamp
```

Main comparisons use five selection seeds for DeepFM and three for DCNv2; controls use seeds 0–2. External rankings use seeds 2027–2029. The MovieLens mRMR runs use all training rows; larger datasets use independent 2M-row samples. SHARK importance uses a 100k-row validation sample.

FairFS uses the authors' unmodified DCN implementation and FuxiCTR v2.1.3. MovieLens selects its regularisation strength using validation AUC only; Avazu and Criteo use fixed author settings. Every selected set uses the same final MOSAIC evaluation protocol. Sensitivity uses one shared model per selection seed and varies only the reported kernel widths, candidate counts and half-row moment estimate. Serving measures DCNv2 CPU batch-256 and GPU batch-4096 forward latency on the first DeepFM run's selected sets.

Third-party provenance is listed in `third_party/SOURCES.json`; upstream licenses and notices are retained. No raw data, credentials or manuscript source are included.
