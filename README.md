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

This release contains the experiments in the current paper. Development logs, checkpoints, experiment queues and exploratory greedy-comparison or tuning histories are excluded. The data under `figures/results/` are the paper's final plotting inputs, not execution histories. `reference_best.json` records the fixed reference values used by the paper's regret metric.

## Install and reproduce figures

Use Python 3.10–3.12. Install a CUDA-enabled PyTorch build appropriate for your machine when running experiments.

```bash
pip install -r requirements.txt
python figures/make_figures.py
python figures/make_toy.py
python -m unittest discover -s tests -v
```

Figures need no GPU. Outputs are written under `figures/figures/`, `figures/tables/` and `figures/tex/`.

## Prepare data

Download MovieLens-1M, Avazu and Criteo from their original public sources; raw data are not distributed here. The CSV encoding and 70/20/10 splits follow ERASE.

```bash
python scripts/prep_ml1m20.py /path/to/ml-1m data/ml1m20.npz
python scripts/export_erase_ml1m.py --npz data/ml1m20.npz --out erase_data_ml1m20
python scripts/prep_ctr.py avazu --inputs /path/to/train.csv --out third_party/ERASE/data/avazu/preprocessed_avazu.csv
python scripts/prep_ctr.py criteo --inputs /path/to/train.txt --out third_party/ERASE/data/criteo/preprocessed_criteo.csv
```

## Run experiments

Run commands from the repository root. They execute one job at a time and write fresh results under `results/`. Add `--dry` to inspect the commands without running them.

```bash
python scripts/reproduce.py main
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

FairFS uses the authors' unmodified DCN implementation and FuxiCTR v2.1.3. Set it up once with `python baselines/setup_fairfs.py`. Use a separate environment with `requirements-fairfs.txt`, and set `FAIRFS_PYTHON` to its Python executable before running baselines. To reproduce MovieLens validation-only regularisation selection, run `python baselines/fairfs.py --dataset ml1m20 --select-norm`. The other two datasets use fixed author settings. FairFS's own final retraining is not used.

Third-party provenance is listed in `third_party/SOURCES.json`; upstream licenses and notices are retained. No raw data, credentials or manuscript source are included.
