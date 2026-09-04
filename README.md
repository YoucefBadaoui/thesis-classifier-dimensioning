# Impact of Machine-Learning Traffic Classification Errors on Multi-Class Capacity Dimensioning in Multirate Loss Systems

Analysis code, notebooks and result archives of the MSc thesis by Youcef
Badaoui, Poznan University of Technology, Faculty of Computing and
Telecommunications, Electronics and Telecommunications (ICT), 2026.
Supervisor: dr hab. inż. Piotr Zwierzykowski, Prof. of PUT, Institute of Communication and Computer Networks.

The framework treats a classifier's confusion matrix `C` as a linear operator
on the per-class offered-load vector, `a_hat = C^T a`, and feeds the apparent
load into the Kaufman-Roberts recursion for a full-availability multirate loss
system. Layout below names the module behind each output.

## Layout

| Path | Contents |
|---|---|
| `src/analytical/` | Kaufman-Roberts recursion, composition, blocking deviation, sensitivity tensor, EFPA cascade, scenario constants, published confusion matrices |
| `src/monte_carlo/` | Event-driven simulators: common-shock correlated errors, two-state Markov error process, path-based multi-link simulator |
| `src/cesnet/` | DataZoo loading and the classifier fitting shared by the CESNET scripts |
| `src/figures/` | One module per figure group plus the shared style; 14 modules render the 22 figures |
| `scripts/` | `regenerate_*.py` drivers at the top level; `cesnet/` (corpus stage), `iscx/` (classifier refreshes), `experiments/` (sweeps and the cascade), `checks/` (verification oracles) |
| `notebooks/` | Notebooks 01 to 04; executed records with their outputs under `executed/` |
| `data/processed/` | Every result archive the thesis quotes, in NumPy `.npz` and JSON form |
| `figures/` | The 22 rendered thesis figures, PDF for the manuscript and PNG for previews |
| `tests/` | pytest suite over the analytical core and the Markov error process |

The raw corpora are not redistributed. `data/processed/` carries every
result archive, so all tables and figures can be regenerated and every
number in the thesis can be checked without the corpora. The cleaned ISCX
table `iscx_5class_15s_clean.csv` is a derivative of the CIC corpus and is
not included either; notebook 01 writes it from the ARFF. Re-running the
classifier stages needs the corpora described below.

## Environment

The interpreter is CPython 3.13.12 and every pin is in `requirements.txt`.
The reference environment was created with `uv`:

```bash
uv venv -p 3.13.12
uv pip install -r requirements.txt
```

`cesnet-datazoo` pulls `torch` and the CUDA runtime; only the CESNET stage
needs it. `notebooks/02_classifiers.ipynb` configures XGBoost with
`device="cuda"`; on a CPU-only machine edit that cell, and pass
`--device cpu` to `scripts/iscx/iscx_seed_sweep.py`. Either change may move
the trained matrices slightly. The CESNET stage does not set `device`.

All notebooks declare the kernel `python3`; activate the environment before
starting a front end so the kernel resolves to it. Headless execution goes
through `nbclient.NotebookClient` with `kernel_name="python3"` and the
working directory set to `notebooks/`.

## Input data

**CESNET-TLS-Year22** (primary corpus). Hynek et al., Scientific Data, 2024,
doi 10.1038/s41597-024-03927-4, distributed through the CESNET DataZoo
library. The thesis uses the size `M` configuration (5e7 flows) through the
DataZoo class `CESNET_TLS_Year22`: train period `M-2022-9`, test period
`M-2022-10`, a validation split carved from the training period, 1,000,000
training rows, 1,000,000 validation rows, 3,000,000 test rows, model seeds
42, 7 and 123. Place the DataZoo download under `data/cesnet/M/`. The
archives record the corpus size and the category names.

**ISCX VPN-nonVPN 2016** (published anchor). Draper-Gil et al., Canadian
Institute for Cybersecurity, University of New Brunswick, time-based
feature ARFF files. Notebooks 01 and 02 read
`data/Scenario B-ARFF/TimeBasedFeatures-Dataset-15s-AllinOne.arff` and
`data/Scenario A2-ARFF/TimeBasedFeatures-Dataset-15s-VPN.arff`.

## Execution order

### Stage 1, ISCX notebooks (run from `notebooks/`)

| Step | Notebook or command | Consumes | Produces |
|---|---|---|---|
| 1 | `01_data_exploration.ipynb` | the AllinOne ARFF | `data/processed/iscx_5class_15s_clean.csv` |
| 2 | `02_classifiers.ipynb` | the CSV, the VPN ARFF | `data/processed/confusion_matrices.npz` |
| 3 | `make tables` | `confusion_matrices.npz` | `data/processed/analytical_results.npz` |
| 4 | `03_blocking_deviation.ipynb` | both archives | inline figures only, writes no file |
| 5 | `04_monte_carlo.ipynb` | both archives | `data/processed/monte_carlo_results.npz` |

Step 3 is a script and is not optional. Step 5 runs 30 replications of
5,000,000 arrivals each. The notebooks under `notebooks/executed/` are
records of past executions; do not execute them in place.

### Stage 2, CESNET scripts (run from the repository root)

```bash
python scripts/cesnet/cesnet_definitive.py --size M     # cesnet_definitive.npz, cesnet_definitive_eda.json
python scripts/cesnet/cesnet_degraded.py --size M       # cesnet_degraded.npz
python scripts/cesnet/cesnet_dimension.py               # cesnet_dimension.npz
python scripts/cesnet/cesnet_highk.py                   # cesnet_highk_real.npz
python scripts/cesnet/cesnet_mc.py                      # cesnet_mc_results.npz
python scripts/cesnet/cesnet_au_robustness.py           # cesnet_au_robustness.npz
python scripts/cesnet/cesnet_bootstrap_ci.py            # cesnet_bootstrap_ci.npz
python scripts/cesnet/cesnet_category_throughput.py --size M  # cesnet_category_throughput.json
python scripts/cesnet/cesnet_duration_weighted.py       # cesnet_duration_weighted.npz
python scripts/cesnet/cesnet_duration_compare.py        # cesnet_duration_compare.npz
```

Steps 1, 2, 8 and 9 read the corpus; the rest read the archives. Each
script's docstring states the rows, seeds and model families it fits.

### Stage 3, remaining archives

```bash
python scripts/experiments/cascade_analysis.py       # cascade_results.npz
python scripts/experiments/highk_power_analysis.py   # highk_power.npz
python scripts/experiments/markov_rho_sweep.py       # markov_rho_sweep_ott.npz, markov_rho_sweep_5g.npz
python scripts/experiments/shock_rho_sweep.py        # monte_carlo_rho_sweep_M{M}_{scenario}.npz
python scripts/iscx/iscx_seed_sweep.py        # iscx_seed_sweep.npz
python scripts/iscx/retrain_mlp_weighted.py   # refreshes mlp_clean and mlp_vpn_shift in confusion_matrices.npz
python scripts/iscx/iscx_reduced_feature_ranking.py   # refreshes the two reduced-feature slots; its comparison JSON is not distributed
```

Both sweeps are deterministic at a fixed M and arrival budget. The
common-shock replication mean is nevertheless unstable in M, because that
design averages a mixture over shock regimes: the shipped M = 30, M = 100 and
M = 300 archives differ materially at rho = 0.6. The figures and the oracle
read the M = 300 archives.

## Tables, figures and checks

```bash
make test       # pytest, 24 tests
make tables     # analytical_results.npz
make figures    # all 22 figures into figures/
make verify     # verify_core_math, verify_jensen_convexity, verify_rebinning_flip, validate_pipeline
```

`validate_pipeline.py` re-reads the raw ISCX ARFF when it is present and
skips that section otherwise, so the check count is lower on a fresh export.

## Citation and license

The thesis quotes the annotated tag `v1.0` of this repository;
`git rev-list -n1 v1.0` resolves it to the commit and `CITATION.cff` records
the same version. Code and result archives are MIT (see `LICENSE`). Work
using the corpora cites Hynek et al. (2024) and Draper-Gil et al. (2016);
CESNET-TLS-Year22 is published under CC BY 4.0 (Zenodo record 10608607),
which the category-level summaries in `data/processed/` inherit with that
attribution.
