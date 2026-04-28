# ProSiT — Experimental Evaluation

Reproducibility bundle for the ProSiT (PROcess SImulation Tool) experimental study against four state-of-the-art simulators (AgentSimulator, SIMOD, RIMS, DSIM) on nine real and synthetic event logs.

This branch contains:

- the experiment driver (`run_experiments.py`) and its helpers (`experimental_utils/`),
- the `prosit` library (`src/prosit/`),
- three reproducibility zip archives (`data.zip`, `sota_results.zip`, `results.zip`),
- the per-dataset metrics and the aggregated `summary.json` (inside `results.zip`),
- this README with the full results.

---

## Table of Contents

- [What is in this branch](#what-is-in-this-branch)
- [Installation](#installation)
- [How to reproduce](#how-to-reproduce)
- [Datasets](#datasets)
- [Methods](#methods)
- [Metrics](#metrics)
- [Distance metrics](#distance-metrics)
- [Resource-based metrics](#resource-based-metrics)
- [Rule-based metrics](#rule-based-metrics)

---

## What is in this branch

```
.
├── README.md                  ← this file (instructions + result tables)
├── requirements.txt           ← pip dependencies (pinned)
├── environment.yml            ← conda environment (Python 3.10 + pinned pip deps)
│
├── run_experiments.py         ← main entry point: discover → simulate → evaluate
├── experimental_utils/        ← helpers imported by run_experiments.py
│   ├── evaluation.py            log-distance metrics
│   ├── tree_accuracy.py         decision-tree (rule-based) fidelity metrics
│   └── simulator_loaders.py     loaders for AgentSim/SIMOD/RIMS/DSIM outputs
│
├── src/prosit/                ← the ProSiT library (simulator + discovery + utils)
│
├── data.zip                   ← the 9 datasets (exp_data/)
├── sota_results.zip           ← the 10 simulated logs per (dataset, SOTA method)
└── results.zip                ← per-method JSONs, simulations CSVs, summary.json
```

---

## Installation

**Requirements:** Python 3.10.

### Option 1 — Conda (recommended)

```bash
git clone https://github.com/franvinci/prosit
cd prosit
git checkout experimental-evaluation
conda env create -f environment.yml
conda activate prosit
```

### Option 2 — pip

```bash
git clone https://github.com/franvinci/prosit
cd prosit
git checkout experimental-evaluation
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## How to reproduce

### Step 1 — Unzip the data and SOTA outputs

```bash
unzip data.zip            # → exp_data/<dataset>/{log_train.xes, log_test.xes, model.pnml}
unzip sota_results.zip    # → agentsimulator/, simod/, RIMS/, RESULT_RIMS_SPECIAL_ISSUE/
```

`sota_results.zip` recreates exactly the four directory trees that `simulator_loaders.py` reads from. No extra setup is needed for the SOTA simulators themselves: the bundle ships their pre-computed simulation logs (10 per dataset/method).

### Step 2 — (a) Re-run the full pipeline

```bash
python run_experiments.py --workers 10
```

For each dataset this:

1. discovers ProSiT simulation parameters (`max_depth_tree=5`) from `log_train.xes`,
2. simulates 10 logs,
3. loads up to 10 simulation logs from each SOTA method,
4. evaluates every simulation against `log_test.xes` (distance and resource metrics) and against the discovered ProSiT decision trees (rule-based metrics).

Outputs land in `results/`:

```
results/
├── <dataset>/
│   ├── prosit.json                    per-metric mean/std/values across 10 runs
│   ├── agentsimulator.json
│   ├── simod.json
│   ├── rims.json
│   ├── dsim.json
│   └── simulations/prosit/simulated_log_*.csv  (ProSiT only — SOTA logs already on disk)
└── summary.json                       one consolidated file across all datasets/methods
```

CLI flags:

- `--datasets <name>...` — restrict to a subset (default: all 9).
- `--workers N` — parallel workers for per-run evaluation (default: 10).

### Step 2 — (b) Skip the run, use pre-computed results

```bash
unzip results.zip
```

The shipped `results/` folder contains exactly what `run_experiments.py` would produce, including the 10 simulated PROSIT logs per dataset. The numbers in this README are computed from the shipped `results/summary.json`.

---

## Datasets

Nine event logs span real-world public BPI Challenge logs, confidential private logs, and synthetic loan/pharmacy scenarios. All are split chronologically into a training half (used for parameter discovery) and a test half (used as ground truth for evaluation).

| Label | Folder | Source |
|---|---|---|
| BPIC12W | `BPI_Challenge_2012_W_Two_TS` | BPI Challenge 2012, W subprocess, two-timestamp |
| BPIC17W | `BPI_Challenge_2017_W_Two_TS` | BPI Challenge 2017, W subprocess, two-timestamp |
| P2P-1000 | `confidential_1000` | Synthetic P2P, 1000 cases |
| P2P-2000 | `confidential_2000` | Synthetic P2P, 2000 cases |
| ACR | `ConsultaDataMining201618` | Consulta Data Mining 2016-18 |
| CVS | `cvs_pharmacy` | CVS Pharmacy retail process |
| Production | `Production` | Manufacturing process |
| PurchasingExample | `PurchasingExample` | Purchasing process textbook example |
| SynLoan | `SynLoan` | Synthetic loan application |

Each dataset folder under `exp_data/` ships with `log_train.xes`, `log_test.xes`, and the inductive-mined `model.pnml`.

---

## Methods

Five methods are compared.

| Method | Mode | Resources | Case attributes | Notes |
|---|---|---|---|---|
| **ProSiT** | white-box, decision-tree rules (`max_depth_tree=5`) | yes | yes | this work |
| AgentSim | white-box | yes | no | runs only on 5 datasets (no public output for `confidential_*`, `cvs_pharmacy`, `SynLoan`) |
| SIMOD | white-box | yes | yes (some datasets) | BPMN-based |
| RIMS | black-box (LSTM) | no (`role` only) | yes (some datasets) | sequence-to-sequence model |
| DSIM | black-box (LSTM) | no | no | sequence-to-sequence model |

Cells marked `—` in the result tables below mean the method does not produce the required signal for that metric, or there is no simulated log for that (dataset, method) pair.

---

## Metrics

Distance metrics are **lower is better** (`↓`); the entropy metric is **higher is better** (`↑`) — it measures how much variability the model preserves, not how close it is to the test log.

### Log-distance metrics (`experimental_utils/evaluation.py`)

| Metric | Type | Description |
|---|---|---|
| `2gd` ↓ | n-gram | 2-gram distribution distance over activity bigrams |
| `car` ↓ | timestamp | case arrival rate distance (Wasserstein over inter-arrival times) |
| `ctd` ↓ | timestamp | cycle time distribution distance (Wasserstein) |
| `etd_entropy` ↑ | entropy | Shannon entropy of execution-time distribution |
| `2rgd` ↓ | resource | 2-gram distribution distance over `(activity, resource)` bigrams; `—` for resource-less methods |

### Rule-based metrics (`experimental_utils/tree_accuracy.py`)

These quantify how faithfully the simulated log respects the conditional structure that ProSiT learned from the training log. Each metric is the gap between ProSiT's discovered decision tree and the simulated log for the corresponding component:

| Metric | What is compared | Direction |
|---|---|---|
| `arrival_tree` | inter-arrival time distribution per arrival-calendar slot | ↓ Wasserstein |
| `execution_tree` | execution time per `(activity, resource, history)` leaf | ↓ Wasserstein |
| `waiting_tree` | waiting time per `(resource, history)` leaf | ↓ Wasserstein |
| `transition_tree` | routing probability per decision-point leaf | ↓ MAE |
| `resource_tree` | resource-selection probability per `(activity, history)` leaf | ↓ MAE |

Methods that do not emit resources or routing-relevant signals report `—` where appropriate.

---

## Distance metrics

Highlight convention used in every table:

- **bold** — best overall (across all 5 methods) per dataset
- *italic* — best white-box (ProSiT / AgentSim / SIMOD) per dataset, only when distinct from the overall best (i.e. when a black-box method wins)

#### `2gd` ↓ — 2-gram distribution distance

| Dataset | ProSiT | AgentSim | SIMOD | RIMS | DSIM |
|---|---|---|---|---|---|
| BPIC12W | **0.398 ± 0.004** | 0.456 ± 0.006 | 0.525 ± 0.006 | 0.525 ± 0.006 | 0.510 ± 0.007 |
| BPIC17W | **0.214 ± 0.004** | 0.258 ± 0.001 | 0.424 ± 0.003 | 0.479 ± 0.003 | 0.549 ± 0.002 |
| P2P-1000 | **0.115 ± 0.011** | — | 0.189 ± 0.009 | 0.180 ± 0.008 | 0.184 ± 0.007 |
| P2P-2000 | **0.104 ± 0.012** | — | 0.187 ± 0.008 | 0.195 ± 0.010 | 0.210 ± 0.012 |
| ACR | *0.266 ± 0.021* | 0.375 ± 0.026 | 0.279 ± 0.015 | **0.221 ± 0.020** | 0.266 ± 0.001 |
| CVS | **0.016 ± 0.001** | — | 0.021 ± 0.002 | 0.221 ± 0.003 | 0.256 ± 0.002 |
| Production | **0.462 ± 0.031** | 0.561 ± 0.023 | 0.565 ± 0.021 | 0.764 ± 0.015 | 0.763 ± 0.017 |
| PurchasingExample | *0.250 ± 0.015* | 0.420 ± 0.011 | 0.253 ± 0.024 | **0.160 ± 0.016** | 0.439 ± 0.005 |
| SynLoan | 0.121 ± 0.006 | — | **0.113 ± 0.007** | 0.400 ± 0.018 | 0.439 ± 0.001 |

#### `car` ↓ — case arrival rate distance

| Dataset | ProSiT | AgentSim | SIMOD | RIMS | DSIM |
|---|---|---|---|---|---|
| BPIC12W | 21.745 ± 9.654 | 51.408 ± 0.136 | **14.350 ± 4.883** | 24.446 ± 4.754 | 23.802 ± 5.022 |
| BPIC17W | 165.327 ± 9.096 | 178.596 ± 0.128 | *156.445 ± 7.511* | 39.051 ± 9.829 | **34.904 ± 10.032** |
| P2P-1000 | 81.935 ± 22.634 | — | **66.852 ± 30.260** | 235.137 ± 9.745 | 236.599 ± 7.354 |
| P2P-2000 | 146.604 ± 45.407 | — | **49.223 ± 19.658** | 646.689 ± 3.856 | 641.835 ± 5.488 |
| ACR | *264.958 ± 33.679* | 368.133 ± 0.037 | 292.398 ± 26.280 | 244.998 ± 6.640 | **239.763 ± 8.422** |
| CVS | **4.994 ± 2.415** | — | 5.184 ± 0.900 | 19.287 ± 1.336 | 20.370 ± 1.751 |
| Production | **30.600 ± 17.695** | 95.884 ± 0.591 | 69.858 ± 28.931 | 167.611 ± 3.652 | 167.733 ± 2.623 |
| PurchasingExample | 664.974 ± 51.290 | 787.604 ± 0.000 | **594.018 ± 34.727** | 900.444 ± 5.919 | 893.916 ± 9.187 |
| SynLoan | **3.420 ± 1.299** | — | 3.498 ± 2.216 | 326.658 ± 3.657 | 326.035 ± 3.963 |

#### `ctd` ↓ — cycle time distribution distance

| Dataset | ProSiT | AgentSim | SIMOD | RIMS | DSIM |
|---|---|---|---|---|---|
| BPIC12W | **66.345 ± 4.369** | 70.855 ± 3.172 | 77.236 ± 2.594 | 100.291 ± 2.747 | 163.183 ± 1.692 |
| BPIC17W | 109.807 ± 2.857 | 67.475 ± 2.660 | **46.082 ± 1.539** | 87.000 ± 0.948 | 115.763 ± 0.723 |
| P2P-1000 | *14.323 ± 1.273* | — | 41.174 ± 12.886 | **9.249 ± 0.256** | 10.052 ± 0.281 |
| P2P-2000 | *11.770 ± 1.483* | — | 78.019 ± 14.557 | 2.483 ± 1.483 | **2.369 ± 0.631** |
| ACR | *244.062 ± 33.255* | 470.875 ± 45.046 | 355.149 ± 28.821 | **46.610 ± 7.250** | 64.036 ± 1.490 |
| CVS | 146.904 ± 0.998 | — | *103.121 ± 1.681* | 97.901 ± 1.064 | **37.627 ± 0.907** |
| Production | *220.122 ± 29.775* | 221.173 ± 29.017 | 410.071 ± 79.595 | 34.816 ± 6.330 | **21.556 ± 4.580** |
| PurchasingExample | **439.985 ± 41.762** | 448.430 ± 15.507 | 445.978 ± 22.374 | 584.773 ± 8.558 | 632.178 ± 4.503 |
| SynLoan | **117.561 ± 12.112** | — | 457.960 ± 27.600 | 290.938 ± 16.227 | 467.776 ± 20.804 |

> Note on ACR and CVS: both datasets have a documented cycle-time distribution shift between train and test halves. ProSiT learns from the training half only, so the train/test gap shows up in `ctd` independently of how the model fits. Black-box LSTM methods (RIMS, DSIM) implicitly absorb the shift through their last-hidden-state warmup and look better on `ctd` for these two datasets.

#### `etd_entropy` ↑ — execution time distribution entropy

| Dataset | ProSiT | AgentSim | SIMOD | RIMS | DSIM |
|---|---|---|---|---|---|
| BPIC12W | **3.027 ± 0.015** | 2.376 ± 0.035 | 2.208 ± 0.010 | 0.298 ± 0.006 | 0.300 ± 0.011 |
| BPIC17W | 2.098 ± 0.008 | **2.804 ± 0.030** | 2.108 ± 0.009 | 0.688 ± 0.006 | 0.512 ± 0.004 |
| P2P-1000 | **1.630 ± 0.017** | — | 0.762 ± 0.028 | 0.296 ± 0.017 | 0.347 ± 0.029 |
| P2P-2000 | **1.903 ± 0.011** | — | 0.655 ± 0.017 | 0.271 ± 0.034 | 0.494 ± 0.025 |
| ACR | **0.881 ± 0.030** | 0.369 ± 0.061 | 0.712 ± 0.032 | 0.873 ± 0.019 | 0.786 ± 0.045 |
| CVS | 0.093 ± 0.002 | — | **0.242 ± 0.003** | 0.145 ± 0.004 | 0.056 ± 0.002 |
| Production | **1.680 ± 0.042** | 1.009 ± 0.083 | 1.287 ± 0.038 | 0.914 ± 0.039 | 1.270 ± 0.050 |
| PurchasingExample | **1.477 ± 0.010** | 0.257 ± 0.012 | 1.102 ± 0.064 | 0.334 ± 0.027 | 0.595 ± 0.047 |
| SynLoan | **2.316 ± 0.012** | — | 2.260 ± 0.008 | 1.057 ± 0.014 | 0.343 ± 0.011 |

---

## Resource-based metrics

Restricted to methods that emit `org:resource`. RIMS and DSIM only attach a coarse `role` label, not a real resource, so their cells are `—`. AgentSimulator outputs are shipped only for 5 of the 9 datasets.

#### `2rgd` ↓ — 2-gram resource bigram distance

| Dataset | ProSiT | AgentSim | SIMOD | RIMS | DSIM |
|---|---|---|---|---|---|
| BPIC12W | **0.626 ± 0.008** | 0.694 ± 0.021 | 0.692 ± 0.004 | — | — |
| BPIC17W | **0.637 ± 0.002** | 0.827 ± 0.007 | 0.777 ± 0.001 | — | — |
| P2P-1000 | **0.123 ± 0.008** | — | 0.188 ± 0.010 | — | — |
| P2P-2000 | **0.115 ± 0.010** | — | 0.216 ± 0.012 | — | — |
| ACR | **0.832 ± 0.007** | 0.969 ± 0.003 | 0.962 ± 0.004 | — | — |
| CVS | **0.018 ± 0.002** | — | 0.368 ± 0.002 | — | — |
| Production | **0.620 ± 0.018** | 0.845 ± 0.018 | 0.833 ± 0.011 | — | — |
| PurchasingExample | **0.384 ± 0.014** | 0.563 ± 0.019 | 0.396 ± 0.015 | — | — |
| SynLoan | **0.086 ± 0.004** | — | 0.087 ± 0.006 | — | — |

---

## Rule-based metrics

How faithfully each simulator reproduces the conditional structure that ProSiT learned from the training log. Lower is better.

#### `arrival_tree` ↓ — inter-arrival fidelity per arrival-calendar slot

| Dataset | ProSiT | AgentSim | SIMOD | RIMS | DSIM |
|---|---|---|---|---|---|
| BPIC12W | **0.529 ± 0.101** | 1.749 ± 0.303 | 3.478 ± 2.091 | 1.120 ± 0.304 | 1.518 ± 0.155 |
| BPIC17W | **0.289 ± 0.019** | 1.465 ± 0.412 | 5.287 ± 0.264 | 2.514 ± 0.116 | 2.506 ± 0.104 |
| P2P-1000 | **7.008 ± 2.272** | — | 32.196 ± 5.778 | 38.432 ± 1.713 | 39.178 ± 0.957 |
| P2P-2000 | **6.076 ± 2.737** | — | 86.109 ± 3.265 | 63.261 ± 0.725 | 62.610 ± 0.798 |
| ACR | **14.192 ± 2.845** | 36.344 ± 0.125 | 57.747 ± 5.720 | 46.070 ± 2.581 | 46.723 ± 2.251 |
| CVS | **0.273 ± 0.016** | — | 0.279 ± 0.010 | 0.605 ± 0.034 | 0.593 ± 0.047 |
| Production | **46.790 ± 17.380** | 130.656 ± 2.707 | 301.825 ± 311.528 | 214.825 ± 6.231 | 213.807 ± 6.891 |
| PurchasingExample | **31.823 ± 10.816** | 94.443 ± 0.000 | 51.727 ± 18.422 | 178.992 ± 3.007 | 175.419 ± 4.915 |
| SynLoan | **2.465 ± 0.776** | — | 12.201 ± 1.305 | 98.475 ± 1.006 | 98.440 ± 0.993 |

#### `execution_tree` ↓ — execution time per (activity, resource, history) leaf

| Dataset | ProSiT | AgentSim | SIMOD | RIMS | DSIM |
|---|---|---|---|---|---|
| BPIC12W | **1.981 ± 0.805** | — | 25.557 ± 1.203 | — | — |
| BPIC17W | **0.358 ± 0.042** | — | 10.711 ± 1.700 | — | — |
| P2P-1000 | **2.943 ± 0.318** | — | 23.211 ± 2.855 | — | — |
| P2P-2000 | **2.396 ± 0.333** | — | 30.103 ± 2.655 | — | — |
| ACR | **12.572 ± 1.400** | 149.799 ± 17.899 | 91.481 ± 14.641 | — | — |
| CVS | **0.465 ± 0.098** | — | 1.078 ± 0.213 | — | — |
| Production | **26.990 ± 5.821** | — | 112.548 ± 28.257 | — | — |
| PurchasingExample | **7.416 ± 1.033** | 143.656 ± 8.605 | 66.457 ± 16.202 | — | — |
| SynLoan | **2.049 ± 0.238** | — | 6.517 ± 0.170 | — | — |

#### `waiting_tree` ↓ — waiting time per (resource, history) leaf

| Dataset | ProSiT | AgentSim | SIMOD | RIMS | DSIM |
|---|---|---|---|---|---|
| BPIC12W | **933.025 ± 188.846** | — | 1954.389 ± 20.516 | — | — |
| BPIC17W | **504.096 ± 57.993** | — | 2372.497 ± 25.522 | — | — |
| P2P-1000 | **8.689 ± 0.832** | — | 67.953 ± 8.948 | — | — |
| P2P-2000 | **6.959 ± 0.538** | — | 226.779 ± 21.798 | — | — |
| ACR | **1073.039 ± 169.500** | 4748.072 ± 606.664 | 3429.664 ± 178.267 | — | — |
| CVS | **5.894 ± 1.262** | — | 554.463 ± 4.149 | — | — |
| Production | **390.361 ± 276.284** | — | 1745.019 ± 229.809 | — | — |
| PurchasingExample | **206.001 ± 40.754** | 547.383 ± 54.020 | 537.600 ± 32.344 | — | — |
| SynLoan | **362.052 ± 60.893** | — | 700.917 ± 50.813 | — | — |

#### `transition_tree` ↓ — routing probability per decision-point leaf

| Dataset | ProSiT | AgentSim | SIMOD | RIMS | DSIM |
|---|---|---|---|---|---|
| BPIC12W | **0.017 ± 0.002** | — | 0.169 ± 0.007 | 0.114 ± 0.004 | — |
| BPIC17W | **0.008 ± 0.001** | — | 0.195 ± 0.002 | 0.136 ± 0.002 | — |
| P2P-1000 | **0.031 ± 0.017** | — | 0.276 ± 0.027 | 0.277 ± 0.025 | 0.289 ± 0.029 |
| P2P-2000 | **0.024 ± 0.007** | — | 0.132 ± 0.009 | 0.142 ± 0.009 | 0.131 ± 0.007 |
| ACR | **0.019 ± 0.004** | 0.048 ± 0.010 | 0.090 ± 0.014 | 0.134 ± 0.014 | 0.292 ± 0.054 |
| CVS | **0.002 ± 0.000** | — | 0.038 ± 0.001 | 0.178 ± 0.002 | 0.081 ± 0.001 |
| Production | **0.046 ± 0.005** | — | 0.134 ± 0.012 | — | — |
| PurchasingExample | **0.027 ± 0.006** | 0.128 ± 0.030 | 0.108 ± 0.024 | 0.357 ± 0.023 | 0.141 ± 0.009 |
| SynLoan | **0.020 ± 0.004** | — | 0.091 ± 0.009 | 0.129 ± 0.004 | — |

#### `resource_tree` ↓ — resource-selection probability per (activity, history) leaf

| Dataset | ProSiT | AgentSim | SIMOD | RIMS | DSIM |
|---|---|---|---|---|---|
| BPIC12W | **0.007 ± 0.000** | — | 0.021 ± 0.000 | — | — |
| BPIC17W | **0.005 ± 0.000** | — | 0.010 ± 0.000 | — | — |
| P2P-1000 | **0.102 ± 0.005** | — | 0.115 ± 0.008 | — | — |
| P2P-2000 | **0.116 ± 0.008** | — | 0.209 ± 0.008 | — | — |
| ACR | **0.005 ± 0.001** | 0.010 ± 0.001 | 0.011 ± 0.001 | — | — |
| CVS | **0.165 ± 0.001** | — | 0.166 ± 0.002 | — | — |
| Production | **0.036 ± 0.005** | — | 0.058 ± 0.003 | — | — |
| PurchasingExample | **0.042 ± 0.008** | 0.104 ± 0.015 | 0.043 ± 0.004 | — | — |
| SynLoan | 0.061 ± 0.005 | — | **0.049 ± 0.004** | — | — |
