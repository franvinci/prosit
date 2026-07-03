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
| BPIC17W | **0.216 ± 0.003** | 0.258 ± 0.001 | 0.424 ± 0.003 | 0.479 ± 0.003 | 0.549 ± 0.002 |
| P2P-1000 | **0.116 ± 0.010** | — | 0.189 ± 0.009 | 0.180 ± 0.008 | 0.184 ± 0.007 |
| P2P-2000 | **0.106 ± 0.011** | — | 0.187 ± 0.008 | 0.195 ± 0.010 | 0.210 ± 0.012 |
| ACR | *0.264 ± 0.023* | 0.375 ± 0.026 | 0.279 ± 0.015 | **0.221 ± 0.020** | 0.266 ± 0.001 |
| CVS | **0.016 ± 0.001** | — | 0.021 ± 0.002 | 0.221 ± 0.003 | 0.256 ± 0.002 |
| Production | **0.486 ± 0.029** | 0.561 ± 0.023 | 0.565 ± 0.021 | 0.764 ± 0.015 | 0.763 ± 0.017 |
| PurchasingExample | *0.248 ± 0.014* | 0.420 ± 0.011 | 0.253 ± 0.024 | **0.160 ± 0.016** | 0.439 ± 0.005 |
| SynLoan | 0.127 ± 0.012 | — | **0.113 ± 0.007** | 0.400 ± 0.018 | 0.439 ± 0.001 |

#### `car` ↓ — case arrival rate distance

| Dataset | ProSiT | AgentSim | SIMOD | RIMS | DSIM |
|---|---|---|---|---|---|
| BPIC12W | **12.519 ± 6.272** | 51.408 ± 0.136 | 14.350 ± 4.883 | 24.446 ± 4.754 | 23.802 ± 5.022 |
| BPIC17W | *141.054 ± 11.784* | 178.596 ± 0.128 | 156.445 ± 7.511 | 39.051 ± 9.829 | **34.904 ± 10.032** |
| P2P-1000 | 71.011 ± 16.839 | — | **66.852 ± 30.260** | 235.137 ± 9.745 | 236.599 ± 7.354 |
| P2P-2000 | 132.228 ± 50.844 | — | **49.223 ± 19.658** | 646.689 ± 3.856 | 641.835 ± 5.488 |
| ACR | *252.393 ± 44.541* | 368.133 ± 0.037 | 292.398 ± 26.280 | 244.998 ± 6.640 | **239.763 ± 8.422** |
| CVS | 5.649 ± 0.772 | — | **5.184 ± 0.900** | 19.287 ± 1.336 | 20.370 ± 1.751 |
| Production | **32.029 ± 13.758** | 95.884 ± 0.591 | 69.858 ± 28.931 | 167.611 ± 3.652 | 167.733 ± 2.623 |
| PurchasingExample | 633.149 ± 48.735 | 787.604 ± 0.000 | **594.018 ± 34.727** | 900.444 ± 5.919 | 893.916 ± 9.187 |
| SynLoan | 3.707 ± 1.978 | — | **3.498 ± 2.216** | 326.658 ± 3.657 | 326.035 ± 3.963 |

#### `ctd` ↓ — cycle time distribution distance

| Dataset | ProSiT | AgentSim | SIMOD | RIMS | DSIM |
|---|---|---|---|---|---|
| BPIC12W | **44.553 ± 3.902** | 70.855 ± 3.172 | 77.236 ± 2.594 | 100.291 ± 2.747 | 163.183 ± 1.692 |
| BPIC17W | 106.389 ± 2.153 | 67.475 ± 2.660 | **46.082 ± 1.539** | 87.000 ± 0.948 | 115.763 ± 0.723 |
| P2P-1000 | *14.850 ± 1.829* | — | 41.174 ± 12.886 | **9.249 ± 0.256** | 10.052 ± 0.281 |
| P2P-2000 | *10.629 ± 1.726* | — | 78.019 ± 14.557 | 2.483 ± 1.483 | **2.369 ± 0.631** |
| ACR | *229.489 ± 18.183* | 470.875 ± 45.046 | 355.149 ± 28.821 | **46.610 ± 7.250** | 64.036 ± 1.490 |
| CVS | 145.686 ± 0.758 | — | *103.121 ± 1.681* | 97.901 ± 1.064 | **37.627 ± 0.907** |
| Production | 335.896 ± 49.422 | *221.173 ± 29.017* | 410.071 ± 79.595 | 34.816 ± 6.330 | **21.556 ± 4.580** |
| PurchasingExample | **402.283 ± 25.978** | 448.430 ± 15.507 | 445.978 ± 22.374 | 584.773 ± 8.558 | 632.178 ± 4.503 |
| SynLoan | **153.435 ± 12.224** | — | 457.960 ± 27.600 | 290.938 ± 16.227 | 467.776 ± 20.804 |

> Note on ACR and CVS: both datasets have a documented cycle-time distribution shift between train and test halves. ProSiT learns from the training half only, so the train/test gap shows up in `ctd` independently of how the model fits. Black-box LSTM methods (RIMS, DSIM) implicitly absorb the shift through their last-hidden-state warmup and look better on `ctd` for these two datasets.

#### `etd_entropy` ↑ — execution time distribution entropy

| Dataset | ProSiT | AgentSim | SIMOD | RIMS | DSIM |
|---|---|---|---|---|---|
| BPIC12W | **2.863 ± 0.014** | 2.376 ± 0.035 | 2.208 ± 0.010 | 0.298 ± 0.006 | 0.300 ± 0.011 |
| BPIC17W | 2.618 ± 0.007 | **2.804 ± 0.030** | 2.108 ± 0.009 | 0.688 ± 0.006 | 0.512 ± 0.004 |
| P2P-1000 | **1.615 ± 0.017** | — | 0.762 ± 0.028 | 0.296 ± 0.017 | 0.347 ± 0.029 |
| P2P-2000 | **1.872 ± 0.009** | — | 0.655 ± 0.017 | 0.271 ± 0.034 | 0.494 ± 0.025 |
| ACR | **0.943 ± 0.021** | 0.369 ± 0.061 | 0.712 ± 0.032 | 0.873 ± 0.019 | 0.786 ± 0.045 |
| CVS | 0.094 ± 0.002 | — | **0.242 ± 0.003** | 0.145 ± 0.004 | 0.056 ± 0.002 |
| Production | **1.641 ± 0.044** | 1.009 ± 0.083 | 1.287 ± 0.038 | 0.914 ± 0.039 | 1.270 ± 0.050 |
| PurchasingExample | **1.460 ± 0.008** | 0.257 ± 0.012 | 1.102 ± 0.064 | 0.334 ± 0.027 | 0.595 ± 0.047 |
| SynLoan | **2.304 ± 0.009** | — | 2.260 ± 0.008 | 1.057 ± 0.014 | 0.343 ± 0.011 |

---

## Resource-based metrics

Restricted to methods that emit `org:resource`. RIMS and DSIM only attach a coarse `role` label, not a real resource, so their cells are `—`. AgentSimulator outputs are shipped only for 5 of the 9 datasets.

#### `2rgd` ↓ — 2-gram resource bigram distance

| Dataset | ProSiT | AgentSim | SIMOD | RIMS | DSIM |
|---|---|---|---|---|---|
| BPIC12W | **0.632 ± 0.006** | 0.694 ± 0.021 | 0.692 ± 0.004 | — | — |
| BPIC17W | **0.638 ± 0.003** | 0.827 ± 0.007 | 0.777 ± 0.001 | — | — |
| P2P-1000 | **0.134 ± 0.007** | — | 0.188 ± 0.010 | — | — |
| P2P-2000 | **0.126 ± 0.009** | — | 0.216 ± 0.012 | — | — |
| ACR | **0.835 ± 0.009** | 0.969 ± 0.003 | 0.962 ± 0.004 | — | — |
| CVS | **0.016 ± 0.003** | — | 0.368 ± 0.002 | — | — |
| Production | **0.624 ± 0.014** | 0.845 ± 0.018 | 0.833 ± 0.011 | — | — |
| PurchasingExample | **0.379 ± 0.011** | 0.563 ± 0.019 | 0.396 ± 0.015 | — | — |
| SynLoan | 0.092 ± 0.005 | — | **0.087 ± 0.006** | — | — |

---

## Rule-based metrics

How faithfully each simulator reproduces the conditional structure that ProSiT learned from the training log. Lower is better.

#### `arrival_tree` ↓ — inter-arrival fidelity per arrival-calendar slot

| Dataset | ProSiT | AgentSim | SIMOD | RIMS | DSIM |
|---|---|---|---|---|---|
| BPIC12W | **1.569 ± 0.192** | 6.647 ± 0.356 | 7.021 ± 2.302 | 6.347 ± 0.629 | 6.873 ± 0.608 |
| BPIC17W | **0.609 ± 0.040** | 3.852 ± 0.374 | 7.636 ± 0.332 | 3.321 ± 0.164 | 3.336 ± 0.127 |
| P2P-1000 | **10.964 ± 4.279** | — | 33.912 ± 4.902 | 38.388 ± 1.780 | 39.166 ± 1.086 |
| P2P-2000 | **14.482 ± 2.454** | — | 92.311 ± 2.829 | 60.282 ± 0.925 | 59.383 ± 0.730 |
| ACR | **24.933 ± 5.863** | 62.140 ± 1.632 | 87.863 ± 12.657 | 64.290 ± 3.194 | 61.118 ± 3.350 |
| CVS | **0.273 ± 0.017** | — | 0.279 ± 0.010 | 0.605 ± 0.034 | 0.593 ± 0.047 |
| Production | **59.243 ± 12.987** | 112.604 ± 2.955 | 358.573 ± 296.214 | 195.779 ± 4.119 | 194.773 ± 5.127 |
| PurchasingExample | **52.333 ± 14.097** | 113.895 ± 0.000 | 52.750 ± 14.503 | 200.353 ± 3.158 | 196.498 ± 4.706 |
| SynLoan | **2.657 ± 0.677** | — | 12.201 ± 1.305 | 98.475 ± 1.006 | 98.440 ± 0.993 |

#### `execution_tree` ↓ — execution time per (activity, resource, history) leaf

| Dataset | ProSiT | AgentSim | SIMOD | RIMS | DSIM |
|---|---|---|---|---|---|
| BPIC12W | **3.676 ± 0.343** | — | 26.675 ± 1.390 | — | — |
| BPIC17W | **1.384 ± 0.088** | — | 11.159 ± 1.683 | — | — |
| P2P-1000 | **4.474 ± 0.496** | — | 23.782 ± 2.889 | — | — |
| P2P-2000 | **4.031 ± 0.323** | — | 31.343 ± 2.202 | — | — |
| ACR | **13.075 ± 1.846** | 151.730 ± 17.644 | 90.974 ± 14.517 | — | — |
| CVS | **0.492 ± 0.075** | — | 1.185 ± 0.219 | — | — |
| Production | **33.297 ± 6.677** | — | 119.619 ± 29.230 | — | — |
| PurchasingExample | **7.732 ± 2.187** | 144.173 ± 8.625 | 66.646 ± 16.199 | — | — |
| SynLoan | **2.629 ± 0.165** | — | 6.042 ± 0.250 | — | — |

#### `waiting_tree` ↓ — waiting time per (resource, history) leaf

| Dataset | ProSiT | AgentSim | SIMOD | RIMS | DSIM |
|---|---|---|---|---|---|
| BPIC12W | **1152.414 ± 195.068** | — | 1960.119 ± 15.590 | — | — |
| BPIC17W | **804.995 ± 101.693** | — | 2366.841 ± 24.195 | — | — |
| P2P-1000 | **14.246 ± 1.557** | — | 75.743 ± 9.620 | — | — |
| P2P-2000 | **8.604 ± 1.389** | — | 228.306 ± 21.740 | — | — |
| ACR | **1125.166 ± 215.739** | 4765.251 ± 601.952 | 3438.467 ± 176.898 | — | — |
| CVS | **28.470 ± 1.328** | — | 540.444 ± 4.188 | — | — |
| Production | **290.748 ± 60.482** | — | 1694.410 ± 243.010 | — | — |
| PurchasingExample | **307.925 ± 83.452** | 436.923 ± 39.425 | 439.847 ± 22.865 | — | — |
| SynLoan | **313.554 ± 55.754** | — | 575.260 ± 33.825 | — | — |

#### `transition_tree` ↓ — routing probability per decision-point leaf

| Dataset | ProSiT | AgentSim | SIMOD | RIMS | DSIM |
|---|---|---|---|---|---|
| BPIC12W | **0.012 ± 0.003** | — | 0.109 ± 0.004 | 0.114 ± 0.004 | — |
| BPIC17W | **0.005 ± 0.000** | — | 0.115 ± 0.001 | 0.136 ± 0.002 | — |
| P2P-1000 | **0.006 ± 0.002** | — | 0.030 ± 0.002 | 0.027 ± 0.003 | 0.028 ± 0.003 |
| P2P-2000 | **0.004 ± 0.001** | — | 0.026 ± 0.002 | 0.027 ± 0.002 | 0.025 ± 0.002 |
| ACR | **0.021 ± 0.006** | 0.044 ± 0.009 | 0.086 ± 0.014 | 0.130 ± 0.014 | 0.292 ± 0.054 |
| CVS | **0.002 ± 0.001** | — | 0.038 ± 0.001 | 0.178 ± 0.002 | 0.081 ± 0.001 |
| Production | **0.035 ± 0.008** | — | 0.072 ± 0.005 | — | — |
| PurchasingExample | **0.012 ± 0.003** | 0.042 ± 0.008 | 0.034 ± 0.007 | 0.153 ± 0.008 | 0.083 ± 0.004 |
| SynLoan | **0.008 ± 0.002** | — | 0.041 ± 0.003 | 0.129 ± 0.004 | — |

#### `resource_tree` ↓ — resource-selection probability per (activity, history) leaf

| Dataset | ProSiT | AgentSim | SIMOD | RIMS | DSIM |
|---|---|---|---|---|---|
| BPIC12W | **0.006 ± 0.001** | — | 0.044 ± 0.000 | — | — |
| BPIC17W | **0.005 ± 0.000** | — | 0.015 ± 0.000 | — | — |
| P2P-1000 | **0.101 ± 0.004** | — | 0.113 ± 0.008 | — | — |
| P2P-2000 | **0.126 ± 0.006** | — | 0.209 ± 0.008 | — | — |
| ACR | **0.005 ± 0.000** | 0.010 ± 0.001 | 0.011 ± 0.001 | — | — |
| CVS | **0.006 ± 0.001** | — | 0.166 ± 0.002 | — | — |
| Production | **0.037 ± 0.006** | — | 0.149 ± 0.017 | — | — |
| PurchasingExample | **0.021 ± 0.002** | 0.115 ± 0.006 | 0.023 ± 0.002 | — | — |
| SynLoan | 0.072 ± 0.004 | — | **0.051 ± 0.004** | — | — |
