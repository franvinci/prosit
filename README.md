# Prosit — PROcess SImulation Tool

Prosit is a Python library for **rule-aware business process simulation**. Given an event log in XES format and a Petri net process model, it automatically discovers simulation parameters (arrival rates, execution times, waiting times, resource assignments, routing probabilities) and runs discrete-event simulations that reproduce the statistical behaviour of the original process.

Unlike basic simulation tools, Prosit builds **conditional models** — decision trees that learn *when* each resource is preferred, *how long* an activity takes depending on the case context, and *which path* is taken at decision points.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [API Reference](#api-reference)
  - [SimulatorParameters](#simulatorparameters)
  - [SimulatorEngine](#simulatorengine)
- [Discovery Options](#discovery-options)
- [Simulation Options](#simulation-options)
- [Save and Load Parameters](#save-and-load-parameters)
- [Advanced Usage](#advanced-usage)

---

## Installation

**Requirements:** Python 3.10

### Option 1 — Conda (recommended)

```bash
git clone https://github.com/franvinci/prosit
cd prosit
conda env create -f environment.yml
conda activate prosit
```

### Option 2 — pip

```bash
git clone https://github.com/franvinci/prosit
cd prosit
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---|---|---|
| `pm4py` | 2.4.1 | Event log parsing, Petri net discovery and conformance |
| `scikit-learn` | 1.1.3 | Decision tree models (batch discovery) |
| `river` | 0.22.0 | Hoeffding Adaptive Tree (incremental discovery) |
| `scipy` | 1.14.1 | Distribution fitting and sampling |
| `numpy` | 1.26.4 | Numerical operations |
| `pandas` | 2.2.3 | Feature DataFrame construction |
| `tqdm` | 4.64.1 | Progress bars |
| `graphviz` | 0.20.3 | Decision tree visualisation |

---

## Quick Start

```python
import sys
sys.path.append("src/")

import warnings
warnings.filterwarnings("ignore")

import pm4py
import pm4py.objects.log.importer.xes.importer as xes_importer
from prosit.simulator import SimulatorParameters, SimulatorEngine

# 1. Load event log
log = xes_importer.apply("data/logs/purchasing.xes")

# 2. Discover Petri net
net, im, fm = pm4py.discover_petri_net_inductive(log)

# 3. Create and populate simulation parameters
params = SimulatorParameters(net, im, fm)
params.discover_from_eventlog(log, max_depth_tree=3)

# 4. Save for later reuse
params.to_json("params_purchasing.json")

# 5. Simulate 500 cases
sim_engine = SimulatorEngine(params)
sim_log = sim_engine.apply(n_traces=500)

print(sim_log.head())
```

The output is a `pandas.DataFrame` with columns:
`case:concept:name`, `concept:name`, `org:resource`, `enabled:timestamp`, `start:timestamp`, `time:timestamp`, plus any case-level data attributes found in the original log.

---

## Core Concepts

### What is discovered

Prosit extracts the following simulation parameters from an event log:

| Parameter | What it models |
|---|---|
| **Arrival time** | Inter-arrival time between consecutive cases, conditional on hour and weekday |
| **Execution time** | Working-hours duration of each activity, conditional on resource and case history |
| **Waiting time** | Queue delay after a resource becomes free, conditional on workload and case context |
| **Control flow** | Routing probability at each decision point, conditional on case history and attributes |
| **Resource selection** | Which eligible resource executes each activity, conditional on workload, handover patterns, and case context |
| **Calendars** | Working hours per resource and for case arrivals |
| **Multitasking** | Maximum concurrent tasks per resource, derived from observed concurrent workload |
| **Data attributes** | Joint or per-attribute distribution of case-level data attributes (e.g. case type, priority) |

### Rules mode vs. no-rules mode

- **Rules mode** (`max_depth_tree >= 1`): Parameters are learned as **Decision Trees**. Time models (arrival, execution, waiting) use **regression trees** — each leaf has its own fitted distribution. Routing and resource selection use **classification trees** that score each candidate transition or resource and sample proportionally.
- **No-rules mode** (`max_depth_tree=0`): Each parameter collapses to a single fitted distribution or frequency weight — simpler, faster, less expressive.

### Batch vs. incremental discovery

- **Batch discovery** (default): Trains Decision Trees using `scikit-learn` with cross-validated hyperparameters (`max_depth`, `min_samples_leaf`, `max_features`).
- **Incremental discovery** (`incremental_discovery=True`): Uses **Hoeffding Adaptive Trees** from the `river` library. Gives more weight to recent traces, suitable for concept drift or evolving processes.

---

## API Reference

### SimulatorParameters

```python
SimulatorParameters(net: PetriNet, initial_marking: Marking, final_marking: Marking)
```

Holds all simulation parameters. Initialise with the Petri net (typically discovered via `pm4py.discover_petri_net_inductive`).

#### `discover_from_eventlog`

```python
params.discover_from_eventlog(
    log,
    max_depth_tree: int = 3,
    min_samples_leaf_cv: list = [5, 10, 20, 30],
    resource_thr: float = 0.95,
    calendar_thr_h: float = 0.95,
    calendar_thr_wd: float = 0.95,
    multitasking_thr: float = 0.05,
    enable_multitasking: bool = True,
    attribute_mode: str = 'empirical',
    incremental_discovery: bool = False,
    grace_period: int = 1000,
    random_state: int = 72,
    verbose: bool = True
)
```

Extracts all simulation parameters from the event log.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `log` | `EventLog` | — | pm4py event log in XES format |
| `max_depth_tree` | `int` | `3` | Maximum depth of decision trees. Higher = more expressive rules. Set to `0` to disable rules (pure distributions) |
| `min_samples_leaf_cv` | `list` | `[5,10,20,30]` | Candidate values for `min_samples_leaf` in cross-validation. Controls the minimum number of samples per leaf — critical for reliable per-leaf distribution fitting |
| `resource_thr` | `float` | `0.95` | Cumulative frequency threshold for resource filtering. Resources whose cumulative share exceeds this value are dropped as outliers |
| `calendar_thr_h` | `float` | `0.95` | Cumulative threshold for working-hour discovery. Hours beyond this share of activity are excluded from calendars |
| `calendar_thr_wd` | `float` | `0.95` | Cumulative threshold for working-weekday discovery |
| `multitasking_thr` | `float` | `0.05` | Minimum fraction of events with concurrent workload > 0 for a resource to be considered multitasking. Below this, capacity is set to 1 |
| `enable_multitasking` | `bool` | `True` | If `False`, all resources are forced to capacity 1 (no parallel task execution) regardless of what the log shows |
| `attribute_mode` | `str` | `'empirical'` | How to model case-level data attributes. `'empirical'`: samples from the joint observed distribution (preserves correlations). `'distribution'`: fits each attribute independently (categorical → frequency table, continuous → best-fitting scipy distribution) |
| `incremental_discovery` | `bool` | `False` | Use Hoeffding Adaptive Trees instead of scikit-learn. Gives more weight to recent traces |
| `grace_period` | `int` | `1000` | (Incremental only) Number of observations before the tree considers splitting a node |
| `random_state` | `int` | `72` | Seed for all random operations (reproducibility) |
| `verbose` | `bool` | `True` | Print discovery progress |

#### `to_json` / `from_json`

```python
params.to_json("params.json")   # save to file

params2 = SimulatorParameters(net, im, fm)
params2.from_json("params.json")  # restore from file
```

Serialises and deserialises all discovered parameters. Allows discovering once and simulating many times without re-running discovery.

---

### SimulatorEngine

```python
SimulatorEngine(simulation_parameters: SimulatorParameters)
```

Discrete-event simulation engine. Takes a `SimulatorParameters` object and runs the simulation.

#### `apply`

```python
sim_log = sim_engine.apply(
    n_traces: int = 1,
    t_start: datetime = None,
    deterministic_time: bool = False
) -> pd.DataFrame
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n_traces` | `int` | `1` | Number of process instances (cases) to simulate |
| `t_start` | `datetime` | `datetime.now()` | Start timestamp of the simulation. Cases arrive from this point onward |
| `deterministic_time` | `bool` | `False` | If `True`, uses the mean value of each distribution instead of sampling. Useful for deterministic analysis or debugging |

Returns a `pandas.DataFrame` sorted by start time, with one row per simulated event.

---

## Discovery Options

### Choosing `max_depth_tree`

The `max_depth_tree` parameter controls the complexity of conditional models:

```python
# No rules — single distribution per activity/resource
params.discover_from_eventlog(log, max_depth_tree=0)

# Shallow rules — fast, interpretable, good for small logs
params.discover_from_eventlog(log, max_depth_tree=2)

# Standard — balances expressiveness and overfitting
params.discover_from_eventlog(log, max_depth_tree=3)

# Deep rules — for complex processes with large logs
params.discover_from_eventlog(log, max_depth_tree=5)
```

Cross-validation automatically selects the best depth up to `max_depth_tree` for each individual model.

### Controlling leaf size (`min_samples_leaf_cv`)

This parameter is especially important for **time models** (execution, waiting, arrival). Each leaf of the regression tree has its own fitted distribution — if a leaf contains too few samples, the distribution fit is unreliable.

```python
# More conservative — larger leaves, smoother distributions
params.discover_from_eventlog(log, min_samples_leaf_cv=[10, 20, 30, 50])

# Less conservative — allows finer segmentation with small logs
params.discover_from_eventlog(log, min_samples_leaf_cv=[1, 5, 10])
```

### Incremental discovery

Use incremental discovery when the process has evolved over time and recent behaviour should dominate the model:

```python
params.discover_from_eventlog(
    log,
    incremental_discovery=True,
    grace_period=500,   # fewer events needed before first split
    max_depth_tree=3
)
```

The `grace_period` controls how quickly the tree adapts: lower values make the model react faster to changes, higher values produce more stable trees.

---

## Simulation Options

### Basic simulation

```python
from datetime import datetime

sim_engine = SimulatorEngine(params)

# Simulate 200 cases starting from a specific date
sim_log = sim_engine.apply(
    n_traces=200,
    t_start=datetime(2024, 1, 1, 8, 0, 0)
)
```

### Deterministic simulation

Uses the mean of each distribution instead of sampling — useful for benchmarking or debugging:

```python
sim_log = sim_engine.apply(n_traces=100, deterministic_time=True)
```

### Multitasking

Resources that handled concurrent work in the log are automatically assigned a maximum concurrency capacity. To disable multitasking entirely (all resources serialised):

```python
params.discover_from_eventlog(log, enable_multitasking=False)
```

To inspect the discovered capacities:

```python
# {resource_name: max_concurrent_tasks}
print(params.max_concurrency)
```

Resources with fewer than `multitasking_thr` fraction of events under concurrent load are treated as non-multitasking (capacity 1) even if `enable_multitasking=True`.

### Output format

```python
print(sim_log.columns.tolist())
# ['case:concept:name', 'concept:name', 'org:resource',
#  'enabled:timestamp', 'start:timestamp', 'time:timestamp',
#  ... (any case-level data attributes from the original log)]

print(sim_log.dtypes)
# All timestamps are datetime objects
# case:concept:name is a string like "case_1", "case_2", ...
```

---

## Save and Load Parameters

Discovered parameters can be saved and reused without re-running the (potentially slow) discovery phase:

```python
# --- Discover once ---
params = SimulatorParameters(net, im, fm)
params.discover_from_eventlog(log, max_depth_tree=3)
params.to_json("my_params.json")

# --- Load and simulate later ---
params2 = SimulatorParameters(net, im, fm)
params2.from_json("my_params.json")

engine = SimulatorEngine(params2)
sim_log = engine.apply(n_traces=1000)
```

---

## Advanced Usage

### Full workflow with evaluation

```python
import sys
sys.path.append("src/")

import warnings
warnings.filterwarnings("ignore")

import pm4py
import pm4py.objects.log.importer.xes.importer as xes_importer
from prosit.simulator import SimulatorParameters, SimulatorEngine
from datetime import datetime

# Load log
log = xes_importer.apply("data/logs/purchasing.xes")

# Split: use 80% for discovery, compare simulation against remaining 20%
n_cases = len(log)
train_log = log[:int(n_cases * 0.8)]
test_log  = log[int(n_cases * 0.8):]

# Discover from training set
net, im, fm = pm4py.discover_petri_net_inductive(train_log)
params = SimulatorParameters(net, im, fm)
params.discover_from_eventlog(
    train_log,
    max_depth_tree=3,
    min_samples_leaf_cv=[5, 10, 20, 30],
    random_state=42,
    verbose=True
)

# Simulate the same number of cases as the test set
engine = SimulatorEngine(params)
sim_log = engine.apply(
    n_traces=len(test_log),
    t_start=datetime(2024, 1, 1, 8, 0, 0)
)

print(f"Simulated {len(sim_log)} events across {sim_log['case:concept:name'].nunique()} cases")
```

### No-rules mode (pure distributions)

For simple processes or small logs where decision trees might overfit:

```python
params.discover_from_eventlog(log, max_depth_tree=0)
engine = SimulatorEngine(params)
sim_log = engine.apply(n_traces=200)
```

### Reproducible simulation

```python
import random
random.seed(42)

params.discover_from_eventlog(log, random_state=42)
sim_log = engine.apply(n_traces=100)
```

### Inspecting discovered parameters

```python
# Resources discovered from the log
print(params.resources)

# Working calendar per resource (weekday -> hour -> bool)
print(params.calendars["Resource A"])

# Per-resource maximum concurrency (1 = no multitasking, >1 = multitasking)
print(params.max_concurrency)

# Which resources can perform each activity
print(params.act_to_resources)

# Whether rules mode (Decision Trees) is active
print(params.rules_mode)

# Arrival time model (DecisionRules in rules mode, distribution tuple in no-rules mode)
print(params.arrival_time_distribution)

# Execution time model per activity (DecisionRules or distribution tuple)
print(params.execution_time_distributions)

# Waiting time model per resource (DecisionRules or distribution tuple)
print(params.waiting_time_distributions)

# Resource selection model per resource (DecisionRules in rules mode, float frequency in no-rules)
print(params.resource_weights)

# Control flow model per transition (DecisionRules in rules mode, float frequency in no-rules)
print(params.transition_weights)

# Case-level data attribute distribution (None if no attributes in log)
print(params.distribution_data_attributes)
# {'mode': 'empirical', 'data': {(val1, val2): frequency, ...}}
# or {'mode': 'distribution', 'data': {attr: {'type': 'categorical'|'continuous', ...}}}
```

---

## What the Models Learn

### Features used per model

| Model | Tree type | Conditional on |
|---|---|---|
| Arrival time | Regressor | Hour of day, weekday |
| Execution time | Regressor | Resource identity (one-hot), activity execution history, case attributes |
| Waiting time | Regressor | Resource workload, activity being executed (one-hot), activity history, case attributes |
| Control flow | Classifier | Activity execution history, last activity executed (one-hot), case attributes |
| Resource selection | Classifier | Resource usage history, last resource used (one-hot handover), last activity (one-hot), case attributes |

History features are expressed as **raw counts** (number of times each activity has been executed in the case so far), so that decision tree rules are directly interpretable (e.g. `"Approve" <= 2` means "Approve has been executed at most 2 times").

### Distribution fitting

For each leaf node of a regression tree, Prosit fits the best distribution among: `fixed`, `normal`, `exponential`, `lognormal`, `gamma`, `uniform`. The best fit is selected by minimising the deterministic Wasserstein distance between the empirical and theoretical quantiles. Outliers are removed using the Median Absolute Deviation method (threshold: 5 MAD) before fitting.

### Data attribute modeling

Case-level data attributes (e.g. `case:type`, `case:priority`) are discovered automatically and sampled at case arrival time. Two modes are available via `attribute_mode`:

- **`'empirical'`** (default): samples complete attribute tuples from the observed joint distribution — preserves correlations between attributes.
- **`'distribution'`**: fits each attribute independently (categorical → frequency table, continuous → best-fitting scipy distribution). Useful when the log is small or attributes are largely independent.
