#!/usr/bin/env python
"""Run PROSIT + SOTA simulator comparison experiments.

For each dataset we:
1. Discover PROSIT simulation parameters from the training log.
2. Generate PROSIT simulation logs.
3. Load up to 10 simulation logs from each of agentsimulator, simod, RIMS, DSIM.
4. Evaluate every log against the test log using the distance metrics in
   ``evaluation.py`` (skipping resource-based metrics when the method has no
   resources).
5. Evaluate every log against the discovered PROSIT trees using
   ``tree_accuracy.evaluate_all`` (each tree metric is ``None`` when the log
   doesn't provide the required features).

Results are saved per dataset/method as JSON, plus a single ``summary.json``.
"""

import sys
sys.path.append("src/")

import os
# Set before any joblib/multiprocessing fork so child workers inherit the
# silence — warnings.filterwarnings alone only affects the parent process.
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import warnings
warnings.filterwarnings("ignore")

import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import pm4py
import pandas as pd
import numpy as np

from prosit.simulator import SimulatorParameters, SimulatorEngine
from experimental_utils.evaluation import evaluate
from experimental_utils import simulator_loaders, tree_accuracy


EXP_DATA_DIR = "exp_data"
RESULTS_DIR = "results"
N_SIMULATIONS = 10

DISTANCE_METRICS = ["2gd", "car", "ctd", "etd_entropy"]
RESOURCE_METRICS = ["2rgd"]
TREE_METRIC_KEYS = [
    "arrival_tree", "execution_tree", "waiting_tree",
    "transition_tree", "resource_tree",
]

DATASETS = [
    "BPI_Challenge_2012_W_Two_TS",
    "BPI_Challenge_2017_W_Two_TS",
    "confidential_1000",
    "confidential_2000",
    "ConsultaDataMining201618",
    "cvs_pharmacy",
    "Production",
    "PurchasingExample",
    "SynLoan",
]

PROSIT_METHOD = "prosit"
PROSIT_MAX_DEPTH = 5
METHODS = [PROSIT_METHOD, "agentsimulator", "simod", "rims", "dsim"]


def load_test_df(dataset_path):
    log = pm4py.read_xes(os.path.join(dataset_path, "log_test.xes"), return_legacy_log_object=True)
    return pm4py.convert_to_dataframe(log)


def load_train_log(dataset_path):
    return pm4py.read_xes(os.path.join(dataset_path, "log_train.xes"), return_legacy_log_object=True)


def count_traces(df):
    return df["case:concept:name"].nunique()


def simulate_prosit(params, n_traces, t_start, n=N_SIMULATIONS, label="PROSIT", save_dir=None):
    engine = SimulatorEngine(params)
    logs = []
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    for i in range(n):
        print(f"  {label} simulation {i + 1}/{n}...", end=" ", flush=True)
        sim = engine.apply(n_traces=n_traces, t_start=t_start)
        print("done")
        logs.append(sim)
        if save_dir is not None:
            sim.to_csv(os.path.join(save_dir, f"simulated_log_{i}.csv"), index=False)
    info = {"has_resources": True, "has_case_attrs": bool(params.label_data_attributes)}
    return logs, info


def eval_run(test_df, sim_df, params, info, distance_metrics, skip_tree=False, skip_distance=False):
    merged = {}
    if not skip_distance:
        dist = evaluate(test_df.copy(), sim_df.copy(), metrics_labels=distance_metrics)
        merged.update(dist)
    if not skip_tree and params is not None:
        tree = tree_accuracy.evaluate_all(params, sim_df.copy(), info)
        for k, v in tree.items():
            if v is not None:
                merged[k] = v
    return merged


def _eval_method_parallel(test_df, logs, params, info, distance_metrics,
                          skip_tree, skip_distance, workers):
    """Run ``eval_run`` across ``logs`` (max N_SIMULATIONS) in parallel when
    ``workers > 1``. Returns a list of per-run dicts (None on failure)."""
    run_slice = logs[:N_SIMULATIONS]
    total = len(run_slice)
    per_run = [None] * total
    if workers and workers > 1 and total > 1:
        with ProcessPoolExecutor(max_workers=min(workers, total)) as ex:
            futures = {
                ex.submit(
                    eval_run, test_df, sim_df, params, info,
                    distance_metrics, skip_tree, skip_distance,
                ): i
                for i, sim_df in enumerate(run_slice)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    per_run[i] = fut.result()
                except Exception as e:
                    print(f"  ERROR: {e}")
    else:
        for i, sim_df in enumerate(run_slice):
            try:
                per_run[i] = eval_run(
                    test_df, sim_df, params, info,
                    distance_metrics, skip_tree, skip_distance,
                )
            except Exception as e:
                print(f"  ERROR: {e}")
    return per_run


def aggregate(per_run):
    """Return {metric: {mean, std, values}} across a list of per-run dicts."""
    out = {}
    if not per_run:
        return out
    keys = set().union(*(d.keys() for d in per_run))
    for k in keys:
        values = [d[k] for d in per_run if k in d and d[k] is not None and isinstance(d[k], (int, float))]
        if not values:
            continue
        out[k] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": [float(v) for v in values],
        }
    return out


SOTA_METHODS = ["agentsimulator", "simod", "rims", "dsim"]


def _load_sota_logs(dataset_name, case_attr_cols):
    logs = {}
    logs["agentsimulator"] = simulator_loaders.load_agentsimulator_sims(dataset_name)
    logs["simod"] = simulator_loaders.load_simod_sims(dataset_name, case_attr_cols=case_attr_cols)
    logs["rims"] = simulator_loaders.load_rims_sims(dataset_name, case_attr_cols=case_attr_cols)
    logs["dsim"] = simulator_loaders.load_dsim_sims(dataset_name)
    # Reconcile simod's has_case_attrs flag with what's actually present.
    simod_logs, simod_info = logs["simod"]
    if case_attr_cols and simod_logs:
        has_attrs = all(a in simod_logs[0].columns for a in case_attr_cols)
        simod_info["has_case_attrs"] = bool(has_attrs)
    logs["simod"] = (simod_logs, simod_info)
    return logs


def run_dataset(dataset_name, workers=1):
    """Run the full experiment for a dataset.

    Returns the per-method results dict."""
    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'=' * 60}")

    dataset_path = os.path.join(EXP_DATA_DIR, dataset_name)

    print("Loading training log and process model...")
    train_log = load_train_log(dataset_path)
    net, im, fm = pm4py.read_pnml(os.path.join(dataset_path, "model.pnml"))

    print("Loading test log...")
    test_df = load_test_df(dataset_path)
    n_traces = count_traces(test_df)
    print(f"Test log has {n_traces} traces")

    t_start = test_df["start:timestamp"].min()
    if pd.isna(t_start):
        t_start = test_df["time:timestamp"].min()

    # Discover PROSIT parameters and simulate.
    print(f"\nDiscovering PROSIT (max_depth_tree={PROSIT_MAX_DEPTH})...")
    prosit_params = SimulatorParameters(net, im, fm)
    prosit_params.discover_from_eventlog(train_log, max_depth_tree=PROSIT_MAX_DEPTH, verbose=True)

    sim_save_dir = os.path.join(RESULTS_DIR, dataset_name, "simulations", PROSIT_METHOD)
    print("--- PROSIT simulations ---")
    prosit_sims, prosit_info = simulate_prosit(
        prosit_params, n_traces, t_start, label="PROSIT",
        save_dir=sim_save_dir,
    )

    ref_params = prosit_params
    case_attr_cols = ref_params.label_data_attributes

    # Load SOTA logs once.
    print("\n--- Loading SOTA simulation logs ---")
    sota_logs = _load_sota_logs(dataset_name, case_attr_cols)
    for m, (logs, _) in sota_logs.items():
        print(f"  {m}: {len(logs)} logs loaded")

    print(f"\n{'-' * 60}")
    print(f"Evaluating methods | out={RESULTS_DIR}/")
    print(f"{'-' * 60}")

    dataset_results = {}

    # PROSIT — full evaluation.
    if prosit_sims:
        print(f"\nEvaluating {PROSIT_METHOD} ({len(prosit_sims)} logs)...")
        distance_metrics = list(DISTANCE_METRICS)
        if prosit_info.get("has_resources"):
            distance_metrics += RESOURCE_METRICS
        per_run = _eval_method_parallel(
            test_df, prosit_sims, ref_params, prosit_info, distance_metrics,
            skip_tree=False, skip_distance=False, workers=workers,
        )
        agg = aggregate([m for m in per_run if m is not None])
        if agg:
            mean_str = " ".join(f"{k}={v['mean']:.4f}" for k, v in agg.items())
            print(f"  {mean_str}")
        dataset_results[PROSIT_METHOD] = agg
    else:
        dataset_results[PROSIT_METHOD] = {"note": "method not available"}

    # SOTA.
    for method, (logs, info) in sota_logs.items():
        if not logs:
            dataset_results[method] = {"note": "method not available"}
            continue
        print(f"\nEvaluating {method} ({len(logs)} logs)...")
        distance_metrics = list(DISTANCE_METRICS)
        if info.get("has_resources"):
            distance_metrics += RESOURCE_METRICS
        per_run = _eval_method_parallel(
            test_df, logs, ref_params, info, distance_metrics,
            skip_tree=False, skip_distance=False, workers=workers,
        )
        agg = aggregate([m for m in per_run if m is not None])
        if agg:
            mean_str = " ".join(f"{k}={v['mean']:.4f}" for k, v in agg.items())
            print(f"  {mean_str}")
        dataset_results[method] = agg

    ds_dir = os.path.join(RESULTS_DIR, dataset_name)
    os.makedirs(ds_dir, exist_ok=True)
    for method, res in dataset_results.items():
        with open(os.path.join(ds_dir, f"{method}.json"), "w") as f:
            json.dump(res, f, indent=2)
    print(f"Per-method results saved under {ds_dir}/")

    return dataset_results


def _print_table(title, col_labels, rows):
    print(title)
    widths = [max(len(str(r[i])) for r in [col_labels] + rows) for i in range(len(col_labels))]
    head = "  ".join(str(c).ljust(w) for c, w in zip(col_labels, widths))
    print(head)
    print("-" * len(head))
    for r in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


def print_summary_tables(summary):
    all_metrics = DISTANCE_METRICS + RESOURCE_METRICS + TREE_METRIC_KEYS
    for metric in all_metrics:
        rows = []
        any_value = False
        for dataset in DATASETS:
            row_cells = [dataset]
            for method in METHODS:
                entry = summary.get(dataset, {}).get(method, {})
                if isinstance(entry, dict) and metric in entry and isinstance(entry[metric], dict):
                    cell = f"{entry[metric]['mean']:.3f}±{entry[metric]['std']:.3f}"
                    any_value = True
                else:
                    cell = "—"
                row_cells.append(cell)
            rows.append(row_cells)
        if not any_value:
            continue
        _print_table(f"\n### {metric}\n", ["Dataset"] + METHODS, rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Subset of dataset names to run (default: all).")
    parser.add_argument("--workers", type=int, default=10,
                        help="Parallel workers for per-run evaluation (default: 10).")
    args = parser.parse_args()

    datasets = args.datasets or DATASETS
    os.makedirs(RESULTS_DIR, exist_ok=True)

    summary = {}
    for dataset_name in datasets:
        try:
            summary[dataset_name] = run_dataset(dataset_name, workers=args.workers)
        except Exception as e:
            print(f"ERROR on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            summary[dataset_name] = {"error": str(e)}

    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull summary saved to {summary_path}")

    print_summary_tables(summary)


if __name__ == "__main__":
    main()
