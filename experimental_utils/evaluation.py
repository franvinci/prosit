import pandas as pd
import numpy as np
from scipy.stats import entropy

from log_distance_measures.config import EventLogIDs

from log_distance_measures.absolute_event_distribution import absolute_event_distribution_distance, discretize_to_hour
from log_distance_measures.case_arrival_distribution import case_arrival_distribution_distance
from log_distance_measures.circadian_event_distribution import circadian_event_distribution_distance
from log_distance_measures.circadian_workforce_distribution import circadian_workforce_distribution_distance
from log_distance_measures.config import AbsoluteTimestampType
from log_distance_measures.control_flow_log_distance import control_flow_log_distance
from log_distance_measures.cycle_time_distribution import cycle_time_distribution_distance
from log_distance_measures.n_gram_distribution import n_gram_distribution_distance
from log_distance_measures.relative_event_distribution import relative_event_distribution_distance

import datetime



def evaluate(original_log, simulated_log, metrics_labels=["cfld", "2gd", "red", "aed", "red", "aed", "car", "ctd", "car_entropy", "ctd_entropy", "etd_entropy"]):

    event_log_ids = EventLogIDs(  
        case="case:concept:name",
        activity="concept:name",
        resource="org:resource",
        start_time="start:timestamp",
        end_time="time:timestamp"
    )


    original_log[event_log_ids.start_time] = pd.to_datetime(original_log[event_log_ids.start_time], format='ISO8601', utc=True)
    original_log[event_log_ids.end_time] = pd.to_datetime(original_log[event_log_ids.end_time], format='ISO8601', utc=True)

    simulated_log[event_log_ids.start_time] = pd.to_datetime(simulated_log[event_log_ids.start_time], utc=True)
    simulated_log[event_log_ids.end_time] = pd.to_datetime(simulated_log[event_log_ids.end_time], utc=True)

    # Normalize resource dtype so that methods storing resources as int (e.g.
    # simod/agentsimulator on ConsultaDataMining) are compared against a
    # string-valued reference on the same alphabet.
    for df in (original_log, simulated_log):
        if event_log_ids.resource in df.columns:
            col = df[event_log_ids.resource]
            df[event_log_ids.resource] = col.map(lambda v: str(v) if pd.notna(v) else v)

    metrics = dict()

    if "cfld" in metrics_labels:
        metrics['cfld'] = control_flow_log_distance(
            original_log, 
            event_log_ids, 
            simulated_log, 
            event_log_ids
        )

    if "2gd" in metrics_labels:
        metrics['2gd'] = n_gram_distribution_distance(
            original_log, 
            event_log_ids, 
            simulated_log, 
            event_log_ids, 
            n=2,
        )

    if "3gd" in metrics_labels:
        metrics['3gd'] = n_gram_distribution_distance(
            original_log, 
            event_log_ids, 
            simulated_log, 
            event_log_ids, 
            n=3,
        )

    if "4gd" in metrics_labels:
        metrics['4gd'] = n_gram_distribution_distance(
            original_log,
            event_log_ids,
            simulated_log,
            event_log_ids,
            n=4,
        )
    
    if "2rgd" in metrics_labels:
        metrics['2rgd'] = compute_nrgd(original_log, simulated_log, 2)

    if "3rgd" in metrics_labels:
        metrics['3rgd'] = compute_nrgd(original_log, simulated_log, 3)

    if "4rgd" in metrics_labels:
        metrics['4rgd'] = compute_nrgd(original_log, simulated_log, 4)

    if "aed" in metrics_labels:
        metrics['aed'] = absolute_event_distribution_distance(
                original_log,
                event_log_ids,
                simulated_log,
                event_log_ids,
                AbsoluteTimestampType.BOTH,
                discretize_to_hour,
        )

    if "red" in metrics_labels:
        metrics['red'] = relative_event_distribution_distance(
            original_log,
            event_log_ids,
            simulated_log,
            event_log_ids,
            AbsoluteTimestampType.BOTH,
        )

    if "ced" in metrics_labels:
        metrics['ced'] = circadian_event_distribution_distance(
            original_log,
            event_log_ids,
            simulated_log,
            event_log_ids,
            AbsoluteTimestampType.BOTH,
        )

    if "cwd" in metrics_labels:
        metrics['cwd'] = circadian_workforce_distribution_distance(
            original_log,
            event_log_ids,
            simulated_log,
            event_log_ids
        )

    if "car" in metrics_labels:
        metrics['car'] = case_arrival_distribution_distance(
            original_log,
            event_log_ids,
            simulated_log,
            event_log_ids,
        )

    if "ctd" in metrics_labels:
        metrics['ctd'] = cycle_time_distribution_distance(
                original_log,
                event_log_ids,
                simulated_log,
                event_log_ids,
                datetime.timedelta(hours=1),
        )

    if "car_entropy" in metrics_labels:
        metrics['car_entropy'] = compute_atd_entropy(simulated_log, original_log)

    if "ctd_entropy" in metrics_labels:
        metrics['ctd_entropy'] = compute_ctd_entropy(simulated_log, original_log)

    if "etd_entropy" in metrics_labels:
        metrics['etd_entropy'] = compute_etd_entropy(simulated_log, original_log)

    return metrics


def _hist_entropy(values, bin_edges) -> float:
    """Entropy of ``values`` histogrammed on the supplied ``bin_edges``.
    Returns 0.0 when no value falls inside the bins (e.g. empty log, or
    the simulator produced values entirely outside the reference range)."""
    hist, _ = np.histogram(values, bins=bin_edges)
    total = hist.sum()
    if total == 0:
        return 0.0
    probs = hist / total
    return float(entropy(probs))


def _auto_bin_edges(values):
    """Bin edges chosen once from the reference values so every simulator
    is scored on the same support. Returns None when there are too few
    reference points to resolve a histogram."""
    if len(values) < 2:
        return None
    return np.histogram_bin_edges(values, bins='auto')


def _inter_arrival_minutes(df_log: pd.DataFrame) -> list:
    first_ts = df_log.groupby('case:concept:name')["start:timestamp"].min()
    ordered = first_ts.sort_values().tolist()
    return [
        (ordered[i] - ordered[i - 1]).total_seconds() / 60
        for i in range(1, len(ordered))
    ]


def _cycle_times_minutes(df_log: pd.DataFrame) -> list:
    grouped = df_log.groupby('case:concept:name')
    starts = grouped["start:timestamp"].min()
    ends = grouped["time:timestamp"].max()
    return ((ends - starts).dt.total_seconds() // 60).tolist()


def _exec_times_minutes(df_log: pd.DataFrame, activity: str) -> list:
    sub = df_log[df_log["concept:name"] == activity]
    if sub.empty:
        return []
    return ((sub["time:timestamp"] - sub["start:timestamp"]).dt.total_seconds() // 60).tolist()


def compute_atd_entropy(sim_log: pd.DataFrame, ref_log: pd.DataFrame) -> float:
    edges = _auto_bin_edges(_inter_arrival_minutes(ref_log))
    if edges is None:
        return 0.0
    return _hist_entropy(_inter_arrival_minutes(sim_log), edges)


def compute_ctd_entropy(sim_log: pd.DataFrame, ref_log: pd.DataFrame) -> float:
    edges = _auto_bin_edges(_cycle_times_minutes(ref_log))
    if edges is None:
        return 0.0
    return _hist_entropy(_cycle_times_minutes(sim_log), edges)


def compute_etd_entropy(sim_log: pd.DataFrame, ref_log: pd.DataFrame) -> float:
    weighted_sum = 0.0
    total_weight = 0
    for act in ref_log["concept:name"].unique():
        ref_ex = _exec_times_minutes(ref_log, act)
        edges = _auto_bin_edges(ref_ex)
        if edges is None:
            continue
        h = _hist_entropy(_exec_times_minutes(sim_log, act), edges)
        w = len(ref_ex)
        weighted_sum += h * w
        total_weight += w
    return weighted_sum / total_weight if total_weight > 0 else 0.0

def compute_nrgd(original_log: pd.DataFrame, simulated_log: pd.DataFrame, n: int) -> float:

    event_log_ids = EventLogIDs(  
        case="case:concept:name",
        activity="org:resource",
        resource="concept:name",
        start_time="start:timestamp",
        end_time="time:timestamp"
    )

    return n_gram_distribution_distance(
        original_log,
        event_log_ids,
        simulated_log,
        event_log_ids,
        n=n,
    )