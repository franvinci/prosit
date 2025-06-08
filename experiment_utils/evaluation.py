import pandas as pd
import pm4py
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



def evaluate(original_log, simulated_log, metrics_labels=["cfld", "ngd", "red", "aed", "red", "aed", "car", "ctd", "car_entropy", "ctd_entropy", "etd_entropy"]):

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

    metrics = dict()

    if "cfld" in metrics_labels:
        metrics['cfld'] = control_flow_log_distance(
            original_log, 
            event_log_ids, 
            simulated_log, 
            event_log_ids
        )

    if "ngd" in metrics_labels:
        metrics['ngd'] = n_gram_distribution_distance(
            original_log, 
            event_log_ids, 
            simulated_log, 
            event_log_ids, 
            n=3,
        )

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
        metrics['car_entropy'] = compute_atd_entropy(simulated_log)

    if "ctd_entropy" in metrics_labels:
        metrics['ctd_entropy'] = compute_ctd_entropy(simulated_log)

    if "etd_entropy" in metrics_labels:
        metrics['etd_entropy'] = compute_etd_entropy(simulated_log)

    return metrics


def compute_atd_entropy(df_log: pd.DataFrame) -> float:

    first_ts = df_log.groupby('case:concept:name')["start:timestamp"].min()
    ordered_first_ts_list = first_ts.sort_values().tolist()

    arrival_times = []
    for i in range(1, len(ordered_first_ts_list)):
        arrival_times.append((ordered_first_ts_list[i] - ordered_first_ts_list[i-1]).total_seconds()/60)

    hist, _ = np.histogram(arrival_times, bins='auto', density=True)
    hist = hist[hist > 0]
    probs = hist / np.sum(hist)
    atd_entr = entropy(probs)

    return atd_entr


def compute_ctd_entropy(df_log: pd.DataFrame) -> float:

    log = pm4py.convert_to_event_log(df_log)
    cycle_times = []
    for trace in log:
        start = trace[0]['start:timestamp']
        end = trace[-1]['time:timestamp']
        cycle_times.append((end-start).total_seconds()//60)
    
    hist, _ = np.histogram(cycle_times, bins='auto', density=True)
    probs = hist / np.sum(hist)
    ctd_entr = entropy(probs)

    return ctd_entr


def compute_etd_entropy(df_log: pd.DataFrame) -> float:

    activities = list(df_log["concept:name"].unique())
    etd_entropies = []
    for act in activities:
        df_log_act = df_log[df_log["concept:name"] == act]
        ex_times = (df_log_act["time:timestamp"] - df_log_act["start:timestamp"]).apply(lambda x: x.total_seconds() // 60)
        hist, _ = np.histogram(list(ex_times), bins='auto', density=True)
        probs = hist / np.sum(hist)
        etd_entropies.append(entropy(probs))

    return np.mean(etd_entropies)