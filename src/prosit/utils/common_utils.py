import random
import pandas as pd
from datetime import datetime, timedelta, timezone
from copy import copy

import pm4py
from tqdm import tqdm
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.log.obj import EventLog

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from prosit.utils.rule_utils import DecisionRules

from river.tree.hoeffding_adaptive_tree_classifier import HoeffdingAdaptiveTreeClassifier



def return_transitions_frequency(
        log: EventLog, 
        net: PetriNet, 
        initial_marking: Marking, 
        final_marking: Marking
    ) -> dict:

    try:
        alignments_ = alignments.apply_multiprocessing(log, net, initial_marking, final_marking, parameters={"ret_tuple_as_trans_desc": True})
    except Exception:
        alignments_ = alignments.apply_log(log, net, initial_marking, final_marking, parameters={"ret_tuple_as_trans_desc": True})
    aligned_traces = [[y[0] for y in x['alignment'] if y[0][1]!='>>'] for x in alignments_]

    frequency_t = {t: 0 for t in net.transitions}
    name_to_trans = {t.name: t for t in net.transitions}
    for trace in aligned_traces:
        for align in trace:
            name_t = align[1]
            if name_t in name_to_trans:
                frequency_t[name_to_trans[name_t]] += 1

    return frequency_t


def return_enabled_and_fired_transitions(
        net: PetriNet, 
        initial_marking: Marking, 
        final_marking: Marking, 
        trace_aligned: list
    ) -> tuple:

    visited_transitions = []
    is_fired = []
    tkns = list(initial_marking)
    enabled_transitions = return_enabled_transitions(net, tkns)
    name_to_trans = {t.name: t for t in net.transitions}
    for t_fired_name in trace_aligned:
        t_fired = name_to_trans[t_fired_name[1]]
        not_fired_transitions = list(enabled_transitions-{t_fired})
        for t_not_fired in not_fired_transitions:
            visited_transitions.append(t_not_fired)
            is_fired.append(0)
        visited_transitions.append(t_fired)
        is_fired.append(1)
        tkns = update_current_marking(tkns, t_fired)
        if set(tkns) == set(final_marking):
            return visited_transitions, is_fired
        enabled_transitions = return_enabled_transitions(net, tkns)

    return visited_transitions, is_fired


def update_current_marking(m: Marking, t_fired: PetriNet.Transition) -> Marking:

    m_out = copy(m)
    for a in t_fired.in_arcs:
        m_out[a.source] -= a.weight
        if m_out[a.source] == 0:
            del m_out[a.source]

    for a in t_fired.out_arcs:
        m_out[a.target] += a.weight

    return m_out


def return_enabled_transitions(net: PetriNet, tkns: Marking) -> set:
    
    enabled_t = set()
    list_transitions = list(net.transitions)
    for t in list_transitions:
        if {a.source for a in t.in_arcs}.issubset(tkns):
            enabled_t.add(t)
    
    return enabled_t


def return_fired_transition(transition_weights: dict, enabled_transitions: list) -> PetriNet.Transition:

    total_weight = sum(transition_weights[s] for s in enabled_transitions)
    random_value = random.uniform(0, total_weight)
    
    cumulative_weight = 0
    for s in enabled_transitions:
        cumulative_weight += transition_weights[s]
        if random_value <= cumulative_weight:
            return s
    return enabled_transitions[-1]


def compute_transition_weights_from_model(models_t: dict, dict_x: dict, enabled_transitions=None) -> dict:
    transition_weights = dict()
    keys = enabled_transitions if enabled_transitions is not None else models_t.keys()
    for t in keys:
        if type(models_t[t]) in [LogisticRegression, DecisionTreeClassifier, RandomForestClassifier]:
            X = pd.DataFrame({k: [dict_x[k]] for k in dict_x.keys()})
            transition_weights[t] = compute_proba(models_t, t, X)
        elif type(models_t[t]) == HoeffdingAdaptiveTreeClassifier:
            try:
                transition_weights[t] = models_t[t].predict_proba_one(dict_x)[1]
            except (KeyError, IndexError, Exception):
                transition_weights[t] = 0
        elif type(models_t[t]) == DecisionRules:
            transition_weights[t] = models_t[t].apply(dict_x)
        elif type(models_t[t]) == float:
            transition_weights[t] = models_t[t]
        else:
            transition_weights[t] = 1
    return transition_weights


def return_resource(resource_weights: dict, enabled_resources: list) -> str:

    total_weight = sum(resource_weights[s] for s in enabled_resources)
    random_value = random.uniform(0, total_weight)
    
    cumulative_weight = 0
    for s in enabled_resources:
        cumulative_weight += resource_weights[s]
        if random_value <= cumulative_weight:
            return s
    return enabled_resources[-1]


def compute_resource_weights_from_model(models_r: dict, enabled_resources: list, dict_x: dict, workloads: dict = None, queue_lengths: dict = None) -> dict:
    resource_weights = dict()
    for r in enabled_resources:
        model = models_r.get(r)
        if workloads is not None or queue_lengths is not None:
            x_r = dict(dict_x)
            if workloads is not None:
                x_r['workload'] = workloads.get(r, 0)
            if queue_lengths is not None:
                x_r['queue_length'] = queue_lengths.get(r, 0)
        else:
            x_r = dict_x
        if type(model) == DecisionRules:
            resource_weights[r] = model.apply(x_r)
        elif type(model) == float:
            resource_weights[r] = model
        else:
            resource_weights[r] = 0
    return resource_weights


def compute_proba(models_t: dict, t: PetriNet.Transition, X:pd.DataFrame) -> float:
    
    clf_t = models_t[t]
    
    return clf_t.predict_proba(X)[0,1]


def count_concurrent_events(schedule, t_enabled) -> int:

    count = 0
    for start, end in reversed(schedule):
        if end <= t_enabled:
            break
        if start <= t_enabled < end:
            count += 1
            
    return count


def count_false_hours(calendar: dict, start_ts: datetime, end_ts: datetime) -> int:
    false_hours_count = 0
    current_time = start_ts
    
    while current_time < end_ts:
        weekday = current_time.weekday()
        hour = current_time.hour
        
        if not calendar.get(weekday, {}).get(hour, False):
            false_hours_count += 1
            
        current_time += timedelta(hours=1)

    return false_hours_count


def calendar_to_working_set(calendar: dict) -> frozenset:
    return frozenset((wd, h) for wd, hours in calendar.items() for h, active in hours.items() if active)


def count_working_minutes(start_ts: datetime, end_ts: datetime, calendar: dict, _working_set: frozenset = None) -> int:
    if end_ts <= start_ts:
        return 0
    working_set = _working_set if _working_set is not None else calendar_to_working_set(calendar)
    working_minutes = 0
    current_time = start_ts
    while current_time < end_ts:
        weekday = current_time.weekday()
        hour = current_time.hour
        if (weekday, hour) in working_set:
            end_of_hour = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            minutes_in_slot = min((min(end_ts, end_of_hour) - current_time).total_seconds() / 60, 60 - current_time.minute)
            working_minutes += max(0, minutes_in_slot)
            current_time = end_of_hour
        else:
            current_time = (current_time + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    return round(working_minutes)


def snap_to_next_working_slot(ts: datetime, calendar: dict, _working_set: frozenset = None) -> datetime:
    """If ts falls outside a working hour, advance to the start of the next working hour."""
    working_set = _working_set if _working_set is not None else calendar_to_working_set(calendar)
    current = ts
    for _ in range(7 * 24):  # max 1 week lookahead
        if (current.weekday(), current.hour) in working_set:
            return current
        current = (current + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    return ts  # calendar is empty or all-false, return as-is


def add_minutes_with_calendar(start_ts: datetime, minutes_to_add: int, calendar: dict, _working_set: frozenset = None) -> datetime:
    working_set = _working_set if _working_set is not None else calendar_to_working_set(calendar)
    remaining_minutes = minutes_to_add
    current_time = start_ts

    while remaining_minutes > 0:
        weekday = current_time.weekday()
        hour = current_time.hour

        if (weekday, hour) in working_set:
            minutes_in_current_hour = min(remaining_minutes, 60 - current_time.minute)

            current_time += timedelta(minutes=minutes_in_current_hour)
            remaining_minutes -= minutes_in_current_hour
        else:
            current_time = (current_time + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

    return current_time


def get_transition_from_name(t_fired_name: str, net: PetriNet) -> PetriNet.Transition:
    for t in net.transitions:
        if t.name == t_fired_name:
            return t
        

def build_df_features(log, net, im, fm, act_to_resources, net_transition_labels, resources, label_data_attributes=[], firing_sequences=None):

    import numpy as np

    df_log = pm4py.convert_to_dataframe(log)
    df_log["start:timestamp"] = df_log["start:timestamp"].apply(lambda x: datetime.fromisoformat(str(x)[:-6]).timestamp())
    df_log["time:timestamp"] = df_log["time:timestamp"].apply(lambda x: datetime.fromisoformat(str(x)[:-6]).timestamp())

    # Pre-group events by resource for fast concurrent-event lookup (avoids full df scan per step)
    resource_events = {}
    for res_name, group in df_log.groupby('org:resource'):
        sorted_group = group.sort_values('start:timestamp')
        resource_events[res_name] = {
            'start_ts': sorted_group['start:timestamp'].values,
            'end_ts': sorted_group['time:timestamp'].values,
        }

    name_to_trans = {t.name: t for t in net.transitions}

    if firing_sequences is not None:
        # Fast path: the simulator already knows which transitions fired per
        # case, so we can fabricate the same alignment structure pm4py would
        # return and skip the (expensive) A* conformance check entirely.
        aligned_traces = []
        for trace in log:
            cid = trace[0]["case:concept:name"]
            seq = firing_sequences.get(cid, [])
            alignment = []
            for t_name in seq:
                t = name_to_trans[t_name]
                label = t.label
                if label is None:
                    # Silent transition → model move (log side is '>>').
                    alignment.append(((t_name, t_name), ('>>', None)))
                else:
                    # Visible transition → sync move.
                    alignment.append(((t_name, t_name), (label, label)))
            aligned_traces.append({"alignment": alignment})
    else:
        try:
            aligned_traces = alignments.apply_multiprocessing(
                log, net, im, fm,
                parameters={"ret_tuple_as_trans_desc": True},
            )
        except Exception:
            aligned_traces = alignments.apply_log(
                log, net, im, fm,
                parameters={"ret_tuple_as_trans_desc": True},
            )

    dataset = []
    for i, trace in enumerate(tqdm(log)):

        trace_aligned = aligned_traces[i]["alignment"]

        case_id = trace[0]["case:concept:name"]
        history = {t_l: 0 for t_l in net_transition_labels}
        history_res = {r: 0 for r in resources}
        last_resource = None
        last_activity = None
        if label_data_attributes:
            try:
                trace_attributes = [trace[a] for a in label_data_attributes]
            except (KeyError, TypeError):
                trace_attributes = [trace[0][a] for a in label_data_attributes]
        else:
            trace_attributes = []

        marking = im
        j = 0
        transition_enabled_times = dict()
        current_t = trace[0]["start:timestamp"]
        for step in trace_aligned:
            if step[0][1] == ">>": # log move
                continue

            transition = name_to_trans[step[0][1]]
            transition_label = transition.label


            prev_enabled_transitions = return_enabled_transitions(net, marking)

            for enabled in prev_enabled_transitions:
                if enabled not in transition_enabled_times:
                    transition_enabled_times[enabled] = current_t

            if step[1][0] == step[1][1]: # sync move
                resource = trace[j]["org:resource"]
                start_t = trace[j]["start:timestamp"]
                end_t = trace[j]["time:timestamp"]
                enabled_t = transition_enabled_times[transition]
                current_t = end_t
                enabled_ts = enabled_t.timestamp()
                if resource in resource_events:
                    re = resource_events[resource]
                    concurrent = (re['start_ts'] < enabled_ts) & (re['end_ts'] > enabled_ts)
                    res_workload = int(concurrent.sum())
                    if res_workload > 0:
                        resource_free_t = datetime.fromtimestamp(re['end_ts'][concurrent].max(), tz=timezone.utc)
                    else:
                        resource_free_t = enabled_t
                else:
                    res_workload = 0
                    resource_free_t = enabled_t
                # Workload at enabled_t for every resource in the role pool of this activity.
                # Used by the resource-assignment classifier to (i) filter negatives to
                # resources that were actually free, and (ii) feed per-candidate workload
                # as a feature, mirroring what the simulator does at inference time.
                candidate_workloads = {}
                for cand in act_to_resources.get(transition_label, []):
                    if cand == resource:
                        candidate_workloads[cand] = res_workload
                    elif cand in resource_events:
                        cre = resource_events[cand]
                        cand_concurrent = (cre['start_ts'] < enabled_ts) & (cre['end_ts'] > enabled_ts)
                        candidate_workloads[cand] = int(cand_concurrent.sum())
                    else:
                        candidate_workloads[cand] = 0
                prev_enabled_resources = act_to_resources[transition_label]
                j += 1
            else: # model move
                resource = None
                enabled_t = None
                start_t = None
                end_t = None
                res_workload = None
                resource_free_t = None
                prev_enabled_resources = None
                candidate_workloads = None

            del transition_enabled_times[transition]
            
            marking = update_current_marking(marking, transition)

            handover_features = tuple(1 if r == last_resource else 0 for r in resources)
            last_activity_features = tuple(1 if t_l == last_activity else 0 for t_l in net_transition_labels)
            dataset.append((case_id, transition, transition_label, resource, enabled_t, start_t, end_t, prev_enabled_transitions, prev_enabled_resources, res_workload, resource_free_t, candidate_workloads) + tuple(trace_attributes) + tuple(history_res.values()) + tuple(history.values()) + handover_features + last_activity_features)

            if transition_label:
                history[transition_label] += 1
                if resource in resources:
                    history_res[resource] += 1
                last_resource = resource
                last_activity = transition_label

    handover_from_cols = ['handover_from_' + r for r in resources]
    last_activity_cols = ['last_activity_' + t_l for t_l in net_transition_labels]
    df = pd.DataFrame(dataset, columns=["case_id", "transition", "transition_label", "resource", "enabled_t", "start_t", "end_t", "prev_enabled_transitions", "prev_enabled_resources", "res_workload", "resource_free_t", "candidate_workloads"] + label_data_attributes + resources + net_transition_labels + handover_from_cols + last_activity_cols)

    # Per-resource sorted arrays of enabled_ts/start_ts, used for both the
    # per-(event, chosen-resource) queue_length column and the per-(event,
    # candidate-resource) queue_length values in candidate_queue_lengths.
    # queue_at_q for r = |{i on r : enabled_ts_i <= q and start_ts_i > q}|
    #                  = searchsorted_right(enabled_sorted, q)
    #                    - searchsorted_right(start_sorted, q)
    # because start_ts_i >= enabled_ts_i always holds.
    per_res_sorted = {}
    sync_mask_full = ~df['resource'].isna() & ~df['enabled_t'].isna() & ~df['start_t'].isna()
    if sync_mask_full.any():
        for resource, group in df[sync_mask_full].groupby('resource'):
            enabled_arr = np.sort(np.array([et.timestamp() for et in group['enabled_t']]))
            start_arr = np.sort(np.array([st.timestamp() for st in group['start_t']]))
            per_res_sorted[resource] = (enabled_arr, start_arr)

    # queue_length of the chosen resource at each event's enabled_t.
    df['queue_length'] = 0
    if sync_mask_full.any():
        idx = df[sync_mask_full].index
        for i in idx:
            r = df.at[i, 'resource']
            if r in per_res_sorted:
                q = df.at[i, 'enabled_t'].timestamp()
                enabled_arr, start_arr = per_res_sorted[r]
                a = int(np.searchsorted(enabled_arr, q, side='right'))
                b = int(np.searchsorted(start_arr, q, side='right'))
                df.at[i, 'queue_length'] = a - b

    # Per-candidate queue length at enabling time (used by the resource
    # assignment classifier — mirrors candidate_workloads).
    df['candidate_queue_lengths'] = None
    if sync_mask_full.any():
        for i in df[sync_mask_full].index:
            cw = df.at[i, 'candidate_workloads']
            if not isinstance(cw, dict):
                continue
            q = df.at[i, 'enabled_t'].timestamp()
            cq = {}
            for cand in cw.keys():
                if cand in per_res_sorted:
                    enabled_arr, start_arr = per_res_sorted[cand]
                    a = int(np.searchsorted(enabled_arr, q, side='right'))
                    b = int(np.searchsorted(start_arr, q, side='right'))
                    cq[cand] = a - b
                else:
                    cq[cand] = 0
            df.at[i, 'candidate_queue_lengths'] = cq

    return df