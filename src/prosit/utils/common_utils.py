import random
import zlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from copy import copy

import pm4py
from tqdm import tqdm
from joblib import Parallel
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.conformance.alignments.petri_net.algorithm import Parameters as AlignParams
from pm4py.objects.petri_net.obj import PetriNet, Marking

from prosit.utils.rule_utils import DecisionRules


# n_jobs=-2 is joblib for "all cores except one" — keeps the machine responsive
# during long discovery runs. Workers must return (key, value) because
# generator_unordered yields results in completion order, not submission order.
DEFAULT_N_JOBS = -2


def parallel_with_progress(delayed_jobs, total, n_jobs=DEFAULT_N_JOBS, desc=None):
    gen = Parallel(n_jobs=n_jobs, return_as='generator_unordered')(delayed_jobs)
    return list(tqdm(gen, total=total, desc=desc))


def seed_worker_from_key(key, base_seed: int) -> None:
    """Deterministic per-worker seed for Python ``random`` and numpy's global RNG.

    Parallel workers don't share RNG state with the parent, so code that
    samples inside a worker (``sampling_from_dist``) is non-reproducible by
    default. CRC32 of ``str(key)`` is stable across processes (unlike
    Python's salted ``hash()``) so the per-entity seed is the same whether
    the worker is dispatched in run A or run B.
    """
    s = (int(base_seed) + zlib.crc32(str(key).encode('utf-8'))) & 0xFFFFFFFF
    random.seed(s)
    np.random.seed(s)



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
    # transition_weights is keyed by transition name (str); enabled_transitions
    # are Transition objects from net.transitions. Keying by name avoids the
    # id-based hash collisions that break object-keyed dicts after a joblib
    # pickle round-trip in discovery workers.
    total_weight = sum(transition_weights[s.name] for s in enabled_transitions)
    random_value = random.uniform(0, total_weight)

    cumulative_weight = 0
    for s in enabled_transitions:
        cumulative_weight += transition_weights[s.name]
        if random_value <= cumulative_weight:
            return s
    return enabled_transitions[-1]


def compute_transition_weights_from_model(models_t: dict, dict_x: dict, enabled_transitions=None) -> dict:
    # models_t is keyed by transition name (str); enabled_transitions are
    # Transition objects. Returned dict is also keyed by name, to feed
    # return_fired_transition directly.
    transition_weights = dict()
    if enabled_transitions is not None:
        keys = [t.name for t in enabled_transitions]
    else:
        keys = list(models_t.keys())
    for k in keys:
        m = models_t.get(k)
        if isinstance(m, DecisionRules):
            transition_weights[k] = m.apply(dict_x)
        elif isinstance(m, (int, float)):
            transition_weights[k] = float(m)
        else:
            transition_weights[k] = 0
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
        if isinstance(model, DecisionRules):
            resource_weights[r] = model.apply(x_r)
        elif isinstance(model, (int, float)):
            resource_weights[r] = float(model)
        else:
            resource_weights[r] = 0
    return resource_weights


def count_concurrent_events(schedule, t_enabled) -> int:

    count = 0
    for start, end in reversed(schedule):
        if end <= t_enabled:
            break
        if start <= t_enabled < end:
            count += 1

    return count


def is_resource_busy(schedule, t_enabled) -> bool:
    for start, end in reversed(schedule):
        if end <= t_enabled:
            return False
        if start <= t_enabled < end:
            return True
    return False


def calendar_to_working_set(calendar: dict) -> frozenset:
    return frozenset((wd, h) for wd, hours in calendar.items() for h, active in hours.items() if active)


def count_working_minutes(start_ts: datetime, end_ts: datetime, calendar: dict, _working_set: frozenset = None) -> int:
    # Same hour-by-hour loop pattern as add_minutes_with_calendar: O(wall time),
    # which kills discovery on long training logs. Skip full weeks in one jump
    # first (each contributes exactly ``minutes_per_week``, because shifting by
    # 7 days preserves the (weekday, hour) pattern), then run the original
    # loop on the <1 week residual.
    if end_ts <= start_ts:
        return 0
    working_set = _working_set if _working_set is not None else calendar_to_working_set(calendar)
    minutes_per_week = len(working_set) * 60

    working_minutes = 0
    current_time = start_ts
    if minutes_per_week > 0:
        week_seconds = 7 * 24 * 3600
        full_weeks = int((end_ts - current_time).total_seconds() // week_seconds)
        if full_weeks > 0:
            working_minutes += full_weeks * minutes_per_week
            current_time = current_time + timedelta(weeks=full_weeks)

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


def add_minutes_with_calendar(start_ts: datetime, minutes_to_add: int, calendar: dict, _working_set: frozenset = None) -> datetime:
    # The minute-by-minute loop below is O(minutes_to_add). For datasets like
    # BPI2017 where sampled waiting/execution times reach tens of thousands
    # of minutes, that turns a single simulation step into a 10–70 ms call,
    # and a full simulation into hours. A full week always contributes the
    # same number of working minutes, so jump full weeks first and only loop
    # over at most ~1 week of wall time afterwards.
    working_set = _working_set if _working_set is not None else calendar_to_working_set(calendar)
    minutes_per_week = len(working_set) * 60
    if minutes_per_week == 0:
        # Empty calendar used to spin forever in the loop below. Fail loudly
        # instead — this only fires when calendar discovery produced a
        # calendar with no active slots (previously a silent hang).
        raise ValueError("Calendar has no working hours; cannot advance time.")

    remaining_minutes = minutes_to_add
    current_time = start_ts
    if remaining_minutes >= minutes_per_week:
        full_weeks, remaining_minutes = divmod(remaining_minutes, minutes_per_week)
        current_time = current_time + timedelta(weeks=full_weeks)

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


def _alignments_to_moves(log, net, im, fm, name_to_trans):
    """Run A* alignments and project each trace to a list of
    ``(move_type, transition)`` tuples.

    ``move_type`` ∈ {``'sync'``, ``'tau'``, ``'model_visible'``}. Log-moves are
    dropped — they correspond to events the model could not consume and thus
    produce no firing. The caller advances the log-event cursor only on
    ``'sync'`` moves.

    ``ret_tuple_as_trans_desc=True`` gives each step as
    ``((trace_t_name, model_t_name), (trace_label, model_label))``, which is
    the only way to tell a visible model move (trace side ``'>>'``, model
    label set) from a sync (both sides set) and from a τ (trace side ``'>>'``,
    model label ``None``).
    """
    params = {AlignParams.PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE: True}
    try:
        aligned = alignments.apply_multiprocessing(log, net, im, fm, parameters=params)
    except Exception:
        aligned = alignments.apply_log(log, net, im, fm, parameters=params)

    per_trace = []
    for trace_align in aligned:
        moves = []
        if not trace_align:
            per_trace.append(moves)
            continue
        for step in trace_align.get("alignment", []):
            (trace_t_name, model_t_name), (_trace_label, model_label) = step
            if model_t_name == '>>':
                # Log move: event dropped, no firing.
                continue
            t = name_to_trans.get(model_t_name)
            if t is None:
                continue
            if trace_t_name == '>>':
                moves.append(('tau' if model_label is None else 'model_visible', t))
            else:
                moves.append(('sync', t))
        per_trace.append(moves)
    return per_trace


def build_df_features(log, net, im, fm, act_to_resources, net_transition_labels, resources, label_data_attributes=[], firing_sequences=None):

    df_log = pm4py.convert_to_dataframe(log)
    # Wall-clock epoch: strip tz if present, then take ``.timestamp()``. Used
    # only for relative comparisons within this function, so the choice of
    # reference tz doesn't matter as long as it's consistent.
    def _to_wall_epoch(x):
        ts = x.tz_localize(None) if getattr(x, 'tzinfo', None) is not None else x
        return ts.timestamp()
    df_log["start:timestamp"] = df_log["start:timestamp"].apply(_to_wall_epoch)
    df_log["time:timestamp"] = df_log["time:timestamp"].apply(_to_wall_epoch)

    # Pre-group events by resource for fast concurrent-event lookup (avoids full df scan per step)
    resource_events = {}
    for res_name, group in df_log.groupby('org:resource'):
        sorted_group = group.sort_values('start:timestamp')
        resource_events[res_name] = {
            'start_ts': sorted_group['start:timestamp'].values,
            'end_ts': sorted_group['time:timestamp'].values,
        }

    name_to_trans = {t.name: t for t in net.transitions}

    # Build, per trace, the sequence of (move_type, transition) pairs to walk.
    # ``move_type`` is one of:
    #   * 'sync'          -> visible transition paired with the log event at
    #                        the current cursor ``j``; produces a full training
    #                        row (resource, start_t, end_t, ...).
    #   * 'tau'           -> invisible model move; marking-only, no row data.
    #   * 'model_visible' -> visible transition with no corresponding log
    #                        event (alignment was forced to fire it to reach
    #                        the final marking); marking-only. The row carries
    #                        ``None`` for resource/start_t/end_t so time- and
    #                        resource-discovery (which filter on those) skip
    #                        it, while cf-discovery still sees a "this
    #                        transition fired at this marking" sample.
    #
    # This is the pm4py-native equivalent of what token replay used to do
    # here, minus the token injection: A* never corrupts the marking, so
    # ``return_enabled_transitions`` downstream always gets a clean enabled
    # set. Log-moves in the alignment drop events (no training row), which is
    # the trade-off for a clean marking.
    if firing_sequences is not None:
        activated_per_trace = [
            [('tau' if name_to_trans[t_name].label is None else 'sync',
              name_to_trans[t_name])
             for t_name in firing_sequences.get(trace[0]["case:concept:name"], [])]
            for trace in log
        ]
    else:
        activated_per_trace = _alignments_to_moves(log, net, im, fm, name_to_trans)

    dataset = []
    for i, trace in enumerate(tqdm(log)):

        activated = activated_per_trace[i]

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
        for move_type, transition in activated:
            transition_label = transition.label

            prev_enabled_transitions = return_enabled_transitions(net, marking)

            for enabled in prev_enabled_transitions:
                if enabled not in transition_enabled_times:
                    transition_enabled_times[enabled] = current_t

            if move_type == 'sync':  # real log event paired with this firing
                resource = trace[j]["org:resource"]
                start_t = trace[j]["start:timestamp"]
                end_t = trace[j]["time:timestamp"]
                # With A* the marking is always consistent, so the transition
                # should be structurally enabled here and ``transition_enabled_times``
                # should have its enable time. Fall back to ``current_t`` only as a
                # defensive measure.
                enabled_t = transition_enabled_times.get(transition, current_t)
                current_t = end_t
                # Match the wall-clock epoch produced by ``_to_wall_epoch``
                # above (pandas' ``.timestamp()`` after ``tz_localize(None)``
                # interprets the naive wall clock as UTC). Force-tagging
                # ``enabled_t`` as UTC gives the same semantics regardless of
                # its original tz (or absence thereof).
                enabled_ts = enabled_t.replace(tzinfo=timezone.utc).timestamp()
                if resource in resource_events:
                    re = resource_events[resource]
                    concurrent = (re['start_ts'] < enabled_ts) & (re['end_ts'] > enabled_ts)
                    res_workload = int(concurrent.sum())
                    if res_workload > 0:
                        max_epoch = re['end_ts'][concurrent].max()
                        # Reverse of ``_to_wall_epoch``: interpret the epoch
                        # as UTC, strip the tzinfo to recover the original
                        # wall clock, then re-attach ``enabled_t``'s tz so
                        # subtraction with ``start_t`` stays tz-compatible.
                        wall = datetime.fromtimestamp(max_epoch, tz=timezone.utc).replace(tzinfo=None)
                        if enabled_t.tzinfo is not None:
                            resource_free_t = wall.replace(tzinfo=enabled_t.tzinfo)
                        else:
                            resource_free_t = wall
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
            else:  # 'tau' or 'model_visible': marking update only, no log event
                resource = None
                enabled_t = None
                start_t = None
                end_t = None
                res_workload = None
                resource_free_t = None
                prev_enabled_resources = None
                candidate_workloads = None

            transition_enabled_times.pop(transition, None)

            marking = update_current_marking(marking, transition)

            handover_features = tuple(1 if r == last_resource else 0 for r in resources)
            last_activity_features = tuple(1 if t_l == last_activity else 0 for t_l in net_transition_labels)
            dataset.append((case_id, transition, transition_label, resource, enabled_t, start_t, end_t, prev_enabled_transitions, prev_enabled_resources, res_workload, resource_free_t, candidate_workloads) + tuple(trace_attributes) + tuple(history_res.values()) + tuple(history.values()) + handover_features + last_activity_features)

            # Only update activity/resource history for real log events. A
            # model_visible move is the aligner firing a visible transition
            # with no corresponding event, so treating it as "activity done"
            # would drift the history away from what the simulator sees at
            # inference time (where every visible firing IS an event).
            if move_type == 'sync' and transition_label:
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