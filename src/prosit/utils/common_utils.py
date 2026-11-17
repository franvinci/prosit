import random
import zlib
from bisect import bisect_left
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from copy import copy

import pm4py
from tqdm import tqdm
from joblib import Parallel, delayed, effective_n_jobs
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


class ModelPlan:
    """The model fits of one discovery step, not yet run.

    ``jobs`` are ``joblib.delayed`` calls returning ``(key, value)``;
    ``finish`` turns the collected ``{key: value}`` into the step's result.
    ``sizes`` (optional, one number per job) is a rough cost used only to
    start the heaviest fits first. Splitting a step this way lets
    :func:`run_model_plans` fit the models of several steps in one worker
    pool, so the wall time is that of the longest fit instead of the sum of
    the steps."""

    __slots__ = ('jobs', 'finish', 'desc', 'sizes')

    def __init__(self, jobs=(), finish=None, desc=None, sizes=None):
        self.jobs = list(jobs)
        self.finish = finish if finish is not None else (lambda fitted: fitted)
        self.desc = desc
        self.sizes = list(sizes) if sizes is not None else [0] * len(self.jobs)

    @classmethod
    def done(cls, result):
        """A step whose result is already known (nothing left to fit)."""
        return cls(finish=lambda fitted: result)


def _run_plan_job(plan_index, func, args, kwargs):
    key, value = func(*args, **kwargs)
    return plan_index, key, value


def run_model_plans(plans, desc=None) -> list:
    """Run the jobs of all ``plans`` in one pool; return each plan's result,
    in the order of ``plans``. Every job seeds what it needs from its own
    key, so the outcome does not depend on the order of execution."""
    plans = list(plans)
    tagged = [
        (size, delayed(_run_plan_job)(i, func, args, kwargs))
        for i, plan in enumerate(plans)
        for size, (func, args, kwargs) in zip(plan.sizes, plan.jobs)
    ]
    fitted = [dict() for _ in plans]
    if tagged:
        if effective_n_jobs(DEFAULT_N_JOBS) > 1:
            # Heaviest first, so a long fit is not left to start last. Without
            # workers the jobs run here, in the caller's process, one after the
            # other: keep the submission order then, because the time-model
            # jobs reseed the global generators as they go.
            tagged.sort(key=lambda item: -item[0])
        if desc is None:
            active = [plan.desc for plan in plans if plan.jobs]
            desc = active[0] if len(active) == 1 else 'model fitting'
        for i, key, value in parallel_with_progress((job for _, job in tagged), total=len(tagged), desc=desc):
            fitted[i][key] = value
    return [plan.finish(result) for plan, result in zip(plans, fitted)]


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
    per_candidate = workloads is not None or queue_lengths is not None
    # One working copy for all candidates: only the two per-candidate entries
    # differ between them, and the models never write to the mapping.
    x_r = dict(dict_x) if per_candidate else dict_x
    for r in enabled_resources:
        model = models_r.get(r)
        if per_candidate:
            if workloads is not None:
                x_r['workload'] = workloads.get(r, 0)
            if queue_lengths is not None:
                x_r['queue_length'] = queue_lengths.get(r, 0)
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


# working set -> weekly table, see ``_calendar_table``. Keyed by the frozenset
# itself (by value), so entries never go stale; emptied when it grows past a
# few thousand calendars.
_CALENDAR_TABLES = {}


def _calendar_table(working_set: frozenset) -> tuple:
    """Weekly lookup table of a calendar, slots numbered ``weekday * 24 + hour``:

    * ``working``: 168 booleans;
    * ``run``: for a working slot, how many consecutive working slots start
      there (wrapping around the week, 168 when every slot is working);
    * ``before``: working minutes before each slot over two consecutive weeks
      (337 entries), so a search started anywhere in the first week never
      has to wrap."""
    # Same membership test the walk does for each (weekday, hour) a clock can show.
    working = [(slot // 24, slot % 24) in working_set for slot in range(168)]
    run = [0] * 168
    if all(working):
        run = [168] * 168
    else:
        # Walk backwards twice around the week so runs that wrap are counted.
        length = 0
        for slot in range(2 * 168 - 1, -1, -1):
            length = length + 1 if working[slot % 168] else 0
            if slot < 168:
                run[slot] = length
    before = [0] * 337
    for slot in range(336):
        before[slot + 1] = before[slot] + (60 if working[slot % 168] else 0)
    table = (tuple(working), tuple(run), before, sum(working) == len(working_set))
    if len(_CALENDAR_TABLES) > 4096:
        _CALENDAR_TABLES.clear()
    _CALENDAR_TABLES[working_set] = table
    return table


def _add_working_minutes(current_time: datetime, remaining_minutes: int, table: tuple) -> datetime:
    """The hour-by-hour walk of ``add_minutes_with_calendar`` for a plain
    ``datetime`` and ``0 < remaining_minutes < minutes per week``, done by
    table lookup. Same result as the walk, including its two quirks: seconds
    and microseconds are carried along while the walk stays inside working
    slots, and dropped the first time a non-working slot is skipped."""
    working, run, before, in_range = table
    if not in_range:
        # A calendar with entries outside weekday 0-6 / hour 0-23: its
        # minutes-per-week does not match the slots a clock can reach.
        return _walk_working_minutes(current_time, remaining_minutes, working)
    slot = current_time.weekday() * 24 + current_time.hour
    if working[slot]:
        length = run[slot]
        if length == 168:
            return current_time + timedelta(minutes=remaining_minutes)
        in_run = (60 - current_time.minute) + 60 * (length - 1)
        if remaining_minutes <= in_run:
            return current_time + timedelta(minutes=remaining_minutes)
        remaining_minutes -= in_run
        skip = length + 1
    else:
        skip = 1
    # The walk is now on the hour, at the first slot after a non-working one.
    on_the_hour = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=skip)
    first = (slot + skip) % 168
    target = before[first] + remaining_minutes
    last = bisect_left(before, target, first + 1) - 1
    return on_the_hour + timedelta(hours=last - first, minutes=target - before[last])


def _walk_working_minutes(current_time: datetime, remaining_minutes: int, working: tuple) -> datetime:
    while remaining_minutes > 0:
        if working[current_time.weekday() * 24 + current_time.hour]:
            minutes_in_current_hour = min(remaining_minutes, 60 - current_time.minute)
            current_time += timedelta(minutes=minutes_in_current_hour)
            remaining_minutes -= minutes_in_current_hour
        else:
            current_time = (current_time + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    return current_time


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

    if type(current_time) is datetime and type(remaining_minutes) is int:
        # Plain ``datetime`` moves on the wall clock, so the walk below can be
        # replaced by a lookup in the weekly table of the calendar. Other
        # types (``pd.Timestamp`` in a DST zone adds elapsed time, not wall
        # time) keep the hour-by-hour walk.
        if remaining_minutes <= 0:
            return current_time
        table = _CALENDAR_TABLES.get(working_set)
        if table is None:
            table = _calendar_table(working_set)
        return _add_working_minutes(current_time, remaining_minutes, table)

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


_EPOCH_UNITS = {'s': 1, 'ms': 10**3, 'us': 10**6, 'ns': 10**9}


def _datetime_epochs(values) -> list:
    """Float epochs of a datetime64 Series, value for value what
    ``Timestamp.timestamp()`` returns (UTC-based for tz-aware data, rounded
    to microseconds), without building one Timestamp per row. Returns
    ``None`` when the Series is not datetime64."""
    if not pd.api.types.is_datetime64_any_dtype(values.dtype):
        return None
    denom = _EPOCH_UNITS.get(getattr(values.dt, 'unit', 'ns'))
    if denom is None:
        return None
    # int / int on Python integers: exact quotient correctly rounded, as in
    # pandas. Going through float64 first would be off by up to 128 ns.
    return [round(v / denom, 6) for v in values.astype('int64').tolist()]


def _wall_epochs(values) -> list:
    """Float epochs of the wall-clock reading (tz dropped, not converted)."""
    if not pd.api.types.is_datetime64_any_dtype(values.dtype) or values.isna().any():
        return None
    if values.dt.tz is not None:
        values = values.dt.tz_localize(None)
    return _datetime_epochs(values)


class _ResourceTimeline:
    """Sorted views over one resource's events for interval-stabbing queries.

    ``workload(q)`` is, for every query time, the number of events with
    ``start < q < end`` and the largest ``end`` among them. Only events with
    ``end > start`` can ever satisfy that, so the others are dropped up
    front; with them gone, every event that ended by ``q`` also started
    before ``q`` and the count is a difference of two rank queries."""

    __slots__ = ('starts', 'ends', 'max_end_by_start')

    def __init__(self, start_ts: np.ndarray, end_ts: np.ndarray):
        valid = end_ts > start_ts
        start_ts = start_ts[valid]
        end_ts = end_ts[valid]
        order = np.argsort(start_ts, kind='stable')
        self.starts = start_ts[order]
        self.max_end_by_start = np.maximum.accumulate(end_ts[order]) if len(order) else end_ts[order]
        self.ends = np.sort(end_ts)

    def workload(self, q: np.ndarray):
        started = np.searchsorted(self.starts, q, side='left')
        count = started - np.searchsorted(self.ends, q, side='right')
        return count, started


def dense_take(dense: dict, mask: np.ndarray) -> dict:
    """Follow a boolean row filter of the feature frame on its companion
    dense candidate block (see :func:`_build_features`)."""
    if dense is None:
        return None
    mask = np.asarray(mask, dtype=bool)
    if len(mask) != dense['n_rows']:
        return None
    kept = np.flatnonzero(mask)
    sel = mask[dense['rows']]
    out = dict(dense)
    out['n_rows'] = int(kept.size)
    out['rows'] = np.searchsorted(kept, dense['rows'][sel])
    out['workloads'] = dense['workloads'][sel]
    if dense.get('queue_lengths') is not None:
        out['queue_lengths'] = dense['queue_lengths'][sel]
    return out


def dense_matrix(dense: dict, n_rows: int, resources: list, key: str):
    """``(n_rows, len(resources))`` matrix ``key`` of a dense candidate block
    when the block covers exactly the rows of the frame, else ``None``."""
    if dense is None or dense.get(key) is None or dense['n_rows'] != n_rows:
        return None
    if list(dense['resources']) != list(resources):
        return None
    rows = dense['rows']
    if len(rows) != n_rows or (n_rows and (rows[0] != 0 or rows[-1] != n_rows - 1)):
        return None
    return dense[key]


def build_df_features(log, net, im, fm, act_to_resources, net_transition_labels, resources, label_data_attributes=[], firing_sequences=None):
    """One row per replayed move (alignment or given firing sequence) with the
    features used by the discovery steps. See :func:`_build_features`."""
    return _build_features(
        log, net, im, fm, act_to_resources, net_transition_labels, resources,
        label_data_attributes, firing_sequences,
    )[0]


def _build_features(log, net, im, fm, act_to_resources, net_transition_labels, resources,
                    label_data_attributes=[], firing_sequences=None, candidate_dicts=True, dense_queue=True):
    """Build the feature frame and its dense companion.

    Returns ``(df, dense)``. ``df`` is the frame documented by the row layout
    below. ``dense`` holds, for the rows backed by a log event, the workload
    (and, with ``dense_queue``, the queue length) of every resource in
    ``resources`` at the enabling time, as ``(n_event_rows, len(resources))``
    matrices with zeros where the resource is not a candidate — the same
    numbers as the ``candidate_workloads`` / ``candidate_queue_lengths``
    cells. ``candidate_dicts=False`` leaves those two dict-valued columns
    empty, for callers that only read the matrices.

    The replay walks every trace once; markings are interned so the enabled
    set and the marking reached by a firing are computed once per distinct
    marking. Everything that only depends on (resource, enabling time) is
    computed after the walk, one rank query per resource instead of one array
    scan per event and candidate.

    ``dense`` is ``None`` when the input falls back to
    :func:`_build_df_features_legacy`."""

    df_log = pm4py.convert_to_dataframe(log)
    start_wall = end_wall = None
    try:
        plain = (
            len(set(resources)) == len(resources)
            and len(set(net_transition_labels)) == len(net_transition_labels)
            and len(df_log) == sum(len(trace) for trace in log)
        )
    except TypeError:
        plain = False
    if plain:
        start_wall = _wall_epochs(df_log["start:timestamp"])
        end_wall = _wall_epochs(df_log["time:timestamp"])
    if start_wall is None or end_wall is None:
        return _build_df_features_legacy(
            log, net, im, fm, act_to_resources, net_transition_labels, resources,
            label_data_attributes, firing_sequences,
        ), None
    start_wall = np.asarray(start_wall, dtype=np.float64)
    end_wall = np.asarray(end_wall, dtype=np.float64)

    # Per-resource timelines over every event of the log (not only the ones
    # the model can replay), keyed like the legacy ``groupby('org:resource')``.
    timelines = {
        res_name: _ResourceTimeline(start_wall[pos], end_wall[pos])
        for res_name, pos in df_log.groupby('org:resource').indices.items()
    }

    name_to_trans = {t.name: t for t in net.transitions}

    # Per trace, the sequence of (move_type, transition) pairs to walk; see
    # ``_build_df_features_legacy`` for the meaning of each move type.
    if firing_sequences is not None:
        activated_per_trace = [
            [('tau' if name_to_trans[t_name].label is None else 'sync',
              name_to_trans[t_name])
             for t_name in firing_sequences.get(trace[0]["case:concept:name"], [])]
            for trace in log
        ]
    else:
        activated_per_trace = _alignments_to_moves(log, net, im, fm, name_to_trans)

    handover_from_cols = ['handover_from_' + r for r in resources]
    last_activity_cols = ['last_activity_' + t_l for t_l in net_transition_labels]
    lead_cols = ["case_id", "transition", "transition_label", "resource", "enabled_t", "start_t", "end_t", "prev_enabled_transitions", "prev_enabled_resources", "res_workload", "resource_free_t", "candidate_workloads"] + label_data_attributes

    n_rows = sum(len(activated) for activated in activated_per_trace)
    if n_rows == 0:
        df = pd.DataFrame([], columns=lead_cols + resources + net_transition_labels + handover_from_cols + last_activity_cols)
        df['queue_length'] = 0
        df['candidate_queue_lengths'] = []
        return df, None
    n_sync = sum(1 for activated in activated_per_trace for move_type, _ in activated if move_type == 'sync')

    res_index = {r: k for k, r in enumerate(resources)}
    act_index = {t_l: k for k, t_l in enumerate(net_transition_labels)}

    col_case = [None] * n_rows
    col_transition = [None] * n_rows
    col_label = [None] * n_rows
    col_resource = [None] * n_rows
    col_enabled = [None] * n_rows
    col_start = [None] * n_rows
    col_end = [None] * n_rows
    col_prev_transitions = [None] * n_rows
    col_prev_resources = [None] * n_rows
    attr_cols = [[None] * n_rows for _ in label_data_attributes]

    # History counts change only when an event is consumed: keep one snapshot
    # per change and remember, per row, which snapshot was current.
    snaps_res = np.zeros((n_sync + 1, len(resources)), dtype=np.int64)
    snaps_act = np.zeros((n_sync + 1, len(net_transition_labels)), dtype=np.int64)
    n_snaps = 1
    row_snap = np.zeros(n_rows, dtype=np.int64)
    row_last_res = np.full(n_rows, -1, dtype=np.int64)
    row_last_act = np.full(n_rows, -1, dtype=np.int64)

    sync_rows = []
    sync_wall = []      # wall-clock epoch of the enabling time of each sync row

    # Marking interning. Keys are the full place -> count content, the only
    # thing ``return_enabled_transitions`` and ``update_current_marking`` look at.
    state_of = {}
    markings = []
    enabled_of = []
    next_state = {}

    def intern(m):
        key = frozenset(m.items())
        sid = state_of.get(key)
        if sid is None:
            sid = state_of[key] = len(markings)
            markings.append(m)
            enabled_of.append(None)
        return sid

    initial_state = intern(im)
    Timestamp = pd.Timestamp
    utc = timezone.utc
    # The precomputed wall-clock epochs stand in for
    # ``ts.replace(tzinfo=utc).timestamp()`` only for Timestamps that carry
    # the very tz of the corresponding ``df_log`` column; anything else is
    # converted with the original expression.
    start_tz = df_log["start:timestamp"].dt.tz
    end_tz = df_log["time:timestamp"].dt.tz

    row = 0
    offset = 0          # position of the trace's first event in ``df_log``
    for i, trace in enumerate(tqdm(log)):

        activated = activated_per_trace[i]

        case_id = trace[0]["case:concept:name"]
        cur_snap = 0
        last_res = -1
        last_act = -1
        if label_data_attributes:
            try:
                trace_attributes = [trace[a] for a in label_data_attributes]
            except (KeyError, TypeError):
                trace_attributes = [trace[0][a] for a in label_data_attributes]
        else:
            trace_attributes = []

        sid = initial_state
        j = 0
        # transition -> (enabling time, its wall-clock epoch). Same bookkeeping
        # as the legacy walk: an entry is only removed when its transition fires.
        transition_enabled_times = dict()
        current_t = trace[0]["start:timestamp"]
        if type(current_t) is Timestamp and current_t.tzinfo is start_tz:
            current = (current_t, start_wall[offset])
        else:
            current = (current_t, current_t.replace(tzinfo=utc).timestamp())
        for move_type, transition in activated:
            transition_label = transition.label

            prev_enabled_transitions = enabled_of[sid]
            if prev_enabled_transitions is None:
                prev_enabled_transitions = enabled_of[sid] = return_enabled_transitions(net, markings[sid])

            for enabled in prev_enabled_transitions:
                if enabled not in transition_enabled_times:
                    transition_enabled_times[enabled] = current

            col_case[row] = case_id
            col_transition[row] = transition
            col_label[row] = transition_label
            col_prev_transitions[row] = prev_enabled_transitions
            row_snap[row] = cur_snap
            row_last_res[row] = last_res
            row_last_act[row] = last_act
            if trace_attributes:
                for k, value in enumerate(trace_attributes):
                    attr_cols[k][row] = value

            if move_type == 'sync':  # real log event paired with this firing
                event = trace[j]
                resource = event["org:resource"]
                end_t = event["time:timestamp"]
                enabled_t, enabled_wall = transition_enabled_times.get(transition, current)
                if type(end_t) is Timestamp and end_t.tzinfo is end_tz:
                    current = (end_t, end_wall[offset + j])
                else:
                    current = (end_t, end_t.replace(tzinfo=utc).timestamp())
                col_resource[row] = resource
                col_enabled[row] = enabled_t
                col_start[row] = event["start:timestamp"]
                col_end[row] = end_t
                col_prev_resources[row] = act_to_resources[transition_label]
                sync_rows.append(row)
                sync_wall.append(enabled_wall)
                j += 1

            transition_enabled_times.pop(transition, None)

            key = (sid, transition)
            next_sid = next_state.get(key)
            if next_sid is None:
                next_sid = next_state[key] = intern(update_current_marking(markings[sid], transition))
            sid = next_sid

            # Only real log events move the activity/resource history (a
            # model_visible move has no event behind it).
            if move_type == 'sync' and transition_label:
                snaps_act[n_snaps] = snaps_act[cur_snap]
                snaps_act[n_snaps, act_index[transition_label]] += 1
                snaps_res[n_snaps] = snaps_res[cur_snap]
                try:
                    res_k = res_index.get(resource, -1)
                except TypeError:
                    res_k = -1
                if res_k >= 0:
                    snaps_res[n_snaps, res_k] += 1
                cur_snap = n_snaps
                n_snaps += 1
                last_res = res_k
                last_act = act_index[transition_label]

            row += 1
        offset += len(trace)

    sync_rows = np.asarray(sync_rows, dtype=np.int64)
    sync_wall = np.asarray(sync_wall, dtype=np.float64)
    n_sync = len(sync_rows)

    # Chosen resource: workload at the enabling time and the moment it gets free.
    col_workload = [None] * n_rows
    col_free = [None] * n_rows
    by_resource = {}
    by_label = {}
    for k in range(n_sync):
        r = sync_rows[k]
        col_workload[r] = 0
        col_free[r] = col_enabled[r]
        try:
            by_resource.setdefault(col_resource[r], []).append(k)
        except TypeError:
            pass
        by_label.setdefault(col_label[r], []).append(k)

    for resource, ks in by_resource.items():
        timeline = timelines.get(resource)
        if timeline is None:
            continue
        ks = np.asarray(ks, dtype=np.int64)
        count, started = timeline.workload(sync_wall[ks])
        rows_r = sync_rows[ks].tolist()
        for r, c in zip(rows_r, count.tolist()):
            col_workload[r] = c
        for b in np.flatnonzero(count > 0).tolist():
            r = rows_r[b]
            enabled_t = col_enabled[r]
            max_epoch = timeline.max_end_by_start[started[b] - 1]
            # Reverse of the wall-clock epoch: read it as UTC, drop the
            # tzinfo, then re-attach ``enabled_t``'s tz so the subtraction
            # with ``start_t`` stays tz-compatible.
            wall = datetime.fromtimestamp(max_epoch, tz=timezone.utc).replace(tzinfo=None)
            if enabled_t.tzinfo is not None:
                col_free[r] = wall.replace(tzinfo=enabled_t.tzinfo)
            else:
                col_free[r] = wall

    # Workload of every candidate resource of the activity at the enabling time.
    workloads = np.zeros((n_sync, len(resources)), dtype=np.int64)
    col_candidate_workloads = [None] * n_rows
    for label, ks in by_label.items():
        candidates = act_to_resources.get(label, [])
        ks = np.asarray(ks, dtype=np.int64)
        q = sync_wall[ks]
        per_candidate = []
        for cand in candidates:
            timeline = timelines.get(cand)
            count = timeline.workload(q)[0] if timeline is not None else np.zeros(len(ks), dtype=np.int64)
            per_candidate.append(count)
            cand_k = res_index.get(cand)
            if cand_k is not None:
                workloads[ks, cand_k] = count
        if candidate_dicts:
            rows_l = sync_rows[ks].tolist()
            if per_candidate:
                for r, values in zip(rows_l, np.column_stack(per_candidate).tolist()):
                    col_candidate_workloads[r] = dict(zip(candidates, values))
            else:
                for r in rows_l:
                    col_candidate_workloads[r] = {}

    # The leading columns go through the same constructor as before so pandas
    # infers the same dtypes; the integer groups are assembled as blocks.
    lead = pd.DataFrame(
        list(zip(col_case, col_transition, col_label, col_resource, col_enabled, col_start, col_end,
                 col_prev_transitions, col_prev_resources, col_workload, col_free, col_candidate_workloads,
                 *attr_cols)),
        columns=lead_cols,
    )
    handover = np.zeros((n_rows, len(resources)), dtype=np.int64)
    has_last_res = np.flatnonzero(row_last_res >= 0)
    handover[has_last_res, row_last_res[has_last_res]] = 1
    last_activity = np.zeros((n_rows, len(net_transition_labels)), dtype=np.int64)
    has_last_act = np.flatnonzero(row_last_act >= 0)
    last_activity[has_last_act, row_last_act[has_last_act]] = 1
    df = pd.concat(
        [
            lead,
            pd.DataFrame(snaps_res[row_snap], columns=resources),
            pd.DataFrame(snaps_act[row_snap], columns=net_transition_labels),
            pd.DataFrame(handover, columns=handover_from_cols),
            pd.DataFrame(last_activity, columns=last_activity_cols),
        ],
        axis=1,
    )

    # Queue lengths. For resource r at query time q:
    #   |{i on r : enabled_ts_i <= q and start_ts_i > q}|
    #       = rank_right(enabled_sorted, q) - rank_right(start_sorted, q)
    # with the true epochs of the frame's own ``enabled_t`` / ``start_t``
    # columns. The two rank queries are taken over the events r performed
    # that do wait, i.e. start after their enabling time: only those can
    # ever be in a queue, and an event recorded as starting before it was
    # enabled would otherwise be subtracted from the count. The event being
    # assigned is one of the counted events whenever it is r that performs
    # it and it waits; it is not part of the queue it finds — the simulator
    # computes the queue before assigning the event — so its own
    # contribution is taken out.
    queue_length = np.zeros(n_rows, dtype=np.int64)
    queues = np.zeros((n_sync, len(resources)), dtype=np.int64) if (dense_queue or candidate_dicts) else None
    col_candidate_queues = [None] * n_rows
    sync_mask_full = (~df['resource'].isna() & ~df['enabled_t'].isna() & ~df['start_t'].isna()).to_numpy()
    if sync_mask_full.any():
        enabled_true = _datetime_epochs(df['enabled_t'])
        start_true = _datetime_epochs(df['start_t'])
        masked_rows = np.flatnonzero(sync_mask_full)
        if enabled_true is None or start_true is None:
            enabled_true = np.zeros(n_rows)
            start_true = np.zeros(n_rows)
            enabled_values = df['enabled_t'].tolist()
            start_values = df['start_t'].tolist()
            for r in masked_rows.tolist():
                enabled_true[r] = enabled_values[r].timestamp()
                start_true[r] = start_values[r].timestamp()
        else:
            enabled_true = np.asarray(enabled_true, dtype=np.float64)
            start_true = np.asarray(start_true, dtype=np.float64)

        per_res_sorted = {}
        rows_of_resource = {}
        for r in masked_rows.tolist():
            rows_of_resource.setdefault(col_resource[r], []).append(r)
        for resource, rows_r in rows_of_resource.items():
            rows_r = np.asarray(rows_r, dtype=np.int64)
            q = enabled_true[rows_r]
            starts = start_true[rows_r]
            waits = starts > q
            enabled_arr = np.sort(q[waits])
            start_arr = np.sort(starts[waits])
            per_res_sorted[resource] = (enabled_arr, start_arr)
            queue_length[rows_r] = (
                np.searchsorted(enabled_arr, q, side='right') - np.searchsorted(start_arr, q, side='right') - waits
            )

        if queues is not None:
            # Per-candidate queue length at enabling time (mirrors the
            # per-candidate workloads above).
            sync_position = np.full(n_rows, -1, dtype=np.int64)
            sync_position[sync_rows] = np.arange(n_sync)
            for label, ks in by_label.items():
                rows_l = sync_rows[np.asarray(ks, dtype=np.int64)]
                rows_l = rows_l[sync_mask_full[rows_l]]
                if len(rows_l) == 0:
                    continue
                candidates = act_to_resources.get(label, [])
                q = enabled_true[rows_l]
                waits = start_true[rows_l] > q
                chosen_codes, chosen = pd.factorize(np.array([col_resource[r] for r in rows_l.tolist()], dtype=object))
                code_of = {resource: code for code, resource in enumerate(chosen)}
                per_candidate = []
                for cand in candidates:
                    arrays = per_res_sorted.get(cand)
                    if arrays is not None:
                        count = np.searchsorted(arrays[0], q, side='right') - np.searchsorted(arrays[1], q, side='right')
                        own = code_of.get(cand)
                        if own is not None:
                            count = count - (waits & (chosen_codes == own))
                    else:
                        count = np.zeros(len(rows_l), dtype=np.int64)
                    per_candidate.append(count)
                    cand_k = res_index.get(cand)
                    if cand_k is not None:
                        queues[sync_position[rows_l], cand_k] = count
                if candidate_dicts:
                    if per_candidate:
                        for r, values in zip(rows_l.tolist(), np.column_stack(per_candidate).tolist()):
                            col_candidate_queues[r] = dict(zip(candidates, values))
                    else:
                        for r in rows_l.tolist():
                            col_candidate_queues[r] = {}

    df['queue_length'] = queue_length
    df['candidate_queue_lengths'] = col_candidate_queues

    dense = {
        'resources': list(resources),
        'n_rows': n_rows,
        'rows': sync_rows,
        'workloads': workloads,
        'queue_lengths': queues,
    }
    return df, dense


def _build_df_features_legacy(log, net, im, fm, act_to_resources, net_transition_labels, resources, label_data_attributes=[], firing_sequences=None):
    """Reference implementation of :func:`build_df_features`: one Python pass
    with per-row array scans. Kept for the inputs the vectorised builder does
    not model (duplicate resource/activity names, non-datetime timestamp
    columns) so those behave exactly as before."""

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
    # over the events of r that start after their enabling time (the others
    # can never be in a queue and would be subtracted from the count). The
    # event being assigned is left out of its own resource's count (it is in
    # the arrays when it starts after its enabling time).
    per_res_sorted = {}
    sync_mask_full = ~df['resource'].isna() & ~df['enabled_t'].isna() & ~df['start_t'].isna()
    if sync_mask_full.any():
        for resource, group in df[sync_mask_full].groupby('resource'):
            enabled_ts = np.array([et.timestamp() for et in group['enabled_t']])
            start_ts = np.array([st.timestamp() for st in group['start_t']])
            waits = start_ts > enabled_ts
            per_res_sorted[resource] = (np.sort(enabled_ts[waits]), np.sort(start_ts[waits]))

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
                own = 1 if df.at[i, 'start_t'].timestamp() > q else 0
                df.at[i, 'queue_length'] = a - b - own

    # Per-candidate queue length at enabling time (used by the resource
    # assignment classifier — mirrors candidate_workloads).
    # Build the column as a positional list and assign it once (mirrors how
    # ``candidate_workloads`` is created via the DataFrame constructor). The
    # previous per-row ``df.at[i, col] = <dict>`` raises "Incompatible indexer
    # with Series" on pandas >= 2.x (it aligns the dict RHS as a Series).
    cql_col = [None] * len(df)
    if sync_mask_full.any():
        pos = {idx: p for p, idx in enumerate(df.index)}
        for i in df[sync_mask_full].index:
            cw = df.at[i, 'candidate_workloads']
            if not isinstance(cw, dict):
                continue
            q = df.at[i, 'enabled_t'].timestamp()
            chosen = df.at[i, 'resource']
            own = 1 if df.at[i, 'start_t'].timestamp() > q else 0
            cq = {}
            for cand in cw.keys():
                if cand in per_res_sorted:
                    enabled_arr, start_arr = per_res_sorted[cand]
                    a = int(np.searchsorted(enabled_arr, q, side='right'))
                    b = int(np.searchsorted(start_arr, q, side='right'))
                    cq[cand] = a - b - (own if cand == chosen else 0)
                else:
                    cq[cand] = 0
            cql_col[pos[i]] = cq
    df['candidate_queue_lengths'] = cql_col

    return df