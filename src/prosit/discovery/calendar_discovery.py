import pm4py
from pm4py.objects.log.obj import EventLog


DEFAULT_MIN_CONFIDENCE = 0.1
DEFAULT_MIN_SUPPORT = 0.1
DEFAULT_MIN_PARTICIPATION = 0.4

# The arrival calendar uses a much stricter support (cover most observed
# arrivals) than resource calendars.
DEFAULT_ARRIVAL_MIN_SUPPORT = 0.7


def _full_week_calendar(active: bool = True) -> dict:
    return {wd: {h: active for h in range(24)} for wd in range(7)}


def _build_calendar_from_slot_data(
    task_wd_dates: dict,
    task_wd_h_dates: dict,
    wd_h_count: dict,
    wd_h_tasks: dict,
    total_events: int,
    min_confidence: float,
    min_support: float,
) -> dict:
    """Apply a confidence + support filter to the accumulated slot data of a
    single resource (or pseudo-resource).

    - **Confidence**: for each (weekday, hour) seen, accept the slot when the
      most frequent task has ``#dates(t,wd,h) / #dates(t,wd) >= min_confidence``.
    - **Support boost**: if accepted slots cover < ``min_support`` of the
      events, greedily re-add discarded slots ordered by frequency until the
      support is met.
    - Empty result falls back to a 24/7 calendar."""

    cal = _full_week_calendar(False)
    if total_events == 0:
        return cal

    accepted_events = 0
    discarded = []
    for wd in range(7):
        for h in range(24):
            slot_count = wd_h_count[wd][h]
            if slot_count == 0:
                continue
            best_conf = 0.0
            for t in wd_h_tasks[wd][h]:
                h_dates = task_wd_h_dates[t][wd][h]
                wd_dates = task_wd_dates[t][wd]
                conf = len(h_dates) / len(wd_dates) if wd_dates else 0
                if conf > best_conf:
                    best_conf = conf
            if best_conf >= min_confidence:
                cal[wd][h] = True
                accepted_events += slot_count
            else:
                discarded.append((wd, h, slot_count))

    support = accepted_events / total_events
    if support < min_support and discarded:
        discarded.sort(key=lambda x: x[2], reverse=True)
        for wd, h, cnt in discarded:
            cal[wd][h] = True
            accepted_events += cnt
            support = accepted_events / total_events
            if support >= min_support:
                break

    if accepted_events == 0:
        cal = _full_week_calendar(True)
    return cal


def discover_arrival_calendar(
    log: EventLog,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_support: float = DEFAULT_ARRIVAL_MIN_SUPPORT,
) -> dict:
    """Return a ``{weekday: {hour: bool}}`` calendar for case arrivals.

    Uses the same confidence + support + greedy-boost pipeline as
    ``discover_res_calendars``, treating all arrivals as a single virtual
    resource performing a single virtual task. The participation filter is
    skipped (only one resource). Granularity is 60 minutes per slot."""

    task = '__arrival__'
    task_wd_dates = {task: {wd: set() for wd in range(7)}}
    task_wd_h_dates = {
        task: {wd: {h: set() for h in range(24)} for wd in range(7)}
    }
    wd_h_count = {wd: {h: 0 for h in range(24)} for wd in range(7)}
    wd_h_tasks = {wd: {h: set() for h in range(24)} for wd in range(7)}
    total_events = 0

    for trace in log:
        ts = trace[0]['start:timestamp']
        wd, h, date_key = ts.weekday(), ts.hour, ts.date()
        task_wd_dates[task][wd].add(date_key)
        task_wd_h_dates[task][wd][h].add(date_key)
        wd_h_count[wd][h] += 1
        wd_h_tasks[wd][h].add(task)
        total_events += 1

    return _build_calendar_from_slot_data(
        task_wd_dates, task_wd_h_dates,
        wd_h_count, wd_h_tasks,
        total_events,
        min_confidence, min_support,
    )


def discover_res_calendars(
    log: EventLog,
    resources: list = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_support: float = DEFAULT_MIN_SUPPORT,
    min_participation: float = DEFAULT_MIN_PARTICIPATION,
) -> dict:
    """Return ``{resource: {weekday: {hour: bool}}}`` at 60-minute granularity.

    - **Participation**: a resource gets a discovered calendar only if
      ``sum_t count[r,t] / sum_t max_r' count[r',t] >= min_participation``;
      otherwise it falls back to a 24/7 calendar.
    - **Confidence + support**: per-resource calendar built by
      ``_build_calendar_from_slot_data``.

    Each event is registered once at its end timestamp."""

    if resources is None:
        resources = []
    if not resources:
        resources = list(pm4py.get_event_attribute_values(log, 'org:resource').keys())

    res_set = set(resources)
    res_task_count = {r: {} for r in resources}
    res_task_wd_dates = {r: {} for r in resources}
    res_task_wd_h_dates = {r: {} for r in resources}
    res_wd_h_count = {
        r: {wd: {h: 0 for h in range(24)} for wd in range(7)} for r in resources
    }
    res_wd_h_tasks = {
        r: {wd: {h: set() for h in range(24)} for wd in range(7)} for r in resources
    }
    res_total_events = {r: 0 for r in resources}

    for trace in log:
        for event in trace:
            r = event.get('org:resource')
            if r not in res_set:
                continue
            t = event.get('concept:name')
            ts = event.get('time:timestamp') or event.get('start:timestamp')
            if t is None or ts is None:
                continue
            wd, h, date_key = ts.weekday(), ts.hour, ts.date()

            res_task_count[r][t] = res_task_count[r].get(t, 0) + 1
            res_task_wd_dates[r].setdefault(t, {}).setdefault(wd, set()).add(date_key)
            res_task_wd_h_dates[r].setdefault(t, {}).setdefault(wd, {}).setdefault(h, set()).add(date_key)
            res_wd_h_count[r][wd][h] += 1
            res_wd_h_tasks[r][wd][h].add(t)
            res_total_events[r] += 1

    max_resource_task_freq = {}
    for r in resources:
        for t, c in res_task_count[r].items():
            if c > max_resource_task_freq.get(t, 0):
                max_resource_task_freq[t] = c

    calendars = {}
    for r in resources:
        if res_total_events[r] == 0:
            calendars[r] = _full_week_calendar(False)
            continue

        denom = sum(max_resource_task_freq.get(t, 0) for t in res_task_count[r])
        numer = sum(res_task_count[r].values())
        participation = numer / denom if denom else 0
        if participation < min_participation:
            calendars[r] = _full_week_calendar(True)
            continue

        calendars[r] = _build_calendar_from_slot_data(
            res_task_wd_dates[r], res_task_wd_h_dates[r],
            res_wd_h_count[r], res_wd_h_tasks[r],
            res_total_events[r],
            min_confidence, min_support,
        )

    return calendars
