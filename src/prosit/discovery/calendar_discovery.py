import pm4py
from pm4py.objects.log.obj import EventLog


def discover_arrival_calendar(log: EventLog, min_participation: float = 0.05) -> dict:
    """Return a ``{weekday: {hour: bool}}`` calendar. A slot is ``True`` when
    its arrival count is at least ``min_participation * max_slot_count``. Set
    ``min_participation=0`` to keep every slot with at least one arrival."""

    counts = {wd: {h: 0 for h in range(24)} for wd in range(7)}
    for trace in log:
        ts = trace[0]['start:timestamp']
        counts[ts.weekday()][ts.hour] += 1

    max_count = max((counts[wd][h] for wd in range(7) for h in range(24)), default=0)
    threshold = min_participation * max_count

    calendar = {wd: {h: False for h in range(24)} for wd in range(7)}
    if max_count == 0:
        return calendar
    for wd in range(7):
        for h in range(24):
            if counts[wd][h] > 0 and counts[wd][h] >= threshold:
                calendar[wd][h] = True
    return calendar


def discover_res_calendars(log: EventLog, resources: list = [], min_participation: float = 0.05) -> dict:
    """Return ``{resource: {weekday: {hour: bool}}}``. Per resource, a slot is
    ``True`` when its event count is at least ``min_participation * max_slot_count``
    for that resource. Set ``min_participation=0`` to keep every slot with at
    least one event."""

    if not resources:
        resources = list(pm4py.get_event_attribute_values(log, 'org:resource').keys())

    counts = {
        res: {wd: {h: 0 for h in range(24)} for wd in range(7)}
        for res in resources
    }

    for trace in log:
        for event in trace:
            res = event['org:resource']
            if res not in counts:
                continue
            for key in ('start:timestamp', 'time:timestamp'):
                ts = event[key]
                counts[res][ts.weekday()][ts.hour] += 1

    calendar_wd_hour_res = {
        res: {wd: {h: False for h in range(24)} for wd in range(7)}
        for res in resources
    }

    for res in resources:
        max_count = max((counts[res][wd][h] for wd in range(7) for h in range(24)), default=0)
        if max_count == 0:
            continue
        threshold = min_participation * max_count
        for wd in range(7):
            for h in range(24):
                if counts[res][wd][h] > 0 and counts[res][wd][h] >= threshold:
                    calendar_wd_hour_res[res][wd][h] = True

    return calendar_wd_hour_res
