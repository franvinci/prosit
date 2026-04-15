import pm4py
from pm4py.objects.log.obj import EventLog


def discover_arrival_calendar(log: EventLog) -> dict:
    """Return a ``{weekday: {hour: bool}}`` calendar where a slot is ``True``
    whenever at least one case arrived in that (weekday, hour) bucket."""

    calendar = {wd: {h: False for h in range(24)} for wd in range(7)}
    for trace in log:
        ts = trace[0]['start:timestamp']
        calendar[ts.weekday()][ts.hour] = True
    return calendar


def discover_res_calendars(log: EventLog, resources: list = []) -> dict:
    """Return ``{resource: {weekday: {hour: bool}}}``. Each slot is ``True``
    whenever at least one event of that resource (start or end timestamp)
    falls in the bucket."""

    if not resources:
        resources = list(pm4py.get_event_attribute_values(log, 'org:resource').keys())

    calendar_wd_hour_res = {
        res: {wd: {h: False for h in range(24)} for wd in range(7)}
        for res in resources
    }

    for trace in log:
        for event in trace:
            res = event['org:resource']
            if res not in calendar_wd_hour_res:
                continue
            for key in ('start:timestamp', 'time:timestamp'):
                ts = event[key]
                calendar_wd_hour_res[res][ts.weekday()][ts.hour] = True

    return calendar_wd_hour_res
