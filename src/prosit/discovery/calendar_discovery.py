import pm4py
from pm4py.objects.log.obj import EventLog

def discover_arrival_calendar(log: EventLog, thr_h: float = 0.0, thr_wd: float = 0.0) -> dict:

    N_events_per_hour = {wd: {h: 0 for h in range(24)} for wd in range(7)}

    for trace in log:
        ts = trace[0]['start:timestamp']
        N_events_per_hour[ts.weekday()][ts.hour] += 1
        ts = trace[0]['time:timestamp']
        N_events_per_hour[ts.weekday()][ts.hour] += 1

    N_events_per_hour_perc = {wd: {h: 0 for h in range(24)} for wd in range(7)}

    for weekday in range(7):
        for h in range(24):
            if sum(N_events_per_hour[weekday].values()):
                N_events_per_hour_perc[weekday][h] = N_events_per_hour[weekday][h] / sum(N_events_per_hour[weekday].values())
            else:
                N_events_per_hour_perc[weekday][h] = 0

    N_events_per_wd = {wd: 0 for wd in range(7)}

    for weekday in range(7):
        N_events_per_wd[weekday] += sum(N_events_per_hour[weekday].values())

    N_events_per_wd_perc = {wd: N_events_per_wd[wd]/sum(N_events_per_wd.values()) if sum(N_events_per_wd.values()) else 0 for wd in range(7)}

    calendar_wd_hour = {wd: {h: False for h in range(24)} for wd in range(7)}

    for wd in range(7):
        if N_events_per_wd_perc[wd] <= thr_wd:
            continue
        else:
            for hour in range(24):
                calendar_wd_hour[wd][hour] = N_events_per_hour_perc[wd][hour] > thr_h

    return calendar_wd_hour



def discover_res_calendars(log: EventLog, thr_h: float = 0.0, thr_wd: float = 0.0) -> dict:

    resources = list(pm4py.get_event_attribute_values(log, 'org:resource').keys())

    N_events_per_hour_res = {res: {wd: {h: 0 for h in range(24)} for wd in range(7)} for res in resources}

    for trace in log:
        for event in trace:
            res = event['org:resource']
            ts = event['start:timestamp']
            N_events_per_hour_res[res][ts.weekday()][ts.hour] += 1
            ts = event['time:timestamp']
            N_events_per_hour_res[res][ts.weekday()][ts.hour] += 1


    N_events_per_hour_res_perc = {res: {wd: {h: 0 for h in range(24)} for wd in range(7)} for res in resources}

    for res in resources:
        for weekday in range(7):
            for h in range(24):
                if sum(N_events_per_hour_res[res][weekday].values()):
                    N_events_per_hour_res_perc[res][weekday][h] = N_events_per_hour_res[res][weekday][h] / sum(N_events_per_hour_res[res][weekday].values())
                else:
                    N_events_per_hour_res_perc[res][weekday][h] = 0


    N_events_per_wd_res = {res: {wd: 0 for wd in range(7)} for res in resources}

    for res in resources:
        for weekday in range(7):
            N_events_per_wd_res[res][weekday] += sum(N_events_per_hour_res[res][weekday].values())


    N_events_per_wd_res_perc = {res: {wd: N_events_per_wd_res[res][wd]/sum(N_events_per_wd_res[res].values()) if sum(N_events_per_wd_res[res].values()) else 0 for wd in range(7)} for res in resources}

    calendar_wd_hour_res = {res: {wd: {h: False for h in range(24)} for wd in range(7)} for res in resources}

    for res in resources:
        for wd in range(7):
            if N_events_per_wd_res_perc[res][wd] <= thr_wd:
                continue
            else:
                for hour in range(24):
                    calendar_wd_hour_res[res][wd][hour] = N_events_per_hour_res_perc[res][wd][hour] > thr_h

    return calendar_wd_hour_res