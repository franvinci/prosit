import pm4py
import pandas as pd
from pm4py.objects.log.obj import EventLog


def discover_resources_list(log: EventLog, thr: float = 0.95) -> list:

    resource_counts = pd.Series(pm4py.get_event_attribute_values(log, 'org:resource'))
    resource_counts = resource_counts / resource_counts.sum()
    resource_counts = resource_counts.sort_values(ascending=False)
    resource_counts_cumsum = resource_counts.cumsum()
    resource_counts = resource_counts[resource_counts_cumsum <= thr]
    resources = resource_counts.index.tolist()

    return resources


def discover_resource_acts_prob(log, resources) -> dict:

    act_resources = dict()
    for trace in log:
        for event in trace:
            res = event['org:resource']
            act = event['concept:name']
            if act not in act_resources.keys():
                act_resources[act] = {r: 0 for r in resources}
            if res not in resources:
                continue
            act_resources[act][res] += 1

    normalized = {}
    for act, res_dict in act_resources.items():
        total = sum(res_dict.values())
        if total == 0:
            normalized[act] = {res: 1/len(resources) for res in res_dict}
        else:
            normalized[act] = {res: val / total for res, val in res_dict.items()}

    for act in list(normalized.keys()):
        normalized[act] = {res: prob for res, prob in normalized[act].items() if prob > 0}

    return normalized