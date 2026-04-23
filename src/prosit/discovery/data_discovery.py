from collections import Counter

import numpy as np
import pm4py
from pm4py.objects.log.obj import EventLog

from prosit.utils.distribution_utils import return_best_distribution

_ATTR_DIST_SEARCH = ['norm', 'expon', 'lognorm', 'gamma', 'uniform']


def discover_attributes_distribution(
        log: EventLog,
        label_data_attributes: list,
        label_data_attributes_categorical: list,
        mode: str = 'empirical'
    ) -> dict:
    """
    Discover the distribution of case-level data attributes.

    mode='empirical'  — samples from the joint empirical distribution of attribute
                        tuples observed in the log. Preserves correlations between
                        attributes but scales poorly with many attributes/values.

    mode='distribution' — models each attribute independently:
                          categorical -> frequency table {value: probability};
                          continuous  -> best-fitting scipy distribution.
                          Scales well but ignores inter-attribute correlations.
    """
    if mode == 'empirical':
        list_data_attributes = []
        for trace in log:
            try:
                list_data_attributes.append(tuple(trace[0][a] for a in label_data_attributes))
            except (KeyError, IndexError):
                continue
        frequency = Counter(list_data_attributes)
        total = len(list_data_attributes)
        data = {t: count / total for t, count in frequency.items()}
        return {'mode': 'empirical', 'data': data}

    elif mode == 'distribution':
        df_log = pm4py.convert_to_dataframe(log)
        # One row per case: take the first event's attributes
        first_events = df_log.groupby('case:concept:name').first().reset_index()

        data = {}
        for a in label_data_attributes:
            values = first_events[a].dropna()
            if a in label_data_attributes_categorical:
                freq = values.value_counts(normalize=True)
                data[a] = {'type': 'categorical', 'values': freq.to_dict()}
            else:
                values_list = values.tolist()
                if len(values_list) == 0:
                    data[a] = {'type': 'continuous', 'dist': 'fixed',
                               'params': [0.0], 'min': 0.0, 'max': 0.0, 'mean': 0.0}
                    continue
                dist, params = return_best_distribution(values_list, dist_search=_ATTR_DIST_SEARCH)
                dist_name = dist if dist == 'fixed' else dist.name
                data[a] = {
                    'type': 'continuous',
                    'dist': dist_name,
                    'params': list(params),
                    'min': float(np.min(values_list)),
                    'max': float(np.max(values_list)),
                    'mean': float(np.mean(values_list))
                }
        return {'mode': 'distribution', 'data': data}

    else:
        raise ValueError(f"Unknown attribute_mode '{mode}'. Use 'empirical' or 'distribution'.")


def return_label_data_attributes(log: EventLog) -> tuple:

    standard_xes_columns = {
        "case:concept:name", "concept:name", "time:timestamp",
        "start:timestamp", "org:resource", "org:role"
    }

    import pandas as pd

    df_log = pm4py.convert_to_dataframe(log)
    label_data_attributes = list(set(df_log.columns) - standard_xes_columns)

    # String-typed attributes that parse as numeric for every non-null value
    # (e.g. BPI12's ``case:AMOUNT_REQ`` stored as "20000") are treated as
    # continuous, not categorical — otherwise each distinct value becomes a
    # one-hot column and most are pruned as low-signal.
    label_data_attributes_categorical = []
    numeric_attrs = []
    for l in label_data_attributes:
        col = df_log[l]
        if col.dtype == 'object':
            coerced = pd.to_numeric(col, errors='coerce')
            non_null_src = col.notna().sum()
            if non_null_src > 0 and coerced.notna().sum() == non_null_src:
                numeric_attrs.append(l)
            else:
                label_data_attributes_categorical.append(l)

    # Coerce values in place on the log so downstream consumers (the training
    # df builders, which read via ``trace[0][a]``) see floats. ``convert_to_dataframe``
    # above has already broadcast case-level attributes onto every event.
    if numeric_attrs:
        for trace in log:
            for a in numeric_attrs:
                short_a = a[5:] if a.startswith('case:') else a
                if short_a in trace.attributes:
                    try:
                        trace.attributes[short_a] = float(trace.attributes[short_a])
                    except (ValueError, TypeError):
                        pass
                for event in trace:
                    if a in event:
                        try:
                            event[a] = float(event[a])
                        except (ValueError, TypeError):
                            pass

    return label_data_attributes, label_data_attributes_categorical
