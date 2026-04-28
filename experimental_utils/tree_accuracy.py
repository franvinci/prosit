"""Tree-accuracy metrics for PROSIT ``SimulatorParameters``.

For each of the five decision trees PROSIT discovers (arrival, execution,
waiting, transition weights, resource weights), this module evaluates how well
the tree's leaf predictions match an arbitrary normalized event log:

- ``arrival_tree`` — Wasserstein distance between empirical inter-arrival times
  (per leaf) and the stored distribution at that leaf.
- ``execution_tree`` — Wasserstein on execution times, per activity per leaf,
  aggregated as support-weighted mean across leaves and then averaged across
  activities.
- ``waiting_tree`` — Wasserstein on waiting times, per resource per leaf.
- ``transition_tree`` — mean absolute difference between empirical firing
  frequency and the leaf's stored probability, per transition per leaf.
- ``resource_tree`` — same as transition tree but per resource.

Each evaluator returns ``None`` when the log lacks the features that the tree
was trained on (e.g., no resources or no case attributes).
"""

import numpy as np
import pandas as pd
import pm4py
from scipy.stats import wasserstein_distance

from prosit.utils.common_utils import (
    count_working_minutes,
    calendar_to_working_set,
    build_df_features,
)
from prosit.utils.rule_utils import DecisionRules
from prosit.utils.distribution_utils import sampling_from_dist


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def _build_feature_matrix(tree_dr, features_list):
    """Return an ``(n_events, n_features)`` array in sklearn's feature order."""
    sk = tree_dr.decision_tree
    cols = list(sk.feature_names_in_)
    rows = np.zeros((len(features_list), len(cols)), dtype=float)
    for i, feat in enumerate(features_list):
        for j, c in enumerate(cols):
            v = feat.get(c, 0)
            try:
                rows[i, j] = float(v)
            except (TypeError, ValueError):
                rows[i, j] = 0.0
    return rows


def _sample_from_leaf(leaf_rule, n=1000):
    if 'sampled' in leaf_rule and leaf_rule['sampled'] is not None and len(leaf_rule['sampled']) > 0:
        return np.asarray(leaf_rule['sampled'], dtype=float)
    dist = leaf_rule.get('dist')
    value = leaf_rule.get('value', 0.0)
    if dist is None:
        return np.array([value], dtype=float)
    dist_obj = dist[0]
    params = dist[1] if len(dist) > 1 else ()
    min_v = dist[2] if len(dist) > 2 else 0
    max_v = dist[3] if len(dist) > 3 else 0
    return np.asarray(
        sampling_from_dist(dist_obj, params, min_v, max_v, value, n_sample=n),
        dtype=float,
    )


def _leaf_time_wasserstein(tree_dr, features_list, y):
    if not features_list:
        return None
    y = np.asarray(y, dtype=float)
    # ``_build_no_rule_decision_rules`` (fallback when no CV-scored tree beats
    # the global baseline) skips ``from_decision_tree`` and therefore has no
    # ``.decision_tree``. It carries a single leaf keyed ``0`` — route all
    # samples there.
    if not hasattr(tree_dr, 'decision_tree'):
        rule = tree_dr.rules.get(0)
        if rule is None or 'value' not in rule:
            return None
        samples = _sample_from_leaf(rule)
        try:
            return float(wasserstein_distance(y, samples))
        except Exception:
            return None

    sk = tree_dr.decision_tree
    X = _build_feature_matrix(tree_dr, features_list)
    leaf_ids = sk.apply(X)

    total_n = 0
    weighted_sum = 0.0
    for leaf_id in np.unique(leaf_ids):
        mask = leaf_ids == leaf_id
        y_leaf = y[mask]
        if len(y_leaf) == 0:
            continue
        rule = tree_dr.rules.get(int(leaf_id))
        if rule is None or 'value' not in rule:
            continue
        samples = _sample_from_leaf(rule)
        try:
            w = wasserstein_distance(y_leaf, samples)
        except Exception:
            continue
        weighted_sum += w * len(y_leaf)
        total_n += len(y_leaf)

    if total_n == 0:
        return None
    return weighted_sum / total_n


def _leaf_probability_error(tree_dr, features_list, y):
    if not features_list:
        return None
    y = np.asarray(y, dtype=float)
    if not hasattr(tree_dr, 'decision_tree'):
        rule = tree_dr.rules.get(0)
        if rule is None or 'value' not in rule:
            return None
        predicted = rule['value']
        if not isinstance(predicted, (int, float)):
            return None
        return float(abs(float(y.mean()) - float(predicted)))

    sk = tree_dr.decision_tree
    X = _build_feature_matrix(tree_dr, features_list)
    leaf_ids = sk.apply(X)

    total_n = 0
    weighted_sum = 0.0
    for leaf_id in np.unique(leaf_ids):
        mask = leaf_ids == leaf_id
        y_leaf = y[mask]
        if len(y_leaf) == 0:
            continue
        rule = tree_dr.rules.get(int(leaf_id))
        if rule is None or 'value' not in rule:
            continue
        predicted = rule['value']
        if not isinstance(predicted, (int, float)):
            continue
        empirical = float(y_leaf.mean())
        err = abs(empirical - float(predicted))
        weighted_sum += err * len(y_leaf)
        total_n += len(y_leaf)

    if total_n == 0:
        return None
    return weighted_sum / total_n


def _scalar_probability_error(predicted, y):
    """Single absolute error for a constant-probability "tree".

    ``transition_weights`` / ``resource_weights`` collapse to a numeric scalar
    when discovery's CV picks the marginal-prior baseline (constant label, or
    DummyClassifier-best). Treat that as a single-leaf model with the scalar
    as the predicted probability.
    """
    if not len(y):
        return None
    y = np.asarray(y, dtype=float)
    return float(abs(float(y.mean()) - float(predicted)))


def _single_dist_wasserstein(dist_tuple, y):
    if not len(y):
        return None
    y = np.asarray(y, dtype=float)
    if len(dist_tuple) >= 5:
        dist_obj, params, min_v, max_v, mean_v = dist_tuple[:5]
    else:
        dist_obj = dist_tuple[0]
        params = dist_tuple[1] if len(dist_tuple) > 1 else ()
        min_v = dist_tuple[2] if len(dist_tuple) > 2 else 0
        max_v = dist_tuple[3] if len(dist_tuple) > 3 else 0
        mean_v = float(np.mean(y))
    samples = np.asarray(
        sampling_from_dist(dist_obj, params, min_v, max_v, mean_v, n_sample=1000),
        dtype=float,
    )
    try:
        return wasserstein_distance(y, samples)
    except Exception:
        return None


def _build_features_with_alignment(params, log_df):
    """Run ``build_df_features`` on a simulated log DataFrame.

    Injects dummy columns for missing case attributes or ``org:resource`` so
    that the alignment pipeline doesn't crash; the per-tree evaluators decide
    whether the fabricated values are usable.
    """
    firing_sequences = log_df.attrs.get("prosit_firing_sequences")
    df = log_df.copy()
    df['start:timestamp'] = pd.to_datetime(df['start:timestamp'], utc=True)
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], utc=True)
    df['case:concept:name'] = df['case:concept:name'].astype(str)
    if 'org:resource' not in df.columns:
        df['org:resource'] = 'unknown'
    # Resources may arrive as int (e.g., simod/agentsimulator on
    # ConsultaDataMining writes numeric IDs). ``params.calendars`` /
    # ``params.resources`` are keyed by string, so the comparison
    # ``row['resource'] in params.calendars`` would silently miss every row
    # — execution_tree / waiting_tree end up empty and resource_tree
    # collapses to a degenerate all-zero label. Cast once here so every
    # downstream evaluator sees the same alphabet.
    df['org:resource'] = df['org:resource'].fillna('unknown').astype(str)
    for attr in params.label_data_attributes:
        if attr not in df.columns:
            default = 0
            if attr in params.label_data_attributes_categorical:
                vals = params.attribute_values_label_categorical.get(attr, [])
                default = vals[0] if vals else 0
            df[attr] = default

    log = pm4py.convert_to_event_log(df)
    return build_df_features(
        log,
        params.net,
        params.initial_marking,
        params.final_marking,
        params.act_to_resources,
        params.net_transition_labels,
        params.resources,
        params.label_data_attributes,
        firing_sequences=firing_sequences,
    )


def _attrs_feature_dict(params, row):
    f = {}
    for a in params.label_data_attributes:
        if a in params.label_data_attributes_categorical:
            for v in params.attribute_values_label_categorical.get(a, []):
                f[a + ' = ' + str(v)] = int(row[a] == v)
        else:
            f[a] = row[a]
    return f


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------


def eval_arrival_tree(params, log_df):
    df = log_df.copy()
    df['start:timestamp'] = pd.to_datetime(df['start:timestamp'], utc=True)

    working_set = calendar_to_working_set(params.arrival_calendar)
    first_ts = df.groupby('case:concept:name')['start:timestamp'].min().sort_values()
    times = first_ts.tolist()
    if len(times) < 2:
        return None

    empirical = []
    feats = []
    for i in range(1, len(times)):
        delta = count_working_minutes(
            times[i - 1], times[i], params.arrival_calendar, working_set
        )
        empirical.append(delta)
        feats.append({'hour': times[i - 1].hour, 'weekday': times[i - 1].weekday()})

    tree = params.arrival_time_distribution
    if isinstance(tree, DecisionRules):
        return _leaf_time_wasserstein(tree, feats, empirical)
    return _single_dist_wasserstein(tree, empirical)


def eval_execution_tree(params, df_features):
    df = df_features[~df_features['start_t'].isna()].copy()
    if df.empty:
        return None

    working_sets = {r: calendar_to_working_set(params.calendars[r]) for r in params.calendars}

    def _exec_t(row):
        if row['resource'] not in params.calendars:
            return None
        return count_working_minutes(
            row['start_t'], row['end_t'],
            params.calendars[row['resource']], working_sets[row['resource']],
        )

    df['execution_time'] = df.apply(_exec_t, axis=1)
    df = df[df['execution_time'].notna()]

    results = []
    for act, tree in params.execution_time_distributions.items():
        df_act = df[df['transition_label'] == act]
        if df_act.empty:
            continue
        feats = []
        for _, row in df_act.iterrows():
            f = {t_l: row[t_l] for t_l in params.net_transition_labels}
            for r in params.resources:
                f['resource = ' + r] = 1 if row['resource'] == r else 0
            f.update(_attrs_feature_dict(params, row))
            feats.append(f)
        y = df_act['execution_time'].tolist()
        if isinstance(tree, DecisionRules):
            w = _leaf_time_wasserstein(tree, feats, y)
        else:
            w = _single_dist_wasserstein(tree, y)
        if w is not None:
            results.append(w)

    if not results:
        return None
    return float(np.mean(results))


def eval_waiting_tree(params, df_features):
    df = df_features[~df_features['start_t'].isna()].copy()
    df = df[~df['resource_free_t'].isna()]
    if df.empty:
        return None

    working_sets = {r: calendar_to_working_set(params.calendars[r]) for r in params.calendars}

    def _wait_t(row):
        if row['resource'] not in params.calendars:
            return None
        return count_working_minutes(
            row['resource_free_t'], row['start_t'],
            params.calendars[row['resource']], working_sets[row['resource']],
        )

    df['waiting_time'] = df.apply(_wait_t, axis=1)
    df = df[df['waiting_time'].notna()]

    results = []
    for res, tree in params.waiting_time_distributions.items():
        df_res = df[df['resource'] == res]
        if df_res.empty:
            continue
        feats = []
        for _, row in df_res.iterrows():
            f = {'workload': row['res_workload'], 'queue_length': row.get('queue_length', 0)}
            for t_l in params.net_transition_labels:
                f[t_l] = row[t_l]
            for act in params.net_transition_labels:
                f['waiting_activity = ' + act] = 1 if row['transition_label'] == act else 0
            f.update(_attrs_feature_dict(params, row))
            feats.append(f)
        y = df_res['waiting_time'].tolist()
        if isinstance(tree, DecisionRules):
            w = _leaf_time_wasserstein(tree, feats, y)
        else:
            w = _single_dist_wasserstein(tree, y)
        if w is not None:
            results.append(w)

    if not results:
        return None
    return float(np.mean(results))


def eval_transition_tree(params, df_features):
    df = df_features.copy()
    if df.empty:
        return None

    results = []
    # params.transition_weights is keyed by transition name (str); df columns
    # hold Transition objects, so compare by .name. Each weight is either a
    # DecisionRules (real or no-rule fallback) or a float scalar — the latter
    # when build_models picks DummyClassifier or the class is constant.
    for t_name, tree in params.transition_weights.items():
        is_tree = isinstance(tree, DecisionRules)
        is_scalar = isinstance(tree, (int, float)) and not isinstance(tree, bool)
        if not is_tree and not is_scalar:
            continue
        mask = df['prev_enabled_transitions'].apply(
            lambda s: any(tr.name == t_name for tr in s) if s is not None else False
        )
        df_t = df[mask]
        if df_t.empty:
            continue
        labels = df_t['transition'].apply(
            lambda tr: 1 if tr.name == t_name else 0
        ).tolist()
        if is_tree:
            feats = []
            for _, row in df_t.iterrows():
                f = {t_l: row[t_l] for t_l in params.net_transition_labels}
                for t_l in params.net_transition_labels:
                    f['last_activity_' + t_l] = row['last_activity_' + t_l]
                f.update(_attrs_feature_dict(params, row))
                feats.append(f)
            err = _leaf_probability_error(tree, feats, labels)
        else:
            err = _scalar_probability_error(tree, labels)
        if err is not None:
            results.append(err)

    if not results:
        return None
    return float(np.mean(results))


def eval_resource_tree(params, df_features):
    df = df_features[~df_features['resource'].isna()].copy()
    if df.empty:
        return None

    # Invert act_to_resources to find, per resource, the activities where
    # the resource was eligible — this is the scope used during training.
    r_to_acts = {r: set() for r in params.resources}
    for a, rs in params.act_to_resources.items():
        for r in rs:
            if r in r_to_acts:
                r_to_acts[r].add(a)

    results = []
    for r, tree in params.resource_weights.items():
        is_tree = isinstance(tree, DecisionRules)
        is_scalar = isinstance(tree, (int, float)) and not isinstance(tree, bool)
        if not is_tree and not is_scalar:
            continue
        eligible_acts = r_to_acts.get(r, set())
        if not eligible_acts:
            continue
        df_r = df[df['transition_label'].isin(eligible_acts)]
        if df_r.empty:
            continue
        labels = (df_r['resource'] == r).astype(int).tolist()
        if is_tree:
            feats = []
            for _, row in df_r.iterrows():
                f = {res: row[res] for res in params.resources}
                f.update(_attrs_feature_dict(params, row))
                feats.append(f)
            err = _leaf_probability_error(tree, feats, labels)
        else:
            err = _scalar_probability_error(tree, labels)
        if err is not None:
            results.append(err)

    if not results:
        return None
    return float(np.mean(results))


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def evaluate_all(params, log_df, info):
    """Compute all tree-accuracy metrics that are applicable to this log.

    ``info`` comes from ``simulator_loaders`` and contains ``has_resources``
    and ``has_case_attrs`` booleans.
    """
    results = {
        'arrival_tree': None,
        'execution_tree': None,
        'waiting_tree': None,
        'transition_tree': None,
        'resource_tree': None,
    }

    results['arrival_tree'] = _safe(eval_arrival_tree, params, log_df)

    has_resources = info.get('has_resources', False)
    has_case_attrs = info.get('has_case_attrs', False)
    requires_attrs = bool(params.label_data_attributes)
    attrs_ok = has_case_attrs or not requires_attrs

    # We can short-circuit if neither alignment-based branch will run.
    if not has_resources and not attrs_ok:
        return results

    df_features = _safe(_build_features_with_alignment, params, log_df)
    if df_features is None:
        return results

    if has_resources:
        if attrs_ok:
            results['execution_tree'] = _safe(eval_execution_tree, params, df_features)
            results['waiting_tree'] = _safe(eval_waiting_tree, params, df_features)
            results['resource_tree'] = _safe(eval_resource_tree, params, df_features)

    if attrs_ok:
        results['transition_tree'] = _safe(eval_transition_tree, params, df_features)

    return results
