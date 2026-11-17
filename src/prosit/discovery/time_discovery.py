from datetime import timezone

import numpy as np
from joblib import Parallel, delayed
import pandas as pd
import pm4py

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
import scipy.stats as stats
from scipy.stats import wasserstein_distance

from pm4py.objects.log.obj import EventLog

from prosit.utils.common_utils import (
    count_working_minutes,
    calendar_to_working_set,
    parallel_with_progress,
    seed_worker_from_key,
    DEFAULT_N_JOBS,
    ModelPlan,
    run_model_plans,
)
from prosit.utils.distribution_utils import return_best_distribution, sampling_from_dist, remove_outliers
from prosit.utils.rule_utils import DecisionRules, prune_low_signal_columns


DIST_SEARCH = ['fixed', 'norm', 'expon', 'lognorm', 'gamma', 'uniform']
DEFAULT_SAMPLE_SIZE = 1000
# Relative cost of a time-model fit versus a classifier fit of the same size,
# only used to order the jobs of a shared pool (heaviest first).
_TIME_JOB_WEIGHT = 3


# DISCOVERY

def discover_arrival_time(
        log: EventLog,
        calendar_arrival: dict,
        max_depths: list = range(1,6),
        min_samples_leaf_cv: list = [50, 100, 200],
        random_state: int = 72
    ) -> DecisionRules:

    if not max_depths:
        arrival_time_distr = find_best_distribution_arrival(log, calendar_arrival)
    else:
        arrival_time_distr = build_model_arrival(log, calendar_arrival, max_depths, min_samples_leaf_cv, random_state=random_state)

    return arrival_time_distr



def discover_execution_time_distributions(
        df_features: pd.DataFrame,
        net_transition_labels: list,
        resources: list,
        calendars: dict,
        max_depths: list = range(1,6),
        min_samples_leaf_cv: list = [50, 100, 200],
        label_data_attributes: list = [],
        label_data_attributes_categorical: list = [],
        values_categorical: dict = dict(),
        random_state: int = 72
    ) -> dict:

    if not max_depths:
        activity_exec_time_distributions = find_best_distribution_ex(df_features, net_transition_labels, calendars)
    else:
        activity_exec_time_distributions = build_models_ex(
                                                            df_features,
                                                            net_transition_labels,
                                                            resources,
                                                            calendars,
                                                            label_data_attributes,
                                                            label_data_attributes_categorical,
                                                            values_categorical,
                                                            max_depths,
                                                            min_samples_leaf_cv,
                                                            random_state=random_state
                                                        )

    return activity_exec_time_distributions



def discover_waiting_time(
        df_features: pd.DataFrame,
        net_transition_labels: list,
        resources: list,
        calendars: dict,
        label_data_attributes: list,
        label_data_attributes_categorical: list,
        values_categorical: dict,
        max_depths: list = range(1,6),
        min_samples_leaf_cv: list = [50, 100, 200],
        random_state: int = 72,
        use_workload_features: bool = False,
    ) -> dict:

    if not max_depths:
        res_waiting_time_distributions = find_best_distribution_wt(df_features, resources, calendars)
    else:
        res_waiting_time_distributions = build_models_wt(
                                                            df_features,
                                                            net_transition_labels,
                                                            resources,
                                                            calendars,
                                                            label_data_attributes,
                                                            label_data_attributes_categorical,
                                                            values_categorical,
                                                            max_depths,
                                                            min_samples_leaf_cv,
                                                            random_state=random_state,
                                                            use_workload_features=use_workload_features,
                                                        )

    return res_waiting_time_distributions



def _global_distribution(dist):
    """The ``scipy.stats`` distribution with the name of ``dist``.

    A distribution object that comes back from a worker process is a copy
    carrying that worker's random state: sampling from it ignores
    ``np.random.seed`` in this process, so a simulation run on freshly
    discovered parameters could not be reproduced. The module-level object
    draws from numpy's global generator, like the ones rebuilt by
    ``from_json``."""
    if isinstance(dist, str):
        return dist
    return getattr(stats, getattr(dist, 'name', ''), dist)


def _with_global_distribution(model):
    """``model`` (a no-rules ``(dist, params, min, max, mean)`` tuple or a
    ``DecisionRules`` with one ``dist`` per leaf) pointing at the global
    scipy distributions."""
    if isinstance(model, tuple) and model:
        return (_global_distribution(model[0]),) + model[1:]
    rules = getattr(model, 'rules', None)
    if isinstance(rules, dict):
        for node in rules.values():
            if isinstance(node, dict) and isinstance(node.get('dist'), tuple) and node['dist']:
                node['dist'] = (_global_distribution(node['dist'][0]),) + node['dist'][1:]
    return model


def plan_execution_time_distributions(
        df_features: pd.DataFrame,
        net_transition_labels: list,
        resources: list,
        calendars: dict,
        max_depths: list = range(1,6),
        min_samples_leaf_cv: list = [50, 100, 200],
        label_data_attributes: list = [],
        label_data_attributes_categorical: list = [],
        values_categorical: dict = dict(),
        random_state: int = 72
    ) -> ModelPlan:
    """:func:`discover_execution_time_distributions` split in two: the training
    data are prepared here, the fits are left to ``run_model_plans``."""

    if not max_depths:
        return _plan_best_distribution_ex(df_features, net_transition_labels, calendars)
    return _plan_models_ex(
        df_features,
        net_transition_labels,
        resources,
        calendars,
        label_data_attributes,
        label_data_attributes_categorical,
        values_categorical,
        max_depths,
        min_samples_leaf_cv,
        random_state=random_state
    )


def plan_waiting_time(
        df_features: pd.DataFrame,
        net_transition_labels: list,
        resources: list,
        calendars: dict,
        label_data_attributes: list,
        label_data_attributes_categorical: list,
        values_categorical: dict,
        max_depths: list = range(1,6),
        min_samples_leaf_cv: list = [50, 100, 200],
        random_state: int = 72,
        use_workload_features: bool = False,
    ) -> ModelPlan:
    """:func:`discover_waiting_time` split in two, like
    :func:`plan_execution_time_distributions`."""

    if not max_depths:
        return _plan_best_distribution_wt(df_features, resources, calendars)
    return _plan_models_wt(
        df_features,
        net_transition_labels,
        resources,
        calendars,
        label_data_attributes,
        label_data_attributes_categorical,
        values_categorical,
        max_depths,
        min_samples_leaf_cv,
        random_state=random_state,
        use_workload_features=use_workload_features,
    )



# BUILD ML MODELS

def _crps_leaf(train_y, test_y, rng, cap=500):
    """Sum of the per-observation CRPS of held-out ``test_y`` under the
    predictive sample given by ``train_y`` (sub-sampled to ``cap`` for the
    O(m^2)/O(m*n) terms), plus the count.

    CRPS(F, y) = E|X - y| - 0.5 E|X - X'|  with X, X' ~ F : the proper-scoring-
    rule generalisation of absolute error to a *predicted distribution*. It is
    minimised when F is both well-placed on y and appropriately spread, so it
    rewards capturing the conditional mean/rate (unlike the per-leaf self-
    Wasserstein, which only measured intra-leaf stability). Lower = better."""
    S = train_y if len(train_y) <= cap else rng.choice(train_y, cap, replace=False)
    half_spread = 0.5 * np.abs(S[:, None] - S[None, :]).mean()
    crps = np.abs(S[:, None] - test_y[None, :]).mean(axis=0) - half_spread
    return float(crps.sum()), len(test_y)


def _crps_cv_score_no_rule(y_arr, random_state, n_splits=5) -> float:
    # Baseline candidate for the CV: no tree at all, just the global empirical
    # distribution as predictive sample. Held-out CRPS, fold-averaged; directly
    # comparable to the tree scorer (same folds). Lower = better.
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rng = np.random.default_rng(random_state)
    fold_scores = []
    for train_idx, test_idx in kf.split(y_arr):
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]
        if len(y_te) == 0 or len(y_tr) == 0:
            continue
        s, n = _crps_leaf(y_tr, y_te, rng)
        if n > 0:
            fold_scores.append(s / n)
    if not fold_scores:
        return float('inf')
    return float(np.mean(fold_scores))


def _fit_leaf_distribution(y_values, use_outlier_removal: bool) -> dict:
    """Fit a single leaf's distribution and return a rule dict with
    ``value``, ``dist``, ``sampled``."""

    y_raw = np.asarray(y_values.values if hasattr(y_values, 'values') else y_values)
    n_sample = max(len(y_raw), DEFAULT_SAMPLE_SIZE)

    y_l = np.asarray(remove_outliers(pd.Series(y_raw))) if use_outlier_removal else y_raw
    if len(y_l) == 0:
        return {'value': 0.0, 'dist': ('fixed', (0,), 0, 0), 'sampled': [0]}
    min_value = float(np.min(y_l))
    max_value = float(np.max(y_l))
    mean_value = float(np.mean(y_l))
    dist, params = return_best_distribution(y_l, dist_search=DIST_SEARCH)
    sampled = list(sampling_from_dist(
        dist, params, min_value, max_value, mean_value, n_sample=n_sample,
    ))
    return {'value': mean_value,
            'dist': (dist, params, min_value, max_value),
            'sampled': sampled}


def _build_no_rule_decision_rules(y, use_outlier_removal: bool = True) -> DecisionRules:
    clf = DecisionRules()
    clf.rules = {0: _fit_leaf_distribution(y, use_outlier_removal)}
    return clf


def _crps_cv_score(X_arr, y_arr, max_depth, min_samples_leaf, random_state, n_splits=5) -> float:
    # Held-out CRPS of the tree's per-leaf predictive distributions: each
    # held-out point is scored against the train values of its own leaf,
    # leaf-size-weighted and fold-averaged. Replaces the per-leaf self-
    # Wasserstein, which measured intra-leaf stability (insensitive to
    # conditional mean/rate shifts and biased against finer splits). Since CRPS
    # is a proper scoring rule it rewards splits that better match the held-out
    # conditional distribution. Lower = better; comparable to the no-rule score.
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rng = np.random.default_rng(random_state)
    fold_scores = []
    for train_idx, test_idx in kf.split(X_arr):
        X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]
        try:
            tree = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=random_state,
            ).fit(X_tr, y_tr)
        except Exception:
            continue
        leaves_tr = tree.apply(X_tr)
        leaves_te = tree.apply(X_te)
        total, weighted = 0, 0.0
        for l in np.unique(leaves_te):
            y_te_l = y_te[leaves_te == l]
            y_tr_l = y_tr[leaves_tr == l]
            if len(y_te_l) == 0 or len(y_tr_l) == 0:
                continue
            s, n = _crps_leaf(y_tr_l, y_te_l, rng)
            weighted += s
            total += n
        if total > 0:
            fold_scores.append(weighted / total)
    if not fold_scores:
        return float('inf')
    return float(np.mean(fold_scores))


def _fit_decision_rules(X, y, param_grid, max_depths, random_state, use_outlier_removal=True) -> DecisionRules:
    if len(X) == 0:
        clf = DecisionRules()
        clf.rules = {0: {'value': 0.0, 'sampled': [0], 'dist': ('fixed', (0,), 0, 0)}}
        return clf

    if isinstance(X, pd.DataFrame):
        X = prune_low_signal_columns(X)

    if hasattr(X, 'shape') and X.shape[1] == 0:
        return _build_no_rule_decision_rules(y, use_outlier_removal=use_outlier_removal)

    if max_depths:
        if len(X) > 6:
            X_arr = X.values if hasattr(X, 'values') else np.asarray(X)
            y_arr = np.asarray(y)
            best_score = _crps_cv_score_no_rule(y_arr, random_state)
            best_params = 'no_rule' if best_score != float('inf') else None
            for md in param_grid['max_depth']:
                for msl in param_grid['min_samples_leaf']:
                    s = _crps_cv_score(X_arr, y_arr, md, msl, random_state)
                    if s < best_score:
                        best_score = s
                        best_params = (md, msl)
            if best_params == 'no_rule':
                return _build_no_rule_decision_rules(y, use_outlier_removal=use_outlier_removal)
            if best_params is None:
                clf_mean = DecisionTreeRegressor(max_depth=1, random_state=random_state).fit(X, y)
            else:
                md, msl = best_params
                clf_mean = DecisionTreeRegressor(
                    max_depth=md, min_samples_leaf=msl, random_state=random_state,
                ).fit(X, y)
        else:
            clf_mean = DecisionTreeRegressor(max_depth=1, random_state=random_state).fit(X, y)
    else:
        clf_mean = DecisionTreeRegressor(max_depth=5, random_state=random_state)
        clf_mean.fit(X, y)

    leaf_indices = clf_mean.apply(X)
    y_leaf = pd.DataFrame({'Leaf': leaf_indices, 'Y': y})

    clf = DecisionRules()
    clf.from_decision_tree(clf_mean)

    for l in y_leaf['Leaf'].unique():
        y_l_raw = y_leaf[y_leaf['Leaf'] == l]['Y'].values
        leaf_fit = _fit_leaf_distribution(y_l_raw, use_outlier_removal)
        clf.rules[l]['dist'] = leaf_fit['dist']
        clf.rules[l]['sampled'] = leaf_fit['sampled']

    return clf


def build_model_arrival(
        log: EventLog,
        calendar_arrival: dict,
        max_depths: list = range(1,6),
        min_samples_leaf_cv: list = [50, 100, 200],
        random_state: int = 72
    ) -> DecisionRules:

    seed_worker_from_key('__arrival__', random_state)
    param_grid = {'max_depth': max_depths, 'min_samples_leaf': min_samples_leaf_cv}
    df = build_training_df_arrival(log, calendar_arrival)
    X = df.drop(columns=['arrival_time'])
    y = df['arrival_time']
    return _fit_decision_rules(X, y, param_grid, max_depths, random_state)



def _fit_one_ex_worker(act, df_act, param_grid, max_depths, random_state):
    seed_worker_from_key(act, random_state)
    X = df_act.drop(columns=['execution_time'])
    y = df_act['execution_time']
    return act, _fit_decision_rules(X, y, param_grid, max_depths, random_state, use_outlier_removal=False)


def _fit_one_wt_worker(res, df_res, param_grid, max_depths, random_state):
    seed_worker_from_key(res, random_state)
    X = df_res.drop(columns=['waiting_time'])
    y = df_res['waiting_time']
    return res, _fit_decision_rules(X, y, param_grid, max_depths, random_state, use_outlier_removal=False)


def build_models_ex(
        df_features: pd.DataFrame,
        activity_labels: list,
        resources: list,
        calendars: dict,
        label_data_attributes: list,
        label_data_attributes_categorical: list,
        values_categorical: dict,
        max_depths: list = range(1,6),
        min_samples_leaf_cv: list = [50, 100, 200],
        random_state: int = 72
    ) -> dict:

    return run_model_plans([_plan_models_ex(
        df_features,
        activity_labels,
        resources,
        calendars,
        label_data_attributes,
        label_data_attributes_categorical,
        values_categorical,
        max_depths,
        min_samples_leaf_cv,
        random_state=random_state
    )])[0]


def _plan_models_ex(
        df_features: pd.DataFrame,
        activity_labels: list,
        resources: list,
        calendars: dict,
        label_data_attributes: list,
        label_data_attributes_categorical: list,
        values_categorical: dict,
        max_depths: list = range(1,6),
        min_samples_leaf_cv: list = [50, 100, 200],
        random_state: int = 72
    ) -> ModelPlan:

    df = build_training_df_ex(
                                df_features,
                                resources,
                                activity_labels,
                                calendars,
                                label_data_attributes,
                                label_data_attributes_categorical,
                                values_categorical
                            )

    param_grid = {'max_depth': max_depths, 'min_samples_leaf': min_samples_leaf_cv}

    # Pre-slice once so each worker only receives its own activity's rows,
    # not the full df.
    df_by_act = {a: grp.iloc[:, 1:] for a, grp in df.groupby('activity_executed', sort=False)}
    empty_slice = df.iloc[0:0, 1:]

    slices = [df_by_act.get(act, empty_slice) for act in activity_labels]
    return ModelPlan(
        jobs=[
            delayed(_fit_one_ex_worker)(act, df_act, param_grid, max_depths, random_state)
            for act, df_act in zip(activity_labels, slices)
        ],
        finish=lambda fitted: {act: _with_global_distribution(fitted[act]) for act in activity_labels},
        desc='exec-time models',
        sizes=[_TIME_JOB_WEIGHT * df_act.shape[0] * df_act.shape[1] for df_act in slices],
    )



def build_models_wt(
        df_features: pd.DataFrame,
        activity_labels: list,
        resources: list,
        calendars: dict,
        label_data_attributes: list,
        label_data_attributes_categorical: list,
        values_categorical: dict,
        max_depths: list = range(1,6),
        min_samples_leaf_cv: list = [50, 100, 200],
        random_state: int = 72,
        use_workload_features: bool = False,
    ) -> dict:

    return run_model_plans([_plan_models_wt(
        df_features,
        activity_labels,
        resources,
        calendars,
        label_data_attributes,
        label_data_attributes_categorical,
        values_categorical,
        max_depths,
        min_samples_leaf_cv,
        random_state=random_state,
        use_workload_features=use_workload_features,
    )])[0]


def _plan_models_wt(
        df_features: pd.DataFrame,
        activity_labels: list,
        resources: list,
        calendars: dict,
        label_data_attributes: list,
        label_data_attributes_categorical: list,
        values_categorical: dict,
        max_depths: list = range(1,6),
        min_samples_leaf_cv: list = [50, 100, 200],
        random_state: int = 72,
        use_workload_features: bool = False,
    ) -> ModelPlan:

    df = build_training_df_wt(
                                df_features,
                                activity_labels,
                                calendars,
                                label_data_attributes,
                                label_data_attributes_categorical,
                                values_categorical,
                                use_workload_features=use_workload_features
                            )

    param_grid = {'max_depth': max_depths, 'min_samples_leaf': min_samples_leaf_cv}

    df_by_res = {r: grp.iloc[:, 1:] for r, grp in df.groupby('resource', sort=False)}
    empty_slice = df.iloc[0:0, 1:]

    slices = [df_by_res.get(res, empty_slice) for res in resources]
    return ModelPlan(
        jobs=[
            delayed(_fit_one_wt_worker)(res, df_res, param_grid, max_depths, random_state)
            for res, df_res in zip(resources, slices)
        ],
        finish=lambda fitted: {res: _with_global_distribution(fitted[res]) for res in resources},
        desc='waiting-time models',
        sizes=[_TIME_JOB_WEIGHT * df_res.shape[0] * df_res.shape[1] for df_res in slices],
    )



# BUILD TRAINING DATASETS

def build_training_df_arrival(
        log: EventLog,
        calendar_arrival: dict
    ) -> pd.DataFrame:

    df_log = pm4py.convert_to_dataframe(log)
    first_ts = df_log.groupby('case:concept:name')["start:timestamp"].min()
    ordered_first_ts = first_ts.sort_values()
    previous_arrivals = ordered_first_ts.tolist()[:-1]

    arrival_working_set = calendar_to_working_set(calendar_arrival)
    dict_df = {
        'hour': [prev_ts.hour for prev_ts in previous_arrivals],
        'weekday': [prev_ts.weekday() for prev_ts in previous_arrivals],
        'arrival_time': _arrival_gaps(ordered_first_ts, calendar_arrival, arrival_working_set),
    }

    df = pd.DataFrame(dict_df)

    return df



def _exec_time_chunk(chunk, calendars, working_sets):
    return chunk.apply(
        lambda x: count_working_minutes(x["start_t"], x["end_t"], calendars[x["resource"]], working_sets[x["resource"]]),
        axis=1,
    )


def _waiting_time_chunk(chunk, calendars, working_sets):
    return chunk.apply(
        lambda x: count_working_minutes(x["resource_free_t"], x["start_t"], calendars[x["resource"]], working_sets[x["resource"]]),
        axis=1,
    )


_HOUR_US = 3600 * 10**6
_WEEK_US = 7 * 24 * _HOUR_US
_MINUTE_US = 60 * 10**6
_MONDAY_US = 4 * 24 * _HOUR_US          # 1970-01-05 00:00, the first Monday after the epoch


def _wall_microseconds(values):
    """Wall-clock microseconds since the epoch of a datetime64 Series, plus a
    mask of the values that carry sub-microsecond digits (which the integer
    arithmetic below cannot follow). ``None`` when the Series is not a
    NaT-free datetime64 in UTC, a fixed offset or no timezone at all: only
    then is elapsed time the same thing as elapsed wall-clock time."""
    if not pd.api.types.is_datetime64_any_dtype(values.dtype) or values.isna().any():
        return None
    tz = values.dt.tz
    if tz is not None:
        if not (isinstance(tz, timezone) or str(tz) == 'UTC'):
            return None
        values = values.dt.tz_localize(None)
    unit = getattr(values.dt, 'unit', 'ns')
    ints = values.astype('int64').to_numpy()
    if unit == 'ns':
        return ints // 1000, (ints % 1000) != 0
    scale = {'us': 1, 'ms': 10**3, 's': 10**6}.get(unit)
    if scale is None:
        return None
    return ints * scale, np.zeros(len(ints), dtype=bool)


def _working_microseconds_since_epoch(wall_us, working_slots):
    """Working time elapsed between the reference Monday and each instant,
    for a weekly calendar given as 168 booleans (weekday * 24 + hour)."""
    rel = wall_us - _MONDAY_US
    weeks = rel // _WEEK_US
    within = rel - weeks * _WEEK_US
    slot = within // _HOUR_US
    before_slot = np.concatenate(([0], np.cumsum(working_slots, dtype=np.int64))) * _HOUR_US
    in_slot = np.where(working_slots[slot], within - slot * _HOUR_US, 0)
    return weeks * before_slot[-1] + before_slot[slot] + in_slot


def _working_minutes(start, end, resource, calendars, working_sets):
    """``count_working_minutes(start, end, calendars[r], working_sets[r])`` for
    every row, as an int64 array.

    Working time is accumulated in whole microseconds from a 168-slot table
    per calendar and rounded to minutes half-to-even, which is what the
    scalar function returns whenever its float sum is not exactly on a half
    minute. Rows that do land on a half minute — where the scalar result
    depends on the order of its additions — and rows with sub-microsecond
    timestamps are handed to the scalar function itself.

    Returns ``None`` when the columns cannot be treated this way (see
    ``_wall_microseconds``) or a resource has no calendar; the caller then
    uses the scalar path for everything."""
    n = len(start)
    if n == 0:
        return None
    if not (pd.api.types.is_datetime64_any_dtype(start.dtype) and pd.api.types.is_datetime64_any_dtype(end.dtype)):
        return None
    if start.dt.tz != end.dt.tz or getattr(start.dt, 'unit', 'ns') != getattr(end.dt, 'unit', 'ns'):
        return None
    start_us = _wall_microseconds(start)
    end_us = _wall_microseconds(end)
    if start_us is None or end_us is None:
        return None
    start_us, start_inexact = start_us
    end_us, end_inexact = end_us
    needs_scalar = start_inexact | end_inexact
    out = np.zeros(n, dtype=np.int64)
    # Same tz and same unit: the raw integers order like the instants.
    positive = end.astype('int64').to_numpy() > start.astype('int64').to_numpy()

    resources = resource.to_numpy()
    try:
        groups = pd.Series(np.arange(n)).groupby(resources, sort=False, dropna=False).indices
    except TypeError:
        return None
    for res, pos in groups.items():
        if res not in working_sets or res not in calendars:
            return None
        working_slots = np.zeros(168, dtype=bool)
        for weekday, hour in working_sets[res]:
            if isinstance(weekday, (int, np.integer)) and isinstance(hour, (int, np.integer)) and 0 <= weekday < 7 and 0 <= hour < 24:
                working_slots[weekday * 24 + hour] = True
        worked = (_working_microseconds_since_epoch(end_us[pos], working_slots)
                  - _working_microseconds_since_epoch(start_us[pos], working_slots))
        minutes, rest = np.divmod(worked, _MINUTE_US)
        is_positive = positive[pos]
        out[pos] = np.where(is_positive, minutes + (2 * rest > _MINUTE_US), 0)
        needs_scalar[pos] |= is_positive & (2 * rest == _MINUTE_US)
    needs_scalar &= positive

    scalar_rows = np.flatnonzero(needs_scalar)
    if len(scalar_rows):
        start_values = start.tolist()
        end_values = end.tolist()
        for i in scalar_rows.tolist():
            res = resources[i]
            out[i] = count_working_minutes(start_values[i], end_values[i], calendars[res], working_sets[res])
    return out


def _hour_and_weekday(values):
    """Wall-clock hour and weekday of a timestamp column, as int64 Series."""
    if len(values) and pd.api.types.is_datetime64_any_dtype(values.dtype) and not values.isna().any():
        return values.dt.hour.astype('int64'), values.dt.weekday.astype('int64')
    return values.apply(lambda ts: ts.hour), values.apply(lambda ts: ts.weekday())


_ARRIVAL_KEY = '__arrival__'


def _arrival_gaps(ordered_first_ts, calendar_arrival, arrival_working_set) -> list:
    """Working minutes between consecutive case arrivals (sorted Series)."""
    n_gaps = len(ordered_first_ts) - 1
    if n_gaps >= 1:
        minutes = _working_minutes(
            ordered_first_ts.iloc[:-1].reset_index(drop=True),
            ordered_first_ts.iloc[1:].reset_index(drop=True),
            pd.Series([_ARRIVAL_KEY] * n_gaps),
            {_ARRIVAL_KEY: calendar_arrival},
            {_ARRIVAL_KEY: arrival_working_set},
        )
        if minutes is not None:
            return minutes.tolist()
    values = ordered_first_ts.tolist()
    return [
        count_working_minutes(values[i - 1], values[i], calendar_arrival, arrival_working_set)
        for i in range(1, len(values))
    ]


def _working_minutes_column(df, start_col, end_col, calendars, working_sets, chunk_fn):
    """Working minutes between two timestamp columns of ``df`` (which also
    holds ``resource``), index-aligned with ``df``."""
    minutes = _working_minutes(df[start_col], df[end_col], df["resource"], calendars, working_sets)
    if minutes is None:
        return _parallel_working_minutes(df, calendars, working_sets, chunk_fn)
    return pd.Series(minutes, index=df.index)


def _parallel_working_minutes(df, calendars, working_sets, chunk_fn):
    # count_working_minutes iterates hour-by-hour in pure Python, so a
    # 100k-row df.apply can take minutes. Split into n_chunks so each worker
    # does ~O(rows/n_chunks) Python iterations. Only ship the three columns
    # the lambda actually needs to keep per-worker pickle cost bounded.
    import os
    n_chunks = max(1, (os.cpu_count() or 2) - 1)
    if len(df) == 0 or n_chunks == 1:
        return chunk_fn(df, calendars, working_sets)
    chunks = np.array_split(df, n_chunks)
    results = Parallel(n_jobs=DEFAULT_N_JOBS)(
        delayed(chunk_fn)(c, calendars, working_sets) for c in chunks
    )
    return pd.concat(results)


def build_training_df_ex(
        df_features: pd.DataFrame,
        resources: list,
        activity_labels: list,
        calendars: dict,
        label_data_attributes: list,
        label_data_attributes_categorical: list,
        values_categorical: dict
    ) -> pd.DataFrame:

    df_et = df_features[["transition_label", "resource", "start_t", "end_t"] + label_data_attributes + list(activity_labels)]
    df_et = df_et[~df_et["start_t"].isna()]
    working_sets_ex = {r: calendar_to_working_set(calendars[r]) for r in calendars}
    df_et["execution_time"] = _working_minutes_column(
        df_et[["start_t", "end_t", "resource"]], "start_t", "end_t", calendars, working_sets_ex, _exec_time_chunk,
    )
    df_et["hour"], df_et["weekday"] = _hour_and_weekday(df_et["start_t"])
    df_et.drop(columns=["start_t", "end_t"], inplace=True)
    df_et.rename(columns={"transition_label": "activity_executed"}, inplace=True)
    df_et.reset_index(drop=True, inplace=True)

    for r in resources:
        df_et['resource = '+r] = (df_et['resource'] == r).astype(int)
    del df_et['resource']

    for a in label_data_attributes_categorical:
        for v in values_categorical[a]:
            df_et[a+' = '+str(v)] = (df_et[a] == v).astype(int)
        del df_et[a]

    return df_et



def build_training_df_wt(
        df_features: pd.DataFrame,
        net_transition_labels: list,
        calendars: dict,
        label_data_attributes: list,
        label_data_attributes_categorical: list,
        values_categorical: dict,
        use_workload_features: bool = False
    ) -> pd.DataFrame:

    base_cols = ["resource", "transition_label", "start_t", "resource_free_t"] + list(label_data_attributes) + list(net_transition_labels)
    if use_workload_features:
        base_cols = base_cols + ["res_workload", "queue_length"]
    df_wt = df_features[base_cols]
    df_wt = df_wt[~df_wt["start_t"].isna()]
    working_sets_wt = {r: calendar_to_working_set(calendars[r]) for r in calendars}
    df_wt["waiting_time"] = _working_minutes_column(
        df_wt[["start_t", "resource_free_t", "resource"]], "resource_free_t", "start_t", calendars, working_sets_wt, _waiting_time_chunk,
    )
    df_wt["hour"], df_wt["weekday"] = _hour_and_weekday(df_wt["resource_free_t"])
    df_wt.drop(columns=["start_t", "resource_free_t"], inplace=True)
    if use_workload_features:
        df_wt.rename(columns={"res_workload": "workload"}, inplace=True)
    df_wt.reset_index(drop=True, inplace=True)

    for act in net_transition_labels:
        df_wt['waiting_activity = ' + act] = (df_wt['transition_label'] == act).astype(int)
    df_wt.drop(columns=["transition_label"], inplace=True)

    for a in label_data_attributes_categorical:
        for v in values_categorical[a]:
            df_wt[a + ' = ' + str(v)] = (df_wt[a] == v).astype(int)
        del df_wt[a]

    return df_wt



# NO RULES MODE

def find_best_distribution_arrival(log: EventLog,
        calendar_arrival: dict
    ) -> tuple:

    df_log = pm4py.convert_to_dataframe(log)
    first_ts = df_log.groupby('case:concept:name')["start:timestamp"].min()
    arrival_working_set = calendar_to_working_set(calendar_arrival)
    arrival_times = _arrival_gaps(first_ts.sort_values(), calendar_arrival, arrival_working_set)

    dist, params = return_best_distribution(arrival_times, dist_search=DIST_SEARCH)
    min_value = np.min(arrival_times)
    max_value = np.max(arrival_times)

    return dist, params, min_value, max_value, np.mean(arrival_times)


def _fit_best_distribution_worker(key, times, use_outlier_removal: bool = True):
    if use_outlier_removal:
        times = remove_outliers(times)
    if len(times) == 0:
        return key, ('fixed', (0,), 0, 0, 0)

    dist, params = return_best_distribution(times, dist_search=DIST_SEARCH)
    return key, (dist, params, np.min(times), np.max(times), np.mean(times))


def find_best_distribution_ex(df_features: pd.DataFrame,
        activity_labels: list,
        calendars: dict
    ) -> dict:

    return run_model_plans([_plan_best_distribution_ex(df_features, activity_labels, calendars)])[0]


def _plan_best_distribution_ex(df_features: pd.DataFrame,
        activity_labels: list,
        calendars: dict
    ) -> ModelPlan:

    df_et = df_features[["transition_label", "resource", "start_t", "end_t"]]
    df_et = df_et[~df_et["start_t"].isna()]
    working_sets_ex2 = {r: calendar_to_working_set(calendars[r]) for r in calendars}
    df_et["execution_time"] = _working_minutes_column(
        df_et[["start_t", "end_t", "resource"]], "start_t", "end_t", calendars, working_sets_ex2, _exec_time_chunk,
    )

    times_by_act = {
        act: grp['execution_time'].dropna().tolist()
        for act, grp in df_et.groupby('transition_label', sort=False)
    }
    times = [times_by_act.get(act, []) for act in activity_labels]
    return ModelPlan(
        jobs=[
            delayed(_fit_best_distribution_worker)(act, act_times, use_outlier_removal=False)
            for act, act_times in zip(activity_labels, times)
        ],
        finish=lambda fitted: {act: _with_global_distribution(fitted[act]) for act in activity_labels},
        desc='exec-time distributions',
        sizes=[len(act_times) for act_times in times],
    )


def find_best_distribution_wt(df_features: pd.DataFrame,
        resources: list,
        calendars: dict,
    ) -> dict:
    """Discover one global waiting-time distribution per resource (no-rules
    mode). Returns ``(dist, params, min, max, mean)`` per resource."""

    return run_model_plans([_plan_best_distribution_wt(df_features, resources, calendars)])[0]


def _plan_best_distribution_wt(df_features: pd.DataFrame,
        resources: list,
        calendars: dict,
    ) -> ModelPlan:

    df_wt = df_features[["resource", "start_t", "resource_free_t"]]
    df_wt = df_wt[~df_wt["start_t"].isna()]
    working_sets_wt2 = {r: calendar_to_working_set(calendars[r]) for r in calendars}
    df_wt["waiting_time"] = _working_minutes_column(
        df_wt[["start_t", "resource_free_t", "resource"]], "resource_free_t", "start_t", calendars, working_sets_wt2, _waiting_time_chunk,
    )
    df_wt.reset_index(drop=True, inplace=True)

    times_by_res = {
        res: grp['waiting_time'].dropna().tolist()
        for res, grp in df_wt.groupby('resource', sort=False)
    }
    times = [times_by_res.get(res, []) for res in resources]
    return ModelPlan(
        jobs=[
            delayed(_fit_best_distribution_worker)(res, res_times, use_outlier_removal=False)
            for res, res_times in zip(resources, times)
        ],
        finish=lambda fitted: {res: _with_global_distribution(fitted[res]) for res in resources},
        desc='waiting-time distributions',
        sizes=[len(res_times) for res_times in times],
    )
