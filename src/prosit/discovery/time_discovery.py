import numpy as np
from joblib import Parallel, delayed
import pandas as pd
import pm4py

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from scipy.stats import wasserstein_distance

from pm4py.objects.log.obj import EventLog

from prosit.utils.common_utils import (
    count_working_minutes,
    calendar_to_working_set,
    parallel_with_progress,
    seed_worker_from_key,
    DEFAULT_N_JOBS,
)
from prosit.utils.distribution_utils import return_best_distribution, sampling_from_dist, remove_outliers
from prosit.utils.rule_utils import DecisionRules, prune_low_signal_columns


DIST_SEARCH = ['fixed', 'norm', 'expon', 'lognorm', 'gamma', 'uniform']
DEFAULT_SAMPLE_SIZE = 1000


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



# BUILD ML MODELS

def _wasserstein_cv_score_no_rule(y_arr, random_state, n_splits=5) -> float:
    # Baseline candidate for the CV: no tree at all, just the global empirical
    # distribution. Score = average per-fold Wasserstein(y_te, y_tr), aligned
    # with the tree scorer so the two are directly comparable.
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores = []
    for train_idx, test_idx in kf.split(y_arr):
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]
        if len(y_te) == 0 or len(y_tr) == 0:
            continue
        try:
            w = wasserstein_distance(y_te, y_tr)
        except Exception:
            continue
        fold_scores.append(w)
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


def _wasserstein_cv_score(X_arr, y_arr, max_depth, min_samples_leaf, random_state, n_splits=5) -> float:
    # Per-leaf empirical Wasserstein averaged across folds, weighted by
    # held-out leaf size. Aligned with the ctd metric (Wasserstein on cycle
    # time) rather than R², and robust to the heavy-tailed y typical of
    # waiting/arrival time.
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
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
            try:
                w = wasserstein_distance(y_te_l, y_tr_l)
            except Exception:
                continue
            weighted += w * len(y_te_l)
            total += len(y_te_l)
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
            best_score = _wasserstein_cv_score_no_rule(y_arr, random_state)
            best_params = 'no_rule' if best_score != float('inf') else None
            for md in param_grid['max_depth']:
                for msl in param_grid['min_samples_leaf']:
                    s = _wasserstein_cv_score(X_arr, y_arr, md, msl, random_state)
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
    return act, _fit_decision_rules(X, y, param_grid, max_depths, random_state)


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

    jobs = (
        delayed(_fit_one_ex_worker)(
            act, df_by_act.get(act, empty_slice),
            param_grid, max_depths, random_state,
        )
        for act in activity_labels
    )
    results = parallel_with_progress(jobs, total=len(activity_labels), desc='exec-time models')
    return dict(results)



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

    jobs = (
        delayed(_fit_one_wt_worker)(
            res, df_by_res.get(res, empty_slice),
            param_grid, max_depths, random_state,
        )
        for res in resources
    )
    results = parallel_with_progress(jobs, total=len(resources), desc='waiting-time models')
    return dict(results)



# BUILD TRAINING DATASETS

def build_training_df_arrival(
        log: EventLog,
        calendar_arrival: dict
    ) -> pd.DataFrame:

    df_log = pm4py.convert_to_dataframe(log)
    first_ts = df_log.groupby('case:concept:name')["start:timestamp"].min()
    ordered_first_ts_list = first_ts.sort_values().tolist()

    dict_df = {'hour': [], 'weekday': [], 'arrival_time': []}
    arrival_working_set = calendar_to_working_set(calendar_arrival)

    for i in range(1, len(ordered_first_ts_list)):
        prev_ts = ordered_first_ts_list[i-1]
        dict_df['hour'].append(prev_ts.hour)
        dict_df['weekday'].append(prev_ts.weekday())
        dict_df['arrival_time'].append(count_working_minutes(prev_ts, ordered_first_ts_list[i], calendar_arrival, arrival_working_set))

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
    df_et["execution_time"] = _parallel_working_minutes(
        df_et[["start_t", "end_t", "resource"]], calendars, working_sets_ex, _exec_time_chunk,
    )
    df_et["hour"] = df_et["start_t"].apply(lambda ts: ts.hour)
    df_et["weekday"] = df_et["start_t"].apply(lambda ts: ts.weekday())
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
    df_wt["waiting_time"] = _parallel_working_minutes(
        df_wt[["start_t", "resource_free_t", "resource"]], calendars, working_sets_wt, _waiting_time_chunk,
    )
    df_wt["hour"] = df_wt["resource_free_t"].apply(lambda ts: ts.hour)
    df_wt["weekday"] = df_wt["resource_free_t"].apply(lambda ts: ts.weekday())
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
    ordered_first_ts_list = first_ts.sort_values().tolist()

    arrival_times = []
    arrival_working_set = calendar_to_working_set(calendar_arrival)
    for i in range(1, len(ordered_first_ts_list)):
        arrival_times.append(count_working_minutes(ordered_first_ts_list[i-1], ordered_first_ts_list[i], calendar_arrival, arrival_working_set))

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

    df_et = df_features[["transition_label", "resource", "start_t", "end_t"]]
    df_et = df_et[~df_et["start_t"].isna()]
    working_sets_ex2 = {r: calendar_to_working_set(calendars[r]) for r in calendars}
    df_et["execution_time"] = _parallel_working_minutes(
        df_et[["start_t", "end_t", "resource"]], calendars, working_sets_ex2, _exec_time_chunk,
    )

    times_by_act = {
        act: grp['execution_time'].dropna().tolist()
        for act, grp in df_et.groupby('transition_label', sort=False)
    }
    jobs = (
        delayed(_fit_best_distribution_worker)(act, times_by_act.get(act, []))
        for act in activity_labels
    )
    results = parallel_with_progress(jobs, total=len(activity_labels), desc='exec-time distributions')
    return dict(results)


def find_best_distribution_wt(df_features: pd.DataFrame,
        resources: list,
        calendars: dict,
    ) -> dict:
    """Discover one global waiting-time distribution per resource (no-rules
    mode). Returns ``(dist, params, min, max, mean)`` per resource."""

    df_wt = df_features[["resource", "start_t", "resource_free_t"]]
    df_wt = df_wt[~df_wt["start_t"].isna()]
    working_sets_wt2 = {r: calendar_to_working_set(calendars[r]) for r in calendars}
    df_wt["waiting_time"] = _parallel_working_minutes(
        df_wt[["start_t", "resource_free_t", "resource"]], calendars, working_sets_wt2, _waiting_time_chunk,
    )
    df_wt.reset_index(drop=True, inplace=True)

    times_by_res = {
        res: grp['waiting_time'].dropna().tolist()
        for res, grp in df_wt.groupby('resource', sort=False)
    }
    jobs = (
        delayed(_fit_best_distribution_worker)(res, times_by_res.get(res, []), use_outlier_removal=False)
        for res in resources
    )
    results = parallel_with_progress(jobs, total=len(resources), desc='waiting-time distributions')
    return dict(results)
