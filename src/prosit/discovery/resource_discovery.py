import pm4py
import pandas as pd
from pm4py.objects.log.obj import EventLog

from tqdm import tqdm

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from prosit.utils.rule_utils import DecisionRules, apply_laplace_smoothing, BINARY_NEG_LOG_LOSS_SCORER


def discover_resources_list(log: EventLog) -> list:

    resource_counts = pd.Series(pm4py.get_event_attribute_values(log, 'org:resource'))
    resource_counts = resource_counts.sort_values(ascending=False)
    return resource_counts.index.tolist()


def discover_resources_per_act(log: EventLog, activities: list, resources: list) -> dict:

    df_log = pm4py.convert_to_dataframe(log)
    df_log = df_log[df_log["concept:name"].isin(activities)]
    df_log = df_log[df_log["org:resource"].isin(resources)]

    R_act = dict()
    for act in activities:
        df_log_act = df_log[df_log["concept:name"] == act]
        res_counts_act = df_log_act["org:resource"].value_counts()
        R_act[act] = res_counts_act.index.tolist()

    return R_act


def return_max_concurrency(df_features: pd.DataFrame, thr: float = 0.05) -> dict:
    """
    Returns a dict {resource: max_concurrent_tasks}.
    Resources where more than `thr` fraction of events have concurrent workload > 0
    are treated as multitasking; their capacity is set to the maximum observed
    concurrency + 1 (the resource itself). Non-multitasking resources get capacity 1.
    """
    max_concurrency = {}
    for resource, group in df_features[~df_features['resource'].isna()].groupby('resource'):
        total = len(group)
        positive = (group['res_workload'] > 0).sum()
        if total > 0 and (positive / total) >= thr:
            max_concurrency[resource] = int(group['res_workload'].max()) + 1
        else:
            max_concurrency[resource] = 1
    return max_concurrency


def discover_weight_resources(
        df_features: pd.DataFrame,
        act_to_resources: dict,
        resources: list,
        max_concurrency: dict,
        max_depths_cv: list = range(1,6),
        min_samples_leaf_cv: list = [5, 10, 20],
        label_data_attributes: list = [],
        label_data_attributes_categorical: list = [],
        values_categorical: dict = dict(),
        random_state: int = 72
    ) -> dict :
    """Return ``{resource: classifier}``.

    One binary classifier per resource, trained on events where the resource
    was a candidate (i.e. in the activity's role pool) AND was actually free
    at the enabling time (workload below its max concurrency). Features are
    per-resource history counts, case attributes, and the resource's own
    workload at enabling time. This matches the simulator's inference path:
    the classifier is invoked only on free candidates, so training must
    mirror that conditional distribution.
    """

    df_features = df_features[~df_features["resource"].isna()]

    if not max_depths_cv:
        weights = {}
        r_to_acts = _resource_to_eligible_acts(act_to_resources, resources)
        for r in resources:
            eligible = r_to_acts.get(r, set())
            if not eligible:
                weights[r] = 0.0
                continue
            scope = df_features[df_features['transition_label'].isin(eligible)]
            if scope.empty:
                weights[r] = 0.0
                continue
            free_mask = scope['candidate_workloads'].apply(
                lambda d: isinstance(d, dict) and d.get(r, 0) < max_concurrency.get(r, 1)
            )
            scope = scope[free_mask]
            if scope.empty:
                weights[r] = 0.0
                continue
            weights[r] = float((scope['resource'] == r).sum()) / len(scope)
        return weights

    return build_models(
        df_features,
        act_to_resources,
        resources,
        max_concurrency,
        max_depths_cv,
        min_samples_leaf_cv,
        label_data_attributes,
        label_data_attributes_categorical,
        values_categorical,
        random_state=random_state
    )


def build_models(
        df_features: pd.DataFrame,
        act_to_resources: dict,
        resources: list,
        max_concurrency: dict,
        max_depths_cv: list = range(1,6),
        min_samples_leaf_cv: list = [5, 10, 20],
        label_data_attributes: list = [],
        label_data_attributes_categorical: list = [],
        values_categorical: dict = dict(),
        random_state: int = 72
    ) -> dict :

    param_grid = {'max_depth': max_depths_cv, 'min_samples_leaf': min_samples_leaf_cv, 'max_features': [None, 'sqrt']}

    datasets = build_training_datasets(
        df_features,
        act_to_resources,
        resources,
        max_concurrency,
        label_data_attributes,
    )

    models = dict()

    for r, data_r in tqdm(datasets.items()):
        if len(data_r['class'].unique()) < 2:
            # Constant label — store scalar probability (0 or 1).
            models[r] = float(data_r['class'].iloc[0]) if len(data_r) else 0.0
            continue

        for a in label_data_attributes_categorical:
            for v in values_categorical[a]:
                data_r[a + ' = ' + str(v)] = (data_r[a] == v).astype(int)
            del data_r[a]

        X = data_r.drop(columns=['class'])
        y = data_r['class']

        if max_depths_cv:
            clf_dtc = DecisionTreeClassifier(random_state=random_state)
            try:
                grid_search = GridSearchCV(
                    estimator=clf_dtc,
                    param_grid=param_grid,
                    cv=3,
                    scoring=BINARY_NEG_LOG_LOSS_SCORER,
                ).fit(X, y)
                clf_dtc = grid_search.best_estimator_
            except Exception:
                clf_dtc = DecisionTreeClassifier(max_depth=2, random_state=random_state)
                clf_dtc.fit(X, y)
        else:
            clf_dtc = DecisionTreeClassifier(random_state=random_state, max_depth=1)
            clf_dtc.fit(X, y)

        apply_laplace_smoothing(clf_dtc, alpha=1.0)

        clf = DecisionRules()
        clf.from_decision_tree(clf_dtc)

        if clf is None:
            clf = float(y.mode().iloc[0])

        models[r] = clf

    return models


def _resource_to_eligible_acts(act_to_resources: dict, resources: list) -> dict:
    """Invert ``act_to_resources`` to ``{resource: set(activities)}``."""
    r_to_acts = {r: set() for r in resources}
    for a, rs in act_to_resources.items():
        for r in rs:
            if r in r_to_acts:
                r_to_acts[r].add(a)
    return r_to_acts


def build_training_datasets(
        df_features: pd.DataFrame,
        act_to_resources: dict,
        resources: list,
        max_concurrency: dict,
        label_data_attributes: list,
    ) -> dict:
    """Return ``{resource: training_df}``.

    One training set per resource. Scope: events whose activity has the
    resource in its candidate pool AND where the resource was actually free
    (``workload < max_concurrency[r]``) at the enabling time. Features:
    per-resource history counts, case attributes, and the resource's own
    workload at enabling time. Target ``class`` is ``1`` when this resource
    actually performed the event, ``0`` otherwise.
    """

    res_history_cols = list(resources)
    feature_cols = res_history_cols + list(label_data_attributes)

    r_to_acts = _resource_to_eligible_acts(act_to_resources, resources)

    datasets = {}
    for r in resources:
        eligible_acts = r_to_acts.get(r, set())
        if not eligible_acts:
            continue
        df_scope = df_features[df_features['transition_label'].isin(eligible_acts)]
        if df_scope.empty:
            continue
        max_c = max_concurrency.get(r, 1)
        workload_r = df_scope['candidate_workloads'].apply(
            lambda d: d.get(r, 0) if isinstance(d, dict) else 0
        )
        free_mask = workload_r < max_c
        df_scope = df_scope[free_mask]
        workload_r = workload_r[free_mask]
        if df_scope.empty:
            continue
        queue_r = df_scope['candidate_queue_lengths'].apply(
            lambda d: d.get(r, 0) if isinstance(d, dict) else 0
        )
        base = df_scope[feature_cols].reset_index(drop=True)
        actual = df_scope['resource'].reset_index(drop=True)
        df_r = base.copy()
        df_r['workload'] = workload_r.reset_index(drop=True).values
        df_r['queue_length'] = queue_r.reset_index(drop=True).values
        df_r['class'] = (actual == r).astype(int).values
        datasets[r] = df_r

    return datasets
