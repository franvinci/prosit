import warnings
import numpy as np
import pm4py
import pandas as pd
from joblib import delayed
from pm4py.objects.log.obj import EventLog
from tqdm import tqdm

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import FitFailedWarning
from sklearn.pipeline import Pipeline

from prosit.utils.common_utils import ModelPlan, run_model_plans, dense_take, dense_matrix
from prosit.utils.rule_utils import (
    DecisionRules,
    BINARY_NEG_LOG_LOSS_SCORER,
    prune_low_signal_columns,
    fit_classifier_task,
)


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
        label_data_attributes: list = [],
        label_data_attributes_categorical: list = [],
        values_categorical: dict = dict(),
        random_state: int = 72,
        use_workload_features: bool = False,
        _dense: dict = None
    ) -> dict:
    """Return ``{resource: classifier}``. See :func:`plan_weight_resources`."""

    return run_model_plans([plan_weight_resources(
        df_features,
        act_to_resources,
        resources,
        max_concurrency,
        max_depths_cv,
        label_data_attributes,
        label_data_attributes_categorical,
        values_categorical,
        random_state=random_state,
        use_workload_features=use_workload_features,
        _dense=_dense,
    )])[0]


def plan_weight_resources(
        df_features: pd.DataFrame,
        act_to_resources: dict,
        resources: list,
        max_concurrency: dict,
        max_depths_cv: list = range(1,6),
        label_data_attributes: list = [],
        label_data_attributes_categorical: list = [],
        values_categorical: dict = dict(),
        random_state: int = 72,
        use_workload_features: bool = False,
        _dense: dict = None
    ) -> ModelPlan:
    """Prepare the ``{resource: classifier}`` fits, left to
    :func:`prosit.utils.common_utils.run_model_plans` to run.

    One binary classifier per resource, trained on events where the resource
    was a candidate (i.e. in the activity's role pool) AND was actually free
    at the enabling time (workload below its max concurrency). Features are
    per-resource history counts (one column per resource), case attributes,
    and — when ``use_workload_features`` is True — the resource's own
    workload and queue length at enabling time. This matches the simulator's
    inference path: the classifier is invoked only on free candidates, so
    training must mirror that conditional distribution.

    ``_dense`` is the optional companion block of
    :func:`prosit.utils.common_utils._build_features`: the per-candidate
    workloads / queue lengths as matrices, read instead of the dict-valued
    ``candidate_*`` columns when it matches the rows of ``df_features``.
    """

    has_resource = ~df_features["resource"].isna()
    _dense = dense_take(_dense, has_resource.to_numpy())
    df_features = df_features[has_resource]

    if not max_depths_cv:
        weights = {}
        r_to_acts = _resource_to_eligible_acts(act_to_resources, resources)
        workloads = dense_matrix(_dense, len(df_features), resources, 'workloads')
        if workloads is not None:
            actual = df_features['resource'].to_numpy()
            for j, r in enumerate(resources):
                eligible = r_to_acts.get(r, set())
                in_scope = df_features['transition_label'].isin(eligible).to_numpy() if eligible else None
                if in_scope is None or not in_scope.any():
                    weights[r] = 0.0
                    continue
                free = in_scope & (workloads[:, j] < max_concurrency.get(r, 1))
                n_free = int(free.sum())
                if n_free == 0:
                    weights[r] = 0.0
                    continue
                weights[r] = float((actual[free] == r).sum()) / n_free
            return ModelPlan.done(weights)
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
        return ModelPlan.done(weights)

    return _plan_models(
        df_features,
        act_to_resources,
        resources,
        max_concurrency,
        max_depths_cv,
        label_data_attributes,
        label_data_attributes_categorical,
        values_categorical,
        random_state=random_state,
        use_workload_features=use_workload_features,
        _dense=_dense
    )


def build_models(
        df_features: pd.DataFrame,
        act_to_resources: dict,
        resources: list,
        max_concurrency: dict,
        max_depths_cv: list = range(1,6),
        label_data_attributes: list = [],
        label_data_attributes_categorical: list = [],
        values_categorical: dict = dict(),
        random_state: int = 72,
        use_workload_features: bool = False,
        _dense: dict = None
    ) -> dict:
    """Fit one classifier per resource.

    The training matrices are the ones described in
    :func:`build_training_datasets`, but they are never materialised together:
    a single numeric matrix is shared by all resources and each worker slices
    out its own rows. Model selection is the cross-validation of
    :func:`prosit.utils.rule_utils._grid_search_cv_equivalent`; the fits run in
    parallel and do not depend on the worker that executes them.

    Inputs the dense layout cannot represent (non-numeric feature columns,
    clashing column names) go through :func:`_build_models_legacy`."""

    return run_model_plans([_plan_models(
        df_features,
        act_to_resources,
        resources,
        max_concurrency,
        max_depths_cv,
        label_data_attributes,
        label_data_attributes_categorical,
        values_categorical,
        random_state=random_state,
        use_workload_features=use_workload_features,
        _dense=_dense,
    )])[0]


def _plan_models(
        df_features: pd.DataFrame,
        act_to_resources: dict,
        resources: list,
        max_concurrency: dict,
        max_depths_cv: list = range(1,6),
        label_data_attributes: list = [],
        label_data_attributes_categorical: list = [],
        values_categorical: dict = dict(),
        random_state: int = 72,
        use_workload_features: bool = False,
        _dense: dict = None
    ) -> ModelPlan:

    plan = _dense_training_plan(
        df_features,
        act_to_resources,
        resources,
        max_concurrency,
        label_data_attributes,
        label_data_attributes_categorical,
        values_categorical,
        use_workload_features,
        _dense,
    )
    if plan is None:
        return ModelPlan.done(_build_models_legacy(
            df_features,
            act_to_resources,
            resources,
            max_concurrency,
            max_depths_cv,
            label_data_attributes,
            label_data_attributes_categorical,
            values_categorical,
            random_state=random_state,
            use_workload_features=use_workload_features,
        ))

    base, names, extra_at, order, constant, tasks = plan
    depths = list(max_depths_cv) if max_depths_cv else []
    return ModelPlan(
        jobs=[
            delayed(fit_classifier_task)(
                r, base, rows, y, names, depths, random_state,
                extra=extra, extra_at=extra_at, marginal_guards=True,
            )
            for r, rows, y, extra in tasks
        ],
        finish=lambda fitted: {r: (constant[r] if r in constant else fitted[r]) for r in order},
        desc='resource models',
        sizes=[len(rows) * base.shape[1] for _, rows, _, _ in tasks],
    )


def _dense_from_dicts(values, res_index: dict) -> np.ndarray:
    """``{resource: int}`` cells of a ``candidate_*`` column as an
    ``(n_rows, n_resources)`` matrix; missing entries and non-dict cells are 0,
    like ``d.get(r, 0)`` in :func:`build_training_datasets`."""
    out = np.zeros((len(values), len(res_index)), dtype=np.int64)
    for i, d in enumerate(values):
        if isinstance(d, dict):
            for k, v in d.items():
                j = res_index.get(k)
                if j is not None:
                    out[i, j] = v
    return out


def _dense_training_plan(
        df_features: pd.DataFrame,
        act_to_resources: dict,
        resources: list,
        max_concurrency: dict,
        label_data_attributes: list,
        label_data_attributes_categorical: list,
        values_categorical: dict,
        use_workload_features: bool,
        _dense: dict = None,
    ):
    """Lay out the per-resource training sets over one shared matrix.

    Returns ``(base, names, extra_at, order, constant, tasks)``:

    * ``base``: float64 matrix over the rows of ``df_features`` with, in this
      order, the per-resource history counts, the non-categorical case
      attributes, the current-activity one-hots and the one-hots of the
      categorical attributes. It is the column order of the legacy frames,
      except for the two per-resource columns (workload, queue length) that go
      at position ``extra_at`` when ``use_workload_features`` is set;
    * ``order``: resources that have a training set, in ``resources`` order;
    * ``constant``: resources whose label never changes, with that label;
    * ``tasks``: ``(resource, row_indices, y, extra_columns)`` for the rest.

    Returns ``None`` when the layout does not apply."""

    cat_set = set(label_data_attributes_categorical)
    if any(a not in set(label_data_attributes) for a in label_data_attributes_categorical):
        return None
    numeric_attrs = [a for a in label_data_attributes if a not in cat_set]
    act_labels = list(act_to_resources.keys())
    names = list(resources) + numeric_attrs + ['activity = ' + a for a in act_labels]
    extra_at = len(names)
    if use_workload_features:
        names += ['workload', 'queue_length']
    names += [a + ' = ' + str(v) for a in label_data_attributes_categorical for v in values_categorical[a]]
    if len(set(names)) != len(names) or 'class' in names or df_features.columns.has_duplicates:
        return None
    numeric_cols = list(resources) + numeric_attrs
    if any(c not in df_features.columns for c in numeric_cols + list(label_data_attributes_categorical)):
        return None
    if not all(isinstance(dt, np.dtype) and dt.kind in 'biuf' for dt in df_features[numeric_cols].dtypes):
        return None

    n = len(df_features)
    labels = df_features['transition_label'].to_numpy()
    actual = df_features['resource'].to_numpy()
    res_index = {r: j for j, r in enumerate(resources)}
    if len(res_index) != len(resources):
        return None

    blocks = [df_features[numeric_cols].to_numpy(dtype=np.float64).reshape(n, len(numeric_cols))]
    blocks.append(np.column_stack([labels == a for a in act_labels]).astype(np.float64) if act_labels else np.zeros((n, 0)))
    for a in label_data_attributes_categorical:
        col = df_features[a]
        for v in values_categorical[a]:
            blocks.append((col == v).to_numpy(dtype=np.float64).reshape(n, 1))
    base = np.ascontiguousarray(np.hstack(blocks))

    workloads = dense_matrix(_dense, n, resources, 'workloads')
    if workloads is None:
        workloads = _dense_from_dicts(df_features['candidate_workloads'].tolist(), res_index)
    queues = None
    if use_workload_features:
        queues = dense_matrix(_dense, n, resources, 'queue_lengths')
        if queues is None:
            queues = _dense_from_dicts(df_features['candidate_queue_lengths'].tolist(), res_index)

    act_index = {a: i for i, a in enumerate(act_labels)}
    label_idx = np.fromiter((act_index.get(l, -1) for l in labels), dtype=np.int64, count=n)
    r_to_acts = _resource_to_eligible_acts(act_to_resources, resources)

    order, constant, tasks = [], {}, []
    for r in resources:
        eligible_acts = r_to_acts.get(r, set())
        if not eligible_acts:
            continue
        eligible = np.zeros(len(act_labels) + 1, dtype=bool)   # last slot: labels outside the model
        eligible[[act_index[a] for a in eligible_acts]] = True
        in_scope = eligible[label_idx]
        if not in_scope.any():
            continue
        j = res_index[r]
        workload_r = workloads[:, j]
        rows = np.flatnonzero(in_scope & (workload_r < max_concurrency.get(r, 1)))
        if rows.size == 0:
            continue
        y = (actual[rows] == r).astype(np.int64)
        order.append(r)
        if y.min() == y.max():
            # Constant label — store scalar probability (0 or 1).
            constant[r] = float(y[0])
            continue
        extra = None
        if use_workload_features:
            extra = np.column_stack([workload_r[rows], queues[rows, j]]).astype(np.float64)
        tasks.append((r, rows, y, extra))
    return base, names, extra_at, order, constant, tasks


def _build_models_legacy(
        df_features: pd.DataFrame,
        act_to_resources: dict,
        resources: list,
        max_concurrency: dict,
        max_depths_cv: list = range(1,6),
        label_data_attributes: list = [],
        label_data_attributes_categorical: list = [],
        values_categorical: dict = dict(),
        random_state: int = 72,
        use_workload_features: bool = False
    ) -> dict:

    param_grid = [
        {
            'clf': [DecisionTreeClassifier(random_state=random_state)],
            'clf__max_depth': list(max_depths_cv),
        },
        {'clf': [DummyClassifier(strategy='prior', random_state=random_state)]},
    ]

    datasets = build_training_datasets(
        df_features,
        act_to_resources,
        resources,
        max_concurrency,
        label_data_attributes,
        use_workload_features=use_workload_features,
    )

    models = dict()
    n_cv = 5

    for r, data_r in tqdm(datasets.items(), desc='resource models'):
        if len(data_r['class'].unique()) < 2:
            # Constant label — store scalar probability (0 or 1).
            models[r] = float(data_r['class'].iloc[0]) if len(data_r) else 0.0
            continue

        for a in label_data_attributes_categorical:
            for v in values_categorical[a]:
                data_r[a + ' = ' + str(v)] = (data_r[a] == v).astype(int)
            del data_r[a]

        X = data_r.drop(columns=['class'])
        X = prune_low_signal_columns(X)
        y = data_r['class']

        if X.shape[1] == 0:
            # No informative features survived pruning — fall back to the
            # marginal probability (same semantics as the Dummy branch).
            models[r] = float(y.mean())
            continue

        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)
        if n_pos < n_cv or n_neg < n_cv:
            # Too few samples in one class for 5-fold CV to produce both classes
            # in every fold — the log-loss scorer errors. Fall back to the
            # marginal.
            models[r] = float(y.mean())
            continue

        if max_depths_cv:
            pipe = Pipeline([('clf', DecisionTreeClassifier(random_state=random_state))])
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', FitFailedWarning)
                    warnings.simplefilter('ignore', UserWarning)
                    grid_search = GridSearchCV(
                        estimator=pipe,
                        param_grid=param_grid,
                        cv=5,
                        scoring=BINARY_NEG_LOG_LOSS_SCORER,
                    ).fit(X, y)
                best_clf = grid_search.best_estimator_.named_steps['clf']
            except Exception:
                best_clf = DecisionTreeClassifier(max_depth=2, random_state=random_state).fit(X, y)

            if isinstance(best_clf, DummyClassifier):
                models[r] = float(y.mean())
                continue
            clf_dtc = best_clf
        else:
            clf_dtc = DecisionTreeClassifier(random_state=random_state, max_depth=1)
            clf_dtc.fit(X, y)

        clf = DecisionRules()
        clf.from_decision_tree(clf_dtc)
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
        use_workload_features: bool = False,
    ) -> dict:
    """Return ``{resource: training_df}``.

    One training set per resource. Scope: events whose activity has the
    resource in its candidate pool AND where the resource was actually free
    (``workload < max_concurrency[r]``) at the enabling time. Features:
    per-resource history counts (one column per resource), current-activity
    one-hot, case attributes, and — when ``use_workload_features`` is True —
    the resource's own workload and queue length at enabling time. Target
    ``class`` is ``1`` when this resource actually performed the event,
    ``0`` otherwise.
    """

    res_history_cols = list(resources)
    act_labels = list(act_to_resources.keys())

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
        base = df_scope[res_history_cols + list(label_data_attributes) + ["transition_label"]].reset_index(drop=True)
        for a in act_labels:
            base['activity = ' + a] = (base['transition_label'] == a).astype(int)
        base = base.drop(columns=['transition_label'])
        actual = df_scope['resource'].reset_index(drop=True)
        df_r = base
        if use_workload_features:
            queue_r = df_scope['candidate_queue_lengths'].apply(
                lambda d: d.get(r, 0) if isinstance(d, dict) else 0
            )
            df_r['workload'] = workload_r.reset_index(drop=True).values
            df_r['queue_length'] = queue_r.reset_index(drop=True).values
        df_r['class'] = (actual == r).astype(int).values
        datasets[r] = df_r

    return datasets
