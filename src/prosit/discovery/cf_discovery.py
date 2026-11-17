import warnings
import numpy as np
import pandas as pd
from joblib import delayed
from tqdm import tqdm

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import FitFailedWarning
from sklearn.pipeline import Pipeline

from prosit.utils.common_utils import ModelPlan, run_model_plans
from prosit.utils.rule_utils import (
    DecisionRules,
    BINARY_NEG_LOG_LOSS_SCORER,
    prune_low_signal_columns,
    fit_classifier_task,
)




def discover_weight_transitions(
        df_features: pd.DataFrame,
        net_transition_labels: list,
        max_depths_cv: list = range(1, 6),
        label_data_attributes: list = [],
        label_data_attributes_categorical: list = [],
        values_categorical: dict = dict(),
        random_state: int = 72
    ) -> dict:

    return run_model_plans([plan_weight_transitions(
        df_features,
        net_transition_labels,
        max_depths_cv=max_depths_cv,
        label_data_attributes=label_data_attributes,
        label_data_attributes_categorical=label_data_attributes_categorical,
        values_categorical=values_categorical,
        random_state=random_state,
    )])[0]


def plan_weight_transitions(
        df_features: pd.DataFrame,
        net_transition_labels: list,
        max_depths_cv: list = range(1, 6),
        label_data_attributes: list = [],
        label_data_attributes_categorical: list = [],
        values_categorical: dict = dict(),
        random_state: int = 72
    ) -> ModelPlan:
    """:func:`discover_weight_transitions` split in two: the training sets are
    prepared here, the fits are left to :func:`run_model_plans`."""

    if not max_depths_cv:
        transitions = df_features['transition'].unique()
        # Keyed by transition name (str): PetriNet.Transition hashes by id, so
        # object keys drift from self.net.transitions under pickle/alignment
        # round-trips and break every downstream lookup done by identity.
        transition_weights = {t.name: (df_features["transition"] == t).sum() / df_features["prev_enabled_transitions"].apply(lambda t_set: t in t_set).sum() for t in transitions}
        return ModelPlan.done(transition_weights)

    return _plan_models(
        df_features,
        net_transition_labels,
        label_data_attributes,
        label_data_attributes_categorical,
        values_categorical,
        max_depths_cv=max_depths_cv,
        random_state=random_state
    )


def build_models(
        df_features: pd.DataFrame,
        net_transition_labels: list,
        label_data_attributes: list,
        label_data_attributes_categorical: list,
        values_categorical: dict,
        max_depths_cv: list = range(1,6),
        random_state: int = 72
    ) -> dict:
    """Fit one classifier per transition.

    Same training sets as :func:`build_training_datasets`, laid out as row
    selections over one shared numeric matrix and fitted in parallel with the
    cross-validation of
    :func:`prosit.utils.rule_utils._grid_search_cv_equivalent`. Inputs the
    dense layout cannot represent go through :func:`_build_models_legacy`."""

    return run_model_plans([_plan_models(
        df_features,
        net_transition_labels,
        label_data_attributes,
        label_data_attributes_categorical,
        values_categorical,
        max_depths_cv=max_depths_cv,
        random_state=random_state,
    )])[0]


def _plan_models(
        df_features: pd.DataFrame,
        net_transition_labels: list,
        label_data_attributes: list,
        label_data_attributes_categorical: list,
        values_categorical: dict,
        max_depths_cv: list = range(1,6),
        random_state: int = 72
    ) -> ModelPlan:

    plan = _dense_training_plan(
        df_features,
        net_transition_labels,
        label_data_attributes,
        label_data_attributes_categorical,
        values_categorical,
    )
    if plan is None:
        return ModelPlan.done(_build_models_legacy(
            df_features,
            net_transition_labels,
            label_data_attributes,
            label_data_attributes_categorical,
            values_categorical,
            max_depths_cv=max_depths_cv,
            random_state=random_state,
        ))

    base, names, order, constant, tasks = plan
    depths = list(max_depths_cv) if max_depths_cv else []
    return ModelPlan(
        jobs=[
            delayed(fit_classifier_task)(t_name, base, rows, y, names, depths, random_state)
            for t_name, rows, y in tasks
        ],
        finish=lambda fitted: {t_name: (constant[t_name] if t_name in constant else fitted[t_name]) for t_name in order},
        desc='transition models',
        sizes=[len(rows) * base.shape[1] for _, rows, _ in tasks],
    )


def _dense_training_plan(
        df_features: pd.DataFrame,
        net_transition_labels: list,
        label_data_attributes: list,
        label_data_attributes_categorical: list,
        values_categorical: dict,
    ):
    """Lay out the per-transition training sets over one shared matrix.

    Returns ``(base, names, order, constant, tasks)``. ``base`` holds, for
    every row of ``df_features``, the non-categorical case attributes, the
    activity history counts and the one-hots of the categorical attributes —
    the column order of the legacy frames. A transition's training set is the
    rows where it was enabled, with label 1 where it is the one that fired.
    ``order`` lists the transition names as the legacy explode/groupby yields
    them, ``constant`` the ones whose label never changes, ``tasks`` the
    ``(name, row_indices, y)`` left to fit.

    Returns ``None`` when the layout does not apply."""

    cat_set = set(label_data_attributes_categorical)
    if any(a not in set(label_data_attributes) for a in label_data_attributes_categorical):
        return None
    numeric_attrs = [a for a in label_data_attributes if a not in cat_set]
    numeric_cols = numeric_attrs + list(net_transition_labels)
    names = numeric_cols + [a + ' = ' + str(v) for a in label_data_attributes_categorical for v in values_categorical[a]]
    if len(set(names)) != len(names) or 'class' in names or df_features.columns.has_duplicates:
        return None
    if any(c not in df_features.columns for c in numeric_cols + list(label_data_attributes_categorical)):
        return None
    if not all(isinstance(dt, np.dtype) and dt.kind in 'biuf' for dt in df_features[numeric_cols].dtypes):
        return None

    n = len(df_features)
    enabled_sets = df_features['prev_enabled_transitions'].tolist()
    fired = df_features['transition'].tolist()

    # Group the rows by enabled set. Rows produced from the same marking
    # usually share one set object; equal sets built separately are merged.
    by_id = {}
    for i, s in enumerate(enabled_sets):
        entry = by_id.get(id(s))
        if entry is None:
            by_id[id(s)] = (s, [i])
        else:
            entry[1].append(i)
    groups = {}
    for s, row_list in by_id.values():
        try:
            key = frozenset(s)
        except TypeError:
            return None
        entry = groups.get(key)
        if entry is None:
            groups[key] = [s, row_list]
        else:
            entry[1] = entry[1] + row_list
    distinct = sorted(groups.values(), key=lambda e: min(e[1]))

    rows_by_name, objects_by_name, order = {}, {}, []
    for s, row_list in distinct:
        for t in s:
            name = t.name
            if name not in rows_by_name:
                rows_by_name[name] = []
                objects_by_name[name] = t
                order.append(name)
            elif objects_by_name[name] is not t:
                return None                      # two transitions sharing a name
            rows_by_name[name].append(row_list)

    blocks = [df_features[numeric_cols].to_numpy(dtype=np.float64).reshape(n, len(numeric_cols))]
    for a in label_data_attributes_categorical:
        col = df_features[a]
        for v in values_categorical[a]:
            blocks.append((col == v).to_numpy(dtype=np.float64).reshape(n, 1))
    base = np.ascontiguousarray(np.hstack(blocks))

    fired_id = np.fromiter((id(t) for t in fired), dtype=np.int64, count=n)
    constant, tasks = {}, []
    for name in order:
        rows = np.sort(np.concatenate([np.asarray(r, dtype=np.int64) for r in rows_by_name[name]]))
        y = (fired_id[rows] == id(objects_by_name[name])).astype(np.int64)
        if y.min() == y.max():
            # Constant label — store scalar probability so the simulator picks
            # it up via the numeric branch in
            # compute_transition_weights_from_model.
            constant[name] = float(y[0])
            continue
        tasks.append((name, rows, y))
    # The order collected above follows the iteration order of sets of
    # transitions, which changes from run to run; report the models by name.
    order.sort()
    return base, names, order, constant, tasks


def _build_models_legacy(
        df_features: pd.DataFrame,
        net_transition_labels: list,
        label_data_attributes: list,
        label_data_attributes_categorical: list,
        values_categorical: dict,
        max_depths_cv: list = range(1,6),
        random_state: int = 72
    ) -> dict:

    datasets_t = build_training_datasets(
                    df_features,
                    net_transition_labels,
                    label_data_attributes
                )

    param_grid = [
        {
            'clf': [DecisionTreeClassifier(random_state=random_state)],
            'clf__max_depth': list(max_depths_cv),
        },
        {'clf': [DummyClassifier(strategy='prior', random_state=random_state)]},
    ]

    models_t = dict()

    for t in tqdm(datasets_t.keys(), desc='transition models'):
        data_t = datasets_t[t]
        if len(data_t['class'].unique()) < 2:
            # Constant label — store scalar probability so the simulator picks
            # it up via the numeric branch in
            # compute_transition_weights_from_model.
            models_t[t] = float(data_t['class'].iloc[0]) if len(data_t) else 0.0
            continue

        for a in label_data_attributes_categorical:
            for v in values_categorical[a]:
                data_t[a + ' = ' + str(v)] = (data_t[a] == v).astype(int)
            del data_t[a]

        X = data_t.drop(columns=['class'])
        X = prune_low_signal_columns(X)
        y = data_t['class']

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
                models_t[t] = float(y.mean())
                continue
            clf_t_dtc = best_clf
        else:
            clf_t_dtc = DecisionTreeClassifier(random_state=random_state, max_depth=1)
            clf_t_dtc.fit(X, y)

        clf_t = DecisionRules()
        clf_t.from_decision_tree(clf_t_dtc)
        models_t[t] = clf_t

    return models_t



def build_training_datasets(
        df_features: pd.DataFrame,
        net_transition_labels: list,
        label_data_attributes: list
    ) -> dict:

    df_cf = df_features[["transition", "prev_enabled_transitions"] + label_data_attributes + net_transition_labels].copy()

    df_cf = df_cf.explode('prev_enabled_transitions')
    df_cf['class'] = (df_cf['prev_enabled_transitions'] == df_cf['transition']).astype(int)

    df_cf = df_cf.drop(columns=['transition'])
    # Group by transition NAME (str) rather than the Transition object: object
    # identity can drift across alignment/pickle boundaries, which would
    # silently break every downstream lookup done by identity.
    df_cf['transition_name'] = df_cf['prev_enabled_transitions'].apply(lambda t: t.name)
    df_cf = df_cf.drop(columns=['prev_enabled_transitions'])

    datasets_t = {
        t_name: group.drop(columns=['transition_name']).reset_index(drop=True)
        for t_name, group in df_cf.groupby('transition_name', sort=False)
    }

    return datasets_t
