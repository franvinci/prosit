import pandas as pd
from prosit.discovery.resource_discovery import build_training_datasets
from tqdm import tqdm
from river import tree
from prosit.utils.rule_utils import DecisionRules


def incremental_resource_weights_learning(
        df_features: pd.DataFrame,
        act_to_resources: dict,
        resources: list,
        max_concurrency: dict,
        max_depth: int = 3,
        grace_period: int = 1000,
        label_data_attributes: list = [],
        label_data_attributes_categorical: list = [],
        values_categorical: dict = dict(),
        use_workload_features: bool = False,
    ):
    """Return ``{resource: classifier}`` using incremental trees.

    Mirrors :func:`prosit.discovery.resource_discovery.build_models` but trains
    a ``HoeffdingAdaptiveTreeClassifier`` per resource on the events where
    the resource was a free candidate at the enabling time.
    """

    df_features = df_features[~df_features["resource"].isna()]

    datasets = build_training_datasets(
        df_features,
        act_to_resources,
        resources,
        max_concurrency,
        label_data_attributes,
        use_workload_features=use_workload_features,
    )

    models = dict()

    for r, data_r in tqdm(datasets.items()):
        if len(data_r['class'].unique()) < 2:
            models[r] = float(data_r['class'].iloc[0]) if len(data_r) else 0.0
            continue

        for a in label_data_attributes_categorical:
            for v in values_categorical[a]:
                data_r[a + ' = ' + str(v)] = (data_r[a] == v).astype(int)
            del data_r[a]

        m = tree.HoeffdingAdaptiveTreeClassifier(
            seed=72, max_depth=max_depth, grace_period=grace_period, leaf_prediction="mc"
        )
        for _, row in data_r.iterrows():
            X_row = row.drop('class').to_dict()
            y_row = row['class']
            m.learn_one(X_row, y_row)

        clf = DecisionRules()
        clf.from_river_decision_tree(m)
        models[r] = clf

    return models
