import re
from sklearn.tree import export_graphviz
from sklearn.metrics import log_loss, make_scorer
import graphviz
import random
import numpy as np
import pandas as pd
import scipy.stats as stats


# Scorer for GridSearchCV on binary classifiers. Uses ``labels=[0, 1]`` so that
# log_loss stays defined even when a CV fold contains only one class (which
# happens on small or heavily imbalanced training sets for per-resource /
# per-transition models). Log loss rewards calibration, which is what matters
# here: the simulator samples resources/transitions proportionally to leaf
# probabilities, so miscalibration biases the sampling distribution.
BINARY_NEG_LOG_LOSS_SCORER = make_scorer(
    log_loss,
    response_method='predict_proba',
    greater_is_better=False,
    labels=[0, 1],
)

MAX_DURATION_MINUTES = 60 * 24  # 1440 minutes = 24 hours
DEFAULT_SAMPLE_SIZE = 1000
MIN_FEATURE_POSITIVES = 20


def prune_low_signal_columns(X: pd.DataFrame, min_positives: int = MIN_FEATURE_POSITIVES) -> pd.DataFrame:
    """Drop columns with zero variance or binary 0/1 columns with fewer than
    ``min_positives`` positives. One-hot features for resources/activities
    rarely or never seen in this training slice contribute only noise and
    bloat the GridSearchCV split candidates.
    """
    if X.empty:
        return X
    keep = []
    for col in X.columns:
        series = X[col]
        unique = pd.unique(series)
        if len(unique) < 2:
            continue
        if set(unique).issubset({0, 1, 0.0, 1.0}):
            if int(series.sum()) < min_positives:
                continue
        keep.append(col)
    return X[keep]


def apply_laplace_smoothing(clf, alpha: float = 1.0):
    """Add-alpha smoothing to leaf class distributions of a fitted sklearn classifier.

    Recent sklearn versions (>=1.3) store ``tree_.value`` already normalized as
    class proportions. This function recovers raw (weighted) counts via
    ``weighted_n_node_samples``, applies ``(n_c + alpha) / (n_total + K*alpha)``
    at each leaf, and writes the smoothed proportions back. This avoids 0/1
    leaves that zero out competitors when the simulator uses these
    probabilities as sampling weights.
    """
    tree_ = clf.tree_
    n_classes = tree_.value.shape[2]
    for node_id in range(tree_.node_count):
        if tree_.children_left[node_id] == tree_.children_right[node_id]:
            n_total = float(tree_.weighted_n_node_samples[node_id])
            proportions = tree_.value[node_id, 0, :]
            counts = proportions * n_total
            smoothed = (counts + alpha) / (n_total + alpha * n_classes)
            tree_.value[node_id, 0, :] = smoothed
    return clf


def build_graph_vis(model_t, model_distributions=False):

    try:
        dot_data = export_graphviz(model_t, 
                feature_names=model_t.feature_names_in_,
                label='none',
                filled=True, 
                rounded=True,
                impurity=False,
                proportion=True)
    except Exception:
        dot_data = export_graphviz(model_t,
                feature_names=model_t.feature_names_in_,
                label='none',
                rounded=True,
                impurity=False,
                proportion=True)

    new_dot_data = reformat_dot_str(dot_data, model_distributions)
    graph = graphviz.Source(new_dot_data)

    return graph


def reformat_dot_str(input_str, dot_distributions=False):

    result = re.sub(r'\d+\.?\d*\s?%\\n', '', input_str)
    result = re.sub(r'\[\d+\.?\d*,\s*(\d+\.?\d*)\]', r'\1', result)

    if dot_distributions:
        pattern = r'\[\[(\d+\.?\d*)\]\\n\[(\d+\.?\d*)\]\]'
            
        def round_and_replace(match):
            num1, num2 = match.groups()
            rounded1 = round(float(num1))
            rounded2 = round(float(num2))
            return f'({rounded1}, {rounded2})'
            
        result = re.sub(pattern, round_and_replace, result)
        
    return result


def parse_tree(dot_string):
    node_pattern = r'(\d+) \[label="([^"]+)"'
    edge_pattern = r'(\d+) -> (\d+)(?: \[labeldistance=[^,]+, labelangle=[^,]+, headlabel="([^"]+)"\])?'

    nodes = {}
    edges = []

    for match in re.findall(node_pattern, dot_string):
        node_id = int(match[0])
        label_info = match[1].split("\\n")
        if len(label_info) > 1:
            feature, threshold = label_info[0].split(" <= ")
            threshold = float(threshold)
            nodes[node_id] = {'feature': feature, 'threshold': threshold}
        else:
            try:
                nodes[node_id] = {'value': float(label_info[0])}  # Leaf node with a value
            except ValueError:
                nodes[node_id] = {'value': (int(label_info[0][1:-1].split(', ')[0]), int(label_info[0][1:-1].split(', ')[1]))}
    for match in re.findall(edge_pattern, dot_string):
        parent, child = int(match[0]), int(match[1])
        edges.append((parent, child))

    return nodes, edges


def build_tree_structure(nodes, edges):
    def add_edge(parent, child, edge_index):
        if 'children' not in nodes[parent]:
            nodes[parent]['children'] = {}
        # First edge encountered for a parent is the left (True) branch, second is right (False).
        condition = (edge_index == 0)
        nodes[parent]['children'][condition] = child

    parent_edge_count = {}
    for parent, child in edges:
        if parent not in parent_edge_count:
            parent_edge_count[parent] = 0
        add_edge(parent, child, parent_edge_count[parent])
        parent_edge_count[parent] += 1

    return nodes


def traverse_tree(tree, features):
    
    if type(tree) != dict:
        return tree
    
    current_node = 0 
    while 'value' not in tree[current_node]:
        feature = tree[current_node]['feature']
        threshold = tree[current_node]['threshold']
        if features[feature] <= threshold:
            current_node = tree[current_node]['children'][True]
        else:
            current_node = tree[current_node]['children'][False]

    return tree[current_node]['value']

def _ensure_sampled(node):
    """Lazy-generate sampled values from the distribution if not yet present."""
    if 'sampled' not in node and 'dist' in node:
        from prosit.utils.distribution_utils import sampling_from_dist
        dist_tuple = node['dist']
        node['sampled'] = list(sampling_from_dist(
            dist_tuple[0], dist_tuple[1], dist_tuple[2], dist_tuple[3],
            node['value'], n_sample=DEFAULT_SAMPLE_SIZE
        ))

def traverse_tree_distribution(tree, features):

    if type(tree) != dict:
        return random.choice(tree)

    current_node = 0
    while 'value' not in tree[current_node]:
        feature = tree[current_node]['feature']
        threshold = tree[current_node]['threshold']
        if features[feature] <= threshold:
            current_node = tree[current_node]['children'][True]
        else:
            current_node = tree[current_node]['children'][False]

    _ensure_sampled(tree[current_node])
    return random.choice(tree[current_node]['sampled'])


def transform_river_decision_tree_data(decision_tree, distribution=True, min_value=0, max_value=MAX_DURATION_MINUTES) -> dict:

    if decision_tree.height == 0:
        if distribution:
            return {0: {'value': 0, 'sampled': [0], 'dist': ("fixed", (0,), 0, 0)}}
        else:
            return {0: {"value": 1}}

    df = decision_tree.to_dataframe()
    if df is None:
        if distribution:
            mean_var = decision_tree.debug_one({})
            pred_split = mean_var.split("\n")[-2].split(" | ")
            value = float(pred_split[0][6:].replace(",", ""))
            variance = float(pred_split[1][5:].replace(",", ""))
            std_dev = np.sqrt(max(0, variance))

            if std_dev == 0:
                sampled_values = [value]
            else:
                sampled_values = np.random.normal(loc=value, scale=std_dev, size=DEFAULT_SAMPLE_SIZE)
                sampled_values[sampled_values < min_value] = value
                sampled_values[sampled_values > max_value] = value
                sampled_values = sampled_values.tolist()

            return {0: {'value': value, 'sampled': sampled_values, 'dist': (getattr(stats, "norm"), (value, std_dev), min_value, max_value)}}
        else:
            value = decision_tree.predict_proba_one({})[1]
            return {0: {"value": value}}

    transformed_data = {}

    for node_id, row in df.iterrows():
        is_leaf = row['is_leaf']

        if not is_leaf and pd.notna(row['feature']) and pd.notna(row['threshold']):
            feature = row['feature']
            threshold = row['threshold']

            children_nodes_df = df[df['parent'] == node_id].sort_index()
            children_node_ids = children_nodes_df.index.tolist()

            children_dict = {}
            # Binary tree: smaller child id is the 'False' branch, larger the 'True' branch.
            if len(children_node_ids) == 2:
                children_dict[False] = int(children_node_ids[0])
                children_dict[True] = int(children_node_ids[1])
            elif len(children_node_ids) == 1:
                children_dict[True] = int(children_node_ids[0])

            transformed_data[node_id] = {
                'feature': feature,
                'threshold': threshold,
                'children': children_dict
            }
        else:
            if distribution:
                value = row['stats'].mean.get()
                variance = row['stats'].get()

                std_dev = np.sqrt(max(0, variance))

                if std_dev == 0:
                    sampled_values = [value]
                else:
                    sampled_values = np.random.normal(loc=value, scale=std_dev, size=max(DEFAULT_SAMPLE_SIZE, int(row['stats'].n)))
                    sampled_values[sampled_values < min_value] = value
                    sampled_values[sampled_values > max_value] = value
                    sampled_values = sampled_values.tolist()

                transformed_data[node_id] = {
                    'value': value,
                    'sampled': sampled_values,
                    'dist': (getattr(stats, "norm"), (value, std_dev), min_value, max_value)
                }
            else:
                value = row['stats'][1]/(row['stats'][0]+row['stats'][1])
                transformed_data[node_id] = {'value': value}

    transformed_data_sorted = dict(sorted(transformed_data.items()))
    return transformed_data_sorted


class DecisionRules:
    def __init__(self):
        self.rules = None
        self.graph = None

    def from_decision_tree(self, decision_tree):
        self.decision_tree = decision_tree
        self.graph = build_graph_vis(decision_tree, True)
        nodes, edges = parse_tree(self.graph.source)
        self.rules = build_tree_structure(nodes, edges)

    def from_river_decision_tree(self, decision_tree, distribution=False, min_value=0, max_value=MAX_DURATION_MINUTES):
        self.decision_tree = decision_tree
        self.rules = transform_river_decision_tree_data(decision_tree, distribution, min_value, max_value)

    def apply(self, features):
        return traverse_tree(self.rules, features)
    
    def apply_distribution(self, features):
        return traverse_tree_distribution(self.rules, features)
    
    def write_dot(self, file_name='decision_tree.dot'):
        if self.graph:
            self.graph.render(file_name)