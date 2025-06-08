import re
from prosit.utils.distribution_utils import sampling_from_dist
from sklearn.tree import export_graphviz
import graphviz
import random

def build_graph_vis(model_t, model_distributions=False):

    try:
        dot_data = export_graphviz(model_t, 
                feature_names=model_t.feature_names_in_,
                label='none',
                filled=True, 
                rounded=True,
                impurity=False,
                proportion=True)
    except:
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
            except:
                nodes[node_id] = {'value': (int(label_info[0][1:-1].split(', ')[0]), int(label_info[0][1:-1].split(', ')[1]))}
    for match in re.findall(edge_pattern, dot_string):
        parent, child = int(match[0]), int(match[1])
        edges.append((parent, child))

    return nodes, edges


def build_tree_structure(nodes, edges):
    tree = {}

    def add_edge(parent, child, edge_index):
        if 'children' not in nodes[parent]:
            nodes[parent]['children'] = {}
        condition = (edge_index == 0)  # True for left (first edge), False for right (second edge)
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

    return random.choice(tree[current_node]['sampled'])


class DecisionRules:
    def __init__(self):
        self.rules = None
        self.graph = None

    def from_decision_tree(self, decision_tree):
        self.decision_tree = decision_tree
        self.graph = build_graph_vis(decision_tree, True)
        nodes, edges = parse_tree(self.graph.source)
        self.rules = build_tree_structure(nodes, edges)

    def from_dict(self, dict_value):
        if type(dict_value) == float:
            self.rules = dict_value
        else:
            self.rules = sampling_from_dist(dict_value[0], dict_value[1], dict_value[2], dict_value[3], dict_value[1])

    def apply(self, features):
        return traverse_tree(self.rules, features)
    
    def apply_distribution(self, features):
        return traverse_tree_distribution(self.rules, features)
    
    def write_dot(self, file_name='decision_tree.dot'):
        if self.graph:
            self.graph.render(file_name)