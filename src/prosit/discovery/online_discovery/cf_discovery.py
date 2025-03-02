from prosit.discovery.cf_discovery import build_training_datasets
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.log.obj import EventLog
from tqdm import tqdm
from river import tree
from prosit.utils.online_utils import return_best_online_model

def incremental_transition_weights_learning(
        models_t: dict,
        log: EventLog,
        net: PetriNet, 
        initial_marking: Marking, 
        final_marking: Marking,
        net_transition_labels: list,
        label_data_attributes: list,
        label_data_attributes_categorical: list,
        values_categorical: dict,
        mode_act_history: bool,
        max_depth: int = 3,
        grace_period: int = 5000
    ) -> dict :
    
    datasets_t = build_training_datasets(
                    log, 
                    net, 
                    initial_marking, 
                    final_marking, 
                    net_transition_labels, 
                    mode_act_history, 
                    label_data_attributes
                )

    if not models_t:
        models_t = dict()
        first_step = True
    else:
        first_step = False


    for t in tqdm(net.transitions):
        data_t = datasets_t[t]
        if len(data_t['class'].unique())<2:
            models_t[t] = None
            continue
        
        for a in label_data_attributes_categorical:
            for v in values_categorical[a]:
                data_t[a+' = '+str(v)] = (data_t[a] == v)*1
            del data_t[a]

        if first_step or t not in models_t.keys() or not models_t[t]:
            models_t[t] = tree.HoeffdingAdaptiveTreeClassifier(seed=72, leaf_prediction="mc", max_depth=max_depth, grace_period=grace_period)

        for _, row in data_t.iterrows():
            X_row = row.drop('class').to_dict()
            y_row = row['class']
            models_t[t].learn_one(X_row, y_row)

    return models_t