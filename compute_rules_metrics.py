import sys
sys.path.append("src/")

import os

import warnings
warnings.filterwarnings("ignore")

from prosit.simulator import SimulatorParameters
from prosit.utils.common_utils import build_df_features
from prosit.discovery.cf_discovery import build_training_datasets
from prosit.discovery.time_discovery import build_training_df_ex, build_training_df_arrival, build_training_df_wt

import pm4py

import numpy as np
import pandas as pd
import scipy.stats as stats

import json

CASE_STUDIES = {
    'purchasing': {
        'PATH_LOG': 'data/logs/purchasing.xes',
        'PATH_MODEL': 'data/models/purchasing.bpmn'
    },
    'cvs': {
        'PATH_LOG': 'data/logs/cvs.xes',
        'PATH_MODEL': 'data/models/cvs.bpmn'
    },
    'acr': {
        'PATH_LOG': 'data/logs/acr.xes',
        'PATH_MODEL': 'data/models/acr.bpmn'
    },
    'bpi12': {
        'PATH_LOG': 'data/logs/bpi12.xes',
        'PATH_MODEL': 'data/models/bpi12.bpmn'
    },
    'bpi17': {
        'PATH_LOG': 'data/logs/bpi17.xes',
        'PATH_MODEL': 'data/models/bpi17.bpmn'
    },
}

CONVERT_IDS = {
    "simod_im": {
        "case_id": "case:concept:name",
        "activity": "concept:name",
        "resource": "org:resource",
        "end_time": "time:timestamp",
        "start_time": "start:timestamp",
    }
}

OUTPUT_PATH = 'outputs'
NOISE_THRESHOLD_IM = 0.0
MAX_DEPTH_TO_ANALYZE = 5
SOTA_METHODS = ["simod"]

if __name__ == "__main__":
    
    dist_cf = dict()
    dist_at = dict()
    dist_et = dict()
    dist_wt = dict()
    
    for case_study in CASE_STUDIES.keys():
        print(case_study)

        df_train_log = pd.read_csv(OUTPUT_PATH+f"/{case_study}/df_train.csv")
        df_test_log = pd.read_csv(OUTPUT_PATH+f"/{case_study}/df_test.csv")
        
        df_train_log["case:concept:name"] = df_train_log["case:concept:name"].astype(str)
        df_train_log["start:timestamp"] = pd.to_datetime(df_train_log["start:timestamp"].apply(lambda x: x[:19]))
        df_train_log["time:timestamp"] = pd.to_datetime(df_train_log["time:timestamp"].apply(lambda x: x[:19]))
        df_train_log["org:resource"] = df_train_log["org:resource"].astype(str)

        df_test_log["case:concept:name"] = df_test_log["case:concept:name"].astype(str)
        df_test_log["start:timestamp"] = pd.to_datetime(df_test_log["start:timestamp"].apply(lambda x: x[:19]))
        df_test_log["time:timestamp"] = pd.to_datetime(df_test_log["time:timestamp"].apply(lambda x: x[:19]))
        df_test_log["org:resource"] = df_test_log["org:resource"].astype(str)

        train_log = pm4py.convert_to_event_log(df_train_log)
        test_log = pm4py.convert_to_event_log(df_test_log)

        if CASE_STUDIES[case_study]["PATH_MODEL"] is not None:
            bpmn_model = pm4py.read_bpmn(CASE_STUDIES[case_study]['PATH_MODEL'])
            net, initial_marking, final_marking = pm4py.convert_to_petri_net(bpmn_model)
        else:
            net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(train_log, noise_threshold=CASE_STUDIES[case_study].get('NOISE_THRESHOLD_IM', NOISE_THRESHOLD_IM))

        simulator_params = SimulatorParameters(net, initial_marking, final_marking)
        simulator_params.discover_from_eventlog(train_log, max_depth_tree=MAX_DEPTH_TO_ANALYZE, verbose=False)

        df_test_features = build_df_features(test_log, net, initial_marking, final_marking, simulator_params.net_transition_labels, simulator_params.label_data_attributes)
        df_test_cf = build_training_datasets(df_test_features, simulator_params.net_transition_labels, simulator_params.label_data_attributes) 
        df_test_features = df_test_features[(df_test_features['transition_label'].isin(simulator_params.net_transition_labels)) & (df_test_features['resource'].isin(simulator_params.resources))]
        df_test_et = build_training_df_ex(df_test_features, simulator_params.resources, simulator_params.net_transition_labels, simulator_params.calendars, simulator_params.label_data_attributes, simulator_params.label_data_attributes_categorical, simulator_params.attribute_values_label_categorical)
        df_test_wt = build_training_df_wt(df_test_features, simulator_params.net_transition_labels, simulator_params.calendars, simulator_params.label_data_attributes, simulator_params.label_data_attributes_categorical, simulator_params.attribute_values_label_categorical)
        df_test_at = build_training_df_arrival(test_log, simulator_params.arrival_calendar)

        dist_cf[case_study] = {method: [] for method in [f"maxdepth_{MAX_DEPTH_TO_ANALYZE}"] + SOTA_METHODS}
        dist_at[case_study] = {method: [] for method in [f"maxdepth_{MAX_DEPTH_TO_ANALYZE}"] + SOTA_METHODS}
        dist_et[case_study] = {method: [] for method in [f"maxdepth_{MAX_DEPTH_TO_ANALYZE}"] + SOTA_METHODS}
        dist_wt[case_study] = {method: [] for method in [f"maxdepth_{MAX_DEPTH_TO_ANALYZE}"] + SOTA_METHODS}
        
        for method in [f"maxdepth_{MAX_DEPTH_TO_ANALYZE}"] + SOTA_METHODS:
            print(f"COMPUTE RULES ERRORS FOR {method} for case study: {case_study}...")

            if method.startswith("maxdepth_"):
                depth = int(method.split("_")[-1])
                path_simulations = os.listdir(OUTPUT_PATH+f"/{case_study}/simulations/prob")
                dfsim_paths = [OUTPUT_PATH+f"/{case_study}/simulations/prob/df_sim_{i}_maxdepth_{depth}.csv" for i in range(5)]
            elif method in SOTA_METHODS:
                path_simulations = os.listdir(f"sota_results/{method}/{case_study}/")
                dfsim_paths = [f"sota_results/{method}/{case_study}/{dfpath}" for dfpath in path_simulations if dfpath.endswith(".csv")]
            
            for dfsim_path in dfsim_paths:

                df_sim = pd.read_csv(dfsim_path)

                if method in SOTA_METHODS:
                    df_sim.rename(columns=CONVERT_IDS[method], inplace=True)
                df_sim["case:concept:name"] = df_sim["case:concept:name"].astype(str)
                df_sim["start:timestamp"] = pd.to_datetime(df_sim["start:timestamp"])
                df_sim["time:timestamp"] = pd.to_datetime(df_sim["time:timestamp"])
                df_sim["org:resource"] = df_sim["org:resource"].astype(str)

                sim_log = pm4py.convert_to_event_log(df_sim)

                df_sim_features = build_df_features(sim_log, net, initial_marking, final_marking, simulator_params.net_transition_labels, simulator_params.label_data_attributes)
                df_sim_cf = build_training_datasets(df_sim_features, simulator_params.net_transition_labels, simulator_params.label_data_attributes) 
                df_sim_features = df_sim_features[(df_sim_features['transition_label'].isin(simulator_params.net_transition_labels)) & (df_sim_features['resource'].isin(simulator_params.resources))]
                df_sim_et = build_training_df_ex(df_sim_features, simulator_params.resources, simulator_params.net_transition_labels, simulator_params.calendars, simulator_params.label_data_attributes, simulator_params.label_data_attributes_categorical, simulator_params.attribute_values_label_categorical)
                df_sim_wt = build_training_df_wt(df_sim_features, simulator_params.net_transition_labels, simulator_params.calendars, simulator_params.label_data_attributes, simulator_params.label_data_attributes_categorical, simulator_params.attribute_values_label_categorical)
                df_sim_at = build_training_df_arrival(sim_log, simulator_params.arrival_calendar)

                print("COMPUTE TRANSITION ERRORS...")
                for t in list(df_test_cf.keys()):

                    data_t = df_test_cf[t]
                    if t not in list(df_sim_cf.keys()):
                        continue
                    data_t_sim = df_sim_cf[t]

                    for a in simulator_params.label_data_attributes_categorical:
                        for v in simulator_params.attribute_values_label_categorical[a]:
                            if a in data_t.columns:
                                data_t[a+' = '+str(v)] = (data_t[a] == v).astype(int)
                            if a in data_t_sim.columns:
                                data_t_sim[a+' = '+str(v)] = (data_t_sim[a] == v).astype(int)
                        if a in data_t.columns:
                            del data_t[a]
                        if a in data_t_sim.columns:
                            del data_t_sim[a]

                    X = data_t.drop(columns=['class'])
                    y = data_t['class']

                    X_sim = data_t_sim.drop(columns=['class'])
                    y_sim = data_t_sim['class']

                    if simulator_params.transition_weights[t] is not None:
                        leaf_indices = simulator_params.transition_weights[t].decision_tree.apply(X)
                        leaf_indices_sim = simulator_params.transition_weights[t].decision_tree.apply(X_sim)
                        if simulator_params.transition_weights[t].decision_tree.get_n_leaves() == 1:
                            if len(y) == 0:
                                true_prob = 0
                            else:
                                true_prob = (y == 1).sum()/len(y)
                            if len(y_sim) == 0:
                                sim_prob = 0
                            else:
                                sim_prob = (y_sim == 1).sum()/len(y_sim)
                            dist_cf[case_study][method].append(abs(true_prob - sim_prob))
                        else:
                            for leaf in range(1, simulator_params.transition_weights[t].decision_tree.get_n_leaves()+1):
                                if len(y[leaf_indices == leaf]) > 0:
                                    true_prob = (y[(leaf_indices == leaf)] == 1).sum()/len(y[(leaf_indices == leaf)])
                                else:
                                    true_prob = 0
                                if len(y_sim[leaf_indices_sim == leaf]) > 0:
                                    sim_prob = (y_sim[(leaf_indices_sim == leaf)] == 1).sum()/len(y_sim[(leaf_indices_sim == leaf)])
                                else:
                                    sim_prob = 0
                                dist_cf[case_study][method].append(abs(true_prob - sim_prob))


                print("COMPUTE ARRIVAL TIME ERRORS...")
                dt_at = simulator_params.arrival_time_distribution.decision_tree

                X = df_test_at.drop(columns=['arrival_time'])
                y = df_test_at['arrival_time']

                X_sim = df_sim_at.drop(columns=['arrival_time'])
                y_sim = df_sim_at['arrival_time']

                if (len(X) == 0) or (len(X_sim) == 0):
                    continue
                leaf_indices = dt_at.apply(X)
                leaf_indices_sim = dt_at.apply(X_sim)

                if dt_at.get_n_leaves() == 1:
                    dist = stats.wasserstein_distance(y, y_sim)
                    dist_at[case_study][method].append(dist)
                else:
                    for i in range(1, dt_at.get_n_leaves()+1):
                        y_i = y[leaf_indices == i]
                        y_sim_i = y_sim[leaf_indices_sim == i]
                        if (len(y_i) > 0) and (len(y_sim_i) > 0):
                            dist = stats.wasserstein_distance(y_i, y_sim_i)
                            dist_at[case_study][method].append(dist)


                print("COMPUTE EXECUTION TIME ERRORS...")
                for act in simulator_params.net_transition_labels:
                    try:
                        dt_act = simulator_params.execution_time_distributions[act].decision_tree
                    except:
                        continue
                    df_test_et_act = df_test_et[df_test_et['activity_executed'] == act].iloc[:,1:]
                    df_sim_et_act = df_sim_et[df_sim_et['activity_executed'] == act].iloc[:,1:]

                    X = df_test_et_act.drop(columns=['execution_time'])
                    y = df_test_et_act['execution_time']

                    X_sim = df_sim_et_act.drop(columns=['execution_time'])
                    y_sim = df_sim_et_act['execution_time']

                    if (len(X) == 0) or (len(X_sim) == 0):
                        continue

                    leaf_indices = dt_act.apply(X)
                    leaf_indices_sim = dt_act.apply(X_sim)

                    if dt_act.get_n_leaves() == 1:
                        dist = stats.wasserstein_distance(y, y_sim)
                        dist_et[case_study][method].append(dist)
                    else:
                        for i in range(1, dt_act.get_n_leaves()+1):
                            y_i = y[leaf_indices == i]
                            y_sim_i = y_sim[leaf_indices_sim == i]
                            if (len(y_i) > 0) and (len(y_sim_i) > 0):
                                dist = stats.wasserstein_distance(y_i, y_sim_i)
                                dist_et[case_study][method].append(dist)
                
                if method in ["dsim", "rims"]:
                    continue

                print("COMPUTE WAITING TIME ERRORS...")
                for res in simulator_params.resources:
                    try:
                        dt_res = simulator_params.waiting_time_distributions[res].decision_tree
                    except:
                        continue
                    df_test_wt_res = df_test_wt[df_test_wt['resource'] == res].iloc[:,1:]
                    df_sim_wt_res = df_sim_wt[df_sim_wt['resource'] == res].iloc[:,1:]

                    X = df_test_wt_res.drop(columns=['waiting_time'])
                    y = df_test_wt_res['waiting_time']

                    X_sim = df_sim_wt_res.drop(columns=['waiting_time'])
                    y_sim = df_sim_wt_res['waiting_time']

                    if (len(X) == 0) or (len(X_sim) == 0):
                        continue
                    leaf_indices = dt_res.apply(X)
                    leaf_indices_sim = dt_res.apply(X_sim)

                    if dt_res.get_n_leaves() == 1:
                        dist = stats.wasserstein_distance(y, y_sim)
                        dist_wt[case_study][method].append(dist)
                    else:
                        for i in range(1, dt_res.get_n_leaves()+1):
                            y_i = y[leaf_indices == i]
                            y_sim_i = y_sim[leaf_indices_sim == i]
                            if (len(y_i) > 0) and (len(y_sim_i) > 0):
                                dist = stats.wasserstein_distance(y_i, y_sim_i)
                                dist_wt[case_study][method].append(dist)
            print()

        evaluations = {
            "rule_cfd": {method: np.mean(dist_cf[case_study][method]) for method in [f"maxdepth_{MAX_DEPTH_TO_ANALYZE}"] +SOTA_METHODS}, 
            "rule_atd": {method: np.mean(dist_at[case_study][method]) for method in [f"maxdepth_{MAX_DEPTH_TO_ANALYZE}"] +SOTA_METHODS}, 
            "rule_etd": {method: np.mean(dist_et[case_study][method]) for method in [f"maxdepth_{MAX_DEPTH_TO_ANALYZE}"] +SOTA_METHODS},  
            "rule_wtd": {method: np.mean(dist_wt[case_study][method]) for method in [f"maxdepth_{MAX_DEPTH_TO_ANALYZE}"] +SOTA_METHODS}, 
            }
        with open(f"{OUTPUT_PATH}/rule_distances_{case_study}.json", 'w') as f:
            json.dump(evaluations, f)