import random
import pandas as pd
import math
import heapq
from datetime import datetime
from tqdm import tqdm

import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.log.obj import EventLog

from prosit.discovery.cf_discovery import discover_weight_transitions
from prosit.discovery.time_discovery import discover_execution_time_distributions, discover_arrival_time, discover_waiting_time
from prosit.discovery.calendar_discovery import discover_res_calendars, discover_arrival_calendar
from prosit.discovery.resource_discovery import discover_resources_list, return_max_concurrency, discover_resources_per_act, discover_weight_resources
from prosit.discovery.data_discovery import discover_attributes_distribution, return_label_data_attributes
from prosit.discovery.online_discovery.cf_discovery import incremental_transition_weights_learning
from prosit.discovery.online_discovery.time_discovery import incremental_execution_time_learning, incremental_model_arrival_learning, incremental_waiting_time_learning
from prosit.discovery.online_discovery.resource_discovery import incremental_resource_weights_learning
from prosit.utils.common_utils import (
    return_enabled_transitions,
    update_current_marking,
    return_fired_transition,
    count_concurrent_events,
    compute_transition_weights_from_model,
    add_minutes_with_calendar,
    build_df_features,
    return_resource,
    compute_resource_weights_from_model,
    )
from prosit.utils.distribution_utils import sampling_from_dist
from prosit.utils.save_and_load_utils import decision_rules_to_dict, transition_to_name, convert_calendar_names, dict_to_decrules, name_to_transition, fromstr_to_scipy

import json


class SimulatorParameters:
    """

    Simulation Parameters
    
    """

    def __init__(
            self, 
            net: PetriNet, 
            initial_marking: Marking,
            final_marking: Marking
        ):
        """ Initilize parameters """
        
        self.net: PetriNet = net
        self.initial_marking: Marking = initial_marking
        self.final_marking: Marking = final_marking
        self.net_transition_labels: list = list(set([t.label for t in net.transitions if t.label]))

        self.label_data_attributes: list = []
        self.label_data_attributes_categorical: list = []
        self.attribute_values_label_categorical: dict = dict()

        self.transition_weights: dict = {t: 1 for t in list(self.net.transitions)}
        self.resources: list = ['auto']
        self.act_to_resources: dict = {act: [r for r in self.resources] for act in self.net_transition_labels}
        self.resource_weights: dict = {"auto": 1}
        self.max_concurrency: dict = {'auto': 1}
        self.calendars: dict = {'auto': {wd: {h: True for h in range(24)} for wd in range(7)}}
        self.arrival_calendar: dict = {wd: {h: True for h in range(24)} for wd in range(7)}

        self.execution_time_distributions: dict = {a: ('fixed', 1, 1, 1, 1) for a in self.net_transition_labels}
        self.arrival_time_distribution: tuple = ('fixed', 1, 1, 1, 1)
        self.waiting_time_distributions: dict = {'auto': ('fixed', 1, 1, 1, 1)}

        self.rules_mode: bool = False


    def discover_from_eventlog(
            self,
            log: EventLog,
            max_depth_tree: int = 5,
            min_samples_leaf_cv: list = [5, 10, 20, 30],
            multitasking_thr: float = 0.05,
            enable_multitasking: bool = False,
            attribute_mode: str = 'distribution',
            incremental_discovery: bool = False,
            grace_period: int = 1000,
            random_state: int = 72,
            verbose: bool = True
        ):
        """ Discovery Parameters from event log data """

        if max_depth_tree < 1:
            self.rules_mode = False
            max_depth_cv = []
        else:
            self.rules_mode = True
            max_depth_cv = range(1, max_depth_tree + 1)
        
        self.label_data_attributes, self.label_data_attributes_categorical = return_label_data_attributes(log)
        
        for a in self.label_data_attributes_categorical:
            self.attribute_values_label_categorical[a] = list(pm4py.get_event_attribute_values(log, a).keys())

        if verbose:
            print("Resources discovery...")
        self.resources = discover_resources_list(log)
        self.act_to_resources = discover_resources_per_act(log, self.net_transition_labels, self.resources)

        if self.label_data_attributes:
            if verbose:
                print("Data attributes discovery...")
            self.distribution_data_attributes = discover_attributes_distribution(
                log, self.label_data_attributes, self.label_data_attributes_categorical, mode=attribute_mode
            )
        else:
            self.distribution_data_attributes = None

        if verbose:
            print("Feature discovery...")
        df_features = build_df_features(log, self.net, self.initial_marking, self.final_marking, self.act_to_resources, self.net_transition_labels, self.resources, self.label_data_attributes)
        df_features = df_features[(df_features['resource'].isin(self.resources)) | (df_features["resource"].isna())]
        df_features.reset_index(drop=True, inplace=True)
        if enable_multitasking:
            self.max_concurrency = return_max_concurrency(df_features, thr=multitasking_thr)
        else:
            self.max_concurrency = {r: 1 for r in self.resources}

        if verbose:
            if incremental_discovery:
                print("Incremental Transition Probabilities discovery...")
            else:
                print("Transition Probabilities discovery...")
        
        if incremental_discovery:
            self.transition_weights = incremental_transition_weights_learning(
                                                                    df_features, 
                                                                    self.net_transition_labels, 
                                                                    max_depth=max_depth_tree,
                                                                    grace_period=grace_period,                  
                                                                    label_data_attributes=self.label_data_attributes, 
                                                                    label_data_attributes_categorical=self.label_data_attributes_categorical, 
                                                                    values_categorical=self.attribute_values_label_categorical
                                                                )
        else:
            self.transition_weights = discover_weight_transitions(
                                                                    df_features,
                                                                    self.net_transition_labels,
                                                                    max_depths_cv=max_depth_cv,
                                                                    min_samples_leaf_cv=min_samples_leaf_cv,
                                                                    label_data_attributes=self.label_data_attributes,
                                                                    label_data_attributes_categorical=self.label_data_attributes_categorical,
                                                                    values_categorical=self.attribute_values_label_categorical,
                                                                    random_state=random_state
                                                                )
        for t in self.net.transitions:
            if t not in self.transition_weights.keys():
                self.transition_weights[t] = 0

        if verbose:
            if incremental_discovery:
                print("Incremental Resource Weights discovery...")
            else:
                print("Resource Weights discovery...")


        if incremental_discovery:
            self.resource_weights = incremental_resource_weights_learning(
                                                                            df_features,
                                                                            self.act_to_resources,
                                                                            self.resources,
                                                                            self.max_concurrency,
                                                                            max_depth_tree,
                                                                            grace_period,
                                                                            self.label_data_attributes,
                                                                            self.label_data_attributes_categorical,
                                                                            self.attribute_values_label_categorical
                                                                        )
        else:
            self.resource_weights = discover_weight_resources(
                                                                df_features,
                                                                self.act_to_resources,
                                                                self.resources,
                                                                self.max_concurrency,
                                                                max_depth_cv,
                                                                min_samples_leaf_cv,
                                                                self.label_data_attributes,
                                                                self.label_data_attributes_categorical,
                                                                self.attribute_values_label_categorical,
                                                                random_state=random_state
                                                            )

        if verbose:
            print("Calendars discovery...")
        self.calendars = discover_res_calendars(log, self.resources)
        self.arrival_calendar = discover_arrival_calendar(log)

        if verbose:
            if incremental_discovery:
                print("Incremental Execution Time discovery...")
            else:
                print("Execution Time discovery...")

        if incremental_discovery:
            self.execution_time_distributions = incremental_execution_time_learning(    
                                                                                        df_features,
                                                                                        self.net_transition_labels,
                                                                                        self.resources,
                                                                                        self.calendars, 
                                                                                        max_depth=max_depth_tree,
                                                                                        grace_period=grace_period,
                                                                                        label_data_attributes=self.label_data_attributes, 
                                                                                        label_data_attributes_categorical=self.label_data_attributes_categorical, 
                                                                                        values_categorical=self.attribute_values_label_categorical
                                                                                    )
        else:
            self.execution_time_distributions = discover_execution_time_distributions(
                                                                                        df_features,
                                                                                        self.net_transition_labels,
                                                                                        self.resources,
                                                                                        self.calendars,
                                                                                        max_depths=max_depth_cv,
                                                                                        min_samples_leaf_cv=min_samples_leaf_cv,
                                                                                        label_data_attributes=self.label_data_attributes,
                                                                                        label_data_attributes_categorical=self.label_data_attributes_categorical,
                                                                                        values_categorical=self.attribute_values_label_categorical,
                                                                                        random_state=random_state
                                                                                    )
        if verbose:
            if incremental_discovery:
                print("Incremental Waiting Time discovery...")
            else:
                print("Waiting Time discovery...")

        if incremental_discovery:
            self.waiting_time_distributions = incremental_waiting_time_learning(
                                                                                    df_features,
                                                                                    self.net_transition_labels,
                                                                                    self.resources, 
                                                                                    self.calendars, 
                                                                                    self.label_data_attributes, 
                                                                                    self.label_data_attributes_categorical, 
                                                                                    self.attribute_values_label_categorical, 
                                                                                    max_depth=max_depth_tree,
                                                                                    grace_period=grace_period
                                                                                )
        else:
            self.waiting_time_distributions = discover_waiting_time(
                                                                        df_features,
                                                                        self.net_transition_labels,
                                                                        self.resources,
                                                                        self.calendars,
                                                                        self.label_data_attributes,
                                                                        self.label_data_attributes_categorical,
                                                                        self.attribute_values_label_categorical,
                                                                        max_depths=max_depth_cv,
                                                                        min_samples_leaf_cv=min_samples_leaf_cv,
                                                                        random_state=random_state
                                                                    )
        
        if verbose:
            if incremental_discovery:
                print("Incremental Arrival Time discovery...")
            else:
                print("Arrival Time discovery...")
        
        if incremental_discovery:
            self.arrival_time_distribution = incremental_model_arrival_learning(log, self.arrival_calendar, max_depth=max_depth_tree, grace_period=grace_period)
        else:
            self.arrival_time_distribution = discover_arrival_time(log, self.arrival_calendar, max_depths=max_depth_cv, min_samples_leaf_cv=min_samples_leaf_cv, random_state=random_state)


    def _serialize_dist_data_attributes(self):
        d = self.distribution_data_attributes
        if d is None:
            return None
        mode = d['mode']
        data = d['data']
        if mode == 'empirical':
            return {'mode': mode, 'data': {str(list(k)): v for k, v in data.items()}}
        else:
            return {'mode': mode, 'data': data}

    def to_dict(self) ->  dict:

        dict_params = {

            "rules_mode": self.rules_mode,

            "transition_params": {
                "transition_weights": {transition_to_name(t): decision_rules_to_dict(dr) for t, dr in self.transition_weights.items()} # ok
                },

            "resource_params": {
                "resources" : self.resources,
                "max_concurrency": self.max_concurrency,
                "act_to_resources": self.act_to_resources,
                "resource_weights": {
                    r: decision_rules_to_dict(dr) for r, dr in self.resource_weights.items()
                },
                "calendars": {r: convert_calendar_names(cal) for r, cal in self.calendars.items()}
                },

            "arrival_params": {
                "arrival_calendar": convert_calendar_names(self.arrival_calendar),
                "arrival_time_distributions": decision_rules_to_dict(self.arrival_time_distribution)
                },

            "execution_time_params": {
                "execution_time_distributions": {a: decision_rules_to_dict(dr) for a, dr in self.execution_time_distributions.items()} 
                },

            "waiting_time_params": {
                "waiting_time_distributions": {r: decision_rules_to_dict(dr) for r, dr in self.waiting_time_distributions.items()} 
                },

            "data_attribute_params": {
                "label_data_attributes": self.label_data_attributes,
                "label_data_attributes_categorical": self.label_data_attributes_categorical,
                "attribute_values_label_categorical": self.attribute_values_label_categorical,
                "distribution_data_attributes": self._serialize_dist_data_attributes()
                }

        }

        return dict_params

    def to_json(self, path: str = "simulator_params.json"):

        dict_params = self.to_dict()
        with open(path, "w") as json_file:
            json.dump(dict_params, json_file, indent=4)


    def from_dict(self, dict_params):

        if "rules_mode" in dict_params:
            self.rules_mode = dict_params["rules_mode"]
        else:
            # Backward compatibility: infer from old format
            self.rules_mode = "mean_value" not in dict_params["arrival_params"]["arrival_time_distributions"].keys()
        self.label_data_attributes, self.label_data_attributes_categorical = dict_params["data_attribute_params"]["label_data_attributes"], dict_params["data_attribute_params"]["label_data_attributes_categorical"]
        self.attribute_values_label_categorical = dict_params["data_attribute_params"]["attribute_values_label_categorical"]
        raw_dist_attrs = dict_params["data_attribute_params"]["distribution_data_attributes"]
        if raw_dist_attrs is not None:
            if 'mode' in raw_dist_attrs:
                # New format: {'mode': ..., 'data': {...}}
                mode = raw_dist_attrs['mode']
                raw_data = raw_dist_attrs['data']
                if mode == 'empirical':
                    import ast
                    data = {tuple(ast.literal_eval(k)): v for k, v in raw_data.items()}
                else:
                    data = raw_data
                self.distribution_data_attributes = {'mode': mode, 'data': data}
            else:
                # Old format: flat dict {str(tuple): probability} — treat as empirical
                import ast
                data = {tuple(ast.literal_eval(k)): v for k, v in raw_dist_attrs.items()}
                self.distribution_data_attributes = {'mode': 'empirical', 'data': data}
        else:
            self.distribution_data_attributes = None

        self.resources = dict_params["resource_params"]["resources"]
        self.act_to_resources = dict_params["resource_params"]["act_to_resources"]
        raw_rw = dict_params["resource_params"]["resource_weights"]
        self.resource_weights = {}
        for res, value in raw_rw.items():
            if isinstance(value, (int, float)):
                self.resource_weights[res] = float(value)
            else:
                self.resource_weights[res] = dict_to_decrules(value)
        resource_params = dict_params["resource_params"]
        if "max_concurrency" in resource_params:
            self.max_concurrency = {r: int(v) for r, v in resource_params["max_concurrency"].items()}
        else:
            # Old format: list of multitasking resources
            old_mt = set(resource_params.get("multitasking_resource", []))
            self.max_concurrency = {r: (100 if r in old_mt else 1) for r in self.resources}

        self.calendars = {r: convert_calendar_names(cal, to_number=True) for r, cal in dict_params["resource_params"]["calendars"].items()}
        self.arrival_calendar = convert_calendar_names(dict_params["arrival_params"]["arrival_calendar"], to_number=True)

        if self.rules_mode:
            self.transition_weights = {name_to_transition(t_name, self.net): dict_to_decrules(value) for t_name, value in dict_params["transition_params"]["transition_weights"].items()}
            self.execution_time_distributions = {act: dict_to_decrules(value) for act, value in dict_params["execution_time_params"]["execution_time_distributions"].items()}
            self.waiting_time_distributions = {res: dict_to_decrules(value) for res, value in dict_params["waiting_time_params"]["waiting_time_distributions"].items()}
            self.arrival_time_distribution = dict_to_decrules(dict_params["arrival_params"]["arrival_time_distributions"])
        else:
            self.transition_weights = {name_to_transition(t_name, self.net): value for t_name, value in dict_params["transition_params"]["transition_weights"].items()}  
            self.execution_time_distributions = {act: (fromstr_to_scipy(value["dist_name"]), tuple(value["params"]), value["min_value"], value["max_value"], value["mean_value"]) for act, value in dict_params["execution_time_params"]["execution_time_distributions"].items()}
            self.waiting_time_distributions = {res: (fromstr_to_scipy(value["dist_name"]), tuple(value["params"]), value["min_value"], value["max_value"], value["mean_value"]) for res, value in dict_params["waiting_time_params"]["waiting_time_distributions"].items()}
            self.arrival_time_distribution = (fromstr_to_scipy(dict_params["arrival_params"]["arrival_time_distributions"]["dist_name"]), tuple(dict_params["arrival_params"]["arrival_time_distributions"]["params"]), dict_params["arrival_params"]["arrival_time_distributions"]["min_value"], dict_params["arrival_params"]["arrival_time_distributions"]["max_value"], dict_params["arrival_params"]["arrival_time_distributions"]["mean_value"])

    def from_json(self, path: str = "simulator_params.json"):
        
        with open(path, "r") as file:
            dict_params = json.load(file)
        self.from_dict(dict_params)



class SimulatorEngine:

    def __init__(
            self, 
            simulation_parameters: SimulatorParameters
        ):

        self.net = simulation_parameters.net
        self.initial_marking = simulation_parameters.initial_marking
        self.final_marking = simulation_parameters.final_marking
        self.simulation_parameters = simulation_parameters


    def apply(self, n_traces: int = 1, t_start: datetime = None, deterministic_time=False) -> pd.DataFrame:

        if t_start is None:
            t_start = datetime.now()

        event_log = []
        enabled_heap = []
        resource_schedule = {r: [] for r in self.simulation_parameters.resources}
        cases = []
        firing_sequences = {f"case_{i+1}": [] for i in range(n_traces)}

        if not self.simulation_parameters.rules_mode:
            if deterministic_time:
                sampled_arrivals = self.simulation_parameters.arrival_time_distribution[-1]
                sampled_waiting_times = {res : self.simulation_parameters.waiting_time_distributions[res][-1] for res in self.simulation_parameters.resources}
                sampled_execution_times = {act: self.simulation_parameters.execution_time_distributions[act][-1] for act in self.simulation_parameters.net_transition_labels}
            else:
                sampled_arrivals = sampling_from_dist(*self.simulation_parameters.arrival_time_distribution, n_sample=n_traces)
                sampled_waiting_times = {res : sampling_from_dist(*self.simulation_parameters.waiting_time_distributions[res], n_sample=n_traces) for res in self.simulation_parameters.resources}
                sampled_execution_times = {act: sampling_from_dist(*self.simulation_parameters.execution_time_distributions[act], n_sample=n_traces) for act in self.simulation_parameters.net_transition_labels}

        if self.simulation_parameters.label_data_attributes and self.simulation_parameters.distribution_data_attributes is not None:
            dist_data = self.simulation_parameters.distribution_data_attributes
            mode = dist_data['mode']
            data = dist_data['data']
            if mode == 'empirical':
                sampled_tuples = random.choices(
                    list(data.keys()),
                    weights=list(data.values()),
                    k=n_traces
                )
                x_attr_list = [list(t) for t in sampled_tuples]
            else:  # 'distribution'
                x_attr_list = []
                for _ in range(n_traces):
                    attrs = []
                    for a in self.simulation_parameters.label_data_attributes:
                        a_data = data[a]
                        if a_data['type'] == 'categorical':
                            vals = list(a_data['values'].keys())
                            probs = list(a_data['values'].values())
                            attrs.append(random.choices(vals, weights=probs, k=1)[0])
                        else:
                            dist_obj = fromstr_to_scipy(a_data['dist'])
                            params = tuple(a_data['params'])
                            sample = sampling_from_dist(dist_obj, params, a_data['min'], a_data['max'], a_data['mean'], n_sample=1)
                            attrs.append(float(sample[0]))
                    x_attr_list.append(attrs)
        else:
            x_attr_list = [[] for _ in range(n_traces)]

        current_arr_ts = t_start

        for i in range(n_traces):

            trace_attributes = dict()
            if x_attr_list[i]:
                for j, a in enumerate(self.simulation_parameters.label_data_attributes):
                    if a in self.simulation_parameters.label_data_attributes_categorical:
                        for v in self.simulation_parameters.attribute_values_label_categorical[a]:
                            trace_attributes[a+' = '+str(v)] = int(x_attr_list[i][j] == v)
                    else:
                        trace_attributes[a] = x_attr_list[i][j]
                
            else:
                trace_attributes = dict()

            if i > 0:
                if not self.simulation_parameters.rules_mode:
                    if deterministic_time:
                        arrival_delta = sampled_arrivals
                    else:
                        arrival_delta = sampled_arrivals[i]
                else:
                    hour = current_arr_ts.hour
                    weekday = current_arr_ts.weekday()
                    arrival_features = {
                        'hour': hour,
                        'weekday': weekday,
                    }
                    if deterministic_time:
                        arrival_delta = self.simulation_parameters.arrival_time_distribution.apply(arrival_features)
                    else:
                        arrival_delta = self.simulation_parameters.arrival_time_distribution.apply_distribution(arrival_features)
                if arrival_delta == 0:
                    arrival_delta = 1
                current_arr_ts = add_minutes_with_calendar(current_arr_ts, round(arrival_delta), self.simulation_parameters.arrival_calendar)

            case = {
                "case_id": i,
                "marking": self.initial_marking,
                "arrival_time": current_arr_ts,
                "place_token_time": {},
                "enabled": {},
                "history": {t: 0 for t in self.simulation_parameters.net_transition_labels},
                "res_history": {r: 0 for r in self.simulation_parameters.resources},
                "attributes": trace_attributes,
                "last_resource": None,
                "last_activity": None
            }
            for place in self.net.places:
                case["place_token_time"][place] = None
            case["place_token_time"][list(self.initial_marking.keys())[0]] = case["arrival_time"]

            enabled = return_enabled_transitions(self.net, case["marking"])
            for t in enabled:
                input_places = [arc.source for arc in t.in_arcs]
                enabled_time = max(case["place_token_time"][p] for p in input_places)
                case["enabled"][t] = enabled_time

            if case["enabled"]:
                enabled_time_case = min(case["enabled"].values())
                heapq.heappush(enabled_heap, (enabled_time_case, i))

            cases.append(case)

        _zero_last_activity = {'last_activity_' + t_l: 0 for t_l in self.simulation_parameters.net_transition_labels}
        _zero_waiting_act = {'waiting_activity = ' + act_label: 0 for act_label in self.simulation_parameters.net_transition_labels}
        _zero_resource_onehot = {'resource = ' + res: 0 for res in self.simulation_parameters.resources}

        completed_cases = set()
        pbar = tqdm(total=n_traces, desc="Simulating Cases")
        while enabled_heap:
            _, case_id = heapq.heappop(enabled_heap)
            case = cases[case_id]

            if not case["enabled"]:
                continue

            enabled_transitions = list(case["enabled"].keys())

            if not self.simulation_parameters.rules_mode:
                transition_weights = self.simulation_parameters.transition_weights
            else:
                last_activity_features = _zero_last_activity.copy()
                if case["last_activity"]:
                    last_activity_features['last_activity_' + case["last_activity"]] = 1
                transition_weights = compute_transition_weights_from_model(self.simulation_parameters.transition_weights, case["attributes"] | case["history"] | last_activity_features, enabled_transitions)
            chosen_transition = return_fired_transition(transition_weights, enabled_transitions)
            activity = chosen_transition.label
            t_enabled = case["enabled"][chosen_transition]
            firing_sequences[f"case_{case_id+1}"].append(chosen_transition.name)

            if activity is not None:
                enabled_resources_act = self.simulation_parameters.act_to_resources[activity]
                workloads = {r: count_concurrent_events(resource_schedule[r], t_enabled) for r in enabled_resources_act}
                queue_lengths_cand = {r: sum(1 for s, _ in resource_schedule[r] if s > t_enabled) for r in enabled_resources_act}
                enabled_resources = []
                for r in enabled_resources_act:
                    if workloads[r] < self.simulation_parameters.max_concurrency.get(r, 1):
                        enabled_resources.append(r)
                if not enabled_resources:
                    t_enabled_enabled_resources = [resource_schedule[r][-1][-1] if resource_schedule[r] else t_enabled for r in enabled_resources_act]
                    index_res, t_enabled_waited = min(enumerate(t_enabled_enabled_resources), key=lambda x: x[1])
                    resource = enabled_resources_act[index_res]
                else:
                    if not self.simulation_parameters.rules_mode:
                        resource_weights = {r: self.simulation_parameters.resource_weights.get(r, 0.0) for r in enabled_resources}
                    else:
                        resource_weights = compute_resource_weights_from_model(
                            self.simulation_parameters.resource_weights,
                            enabled_resources,
                            case["res_history"] | case["attributes"],
                            workloads=workloads,
                            queue_lengths=queue_lengths_cand,
                        )
                    resource = return_resource(resource_weights, enabled_resources)
                    t_enabled_waited = t_enabled
                r_workload = workloads[resource]
                r_queue_length = sum(1 for s, e in resource_schedule[resource] if s > t_enabled)
                case["res_history"][resource] += 1

                if sum(case["history"].values()) == 0:
                    waiting_time = 0
                else:
                    if not self.simulation_parameters.rules_mode:
                        if deterministic_time:
                            waiting_time = sampled_waiting_times[resource]
                            if not isinstance(waiting_time, (int, float)):
                                waiting_time = 0
                        else:
                            waiting_time = random.choice(sampled_waiting_times[resource])
                    else:
                        waiting_activity_features = _zero_waiting_act.copy()
                        waiting_activity_features['waiting_activity = ' + activity] = 1
                        if deterministic_time:
                            waiting_time = self.simulation_parameters.waiting_time_distributions[resource].apply({'workload': r_workload, 'queue_length': r_queue_length} | case["history"] | case["attributes"] | waiting_activity_features)
                            if not isinstance(waiting_time, (int, float)):
                                waiting_time = 0
                        else:
                            waiting_time = self.simulation_parameters.waiting_time_distributions[resource].apply_distribution({'workload': r_workload, 'queue_length': r_queue_length} | case["history"] | case["attributes"] | waiting_activity_features)

                t_start_exec = add_minutes_with_calendar(t_enabled_waited, round(max(0, waiting_time)), self.simulation_parameters.calendars[resource])

                if not self.simulation_parameters.rules_mode:
                    if deterministic_time:
                        ex_time = sampled_execution_times[activity]
                        if not isinstance(ex_time, (int, float)):
                            ex_time = 0
                    else:
                        ex_time = random.choice(sampled_execution_times[activity])
                else:
                    resource_onehot = _zero_resource_onehot.copy()
                    resource_onehot['resource = ' + resource] = 1
                    if deterministic_time:
                        ex_time = self.simulation_parameters.execution_time_distributions[activity].apply(resource_onehot | case["history"] | case["attributes"])
                        if not isinstance(ex_time, (int, float)):
                            ex_time = 0
                    else:
                        ex_time = self.simulation_parameters.execution_time_distributions[activity].apply_distribution(resource_onehot | case["history"] | case["attributes"])

                
                t_end = add_minutes_with_calendar(t_start_exec, round(ex_time), self.simulation_parameters.calendars[resource])

                event_log.append((case_id, activity, resource, t_enabled, t_start_exec, t_end) + tuple(x_attr_list[case_id]))
                resource_schedule[resource].append((t_start_exec, t_end))
                case["history"][activity] += 1
                case["last_resource"] = resource
                case["last_activity"] = activity
                if len(event_log) % 1000 == 0:
                    min_enabled = min((min(c["enabled"].values()) for c in cases if c["enabled"]), default=t_end)
                    for r in resource_schedule:
                        resource_schedule[r] = [(s, e) for s, e in resource_schedule[r] if e > min_enabled]
            else:
                t_end = t_enabled

            for arc in chosen_transition.out_arcs:
                case["place_token_time"][arc.target] = t_end

            case["enabled"] = {}
            case["marking"] = update_current_marking(case["marking"], chosen_transition)
            if case["marking"] == self.final_marking:
                if case_id not in completed_cases:
                    pbar.update(1)
                    completed_cases.add(case_id)
                continue
            enabled = return_enabled_transitions(self.net, case["marking"])
            for t in enabled:
                input_places = [arc.source for arc in t.in_arcs]
                enabled_time = max(case["place_token_time"][p] for p in input_places)
                case["enabled"][t] = enabled_time

            if case["enabled"]:
                next_enabled_time = min(case["enabled"].values())
                heapq.heappush(enabled_heap, (next_enabled_time, case_id))

        pbar.close()
        df_log = pd.DataFrame(event_log, columns=["case:concept:name", "concept:name", "org:resource", "enabled:timestamp", "start:timestamp", "time:timestamp"] + self.simulation_parameters.label_data_attributes)
        df_log["case:concept:name"] = df_log["case:concept:name"].apply(lambda x: f"case_{x+1}")
        df_log.sort_values(by=["start:timestamp", "time:timestamp"], inplace=True)
        df_log.reset_index(drop=True, inplace=True)
        df_log.attrs["prosit_firing_sequences"] = firing_sequences

        return df_log