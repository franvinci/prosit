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
from prosit.discovery.resource_discovery import discover_resources_list, discover_resource_acts_prob
from prosit.discovery.data_discovery import discover_attributes_distribution, return_label_data_attributes
from prosit.discovery.online_discovery.cf_discovery import incremental_transition_weights_learning
from prosit.discovery.online_discovery.time_discovery import incremental_execution_time_learning, incremental_model_arrival_learning, incremental_waiting_time_learning
from prosit.utils.common_utils import (
    return_enabled_transitions, 
    update_current_marking, 
    return_fired_transition, 
    count_concurrent_events,
    compute_transition_weights_from_model, 
    add_minutes_with_calendar,
    build_df_features
    )
from prosit.utils.distribution_utils import sampling_from_dist


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
        self.act_resource_prob: dict = {act: {"auto": 1} for act in self.net_transition_labels}
        self.calendars: dict = {'auto': {wd: {h: True for h in range(24)} for wd in range(7)}}
        self.arrival_calendar: dict = {wd: {h: True for h in range(24)} for wd in range(7)}

        self.execution_time_distributions: dict = {a: ('fixed', 1, 1, 1, 1) for a in self.net_transition_labels}
        self.arrival_time_distribution: tuple = ('fixed', 1, 1, 1, 1)
        self.waiting_time_distributions: dict = {'auto': ('fixed', 1, 1, 1, 1)}

        self.rules_mode: bool = False


    def discover_from_eventlog(
            self, 
            log: EventLog, 
            max_depth_tree: int = 3,
            incremental_discovery: bool = False,
            grace_period: int = 1000,
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

        if self.label_data_attributes:
            if verbose:
                print("Data attributes discovery...")
            self.distribution_data_attributes = discover_attributes_distribution(log, self.label_data_attributes)

        if verbose:
            print("Feature discovery...")
        df_features = build_df_features(log, self.net, self.initial_marking, self.final_marking, self.net_transition_labels, self.label_data_attributes)

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
                                                                    label_data_attributes=self.label_data_attributes, 
                                                                    label_data_attributes_categorical=self.label_data_attributes_categorical, 
                                                                    values_categorical=self.attribute_values_label_categorical
                                                                )
        for t in self.net.transitions:
            if t not in self.transition_weights.keys():
                self.transition_weights[t] = 0

        if verbose:
            print("Resources discovery...")
        self.resources = discover_resources_list(log)
        self.act_resource_prob = discover_resource_acts_prob(log, self.resources)
        df_features = df_features[df_features['resource'].isin(self.resources)]
        df_features.reset_index(drop=True, inplace=True)

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
                                                                                        label_data_attributes=self.label_data_attributes, 
                                                                                        label_data_attributes_categorical=self.label_data_attributes_categorical, 
                                                                                        values_categorical=self.attribute_values_label_categorical
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
                                                                        max_depths=max_depth_cv
                                                                    )
        
        if verbose:
            if incremental_discovery:
                print("Incremental Arrival Time discovery...")
            else:
                print("Arrival Time discovery...")
        
        if incremental_discovery:
            self.arrival_time_distribution = incremental_model_arrival_learning(log, self.arrival_calendar, max_depth=max_depth_tree, grace_period=grace_period)
        else:
            self.arrival_time_distribution = discover_arrival_time(log, self.arrival_calendar, max_depths=max_depth_cv)



class SimulatorEngine:

    def __init__(
            self, 
            simulation_parameters: SimulatorParameters
        ):

        self.net = simulation_parameters.net
        self.initial_marking = simulation_parameters.initial_marking
        self.final_marking = simulation_parameters.final_marking
        self.simulation_parameters = simulation_parameters


    def apply(self, n_traces: int = 1, t_start: datetime = datetime.now(), deterministic_time=False) -> pd.DataFrame:

        event_log = []
        enabled_heap = []
        resource_schedule = {r: [] for r in self.simulation_parameters.resources}
        cases = []

        if not self.simulation_parameters.rules_mode:
            if deterministic_time:
                sampled_arrivals = self.simulation_parameters.arrival_time_distribution[-1]
                sampled_waiting_times = {res : self.simulation_parameters.waiting_time_distributions[res][-1] for res in self.simulation_parameters.resources}
                sampled_execution_times = {act: self.simulation_parameters.execution_time_distributions[act][-1] for act in self.simulation_parameters.net_transition_labels}
            else:
                sampled_arrivals = sampling_from_dist(*self.simulation_parameters.arrival_time_distribution, n_sample=n_traces)
                sampled_waiting_times = {res : sampling_from_dist(*self.simulation_parameters.waiting_time_distributions[res], n_sample=n_traces) for res in self.simulation_parameters.resources}
                sampled_execution_times = {act: sampling_from_dist(*self.simulation_parameters.execution_time_distributions[act], n_sample=n_traces) for act in self.simulation_parameters.net_transition_labels}

        if self.simulation_parameters.label_data_attributes:
            x_attr_list = random.choices(
                list(self.simulation_parameters.distribution_data_attributes.keys()), 
                weights=list(self.simulation_parameters.distribution_data_attributes.values()),
                k = n_traces
                )
            x_attr_list = [list(attr) for attr in x_attr_list]
        else:
            x_attr_list = [[]]*n_traces

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

            if not self.simulation_parameters.rules_mode:
                if deterministic_time:
                    arrival_delta = sampled_arrivals
                else:
                    arrival_delta = sampled_arrivals[i]
            else:
                if deterministic_time:
                    arrival_delta = self.simulation_parameters.arrival_time_distribution.apply({'hour': current_arr_ts.hour,'weekday': current_arr_ts.weekday()})
                else:
                    arrival_delta = self.simulation_parameters.arrival_time_distribution.apply_distribution({'hour': current_arr_ts.hour,'weekday': current_arr_ts.weekday()})
            if arrival_delta == 0:
                arrival_delta = 1
            current_arr_ts = add_minutes_with_calendar(current_arr_ts, int(arrival_delta), self.simulation_parameters.arrival_calendar)

            case = {
                "case_id": i,
                "marking": self.initial_marking,
                "arrival_time": current_arr_ts,
                "place_token_time": {},
                "enabled": {},
                "history": {t: 0 for t in self.simulation_parameters.net_transition_labels},
                "attributes": trace_attributes
            }
            for place in self.net.places:
                case["place_token_time"][place] = None
            case["place_token_time"][list(self.initial_marking.keys())[0]] = case["arrival_time"]

            enabled = return_enabled_transitions(self.net, case["marking"])
            for t in enabled:
                input_places = [arc.source for arc in self.net.arcs if arc.target == t]
                enabled_time = max(case["place_token_time"][p] for p in input_places)
                case["enabled"][t] = enabled_time

            if case["enabled"]:
                enabled_time_case = min(case["enabled"].values())
                heapq.heappush(enabled_heap, (enabled_time_case, i))

            cases.append(case)

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
                transition_weights = compute_transition_weights_from_model(self.simulation_parameters.transition_weights, case["attributes"] | case["history"])
            chosen_transition = return_fired_transition(transition_weights, enabled_transitions)
            activity = chosen_transition.label
            t_enabled = case["enabled"][chosen_transition]

            if activity is not None:
                resources = list(self.simulation_parameters.act_resource_prob[activity].keys())
                resource_weights = list(self.simulation_parameters.act_resource_prob[activity].values())
                resource = random.choices(resources, weights=resource_weights, k=1)[0]
                r_workload = count_concurrent_events(resource_schedule[resource], t_enabled)
                
                if sum(case["history"].values()) == 0:
                    waiting_time = 0
                else:
                    if not self.simulation_parameters.rules_mode:
                        if deterministic_time:
                            waiting_time = sampled_waiting_times[resource]
                            try:
                                int(waiting_time)
                            except:
                                waiting_time = 0
                        else:
                            waiting_time = random.choice(sampled_waiting_times[resource])
                    else:
                        if deterministic_time:
                            waiting_time = self.simulation_parameters.waiting_time_distributions[resource].apply({'workload': r_workload} | case["history"] | case["attributes"])
                            try:
                                int(waiting_time)
                            except:
                                waiting_time = 0
                        else:
                            waiting_time = self.simulation_parameters.waiting_time_distributions[resource].apply_distribution({'workload': r_workload} | case["history"] | case["attributes"])

                t_start_exec = add_minutes_with_calendar(t_enabled, int(waiting_time), self.simulation_parameters.calendars[resource])

                if not self.simulation_parameters.rules_mode:
                    if deterministic_time:
                        ex_time = sampled_execution_times[activity]
                        try:
                            int(ex_time)
                        except:
                            ex_time = 0
                    else:
                        ex_time = random.choice(sampled_execution_times[activity])
                else:    
                    if deterministic_time:
                        ex_time = self.simulation_parameters.execution_time_distributions[activity].apply({'resource = '+res: (res == resource)*1 for res in self.simulation_parameters.resources} | case["history"] | case["attributes"])
                        try:
                            int(ex_time)
                        except: 
                            ex_time = 0
                    else:
                        ex_time = self.simulation_parameters.execution_time_distributions[activity].apply_distribution({'resource = '+res: (res == resource)*1 for res in self.simulation_parameters.resources} | case["history"] | case["attributes"])

                
                t_end = add_minutes_with_calendar(t_start_exec, int(ex_time), self.simulation_parameters.calendars[resource])

                event_log.append((case_id, activity, resource, t_enabled, t_start_exec, t_end) + tuple(x_attr_list[case_id]))
                resource_schedule[resource].append((t_start_exec, t_end))
                case["history"][activity] += 1
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
                input_places = [arc.source for arc in self.net.arcs if arc.target == t]
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

        return df_log