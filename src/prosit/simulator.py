import random
import pandas as pd
import math
import heapq
from datetime import datetime
from tqdm import tqdm

import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.log.obj import EventLog

from cortado_core.lca_approach import add_trace_to_pt_language

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
            final_marking: Marking,
            grace_period: int = 1000,
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

        self.online_discovery = None
        self.grace_period = grace_period


    def discover_from_eventlog(
            self, 
            log: EventLog, 
            max_depth_tree: int = 3,
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
            print("Transition probabilities discovery...")
        self.transition_weights = discover_weight_transitions(
                                                                df_features, 
                                                                self.net_transition_labels, 
                                                                max_depths_cv=max_depth_cv,                  
                                                                label_data_attributes=self.label_data_attributes, 
                                                                label_data_attributes_categorical=self.label_data_attributes_categorical, 
                                                                values_categorical=self.attribute_values_label_categorical
                                                            )

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
            print("Execution Time discovery...")
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
            print("Waiting Time discovery...")
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
            print("Arrival Time discovery...")
        self.arrival_time_distribution = discover_arrival_time(log, self.arrival_calendar, max_depths=max_depth_cv)
        

    def _firststep_incremental_discovery(
            self, 
            log: EventLog,
            mode_act_history: bool = True,
            verbose: bool = True
        ):

        self.mode_act_history = mode_act_history
        self.label_data_attributes, self.label_data_attributes_categorical = return_label_data_attributes(log)
        
        for a in self.label_data_attributes_categorical:
            self.attribute_values_label_categorical[a] = list(pm4py.get_event_attribute_values(log, a).keys())

        if self.label_data_attributes:
            if verbose:
                print("Data attributes discovery...")
            self.distribution_data_attributes = discover_attributes_distribution(log, self.label_data_attributes)

        if verbose:
            print("Transition probabilities discovery...")
        self.transition_weights = incremental_transition_weights_learning(
                                                                            None,
                                                                            log,
                                                                            self.net, 
                                                                            self.initial_marking, 
                                                                            self.final_marking,
                                                                            self.net_transition_labels,
                                                                            self.label_data_attributes,
                                                                            self.label_data_attributes_categorical,
                                                                            self.attribute_values_label_categorical,
                                                                            self.mode_act_history,
                                                                            grace_period=self.grace_period
                                                                        )

        if verbose:
            print("Calendars discovery...")
        self.calendars = discover_res_calendars(log)
        self.arrival_calendar = discover_arrival_calendar(log)

        if verbose:
            print("Resources discovery...")
        self.resources = list(pm4py.get_event_attribute_values(log, 'org:resource').keys())
        self.act_resource_prob = discover_resource_acts_prob(log)


        if verbose:
            print("Execution Time discovery...")
        self.execution_time_distributions, self.min_et, self.max_et = incremental_execution_time_learning(
                                                                                    None,
                                                                                    log, 
                                                                                    self.net_transition_labels, 
                                                                                    self.calendars,
                                                                                    self.label_data_attributes, 
                                                                                    self.label_data_attributes_categorical, 
                                                                                    self.attribute_values_label_categorical, 
                                                                                    self.mode_act_history,
                                                                                    grace_period=self.grace_period
                                                                                )
        if verbose:
            print("Arrival discovery...")
        self.arrival_time_distribution, self.min_at, self.max_at = incremental_model_arrival_learning(None, log, self.arrival_calendar, grace_period=self.grace_period)

        if verbose:
            print("Waiting Time discovery...")
        self.waiting_time_distributions, self.min_wt, self.max_wt = incremental_waiting_time_learning(
                                                                                None,
                                                                                log, 
                                                                                self.calendars, 
                                                                                self.label_data_attributes, 
                                                                                self.label_data_attributes_categorical, 
                                                                                self.attribute_values_label_categorical,
                                                                                grace_period=self.grace_period
                                                                            )




    def incremental_discovery(self,
            log: EventLog,
            log_completed_traces: EventLog = None,
            verbose: bool = True,
            ipmd: bool = True
        ):

        if not log_completed_traces:
            self.online_discovery = True
            self._first_incremental = True
            self.prev_caseids = list(set(pm4py.convert_to_dataframe(log)["case:concept:name"]))
            self.prev_log = log
            self._firststep_incremental_discovery(log, verbose=verbose)

        else:
            df_new_log = pm4py.convert_to_dataframe(log)
            df_arrival = df_new_log[~df_new_log["case:concept:name"].isin(self.prev_caseids)]
            log_arrivals = pm4py.convert_to_event_log(df_arrival)

            if ipmd:
                if verbose:
                    print("Process Model Incremental Discovery...")
                pt = pm4py.convert_to_process_tree(self.net, self.initial_marking, self.final_marking)        
                for i in range(len(log)):
                    try:
                        pt = add_trace_to_pt_language(pt, self.prev_log, log[i])
                    except:
                        continue

                self.net, self.initial_marking, self.final_marking = pm4py.convert_to_petri_net(pt)
                # self.net, self.initial_marking, self.final_marking = pm4py.discover_petri_net_inductive(log_completed_traces)
            self.net_transition_labels = list(set([t.label for t in self.net.transitions if t.label]))

            for a in self.label_data_attributes_categorical:
                values_a = list(pm4py.get_event_attribute_values(log, a).keys())
                for v in values_a:
                    if v not in self.attribute_values_label_categorical[a]:
                        self.attribute_values_label_categorical[a].append(v)

            if self.label_data_attributes:
                
                if verbose:
                    print("Data attributes discovery...")
                self.distribution_data_attributes = discover_attributes_distribution(EventLog(list(log_completed_traces)+list(log_arrivals)), self.label_data_attributes)

            if verbose:
                print("Transition probabilities discovery...")

            self.transition_weights = incremental_transition_weights_learning(
                                                                                None,
                                                                                log_completed_traces,
                                                                                self.net, 
                                                                                self.initial_marking, 
                                                                                self.final_marking,
                                                                                self.net_transition_labels,
                                                                                self.label_data_attributes,
                                                                                self.label_data_attributes_categorical,
                                                                                self.attribute_values_label_categorical,
                                                                                self.mode_act_history,
                                                                                grace_period=self.grace_period
                                                                            )
            
            if verbose:
                print("Calendars discovery...")
            calendars = discover_res_calendars(log)

            for r in calendars.keys():
                self.calendars[r] = calendars[r]

            self.arrival_calendar = discover_arrival_calendar(log_completed_traces)

            if verbose:
                print("Resources discovery...")
            self.resources = list(pm4py.get_event_attribute_values(log, 'org:resource').keys())
            act_resource_prob = discover_resource_acts_prob(log)
            for a in act_resource_prob.keys():
                if a not in self.act_resource_prob.keys():
                    self.act_resource_prob[a] = act_resource_prob[a]
                else:
                    for r in act_resource_prob[a].keys():
                        self.act_resource_prob[a][r] = act_resource_prob[a][r]
                    for r in self.act_resource_prob[a].keys():
                        if r not in act_resource_prob[a].keys():
                            self.act_resource_prob[a][r] = 0

            if verbose:
                print("Execution Time discovery...")
            self.execution_time_distributions, min_et, max_et = incremental_execution_time_learning(
                                                                                        self.execution_time_distributions,
                                                                                        log, 
                                                                                        self.net_transition_labels, 
                                                                                        self.calendars,
                                                                                        self.label_data_attributes, 
                                                                                        self.label_data_attributes_categorical, 
                                                                                        self.attribute_values_label_categorical, 
                                                                                        self.mode_act_history,
                                                                                        grace_period=self.grace_period
                                                                                    )
            
            if verbose:
                print("Arrival Time discovery...")
            self.arrival_time_distribution, self.min_at, self.max_at = incremental_model_arrival_learning(self.arrival_time_distribution, log_arrivals, self.arrival_calendar, grace_period=self.grace_period)
            
            if verbose:
                print("Waiting Time discovery...")
            self.waiting_time_distributions, min_wt, max_wt = incremental_waiting_time_learning(
                                                                                    self.waiting_time_distributions,
                                                                                    log, 
                                                                                    self.calendars, 
                                                                                    self.label_data_attributes, 
                                                                                    self.label_data_attributes_categorical, 
                                                                                    self.attribute_values_label_categorical,
                                                                                    grace_period=self.grace_period
                                                                                )
            for r in min_wt.keys():
                self.min_wt[r] = min_wt[r]
                self.max_wt[r] = max_wt[r]

            for a in min_et.keys():
                self.min_et[a] = min_et[a]
                self.max_et[a] = max_et[a]

            self.prev_log = log_completed_traces
            self._first_incremental = False
            self.prev_caseids.extend(list(set(df_arrival["case:concept:name"])))



class SimulatorEngine:

    def __init__(
            self, 
            simulation_parameters: SimulatorParameters
        ):

        self.net = simulation_parameters.net
        self.initial_marking = simulation_parameters.initial_marking
        self.final_marking = simulation_parameters.final_marking
        self.simulation_parameters = simulation_parameters


    def apply(self, n_traces: int = 1, t_start: datetime = datetime.now()) -> pd.DataFrame:

        event_log = []
        enabled_heap = []
        resource_schedule = {r: [] for r in self.simulation_parameters.resources}
        cases = []

        if not self.simulation_parameters.rules_mode:
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
                arrival_delta = sampled_arrivals[i]
            else:
                if self.simulation_parameters.online_discovery:
                    arrival_pred = self.simulation_parameters.arrival_time_distribution.debug_one({'hour': current_arr_ts.hour,'weekday': current_arr_ts.weekday()})
                    if not arrival_pred:
                        arrival_delta = self.simulation_parameters.arrival_time_distribution.predict_one({'hour': current_arr_ts.hour,'weekday': current_arr_ts.weekday()})
                    else:
                        pred_split = arrival_pred.split("\n")[-2].split(" | ")
                        mean = float(pred_split[0][6:].replace(",", ""))
                        var = float(pred_split[1][5:].replace(",", ""))
                        arrival_delta = random.normalvariate(mean, math.sqrt(var))
                        if arrival_delta < self.simulation_parameters.min_at:
                            arrival_delta = mean
                        elif arrival_delta > self.simulation_parameters.max_at:
                            arrival_delta = mean
                else:   
                    arrival_delta = self.simulation_parameters.arrival_time_distribution.apply_distribution({'hour': current_arr_ts.hour,'weekday': current_arr_ts.weekday()})
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
                        waiting_time = random.choice(sampled_waiting_times[resource])
                    else:
                        if self.simulation_parameters.online_discovery:
                            waiting_time_pred = self.simulation_parameters.waiting_time_distributions[resource].debug_one({'workload': r_workload} | case["history"] | case["attributes"])
                            if not waiting_time_pred:
                                waiting_time = self.simulation_parameters.waiting_time_distributions[resource].predict_one({'workload': r_workload} | case["history"] | case["attributes"])
                            else:
                                pred_split = waiting_time_pred.split("\n")[-2].split(" | ")
                                mean = float(pred_split[0][6:].replace(",", ""))
                                var = float(pred_split[1][5:].replace(",", ""))
                                waiting_time = random.normalvariate(mean, math.sqrt(var))
                                if waiting_time < self.simulation_parameters.min_wt[resource]:
                                    waiting_time = mean
                                elif waiting_time > self.simulation_parameters.max_wt[resource]:
                                    waiting_time = mean
                        else:
                            waiting_time = self.simulation_parameters.waiting_time_distributions[resource].apply_distribution({'workload': r_workload} | case["history"] | case["attributes"])

                if not self.simulation_parameters.rules_mode:
                    ex_time = random.choice(sampled_execution_times[activity])
                else:
                    if self.simulation_parameters.online_discovery:
                        ex_time_pred = self.simulation_parameters.execution_time_distributions[activity].debug_one({'resource = '+res: (res == resource)*1 for res in self.simulation_parameters.resources} | case["history"] | case["attributes"])
                        if not ex_time_pred:
                            ex_time = self.simulation_parameters.execution_time_distributions[activity].predict_one({'resource = '+res: (res == resource)*1 for res in self.simulation_parameters.resources} | case["history"] | case["attributes"])
                        else:
                            pred_split = ex_time_pred.split("\n")[-2].split(" | ")
                            mean = float(pred_split[0][6:].replace(",", ""))
                            var = float(pred_split[1][5:].replace(",", ""))
                            ex_time = random.normalvariate(mean, math.sqrt(var))
                            if ex_time < self.simulation_parameters.min_et[activity]:
                                ex_time = mean
                            elif ex_time > self.simulation_parameters.max_et[activity]:
                                ex_time = mean         
                    else:
                        ex_time = self.simulation_parameters.execution_time_distributions[activity].apply_distribution({'resource = '+res: (res == resource)*1 for res in self.simulation_parameters.resources} | case["history"] | case["attributes"])

                
                t_start_exec = add_minutes_with_calendar(t_enabled, int(waiting_time), self.simulation_parameters.calendars[resource])
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