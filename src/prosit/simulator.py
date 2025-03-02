import random
from tqdm import tqdm
import datetime
import pandas as pd
import multiprocessing as mp
import math

import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.log.obj import EventLog

from cortado_core.lca_approach import add_trace_to_pt_language

from prosit.discovery.cf_discovery import discover_weight_transitions
from prosit.discovery.time_discovery import discover_execution_time_distributions, discover_arrival_time, discover_waiting_time
from prosit.discovery.calendar_discovery import discover_res_calendars, discover_arrival_calendar
from prosit.discovery.resource_discovery import discover_resource_acts_prob
from prosit.discovery.data_discovery import discover_attributes_distribution, return_label_data_attributes
from prosit.discovery.online_discovery.cf_discovery import incremental_transition_weights_learning
from prosit.discovery.online_discovery.time_discovery import incremental_execution_time_learning, incremental_model_arrival_learning, incremental_waiting_time_learning
from prosit.utils.common_utils import (
    return_enabled_transitions, 
    update_markings, 
    return_fired_transition, 
    compute_transition_weights_from_model, 
    add_minutes_with_calendar
    )



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

        self.mode_act_history: bool = False
        self.label_data_attributes: list = []
        self.label_data_attributes_categorical: list = []
        self.attribute_values_label_categorical: dict = dict()
        self.distribution_data_attributes: dict = dict()

        self.transition_weights: dict = {t: 1 for t in list(self.net.transitions)}
        self.resources: list = ['res']
        self.calendars: dict = {'res': {wd: {h: True for h in range(24)} for wd in range(7)}}
        self.arrival_calendar: dict = {wd: {h: True for h in range(24)} for wd in range(7)}

        self.execution_time_distributions: dict = {a: ('fixed', 1, 1, 1) for a in self.net_transition_labels}
        self.arrival_time_distribution: tuple = ('fixed', 1, 1, 1)
        self.waiting_time_distributions: dict = {'res': ('fixed', 1, 1, 1)}

        self.online_discovery = None
        self.grace_period = grace_period


    def discover_from_eventlog(
            self, 
            log: EventLog, 
            max_depths_cv: list = range(1, 6),
            mode_act_history: bool = True,
            verbose: bool = True
        ):
        """ Discovery Parameters from event log data """
        
        self.mode_act_history = mode_act_history
        self.label_data_attributes, self.label_data_attributes_categorical = return_label_data_attributes(log)
        
        for a in self.label_data_attributes_categorical:
            self.attribute_values_label_categorical[a] = list(pm4py.get_event_attribute_values(log, a).keys())

        if self.label_data_attributes:
            if verbose:
                print("Data attributes discovery...")
            self.distribution_data_attributes = discover_attributes_distribution(log, self.label_data_attributes)

        self.transition_weights = discover_weight_transitions(
                                                                log, 
                                                                self.net, self.initial_marking, self.final_marking, 
                                                                self.net_transition_labels, 
                                                                max_depths_cv=max_depths_cv,                  
                                                                label_data_attributes=self.label_data_attributes, 
                                                                label_data_attributes_categorical=self.label_data_attributes_categorical, 
                                                                values_categorical=self.attribute_values_label_categorical,
                                                                mode_act_history=mode_act_history,
                                                                verbose=verbose
                                                            )

        if verbose:
            print("Calendars discovery...")
        self.calendars = discover_res_calendars(log)
        self.arrival_calendar = discover_arrival_calendar(log)

        if verbose:
            print("Resources discovery...")
        self.resources = list(pm4py.get_event_attribute_values(log, 'org:resource').keys())
        self.act_resource_prob = discover_resource_acts_prob(log)


        self.execution_time_distributions = discover_execution_time_distributions(
                                                                                    log, self.net_transition_labels, self.calendars, 
                                                                                    max_depths=max_depths_cv,
                                                                                    label_data_attributes=self.label_data_attributes, 
                                                                                    label_data_attributes_categorical=self.label_data_attributes_categorical, 
                                                                                    values_categorical=self.attribute_values_label_categorical,
                                                                                    mode_act_history=mode_act_history
                                                                                )
        
        self.arrival_time_distribution = discover_arrival_time(log, self.arrival_calendar, max_depths=max_depths_cv)

        self.waiting_time_distributions = discover_waiting_time(
                                                                    log, 
                                                                    self.calendars, 
                                                                    self.label_data_attributes, 
                                                                    self.label_data_attributes_categorical, 
                                                                    self.attribute_values_label_categorical, 
                                                                    max_depths=max_depths_cv
                                                                )
        

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
        self.case_id = 1
        self.current_timestamp = None


    def _generate_activities(self, x_attr=[]):
        """ Generate Activity Traces """

        trace = []
        trace_attributes = dict()
        for i, l in enumerate(self.simulation_parameters.label_data_attributes):
            trace_attributes[l] = x_attr[i]

        tkns = list(self.initial_marking)
        enabled_transitions = return_enabled_transitions(self.net, tkns)

        if self.simulation_parameters.mode_act_history:
            x_history = {t_l: 0 for t_l in self.simulation_parameters.net_transition_labels}
            X = x_attr + list(x_history.values())
        else:
            X = x_attr

        if not X:
            transition_weights = self.simulation_parameters.transition_weights
        else:
            dict_x = dict(zip(self.simulation_parameters.label_data_attributes + self.simulation_parameters.net_transition_labels, X))
            for a in self.simulation_parameters.label_data_attributes_categorical:
                for v in self.simulation_parameters.attribute_values_label_categorical[a]:
                    dict_x[a+' = '+str(v)] = (dict_x[a] == v)*1
                del dict_x[a]
            transition_weights = compute_transition_weights_from_model(self.simulation_parameters.transition_weights, dict_x)
        
        t_fired = return_fired_transition(transition_weights, enabled_transitions)
        if t_fired.label:
            trace.append(t_fired.label)

        tkns = update_markings(tkns, t_fired)
        while set(tkns) != set(self.final_marking):
            if t_fired.label:
                if self.simulation_parameters.mode_act_history:
                    dict_x[t_fired.label] += 1
            transition_weights = compute_transition_weights_from_model(self.simulation_parameters.transition_weights, dict_x)
            enabled_transitions = return_enabled_transitions(self.net, tkns)
            t_fired = return_fired_transition(transition_weights, enabled_transitions)
            if t_fired.label:
                trace.append(t_fired.label)
            tkns = update_markings(tkns, t_fired)

        return trace, trace_attributes


    def _generate_events(self, curr_traces_acts, start_ts_simulation, curr_trace_attributes=[]) -> pd.DataFrame:

        n_sim = len(curr_traces_acts)

        current_arr_ts = start_ts_simulation

        arrival_timestamps = dict()
        for id in range(self.case_id, self.case_id+n_sim):
            arrival_timestamps[f'case_{id}'] = current_arr_ts.timestamp()
            if self.simulation_parameters.online_discovery:
                arrival_pred = self.simulation_parameters.arrival_time_distribution.debug_one({
                                                                                                        'hour': current_arr_ts.hour,
                                                                                                        'weekday': current_arr_ts.weekday()
                                                                                                    })
                if not arrival_pred:
                    arrival_delta = self.simulation_parameters.arrival_time_distribution.predict_one({
                                                                                                        'hour': current_arr_ts.hour,
                                                                                                        'weekday': current_arr_ts.weekday()
                                                                                                    })
                else:
                    pred_split = arrival_pred.split("\n")[-2].split(" | ")
                    mean = float(pred_split[0][6:].replace(",", ""))
                    var = float(pred_split[1][5:].replace(",", ""))
                    # arrival_delta = random.expovariate(1/arrival_pred)
                    arrival_delta = random.normalvariate(mean, math.sqrt(var))
                    if arrival_delta < self.simulation_parameters.min_at:
                        arrival_delta = mean
                    elif arrival_delta > self.simulation_parameters.max_at:
                        arrival_delta = mean
            else:   
                arrival_delta = self.simulation_parameters.arrival_time_distribution.apply_distribution({
                                                                                                            'hour': current_arr_ts.hour,
                                                                                                            'weekday': current_arr_ts.weekday()
                                                                                                        })
            current_arr_ts = add_minutes_with_calendar(current_arr_ts, int(arrival_delta), self.simulation_parameters.arrival_calendar)

        flag_active_cases = {f'case_{id}': False for id in range(self.case_id, self.case_id+n_sim)}

        if self.simulation_parameters.mode_act_history:
            hystory_active_traces = {
                                        f'case_{id}': {l: 0 for l in self.simulation_parameters.net_transition_labels} 
                                        for id in range(self.case_id, self.case_id+n_sim)
                                    }
        events = []

        curr_trace_attribute_features = dict()
        if curr_trace_attributes:
            for case_id in curr_trace_attributes.keys():
                curr_trace_attribute_features[case_id] = dict()
                for a in self.simulation_parameters.label_data_attributes:
                    if a in self.simulation_parameters.label_data_attributes_categorical:
                        for v in self.simulation_parameters.attribute_values_label_categorical[a]:
                            curr_trace_attribute_features[case_id][a+' = '+str(v)] = (curr_trace_attributes[case_id][a] == v)*1
                    else:
                        curr_trace_attribute_features[case_id][a] = curr_trace_attributes[case_id][a]

        while len(curr_traces_acts) > 0:

            curr_case_id = min(arrival_timestamps, key=arrival_timestamps.get)
            current_ts = arrival_timestamps[curr_case_id]

            if len(curr_traces_acts[curr_case_id]) < 1:
                del curr_traces_acts[curr_case_id]
                if self.simulation_parameters.mode_act_history:
                    del hystory_active_traces[curr_case_id]
                del arrival_timestamps[curr_case_id]
                del curr_trace_attribute_features[curr_case_id]
                del curr_trace_attributes[curr_case_id]
                continue

            curr_act = curr_traces_acts[curr_case_id].pop(0)

            curr_res = random.choices(
                                        list(self.simulation_parameters.act_resource_prob[curr_act].keys()), 
                                        weights=list(self.simulation_parameters.act_resource_prob[curr_act].values())
                                    )[0]

            n_active = 0
            if len(events) > 0:
                for e in events:
                    if e['org:resource'] != curr_res:
                        continue
                    if e['time:timestamp'] > current_ts and e['start:timestamp'] < current_ts:
                        n_active += 1

            current_ts_datetime = datetime.datetime.fromtimestamp(current_ts)

            if not flag_active_cases[curr_case_id]:
                waiting_time = 0
                flag_active_cases[curr_case_id] = True
            else:
                n_running_events = n_active
                if curr_res in self.simulation_parameters.waiting_time_distributions.keys():
                    if sum(hystory_active_traces[curr_case_id].values()) == 0:
                        waiting_time = 0
                    else:
                        if self.simulation_parameters.online_discovery:
                            waiting_time_pred = self.simulation_parameters.waiting_time_distributions[curr_res].debug_one(
                                                                                                                            {
                                                                                                                                'hour': current_ts_datetime.hour, 
                                                                                                                                'weekday': current_ts_datetime.weekday(), 
                                                                                                                                'n. running events': n_running_events
                                                                                                                            } | curr_trace_attribute_features[curr_case_id]
                                                                                                                        )
                            if not waiting_time_pred:
                                waiting_time = self.simulation_parameters.waiting_time_distributions[curr_res].predict_one(
                                                                                                                            {
                                                                                                                                'hour': current_ts_datetime.hour, 
                                                                                                                                'weekday': current_ts_datetime.weekday(), 
                                                                                                                                'n. running events': n_running_events
                                                                                                                            } | curr_trace_attribute_features[curr_case_id]
                                                                                                                        )
                            else:
                                pred_split = waiting_time_pred.split("\n")[-2].split(" | ")
                                mean = float(pred_split[0][6:].replace(",", ""))
                                var = float(pred_split[1][5:].replace(",", ""))
                                waiting_time = random.normalvariate(mean, math.sqrt(var))
                                if waiting_time < self.simulation_parameters.min_wt[curr_res]:
                                    waiting_time = mean
                                elif waiting_time > self.simulation_parameters.max_wt[curr_res]:
                                    waiting_time = mean
                        else:
                            waiting_time = self.simulation_parameters.waiting_time_distributions[curr_res].apply_distribution(
                                                                                                                                {
                                                                                                                                    'hour': current_ts_datetime.hour, 
                                                                                                                                    'weekday': current_ts_datetime.weekday(), 
                                                                                                                                    'n. running events': n_running_events
                                                                                                                                } | curr_trace_attribute_features[curr_case_id]
                                                                                                                            )
                else:
                    waiting_time = 0

            start_ts_datetime = add_minutes_with_calendar(current_ts_datetime, int(waiting_time), self.simulation_parameters.calendars[curr_res])
            start_ts = start_ts_datetime.timestamp()


            if self.simulation_parameters.mode_act_history:
                if self.simulation_parameters.online_discovery:
                    ex_time_pred = self.simulation_parameters.execution_time_distributions[curr_act].debug_one(
                                                                                                                {'resource = '+res: (res == curr_res)*1 for res in self.simulation_parameters.resources} | 
                                                                                                                hystory_active_traces[curr_case_id] | 
                                                                                                                curr_trace_attribute_features[curr_case_id] | 
                                                                                                                {'hour': start_ts_datetime.hour, 'weekday': start_ts_datetime.weekday()}
                                                                                                            )
                    if not ex_time_pred:
                        ex_time = self.simulation_parameters.execution_time_distributions[curr_act].predict_one(
                                                                                                                {'resource = '+res: (res == curr_res)*1 for res in self.simulation_parameters.resources} | 
                                                                                                                hystory_active_traces[curr_case_id] | 
                                                                                                                curr_trace_attribute_features[curr_case_id] | 
                                                                                                                {'hour': start_ts_datetime.hour, 'weekday': start_ts_datetime.weekday()}
                                                                                                            )
                    else:
                        pred_split = ex_time_pred.split("\n")[-2].split(" | ")
                        mean = float(pred_split[0][6:].replace(",", ""))
                        var = float(pred_split[1][5:].replace(",", ""))
                        ex_time = random.normalvariate(mean, math.sqrt(var))
                        if ex_time < self.simulation_parameters.min_et[curr_act]:
                            ex_time = mean
                        elif ex_time > self.simulation_parameters.max_et[curr_act]:
                            ex_time = mean         
                else:
                    ex_time = self.simulation_parameters.execution_time_distributions[curr_act].apply_distribution(
                                                                                                                    {'resource = '+res: (res == curr_res)*1 for res in self.simulation_parameters.resources} | 
                                                                                                                    hystory_active_traces[curr_case_id] | 
                                                                                                                    curr_trace_attribute_features[curr_case_id] | 
                                                                                                                    {'hour': start_ts_datetime.hour, 'weekday': start_ts_datetime.weekday()}
                                                                                                                )
            else:
                if self.simulation_parameters.online_discovery:
                    ex_time_pred = self.simulation_parameters.execution_time_distributions[curr_act].debug_one(
                                                                                                                {res: (res == curr_res)*1 for res in self.simulation_parameters.resources} | 
                                                                                                                curr_trace_attribute_features[curr_case_id] | 
                                                                                                                {'hour': start_ts_datetime.hour, 'weekday': start_ts_datetime.weekday()}
                                                                                                            )
                    if not ex_time_pred:
                        ex_time = self.simulation_parameters.execution_time_distributions[curr_act].predict_one(
                                                                                                                {res: (res == curr_res)*1 for res in self.simulation_parameters.resources} | 
                                                                                                                curr_trace_attribute_features[curr_case_id] | 
                                                                                                                {'hour': start_ts_datetime.hour, 'weekday': start_ts_datetime.weekday()}
                                                                                                            )
                    else:
                        pred_split = ex_time_pred.split("\n")[-2].split(" | ")
                        mean = float(pred_split[0][6:].replace(",", ""))
                        var = float(pred_split[1][5:].replace(",", ""))
                        ex_time = random.normalvariate(mean, math.sqrt(var))
                        if ex_time < self.simulation_parameters.min_et[curr_act]:
                            ex_time = mean
                        elif ex_time > self.simulation_parameters.max_et[curr_act]:
                            ex_time = mean
                else:
                    ex_time = self.simulation_parameters.execution_time_distributions[curr_act].apply_distribution(
                                                                                                                    {res: (res == curr_res)*1 for res in self.simulation_parameters.resources} | 
                                                                                                                    curr_trace_attribute_features[curr_case_id] | 
                                                                                                                    {'hour': start_ts_datetime.hour, 'weekday': start_ts_datetime.weekday()}
                                                                                                                )
            
            end_ts_datetime = add_minutes_with_calendar(start_ts_datetime, int(ex_time), self.simulation_parameters.calendars[curr_res])
            end_ts = end_ts_datetime.timestamp()
            arrival_timestamps[curr_case_id] = end_ts


            events.append(
                            {
                                'case:concept:name': curr_case_id, 
                                'concept:name': curr_act, 
                                'start:timestamp': start_ts, 
                                'time:timestamp': end_ts, 
                                'org:resource': curr_res
                            } | {a: curr_trace_attributes[curr_case_id][a] for a in self.simulation_parameters.label_data_attributes}
                        )

            if self.simulation_parameters.mode_act_history:
                hystory_active_traces[curr_case_id][curr_act] += 1
            if len(curr_traces_acts[curr_case_id]) < 1:
                del curr_traces_acts[curr_case_id]
                if self.simulation_parameters.mode_act_history:
                    del hystory_active_traces[curr_case_id]
                del arrival_timestamps[curr_case_id]
                del curr_trace_attribute_features[curr_case_id]
                del curr_trace_attributes[curr_case_id]

        df_events_sim = pd.DataFrame(events)
        df_events_sim['start:timestamp'] = df_events_sim['start:timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x))
        df_events_sim['time:timestamp'] = df_events_sim['time:timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x))

        self.current_timestamp = datetime.datetime.fromtimestamp(end_ts)

        return df_events_sim


    def apply_trace(self, start_ts_simulation=None, x_attr=[])-> pd.DataFrame:

        trace_acts, trace_attributes = self._generate_activities(x_attr)
        curr_traces_acts = {f'case_{self.case_id}': trace_acts}
        if start_ts_simulation:
            self.current_timestamp = start_ts_simulation
        sim_trace_df = self._generate_events(curr_traces_acts, self.current_timestamp, {f'case_{self.case_id}': trace_attributes})

        return sim_trace_df


    def apply(self, n_sim: int, start_ts_simulation: pd.Timestamp, multiprocessing_cf: bool = True) -> pd.DataFrame:

        num_cores = mp.cpu_count()
        if self.simulation_parameters.label_data_attributes:
            x_attr_list = random.choices(
                list(self.simulation_parameters.distribution_data_attributes.keys()), 
                weights=list(self.simulation_parameters.distribution_data_attributes.values()),
                k = n_sim
                )
            x_attr_list = [list(attr) for attr in x_attr_list]
        else:
            x_attr_list = [[]]*n_sim

        if multiprocessing_cf:
            with mp.Pool(processes=num_cores) as pool:
                traces_acts_attributes = pool.map(self._generate_activities, tqdm(x_attr_list, desc="Processing"))
        else:
            traces_acts_attributes = []
            for i in tqdm(range(n_sim)):
                trace_act = self._generate_activities(x_attr_list[i])
                traces_acts_attributes.append(trace_act)

        curr_traces_acts = dict()
        curr_trace_attributes = dict()
        for j, id in enumerate(range(self.case_id, self.case_id+n_sim)):
            curr_traces_acts[f'case_{id}'] = traces_acts_attributes[j][0]
            curr_trace_attributes[f'case_{id}'] = traces_acts_attributes[j][1]

        sim_log_df = self._generate_events(curr_traces_acts, start_ts_simulation, curr_trace_attributes)

        self.case_id += n_sim

        return sim_log_df
