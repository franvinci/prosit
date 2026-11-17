import random
import warnings
import pandas as pd
import heapq
from datetime import datetime, timedelta, timezone
from tqdm import tqdm

import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.log.obj import EventLog

from prosit.discovery.cf_discovery import discover_weight_transitions, plan_weight_transitions
from prosit.discovery.time_discovery import discover_execution_time_distributions, discover_arrival_time, discover_waiting_time, plan_execution_time_distributions, plan_waiting_time
from prosit.discovery.calendar_discovery import discover_res_calendars, discover_arrival_calendar
from prosit.discovery.resource_discovery import discover_resources_list, return_max_concurrency, discover_resources_per_act, discover_weight_resources, plan_weight_resources
from prosit.discovery.data_discovery import discover_attributes_distribution, return_label_data_attributes
from prosit.discovery.online_discovery.cf_discovery import incremental_transition_weights_learning
from prosit.discovery.online_discovery.time_discovery import incremental_execution_time_learning, incremental_model_arrival_learning, incremental_waiting_time_learning
from prosit.discovery.online_discovery.resource_discovery import incremental_resource_weights_learning
from prosit.utils.common_utils import (
    return_enabled_transitions,
    update_current_marking,
    return_fired_transition,
    count_concurrent_events,
    is_resource_busy,
    compute_transition_weights_from_model,
    add_minutes_with_calendar,
    calendar_to_working_set,
    build_df_features,
    _build_features,
    dense_take,
    run_model_plans,
    return_resource,
    compute_resource_weights_from_model,
    )
from prosit.utils.distribution_utils import sampling_from_dist
from prosit.utils.save_and_load_utils import decision_rules_to_dict, convert_calendar_names, dict_to_decrules, fromstr_to_scipy

import json


def _transition_order(t):
    """Sort key giving transitions a stable order. They hash by identity, so
    a set of them iterates in an order that changes with every run."""
    return (t.name, str(t.label))


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
        """ Initialize parameters """
        
        self.net: PetriNet = net
        self.initial_marking: Marking = initial_marking
        self.final_marking: Marking = final_marking
        # Sorted: the order of a set of strings changes from one Python process
        # to the next, and with it the column order of every training matrix
        # (hence which of two equally good splits a tree picks).
        self.net_transition_labels: list = sorted(set([t.label for t in net.transitions if t.label]))

        self.label_data_attributes: list = []
        self.label_data_attributes_categorical: list = []
        self.attribute_values_label_categorical: dict = dict()

        self.transition_weights: dict = {t.name: 1 for t in sorted(self.net.transitions, key=_transition_order)}
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
        self.use_workload_features: bool = False


    def discover_from_eventlog(
            self,
            log: EventLog,
            max_depth_tree: int = 5,
            min_samples_leaf_cv: list = [50, 100, 200],
            multitasking_thr: float = 0.05,
            enable_multitasking: bool = False,
            arrival_calendar_min_confidence: float = 0.1,
            arrival_calendar_min_support: float = 0.7,
            res_calendar_min_confidence: float = 0.1,
            res_calendar_min_support: float = 0.1,
            res_calendar_min_participation: float = 0.4,
            attribute_mode: str = 'distribution',
            incremental_discovery: bool = False,
            grace_period: int = 1000,
            random_state: int = 72,
            verbose: bool = True,
            use_workload_features: bool = False,
        ):
        """ Discovery Parameters from event log data """

        self.use_workload_features = use_workload_features

        if max_depth_tree < 1:
            self.rules_mode = False
            max_depth_cv = []
        else:
            self.rules_mode = True
            max_depth_cv = list(range(1, max_depth_tree + 1))
        
        self.label_data_attributes, self.label_data_attributes_categorical = return_label_data_attributes(log)
        
        for a in self.label_data_attributes_categorical:
            self.attribute_values_label_categorical[a] = list(pm4py.get_event_attribute_values(log, a).keys())

        if verbose:
            print("Resources discovery...")
        self.resources = discover_resources_list(log)
        self.act_to_resources = discover_resources_per_act(log, self.net_transition_labels, self.resources)
        unstaffed = sorted(str(act) for act, act_resources in self.act_to_resources.items() if not act_resources)
        if unstaffed:
            warnings.warn(
                "No resource performs these activities of the model in the log: "
                + ", ".join(unstaffed)
                + ". Whenever the model fires one of them it is simulated without a resource.",
                stacklevel=2,
            )

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
        # The batch models read the per-candidate workloads / queue lengths from
        # the dense block; only the incremental learners still walk the
        # dict-valued ``candidate_*`` columns, so those are built just for them.
        df_features, dense_candidates = _build_features(
            log, self.net, self.initial_marking, self.final_marking, self.act_to_resources, self.net_transition_labels, self.resources, self.label_data_attributes,
            candidate_dicts=incremental_discovery, dense_queue=use_workload_features,
        )
        known_resource = (df_features['resource'].isin(self.resources)) | (df_features["resource"].isna())
        dense_candidates = dense_take(dense_candidates, known_resource.to_numpy())
        df_features = df_features[known_resource]
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
            # Batch mode: each step only prepares its fits here; they all run
            # together further down, in one worker pool.
            transition_plan = plan_weight_transitions(
                                                                    df_features,
                                                                    self.net_transition_labels,
                                                                    max_depths_cv=max_depth_cv,
                                                                    label_data_attributes=self.label_data_attributes,
                                                                    label_data_attributes_categorical=self.label_data_attributes_categorical,
                                                                    values_categorical=self.attribute_values_label_categorical,
                                                                    random_state=random_state
                                                                )

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
                                                                            self.attribute_values_label_categorical,
                                                                            use_workload_features=use_workload_features,
                                                                        )
        else:
            resource_plan = plan_weight_resources(
                                                                df_features,
                                                                self.act_to_resources,
                                                                self.resources,
                                                                self.max_concurrency,
                                                                max_depth_cv,
                                                                self.label_data_attributes,
                                                                self.label_data_attributes_categorical,
                                                                self.attribute_values_label_categorical,
                                                                random_state=random_state,
                                                                use_workload_features=use_workload_features,
                                                                _dense=dense_candidates
                                                            )

        if verbose:
            print("Calendars discovery...")
        self.calendars = discover_res_calendars(
            log, self.resources,
            min_confidence=res_calendar_min_confidence,
            min_support=res_calendar_min_support,
            min_participation=res_calendar_min_participation,
        )
        self.arrival_calendar = discover_arrival_calendar(
            log,
            min_confidence=arrival_calendar_min_confidence,
            min_support=arrival_calendar_min_support,
        )

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
            execution_plan = plan_execution_time_distributions(
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
                                                                                    grace_period=grace_period,
                                                                                    use_workload_features=use_workload_features,
                                                                                )
        else:
            waiting_plan = plan_waiting_time(
                                                                        df_features,
                                                                        self.net_transition_labels,
                                                                        self.resources,
                                                                        self.calendars,
                                                                        self.label_data_attributes,
                                                                        self.label_data_attributes_categorical,
                                                                        self.attribute_values_label_categorical,
                                                                        max_depths=max_depth_cv,
                                                                        min_samples_leaf_cv=min_samples_leaf_cv,
                                                                        random_state=random_state,
                                                                        use_workload_features=use_workload_features,
                                                                    )

        if not incremental_discovery:
            if verbose:
                print("Model fitting...")
            (
                self.transition_weights,
                self.resource_weights,
                self.execution_time_distributions,
                self.waiting_time_distributions,
            ) = run_model_plans([transition_plan, resource_plan, execution_plan, waiting_plan])

        for t in sorted(self.net.transitions, key=_transition_order):
            if t.name not in self.transition_weights:
                self.transition_weights[t.name] = 0
        
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

    def to_dict(self) -> dict:

        dict_params = {

            "rules_mode": self.rules_mode,
            "use_workload_features": self.use_workload_features,

            "transition_params": {
                "transition_weights": {t_name: decision_rules_to_dict(dr) for t_name, dr in self.transition_weights.items()}
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
        self.use_workload_features = dict_params.get("use_workload_features", False)
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
            self.transition_weights = {t_name: dict_to_decrules(value) for t_name, value in dict_params["transition_params"]["transition_weights"].items()}
            self.execution_time_distributions = {act: dict_to_decrules(value) for act, value in dict_params["execution_time_params"]["execution_time_distributions"].items()}
            self.waiting_time_distributions = {res: dict_to_decrules(value) for res, value in dict_params["waiting_time_params"]["waiting_time_distributions"].items()}
            self.arrival_time_distribution = dict_to_decrules(dict_params["arrival_params"]["arrival_time_distributions"])
        else:
            self.transition_weights = {t_name: value for t_name, value in dict_params["transition_params"]["transition_weights"].items()}
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
        elif isinstance(t_start, pd.Timestamp) and t_start is not pd.NaT:
            # A Timestamp with no timezone (or UTC) and no nanoseconds moves
            # exactly like the plain ``datetime`` it wraps: use the latter,
            # whose arithmetic is several times cheaper. Other Timestamps
            # (DST zones add elapsed time, not wall time) are left alone.
            tz = t_start.tzinfo
            if (tz is None or tz is timezone.utc or str(tz) == 'UTC') and t_start.nanosecond == 0:
                t_start = t_start.to_pydatetime()

        sim = self.simulation_parameters
        net = self.net
        final_marking = self.final_marking
        rules_mode = sim.rules_mode
        uwf = sim.use_workload_features
        mc = sim.max_concurrency
        calendars = sim.calendars

        event_log = []
        enabled_heap = []
        resource_schedule = {r: [] for r in sim.resources}
        cases = []
        fired = [[] for _ in range(n_traces)]

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

        # Markings are interned: the enabled transitions of a marking, the
        # marking reached by a firing and "is this the final marking" are each
        # computed once per distinct marking instead of once per step. The
        # enabled transitions are kept sorted by name: the random choice walks
        # them in order, so a fixed order is what makes a run with a given
        # seed reproducible.
        state_of = {}
        state_markings = []
        state_enabled = []
        state_final = []
        next_state = {}
        input_places = {}

        def intern(marking):
            key = frozenset(marking.items())
            state = state_of.get(key)
            if state is None:
                state = state_of[key] = len(state_markings)
                state_markings.append(marking)
                state_enabled.append(None)
                state_final.append(marking == final_marking)
            return state

        def enabled_in(state):
            enabled = state_enabled[state]
            if enabled is None:
                enabled = state_enabled[state] = tuple(sorted(return_enabled_transitions(net, state_markings[state]), key=_transition_order))
                for t in enabled:
                    if t not in input_places:
                        input_places[t] = [arc.source for arc in t.in_arcs]
            return enabled

        initial_state = intern(self.initial_marking)
        no_token_times = {place: None for place in net.places}
        working_sets = {}
        arrival_working_set = calendar_to_working_set(sim.arrival_calendar)

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

            if i > 0:
                if not self.simulation_parameters.rules_mode:
                    if deterministic_time:
                        arrival_delta = sampled_arrivals
                    else:
                        arrival_delta = sampled_arrivals[i]
                else:
                    arrival_features = {
                        'hour': current_arr_ts.hour,
                        'weekday': current_arr_ts.weekday(),
                    }
                    if deterministic_time:
                        arrival_delta = self.simulation_parameters.arrival_time_distribution.apply(arrival_features)
                    else:
                        arrival_delta = self.simulation_parameters.arrival_time_distribution.apply_distribution(arrival_features)
                # Advance whole active minutes via the calendar, then add the
                # residual seconds so sub-minute inter-arrivals (bursts) survive
                # the fit→simulate round-trip. Floor to 1 second so two
                # consecutive cases never share a timestamp.
                arrival_delta_sec = max(1, round(arrival_delta * 60))
                whole_min, frac_sec = divmod(arrival_delta_sec, 60)
                if whole_min > 0:
                    current_arr_ts = add_minutes_with_calendar(
                        current_arr_ts, whole_min, self.simulation_parameters.arrival_calendar, arrival_working_set,
                    )
                if frac_sec > 0:
                    current_arr_ts = current_arr_ts + timedelta(seconds=frac_sec)

            # ``features`` (case attributes overlaid with the activity counts)
            # and ``res_history`` are created when the case takes its first
            # step and dropped when it completes.
            case = {
                "arrival_time": current_arr_ts,
                "state": initial_state,
                "place_token_time": dict(no_token_times),
                "enabled": {},
                "attributes": trace_attributes,
                "attribute_values": tuple(x_attr_list[i]),
                "features": None,
                "res_history": None,
                "n_events": 0,
            }
            case["place_token_time"][list(self.initial_marking.keys())[0]] = case["arrival_time"]

            for t in enabled_in(initial_state):
                enabled_time = max(case["place_token_time"][p] for p in input_places[t])
                case["enabled"][t] = enabled_time

            if case["enabled"]:
                enabled_time_case = min(case["enabled"].values())
                heapq.heappush(enabled_heap, (enabled_time_case, i))

            cases.append(case)

        _zero_history = {t: 0 for t in self.simulation_parameters.net_transition_labels}
        _zero_res_history = {r: 0 for r in self.simulation_parameters.resources}
        _zero_waiting_act = {'waiting_activity = ' + act_label: 0 for act_label in self.simulation_parameters.net_transition_labels}
        _zero_resource_onehot = {'resource = ' + res: 0 for res in self.simulation_parameters.resources}
        _zero_activity_onehot = {'activity = ' + act_label: 0 for act_label in self.simulation_parameters.net_transition_labels}

        completed_cases = set()
        pbar = tqdm(total=n_traces, desc="Simulating Cases")
        while enabled_heap:
            decision_time, case_id = heapq.heappop(enabled_heap)
            case = cases[case_id]

            if not case["enabled"]:
                continue

            # Same mapping as ``case attributes | activity history``: the
            # history counts are kept inside it and updated in place.
            features = case["features"]
            if features is None:
                features = case["features"] = case["attributes"] | _zero_history
                case["res_history"] = dict(_zero_res_history)

            enabled_transitions = list(case["enabled"].keys())

            if not rules_mode:
                transition_weights = self.simulation_parameters.transition_weights
            else:
                transition_weights = compute_transition_weights_from_model(self.simulation_parameters.transition_weights, features, enabled_transitions)
            chosen_transition = return_fired_transition(transition_weights, enabled_transitions)
            activity = chosen_transition.label
            t_enabled = case["enabled"][chosen_transition]
            fired[case_id].append(chosen_transition.name)

            if activity is not None:
                enabled_resources_act = self.simulation_parameters.act_to_resources[activity]
                if uwf:
                    workloads = {}
                    queue_lengths_cand = {}
                    for r in enabled_resources_act:
                        schedule = resource_schedule[r]
                        if schedule:
                            workloads[r] = count_concurrent_events(schedule, t_enabled)
                            queue_lengths_cand[r] = sum(1 for s, _ in schedule if s > t_enabled)
                        else:
                            workloads[r] = 0
                            queue_lengths_cand[r] = 0
                    enabled_resources = [r for r in enabled_resources_act if workloads[r] < mc.get(r, 1)]
                else:
                    workloads = None
                    queue_lengths_cand = None
                    enabled_resources = []
                    for r in enabled_resources_act:
                        max_c = mc.get(r, 1)
                        if max_c == 1:
                            schedule = resource_schedule[r]
                            if not schedule or not is_resource_busy(schedule, t_enabled):
                                enabled_resources.append(r)
                        else:
                            if count_concurrent_events(resource_schedule[r], t_enabled) < max_c:
                                enabled_resources.append(r)
                if not enabled_resources_act:
                    # The model fires an activity that nobody performs in the
                    # log: there is no resource to pick, none to wait for and
                    # no calendar to follow. The event carries no resource,
                    # starts when it is enabled and lasts what the activity's
                    # own model says, around the clock.
                    resource = None
                    t_enabled_waited = t_enabled
                elif not enabled_resources:
                    t_enabled_enabled_resources = [resource_schedule[r][-1][-1] if resource_schedule[r] else t_enabled for r in enabled_resources_act]
                    index_res, t_enabled_waited = min(enumerate(t_enabled_enabled_resources), key=lambda indexed: indexed[1])
                    resource = enabled_resources_act[index_res]
                else:
                    if not rules_mode:
                        resource_weights = {r: self.simulation_parameters.resource_weights.get(r, 0.0) for r in enabled_resources}
                    else:
                        activity_onehot = _zero_activity_onehot.copy()
                        activity_onehot['activity = ' + activity] = 1
                        res_features = dict(case["res_history"]) | activity_onehot | case["attributes"]
                        if uwf:
                            resource_weights = compute_resource_weights_from_model(
                                self.simulation_parameters.resource_weights,
                                enabled_resources,
                                res_features,
                                workloads=workloads,
                                queue_lengths=queue_lengths_cand,
                            )
                        else:
                            resource_weights = compute_resource_weights_from_model(
                                self.simulation_parameters.resource_weights,
                                enabled_resources,
                                res_features,
                            )
                    resource = return_resource(resource_weights, enabled_resources)
                    t_enabled_waited = t_enabled
                if resource is not None:
                    if uwf:
                        r_workload = workloads[resource]
                        r_queue_length = queue_lengths_cand[resource]
                    case["res_history"][resource] += 1

                    calendar = calendars[resource]
                    working_set = working_sets.get(resource)
                    if working_set is None:
                        working_set = working_sets[resource] = calendar_to_working_set(calendar)

                if case["n_events"] == 0 or resource is None:
                    waiting_time = 0
                else:
                    if not rules_mode:
                        if deterministic_time:
                            waiting_time = sampled_waiting_times[resource]
                            if not isinstance(waiting_time, (int, float)):
                                waiting_time = 0
                        else:
                            waiting_time = random.choice(sampled_waiting_times[resource])
                    else:
                        waiting_activity_features = _zero_waiting_act.copy()
                        waiting_activity_features['waiting_activity = ' + activity] = 1
                        time_features = {'hour': t_enabled_waited.hour, 'weekday': t_enabled_waited.weekday()}
                        if uwf:
                            wt_features = {'workload': r_workload, 'queue_length': r_queue_length} | time_features | waiting_activity_features | features
                        else:
                            wt_features = time_features | waiting_activity_features | features
                        if deterministic_time:
                            waiting_time = self.simulation_parameters.waiting_time_distributions[resource].apply(wt_features)
                            if not isinstance(waiting_time, (int, float)):
                                waiting_time = 0
                        else:
                            waiting_time = self.simulation_parameters.waiting_time_distributions[resource].apply_distribution(wt_features)

                if resource is None:
                    t_start_exec = t_enabled_waited
                else:
                    t_start_exec = add_minutes_with_calendar(t_enabled_waited, round(max(0, waiting_time)), calendar, working_set)

                if not rules_mode:
                    if deterministic_time:
                        ex_time = sampled_execution_times[activity]
                        if not isinstance(ex_time, (int, float)):
                            ex_time = 0
                    else:
                        ex_time = random.choice(sampled_execution_times[activity])
                else:
                    resource_onehot = _zero_resource_onehot.copy()
                    if resource is not None:
                        resource_onehot['resource = ' + resource] = 1
                    ex_time_features = {'hour': t_start_exec.hour, 'weekday': t_start_exec.weekday()}
                    ex_features = resource_onehot | ex_time_features | features
                    if deterministic_time:
                        ex_time = self.simulation_parameters.execution_time_distributions[activity].apply(ex_features)
                        if not isinstance(ex_time, (int, float)):
                            ex_time = 0
                    else:
                        ex_time = self.simulation_parameters.execution_time_distributions[activity].apply_distribution(ex_features)

                
                if resource is None:
                    t_end = t_start_exec + timedelta(minutes=round(max(0, ex_time)))
                else:
                    t_end = add_minutes_with_calendar(t_start_exec, round(max(0, ex_time)), calendar, working_set)

                event_log.append((case_id, activity, resource, t_enabled, t_start_exec, t_end) + case["attribute_values"])
                if resource is not None:
                    resource_schedule[resource].append((t_start_exec, t_end))
                features[activity] += 1
                case["n_events"] += 1
                if len(event_log) % 1000 == 0:
                    # Every case still to be stepped sits in the heap with a key
                    # no smaller than the one just popped, and no later step
                    # looks further back than its own key: intervals that ended
                    # by ``decision_time`` can go.
                    for r in resource_schedule:
                        resource_schedule[r] = [(s, e) for s, e in resource_schedule[r] if e > decision_time]
            else:
                t_end = t_enabled

            token_times = case["place_token_time"]
            for arc in chosen_transition.out_arcs:
                token_times[arc.target] = t_end

            case["enabled"] = {}
            step = (case["state"], chosen_transition)
            state = next_state.get(step)
            if state is None:
                state = next_state[step] = intern(update_current_marking(state_markings[case["state"]], chosen_transition))
            case["state"] = state
            if state_final[state]:
                if case_id not in completed_cases:
                    pbar.update(1)
                    completed_cases.add(case_id)
                case["features"] = case["res_history"] = None
                continue
            for t in enabled_in(state):
                enabled_time = max(token_times[p] for p in input_places[t])
                case["enabled"][t] = enabled_time

            if case["enabled"]:
                next_enabled_time = min(case["enabled"].values())
                heapq.heappush(enabled_heap, (next_enabled_time, case_id))

        pbar.close()
        df_log = pd.DataFrame(event_log, columns=["case:concept:name", "concept:name", "org:resource", "enabled:timestamp", "start:timestamp", "time:timestamp"] + self.simulation_parameters.label_data_attributes)
        df_log["case:concept:name"] = df_log["case:concept:name"].apply(lambda x: f"case_{x+1}")
        df_log.sort_values(by=["start:timestamp", "time:timestamp"], inplace=True)
        df_log.reset_index(drop=True, inplace=True)
        df_log.attrs["prosit_firing_sequences"] = {f"case_{i+1}": fired[i] for i in range(n_traces)}

        return df_log
