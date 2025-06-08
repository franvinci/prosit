import sys
sys.path.append("../src")

from pm4py.discovery import discover_petri_net_inductive
import prosit.simulator as simulator

import numpy as np
import pandas as pd
import pm4py

import os


def discovery_and_simulate(
        log, net=None, initial_marking=None, final_marking=None, 
        max_depth=3, 
        noise_threshold_im=0.0, 
        start_ts_simulation=None,
        n_sim_traces=1000, n_sim=5
        ):

    print(f'DISCOVERY WITH MAX DEPTH {max_depth}...')
    if net is None:
        print("Process model discovery...")
        net, initial_marking, final_marking = discover_petri_net_inductive(log, noise_threshold=noise_threshold_im)

    parameters = simulator.SimulatorParameters(net, initial_marking, final_marking)
    parameters.discover_from_eventlog(log, max_depth_tree=max_depth)

    sim_logs = []
    sim_logs_dets = []
    for i in range(1, n_sim+1):

        simulator_eng = simulator.SimulatorEngine(parameters)
        print(f'SIMULATION {i}...')
        if start_ts_simulation is None:
            start_ts_simulation = log[0][0]['start:timestamp']
        sim_log = simulator_eng.apply(n_sim_traces, t_start=start_ts_simulation)
        sim_logs.append(sim_log)

        simulator_eng = simulator.SimulatorEngine(parameters)
        print(f'SIMULATION {i} (DETERMINISTIC)...')
        if start_ts_simulation is None:
            start_ts_simulation = log[0][0]['start:timestamp']
        sim_log = simulator_eng.apply(n_sim_traces, t_start=start_ts_simulation, deterministic_time=True)
        sim_logs_dets.append(sim_log)

    return sim_logs, sim_logs_dets


def split_event_log(df_log, perc=0.8):

    map_new_cases = dict(zip(df_log['case:concept:name'].unique(), range(len(df_log['case:concept:name'].unique()))))
    df_log['case:concept:name'] = df_log['case:concept:name'].apply(lambda x: map_new_cases[x])
    df_train_log = df_log[df_log['case:concept:name'] < int(len(df_log['case:concept:name'].unique())*perc)]
    df_test_log = df_log[df_log['case:concept:name'] >= int(len(df_log['case:concept:name'].unique())*perc)]
    
    df_train_log['case:concept:name'] = df_train_log['case:concept:name'].astype(str)
    df_test_log['case:concept:name'] = df_test_log['case:concept:name'].astype(str)

    return df_train_log, df_test_log


def preprocessing_log(df_log: pd.DataFrame):

    df_log['case:concept:name'] = df_log['case:concept:name'].astype(str)
    df_log['time:timestamp'] = pd.to_datetime(df_log['time:timestamp'])
    df_log['start:timestamp'] = pd.to_datetime(df_log['start:timestamp'])

    df_log.sort_values(by=['start:timestamp', 'time:timestamp'], inplace=True)
    df_log.reset_index(drop=True, inplace=True)

    return df_log