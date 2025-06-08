import sys
sys.path.append('src/')

import os

import warnings
warnings.filterwarnings('ignore')

import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer

from experiment_utils.utils import discovery_and_simulate, split_event_log, preprocessing_log
from experiment_utils.evaluation import evaluate

import numpy as np

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

N_SIM = 5
OUTPUT_PATH = 'outputs_new'
MAX_DEPTHS = [0, 1, 2, 3, 4, 5]
NOISE_THRESHOLD_IM = 0.2
METRICS = ["ngd", "car", "ctd", "car_entropy", "ctd_entropy", "etd_entropy"] 

if __name__ == "__main__":

    os.mkdir(OUTPUT_PATH) if OUTPUT_PATH not in os.listdir() else None

    for case_study in CASE_STUDIES.keys():

        if f'/{case_study}' not in os.listdir(OUTPUT_PATH):
            os.mkdir(OUTPUT_PATH+f'/{case_study}')
            os.mkdir(OUTPUT_PATH+f'/{case_study}/simulations/')
            os.mkdir(OUTPUT_PATH+f'/{case_study}/simulations/prob')
            os.mkdir(OUTPUT_PATH+f'/{case_study}/simulations/det')

        print(f'\nRun Experiments for case study: {case_study}')

        log = xes_importer.apply(CASE_STUDIES[case_study]['PATH_LOG'])

        print('PREPROCESSING...')
        df_log = pm4py.convert_to_dataframe(log)
        df_log = preprocessing_log(df_log)
        print('SPLIT TRAIN-TEST...')
        df_train_log, df_test_log = split_event_log(df_log)
        df_train_log.to_csv(OUTPUT_PATH+f'/{case_study}/df_train.csv', index=False)
        df_test_log.to_csv(OUTPUT_PATH+f'/{case_study}/df_test.csv', index=False)

        train_log = pm4py.convert_to_event_log(df_train_log)

        n_sim_traces = len(df_test_log['case:concept:name'].unique())

        if CASE_STUDIES[case_study]['PATH_MODEL'] is not None:
            if CASE_STUDIES[case_study]['PATH_MODEL'].endswith('.bpmn'):
                bpmn_model = pm4py.read_bpmn(CASE_STUDIES[case_study]['PATH_MODEL'])
                net, initial_marking, final_marking = pm4py.convert_to_petri_net(bpmn_model)
                # check soundness
                soundness = pm4py.analysis.check_soundness(net, initial_marking, final_marking)
                if soundness:
                    print("The model is sound.")
                else:
                    print("WARNING: The BPMN model is not sound. Proceeding with Inductive Miner Discovery.")
                    net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(log, noise_threshold=CASE_STUDIES[case_study].get('NOISE_THRESHOLD_IM', NOISE_THRESHOLD_IM))
            elif CASE_STUDIES[case_study]['PATH_MODEL'].endswith('.pnml'):
                net, initial_marking, final_marking = pm4py.read_pnml(CASE_STUDIES[case_study]['PATH_MODEL'])
        else:
            print("Process Model Discovery...")
            net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(log, noise_threshold=CASE_STUDIES[case_study].get('NOISE_THRESHOLD_IM', NOISE_THRESHOLD_IM))

        pm4py.write_pnml(net, initial_marking, final_marking, OUTPUT_PATH+f'/{case_study}/model.pnml')
        
        evaluations = {"prob": dict(), "det": dict()}
        for max_depth in MAX_DEPTHS:
            df_sim_logs, df_sim_logs_det = discovery_and_simulate(
                train_log,
                net, initial_marking, final_marking,
                max_depth=max_depth,
                noise_threshold_im=CASE_STUDIES[case_study].get('NOISE_THRESHOLD_IM', NOISE_THRESHOLD_IM),
                start_ts_simulation=df_test_log.iloc[0]['start:timestamp'],
                n_sim_traces=n_sim_traces,
                n_sim=N_SIM
            )

            for i in range(N_SIM):
                df_sim_logs[i].to_csv(OUTPUT_PATH+f'/{case_study}/simulations/prob/df_sim_{i}_maxdepth_{max_depth}.csv', index=False)
                df_sim_logs_det[i].to_csv(OUTPUT_PATH+f'/{case_study}/simulations/det/df_sim_{i}_maxdepth_{max_depth}.csv', index=False)

            print(f'EVALUATION WITH MAX DEPTH {max_depth}...')

            evaluations["prob"][f"maxdepth_{max_depth}"] = dict()
            for i in range(N_SIM):
                metrics = evaluate(df_test_log, df_sim_logs[i], metrics_labels=METRICS)
                for metric in metrics.keys():
                    if i == 0:
                        evaluations["prob"][f"maxdepth_{max_depth}"][metric] = [metrics[metric]]
                    else:
                        evaluations["prob"][f"maxdepth_{max_depth}"][metric].append(metrics[metric])

            for metric in evaluations["prob"][f"maxdepth_{max_depth}"].keys():
                print(metric, ': ', np.mean(evaluations["prob"][f"maxdepth_{max_depth}"][metric]))
            
            print(f'EVALUATION WITH MAX DEPTH {max_depth} (DETERMINISTIC)...')

            evaluations["det"][f"maxdepth_{max_depth}"] = dict()
            for i in range(N_SIM):
                metrics = evaluate(df_test_log, df_sim_logs_det[i], METRICS)
                for metric in metrics.keys():
                    if i == 0:
                        evaluations["det"][f"maxdepth_{max_depth}"][metric] = [metrics[metric]]
                    else:
                        evaluations["det"][f"maxdepth_{max_depth}"][metric].append(metrics[metric])

            for metric in evaluations["det"][f"maxdepth_{max_depth}"].keys():
                print(metric, ': ', np.mean(evaluations["det"][f"maxdepth_{max_depth}"][metric]))

        with open(OUTPUT_PATH+f'/{case_study}/'+f"distances.json", 'w') as f:
            json.dump(evaluations, f)
