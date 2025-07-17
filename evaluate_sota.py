import os
import numpy as np
import pandas as pd
from experiment_utils.evaluation import evaluate
import json
import warnings
warnings.filterwarnings("ignore")


SOTA = ["dsim", "rims", "simod"]
CASE_STUDIES = ["purchasing", "acr", "cvs", "bpi12",  "bpi17"]
CONVERT_IDS = {
    "dsim": {
        "caseid": "case:concept:name",
        "task": "concept:name",
        "resource": "org:resource",
        "end_timestamp": "time:timestamp",
        "start_timestamp": "start:timestamp",
    },
    "rims": {
        "caseid": "case:concept:name",
        "task": "concept:name",
    },
    "simod": {
        "case_id": "case:concept:name",
        "activity": "concept:name",
        "resource": "org:resource",
        "end_time": "time:timestamp",
        "start_time": "start:timestamp",
    }
}
METRICS = ["ngd", "car", "ctd", "etd_entropy"]

if __name__ == "__main__":
    for case_study in CASE_STUDIES:
        df_test = pd.read_csv(f"outputs/{case_study}/df_test.csv")
        df_test["case:concept:name"] = df_test["case:concept:name"].astype(str)

        for sota in SOTA:
            sota_path = f"sota_results/{sota}"    
            case_study_path = f"{sota_path}/{case_study}"
            sim_logs_path = os.listdir(case_study_path)
            sim_logs_path = [f"{case_study_path}/{log}" for log in sim_logs_path if log.endswith('.csv')]
            print(f"Evaluating {sota} on {case_study}...")
            
            evaluations = {metric: [] for metric in METRICS}
            for sim_log_path in sim_logs_path:
                df_sim_log = pd.read_csv(sim_log_path)
                df_sim_log.rename(columns=CONVERT_IDS[sota], inplace=True)
                df_sim_log["case:concept:name"] = df_sim_log["case:concept:name"].astype(str)
                metrics = evaluate(df_test, df_sim_log, metrics_labels=METRICS)
                for metric in metrics.keys():
                    evaluations[metric].append(metrics[metric])

            for metric in evaluations.keys():
                print(metric, ': ', np.mean(evaluations[metric]))
            
            with open(case_study_path+f"/distances.json", 'w') as f:
                json.dump(evaluations, f)

        print()