from pm4py.objects.log.obj import EventLog
from prosit.discovery.time_discovery import build_training_df_ex, build_training_df_arrival, build_training_df_wt
from tqdm import tqdm
from river import tree
import pm4py
from prosit.utils.online_utils import return_best_online_model


def incremental_model_arrival_learning(
        model_arrival,
        log: EventLog, 
        calendar_arrival: dict,
        max_depth: int = 5,
        grace_period: int = 5000
    ):

    df = build_training_df_arrival(log, calendar_arrival)
    
    if not model_arrival:
        model_arrival = tree.HoeffdingAdaptiveTreeRegressor(seed=72, leaf_prediction="mean", max_depth=max_depth, grace_period=grace_period)
    
    for _, row in df.iterrows():
        X_row = row.drop('arrival_time').to_dict()
        y_row = row['arrival_time']
        model_arrival.learn_one(X_row, y_row)
    
    min_at = df["arrival_time"].min()
    max_at = df["arrival_time"].max()

    return model_arrival, min_at, max_at


def incremental_execution_time_learning(
        models_act: dict,
        log: EventLog, 
        activity_labels: list, 
        calendars: dict,
        label_data_attributes: list, 
        label_data_attributes_categorical: list, 
        values_categorical: dict, 
        mode_act_history: bool,
        max_depth: int = 5,
        grace_period: int = 5000
    ) -> dict:

    df = build_training_df_ex(
                                log, 
                                activity_labels, 
                                calendars,
                                label_data_attributes, 
                                label_data_attributes_categorical, 
                                values_categorical, 
                                mode_act_history
                            )
    
    if not models_act:
        first_step = True
        models_act = dict()
    else:
        first_step = False

    min_et = dict()
    max_et = dict()

    for act in tqdm(activity_labels):

        df_act = df[df['activity_executed'] == act].iloc[:,1:]

        if first_step or act not in models_act.keys():
            models_act[act] = tree.HoeffdingAdaptiveTreeRegressor(max_depth=max_depth, leaf_prediction="mean", seed=72, grace_period=grace_period)

        for _, row in df_act.iterrows():
            X_row = row.drop('execution_time').to_dict()
            y_row = row['execution_time']
            models_act[act].learn_one(X_row, y_row)

        min_et[act] = df_act["execution_time"].min()
        max_et[act] = df_act["execution_time"].max()

    return models_act, min_et, max_et


def incremental_waiting_time_learning(
        models_res: dict,
        log: EventLog, 
        calendars: dict, 
        label_data_attributes: list, 
        label_data_attributes_categorical: list, 
        values_categorical: dict,
        max_depth: int = 5,
        grace_period: int = 5000
    ):

    resources = list(pm4py.get_event_attribute_values(log, 'org:resource').keys())
    df_per_res = build_training_df_wt(log, calendars, label_data_attributes)

    if not models_res:
        first_step = True
        models_res = dict()
    else:
        first_step = False

    min_wt = dict()
    max_wt = dict()

    for res in tqdm(resources):
        df_res = df_per_res[res]
        if len(df_res) < 1:
            continue

        for a in label_data_attributes_categorical:
            for v in values_categorical[a]:
                df_res[a+' = '+str(v)] = (df_res[a] == v)*1
            del df_res[a]

        if first_step or res not in models_res.keys():
            models_res[res] = tree.HoeffdingAdaptiveTreeRegressor(seed=72, leaf_prediction="mean", max_depth=max_depth, grace_period=grace_period)

        min_wt[res] = df_res["waiting_time"].min()
        max_wt[res] = df_res["waiting_time"].max()

        for _, row in df_res.iterrows():
            X_row = row.drop('waiting_time').to_dict()
            y_row = row['waiting_time']
            models_res[res].learn_one(X_row, y_row)

    return models_res, min_wt, max_wt