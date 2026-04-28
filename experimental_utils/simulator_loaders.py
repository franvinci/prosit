"""Loaders for simulated event logs from state-of-the-art methods.

Each loader returns ``(logs, info)`` where ``logs`` is a list of up to ``n``
normalized ``pd.DataFrame`` (empty list if method unavailable) and ``info`` is a
dict with flags ``has_resources`` and ``has_case_attrs``.

Normalized schema matches ``evaluation.py``:
    case:concept:name, concept:name, org:resource (optional),
    start:timestamp, time:timestamp, + case-attribute columns
"""

import os
import re
import ast
import pandas as pd


EXP_DATA_DIR = "exp_data"
AGENTSIM_DIR = "agentsimulator"
SIMOD_DIR = "simod"
RIMS_DIR = "RESULT_RIMS_SPECIAL_ISSUE/RESULT_RIMS_SPECIAL_ISSUE"
DSIM_DIR = "RIMS/RIMS/DSIM"

N_SIMS = 10

# RIMS / DSIM folder names differ from exp_data names for a couple of datasets.
RIMS_DATASET_ALIAS = {
    "BPI_Challenge_2012_W_Two_TS": "BPI_Challenge_2012_W_Two_TS_1",
    "Production": "Productions",
}
DSIM_DATASET_ALIAS = {
    "Production": "Productions",
}

# Some DSIM folders contain files whose stem does not match the folder name.
DSIM_FILENAME_STEM_ALIAS = {
    "SynLoan": "synthetic_log_2000_11_5",
}

# RIMS encodes per-trace case attributes in an event-level ``attrib`` column
# as a stringified Python dict (e.g. "{'AMOUNT_REQ': 12333.21}"), repeating
# the same dict on every event of the case. The keys often DON'T match the
# pm4py ``case:*`` names used by the discovered model, so each dataset that
# actually carries attributes needs its own translation table. Datasets not
# in this map are treated as attribute-free.
RIMS_ATTRIB_RENAME = {
    "BPI_Challenge_2012_W_Two_TS": {"AMOUNT_REQ": "case:AMOUNT_REQ"},
    "BPI_Challenge_2017_W_Two_TS": {
        "amount": "case:RequestedAmount",
        "goal": "case:LoanGoal",
        "type": "case:ApplicationType",
    },
    "SynLoan": {"AMOUNT_REQ": "case:amount"},
}


def _parse_rims_attribs(df, rename_map):
    """Pull RIMS' per-event ``attrib`` dicts up to per-case columns.

    Returns the list of newly-added column names. Does nothing (returns ``[]``)
    if the column is missing, all dicts are empty, or no key in ``rename_map``
    actually appears in the data.
    """
    if "attrib" not in df.columns or not rename_map:
        return []
    parsed = df["attrib"].fillna("{}").apply(_safe_literal_eval)
    if parsed.apply(lambda d: not d).all():
        return []
    # Per-case attribute = first non-empty dict on the case (RIMS repeats the
    # same dict on every event so this is just a deduplication).
    first_per_case = parsed.groupby(df["case:concept:name"]).agg(_first_nonempty)

    added = []
    for src_key, dst_col in rename_map.items():
        if not first_per_case.apply(lambda d: src_key in d).any():
            continue
        case_to_value = first_per_case.apply(lambda d: d.get(src_key))
        df[dst_col] = df["case:concept:name"].map(case_to_value)
        added.append(dst_col)
    return added


def _safe_literal_eval(s):
    try:
        v = ast.literal_eval(s) if isinstance(s, str) else s
        return v if isinstance(v, dict) else {}
    except (ValueError, SyntaxError):
        return {}


def _first_nonempty(series):
    for d in series:
        if isinstance(d, dict) and d:
            return d
    return {}


def _to_utc(df, cols):
    for c in cols:
        # Parse as ISO8601 (handles nanosecond precision in sources like
        # agentsimulator) and truncate to microseconds — downstream code
        # (``build_df_features``) uses ``datetime.fromisoformat`` which on
        # Python 3.10 rejects sub-microsecond precision.
        df[c] = pd.to_datetime(df[c], utc=True, format="ISO8601").dt.floor("us")
    return df


def _resource_to_str(df):
    # ConsultaDataMining stores resource as a numeric ID; pandas reads it as
    # int64. ``params.calendars`` and ``params.resources`` are keyed by str,
    # so without this cast every per-resource lookup silently misses (no
    # exec/wait tree, degenerate resource_tree, mismatched 2rgd alphabet).
    if "org:resource" in df.columns:
        df["org:resource"] = df["org:resource"].map(
            lambda v: str(v) if pd.notna(v) else v
        )
    return df


def _info(has_resources, has_case_attrs):
    return {"has_resources": has_resources, "has_case_attrs": has_case_attrs}


def load_agentsimulator_sims(dataset, n=N_SIMS):
    base = os.path.join(AGENTSIM_DIR, dataset, "orchestrated_IM")
    if not os.path.isdir(base):
        return [], _info(True, False)

    logs = []
    for i in range(n):
        path = os.path.join(base, f"simulated_log_{i}.csv")
        if not os.path.exists(path):
            break
        df = pd.read_csv(path)
        df = df.rename(columns={
            "case_id": "case:concept:name",
            "activity_name": "concept:name",
            "resource": "org:resource",
            "start_timestamp": "start:timestamp",
            "end_timestamp": "time:timestamp",
        })
        df["case:concept:name"] = df["case:concept:name"].astype(str)
        df = _to_utc(df, ["start:timestamp", "time:timestamp"])
        df = _resource_to_str(df)
        keep = ["case:concept:name", "concept:name", "org:resource",
                "start:timestamp", "time:timestamp"]
        logs.append(df[keep])

    return logs, _info(True, False)


def load_simod_sims(dataset, n=N_SIMS, case_attr_cols=None):
    base = os.path.join(SIMOD_DIR, dataset, "best_result", "evaluation")
    if not os.path.isdir(base):
        return [], _info(True, False)

    case_attr_cols = case_attr_cols or []
    logs = []
    has_attrs_any = False
    for i in range(n):
        path = os.path.join(base, f"simulated_log_{i}.csv")
        if not os.path.exists(path):
            break
        df = pd.read_csv(path)
        df = df.rename(columns={
            "case_id": "case:concept:name",
            "activity": "concept:name",
            "resource": "org:resource",
            "start_time": "start:timestamp",
            "end_time": "time:timestamp",
        })
        df["case:concept:name"] = df["case:concept:name"].astype(str)
        df = _to_utc(df, ["start:timestamp", "time:timestamp"])
        df = _resource_to_str(df)

        # SIMOD writes case attributes without the pm4py ``case:`` prefix —
        # re-add it so downstream checks against ``params.label_data_attributes``
        # (which use the prefixed form) match.
        attr_rename = {
            a.removeprefix("case:"): a
            for a in case_attr_cols
            if a.startswith("case:") and a.removeprefix("case:") in df.columns
        }
        if attr_rename:
            df = df.rename(columns=attr_rename)

        present_attrs = [c for c in case_attr_cols if c in df.columns]
        if present_attrs:
            has_attrs_any = True
        keep = ["case:concept:name", "concept:name", "org:resource",
                "start:timestamp", "time:timestamp"] + present_attrs
        logs.append(df[keep])

    return logs, _info(True, has_attrs_any)


def load_rims_sims(dataset, n=N_SIMS, case_attr_cols=None):
    """Load RIMS simulations from the SPECIAL_ISSUE result bundle.

    Path: ``RESULT_RIMS_SPECIAL_ISSUE/RESULT_RIMS_SPECIAL_ISSUE/<dataset>/results/rims_last_payload_prob/``.
    Filenames are ``simulated_log_LSTM_<dataset><i>.csv`` for ``i`` in
    ``0..9``. RIMS doesn't emit a real ``org:resource`` (only a coarse
    ``role`` label), but for datasets in ``RIMS_ATTRIB_RENAME`` it carries
    case attributes inside the ``attrib`` column.
    """
    rims_dataset = RIMS_DATASET_ALIAS.get(dataset, dataset)
    base = os.path.join(RIMS_DIR, rims_dataset, "results", "rims_last_payload_prob")
    if not os.path.isdir(base):
        return [], _info(False, False)

    # Dataset aliases like ``BPI_Challenge_2012_W_Two_TS_1`` map back to the
    # plain dataset stem in the filename (no trailing ``_1``).
    filename_stem = re.sub(r"_1$", "", rims_dataset)

    rename_map = RIMS_ATTRIB_RENAME.get(dataset, {})
    required_attrs = list(case_attr_cols or [])
    if required_attrs:
        rename_map = {k: v for k, v in rename_map.items() if v in required_attrs}

    logs = []
    added_per_log = []
    for i in range(n):
        path = os.path.join(base, f"simulated_log_LSTM_{filename_stem}{i}.csv")
        if not os.path.exists(path):
            alt = os.path.join(base, f"simulated_log_LSTM_{rims_dataset}{i}.csv")
            if os.path.exists(alt):
                path = alt
            else:
                break
        df = pd.read_csv(path)
        df = df.rename(columns={
            "caseid": "case:concept:name",
            "task": "concept:name",
        })
        df["case:concept:name"] = df["case:concept:name"].astype(str)
        df = _to_utc(df, ["start:timestamp", "time:timestamp"])

        added_attrs = _parse_rims_attribs(df, rename_map)
        added_per_log.append(set(added_attrs))
        keep = ["case:concept:name", "concept:name",
                "start:timestamp", "time:timestamp"] + added_attrs
        logs.append(df[keep])

    # Match the run_experiments simod check: has_case_attrs = True only when
    # every required attr was successfully extracted in every loaded log,
    # otherwise the tree evaluators would silently fill the missing ones with
    # 0 and produce biased numbers.
    has_attrs = bool(required_attrs) and all(
        all(a in s for a in required_attrs) for s in added_per_log
    ) and bool(added_per_log)

    return logs, _info(False, has_attrs)


def load_dsim_sims(dataset, n=N_SIMS):
    dsim_dataset = DSIM_DATASET_ALIAS.get(dataset, dataset)
    base = os.path.join(DSIM_DIR, dsim_dataset)
    if not os.path.isdir(base):
        return [], _info(False, False)

    filename_stem = DSIM_FILENAME_STEM_ALIAS.get(dataset, dsim_dataset)

    logs = []
    for i in range(1, n + 1):
        path = os.path.join(base, f"gen_{filename_stem}_{i}.csv")
        if not os.path.exists(path):
            break
        df = pd.read_csv(path)
        df = df.rename(columns={
            "caseid": "case:concept:name",
            "task": "concept:name",
        })
        df["case:concept:name"] = df["case:concept:name"].astype(str)
        df = _to_utc(df, ["start:timestamp", "time:timestamp"])
        keep = ["case:concept:name", "concept:name",
                "start:timestamp", "time:timestamp"]
        logs.append(df[keep])

    return logs, _info(False, False)
