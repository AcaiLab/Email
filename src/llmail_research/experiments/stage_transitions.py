import argparse

import pandas as pd

from llmail_research.config import RESULTS_DIR
from llmail_research.data import load_clean_dataset
from llmail_research.modeling import prepare_modeling_frame, run_split_experiment, save_results, split_random


TRANSITIONS = [
    {
        "name": "retrieved_to_bypass",
        "filter_col": "email_retrieved",
        "filter_value": 1,
        "target": "defense_bypassed",
        "description": "Given retrieval, did the defense miss the injection?",
    },
    {
        "name": "bypass_to_tool_call",
        "filter_col": "defense_bypassed",
        "filter_value": 1,
        "target": "tool_called",
        "description": "Given defense bypass, did the model call the tool?",
    },
    {
        "name": "tool_call_to_destination",
        "filter_col": "tool_called",
        "filter_value": 1,
        "target": "destination_correct",
        "description": "Given a tool call, was the destination correct?",
    },
    {
        "name": "tool_call_to_content",
        "filter_col": "tool_called",
        "filter_value": 1,
        "target": "content_correct",
        "description": "Given a tool call, was the content correct?",
    },
    {
        "name": "tool_call_to_full_success",
        "filter_col": "tool_called",
        "filter_value": 1,
        "target": "full_success",
        "description": "Given a tool call, did the attack fully succeed?",
    },
]


def run_random_transition(data: pd.DataFrame, transition: dict, max_rows: int, max_features: int) -> pd.DataFrame:
    subset = data[data[transition["filter_col"]].eq(transition["filter_value"])].reset_index(drop=True)
    model_df = prepare_modeling_frame(subset, transition["target"], max_rows=max_rows)
    train, val, test = split_random(model_df, transition["target"])
    result = run_split_experiment(
        train,
        val,
        test,
        transition["target"],
        f"stage_random_{transition['name']}",
        models_to_use=["SGD_Logistic"],
        max_features=max_features,
    )
    result["transition"] = transition["name"]
    result["condition"] = f"{transition['filter_col']} == {transition['filter_value']}"
    result["description"] = transition["description"]
    return result


def run_phase_transition(data: pd.DataFrame, transition: dict, max_rows: int, max_features: int) -> pd.DataFrame:
    subset = data[data[transition["filter_col"]].eq(transition["filter_value"])].reset_index(drop=True)
    model_df = prepare_modeling_frame(subset, transition["target"], max_rows=max_rows)
    phase1 = model_df[model_df["phase"].eq("Phase1")].reset_index(drop=True)
    phase2 = model_df[model_df["phase"].eq("Phase2")].reset_index(drop=True)
    if len(phase1) == 0 or len(phase2) == 0:
        return pd.DataFrame()
    train, val, _ = split_random(phase1, transition["target"])
    result = run_split_experiment(
        train,
        val,
        phase2,
        transition["target"],
        f"stage_phase_transfer_{transition['name']}",
        models_to_use=["SGD_Logistic"],
        max_features=max_features,
    )
    result["transition"] = transition["name"]
    result["condition"] = f"{transition['filter_col']} == {transition['filter_value']}"
    result["description"] = transition["description"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run conditional stage-transition detector experiments.")
    parser.add_argument("--mode", choices=["random", "phase", "all"], default="all")
    parser.add_argument("--max-rows", type=int, default=80_000)
    parser.add_argument("--max-features", type=int, default=30_000)
    args = parser.parse_args()

    data = load_clean_dataset()
    rows = []
    for transition in TRANSITIONS:
        if args.mode in {"random", "all"}:
            rows.append(run_random_transition(data, transition, args.max_rows, args.max_features))
        if args.mode in {"phase", "all"}:
            rows.append(run_phase_transition(data, transition, args.max_rows, args.max_features))
    result = pd.concat([frame for frame in rows if len(frame)], ignore_index=True)
    output = RESULTS_DIR / "models" / "stage_transition_results.csv"
    save_results(result, output)
    print(result[["experiment", "transition", "target", "n_test", "test_f1", "test_roc_auc", "test_precision", "test_recall"]].to_string(index=False))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
