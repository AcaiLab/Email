import argparse

import pandas as pd

from llmail_research.config import RESULTS_DIR, TARGET_COLUMNS
from llmail_research.data import load_clean_dataset
from llmail_research.modeling import prepare_modeling_frame, run_split_experiment, save_results, split_random


def run_random(data: pd.DataFrame, max_rows: int, max_features: int) -> pd.DataFrame:
    rows = []
    for target in TARGET_COLUMNS:
        model_df = prepare_modeling_frame(data, target, max_rows=max_rows)
        train, val, test = split_random(model_df, target)
        rows.append(run_split_experiment(train, val, test, target, f"random_split_{target}", max_features=max_features))
    return pd.concat(rows, ignore_index=True)


def run_phase_transfer(data: pd.DataFrame, max_rows: int, max_features: int) -> pd.DataFrame:
    rows = []
    for target in TARGET_COLUMNS:
        model_df = prepare_modeling_frame(data, target, max_rows=max_rows)
        phase1 = model_df[model_df["phase"].eq("Phase1")].reset_index(drop=True)
        phase2 = model_df[model_df["phase"].eq("Phase2")].reset_index(drop=True)
        if len(phase1) == 0 or len(phase2) == 0:
            continue
        train, val, _ = split_random(phase1, target)
        rows.append(run_split_experiment(train, val, phase2, target, f"phase_transfer_phase1_to_phase2_{target}", max_features=max_features))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_loso(data: pd.DataFrame, max_rows: int, max_features: int) -> pd.DataFrame:
    rows = []
    for target in TARGET_COLUMNS:
        model_df = prepare_modeling_frame(data, target, max_rows=max_rows)
        for holdout in sorted(group for group in model_df["scenario_group"].dropna().unique() if group != "unknown"):
            train_source = model_df[~model_df["scenario_group"].eq(holdout)].reset_index(drop=True)
            test = model_df[model_df["scenario_group"].eq(holdout)].reset_index(drop=True)
            if len(train_source) == 0 or len(test) == 0:
                continue
            train, val, _ = split_random(train_source, target)
            result = run_split_experiment(train, val, test, target, f"leave_one_scenario_group_out_{holdout}_{target}", max_features=max_features)
            if len(result):
                result["holdout_scenario_group"] = holdout
                rows.append(result)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_hybrid(data: pd.DataFrame, max_rows: int, max_features: int) -> pd.DataFrame:
    rows = []
    for target in TARGET_COLUMNS:
        model_df = prepare_modeling_frame(data, target, max_rows=max_rows)
        train, val, test = split_random(model_df, target)
        rows.append(
            run_split_experiment(
                train,
                val,
                test,
                target,
                f"hybrid_tfidf_heuristics_{target}",
                hybrid=True,
                models_to_use=["SGD_Logistic"],
                max_features=max_features,
            )
        )
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLMail text detector experiments.")
    parser.add_argument("--stage", choices=["random", "phase", "loso", "hybrid", "all"], default="all")
    parser.add_argument("--max-rows", type=int, default=80_000)
    parser.add_argument("--max-features", type=int, default=30_000)
    args = parser.parse_args()

    data = load_clean_dataset()
    stage_map = {
        "random": run_random,
        "phase": run_phase_transfer,
        "loso": run_loso,
        "hybrid": run_hybrid,
    }
    selected = list(stage_map) if args.stage == "all" else [args.stage]
    for stage in selected:
        print(f"\n=== Running {stage} experiments ===")
        result = stage_map[stage](data, args.max_rows, args.max_features)
        path = RESULTS_DIR / "models" / f"{stage}_results.csv"
        save_results(result, path)
        if len(result):
            print(result[["experiment", "target", "model", "test_f1", "test_roc_auc", "test_precision", "test_recall"]].to_string(index=False))
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
