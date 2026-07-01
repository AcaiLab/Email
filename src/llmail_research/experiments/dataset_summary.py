import pandas as pd

from llmail_research.config import EDA_SUMMARY_PATH, STAGE_SUMMARY_PATH
from llmail_research.data import load_clean_dataset


def summarize_rates(data: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    targets = [
        "email_retrieved",
        "defense_bypassed",
        "tool_called",
        "destination_correct",
        "content_correct",
        "full_success",
    ]
    rows = []
    for group_key, group in data.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = dict(zip(group_cols, group_key))
        row["n"] = len(group)
        for target in targets:
            row[f"{target}_rate"] = float(group[target].mean())
            row[f"{target}_count"] = int(group[target].sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def main() -> None:
    data = load_clean_dataset()
    EDA_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    summaries = []
    for cols, name in [(["phase"], "phase"), (["scenario_group"], "scenario_group"), (["phase", "scenario_group"], "phase_scenario_group")]:
        summary = summarize_rates(data, cols)
        summary.insert(0, "summary", name)
        summaries.append(summary)
    pd.concat(summaries, ignore_index=True, sort=False).to_csv(EDA_SUMMARY_PATH, index=False)

    stage = data["attack_chain_stage"].value_counts().rename_axis("stage").reset_index(name="n")
    stage["rate"] = stage["n"] / len(data)
    stage.to_csv(STAGE_SUMMARY_PATH, index=False)

    print(f"Rows: {len(data):,}")
    print(f"Saved: {EDA_SUMMARY_PATH}")
    print(f"Saved: {STAGE_SUMMARY_PATH}")
    print(stage.to_string(index=False))


if __name__ == "__main__":
    main()
