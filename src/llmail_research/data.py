import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from .config import CLEAN_DATASET_PATH, PROCESSED_DIR, RAW_DIR, REPO_ID
from .objectives import add_objective_flags, attack_chain_stage


def ensure_dirs() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, CLEAN_DATASET_PATH.parent]:
        path.mkdir(parents=True, exist_ok=True)


def normalize_text_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["subject_text"] = out.get("subject", "").fillna("").astype(str)
    out["body_text"] = out.get("body", "").fillna("").astype(str)
    out["output_text"] = out.get("output", "").fillna("").astype(str)
    out["scenario_key"] = out.get("scenario", "unknown").fillna("unknown").astype(str)
    out["scenario_group"] = (
        out["scenario_key"].str.extract(r"(level\d+)", flags=re.IGNORECASE, expand=False).fillna("unknown").str.lower()
    )
    out["text"] = (
        "Subject: "
        + out["subject_text"]
        + "\nBody: "
        + out["body_text"]
    )
    return out


def load_raw_splits() -> pd.DataFrame:
    dataset = load_dataset(REPO_ID)
    frames = []
    keep_columns = [
        "RowKey",
        "Timestamp",
        "body",
        "completed_time",
        "job_id",
        "objectives",
        "output",
        "scenario",
        "scheduled_time",
        "started_time",
        "subject",
        "team_id",
    ]
    for split_name, split in dataset.items():
        split_columns = [col for col in keep_columns if col in split.column_names]
        frame = split.select_columns(split_columns).to_pandas()
        frame["phase"] = split_name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def prepare_clean_dataset(force: bool = False, sample_rows: int | None = None) -> Path:
    ensure_dirs()
    output_path = CLEAN_DATASET_PATH
    if sample_rows is not None:
        output_path = PROCESSED_DIR / f"llmail_cleaned_sample_{sample_rows}.parquet"
    if output_path.exists() and not force:
        return output_path

    raw = load_raw_splits()
    if sample_rows is not None and len(raw) > sample_rows:
        raw = raw.sample(sample_rows, random_state=2026).reset_index(drop=True)

    clean = normalize_text_columns(add_objective_flags(raw))
    clean["attack_chain_stage"] = clean.apply(attack_chain_stage, axis=1)
    clean.to_parquet(output_path, index=False)
    return output_path


def load_clean_dataset(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else CLEAN_DATASET_PATH
    if not path.exists():
        prepare_clean_dataset()
    return pd.read_parquet(path)
