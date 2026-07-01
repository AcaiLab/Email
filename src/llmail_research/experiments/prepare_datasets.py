import argparse
from pathlib import Path

import pandas as pd

from llmail_research.config import CLEAN_DATASET_PATH, PROCESSED_DIR
from llmail_research.data import prepare_clean_dataset
from llmail_research.external_datasets import (
    load_neuralchemy_core,
    load_nvidia_agentic_ipi,
    load_promptshield,
    load_shieldlm,
    prepare_neuralchemy_core,
    prepare_nvidia_agentic_ipi,
    prepare_promptshield,
    prepare_shieldlm,
)


def summarize_frame(name: str, path: Path, frame: pd.DataFrame) -> dict:
    attack_rate = frame["label_attack"].mean() if "label_attack" in frame.columns else None
    return {
        "dataset": name,
        "rows": len(frame),
        "attack_rows": int(frame["label_attack"].sum()) if "label_attack" in frame.columns else None,
        "benign_rows": int((1 - frame["label_attack"]).sum()) if "label_attack" in frame.columns else None,
        "attack_rate": attack_rate,
        "splits": ", ".join(map(str, sorted(frame["split"].dropna().unique()))) if "split" in frame.columns else "",
        "domains": frame["domain"].nunique() if "domain" in frame.columns else None,
        "path": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LLMail and external evaluation datasets.")
    parser.add_argument("--force", action="store_true", help="Rebuild processed parquet files.")
    parser.add_argument("--sample-rows", type=int, default=None, help="Optional LLMail sample size for smoke tests.")
    args = parser.parse_args()

    llmail_path = prepare_clean_dataset(force=args.force, sample_rows=args.sample_rows)
    print(f"Prepared LLMail dataset: {llmail_path}")
    if llmail_path == CLEAN_DATASET_PATH:
        print("This is the canonical cleaned LLMail dataset used by experiments.")

    preparations = [
        ("NVIDIA_Agentic_IPI", prepare_nvidia_agentic_ipi, load_nvidia_agentic_ipi),
        ("PromptShield", prepare_promptshield, load_promptshield),
        ("ShieldLM", prepare_shieldlm, load_shieldlm),
        ("Neuralchemy_PI_Core", prepare_neuralchemy_core, load_neuralchemy_core),
    ]

    summaries = []
    for name, prepare_fn, load_fn in preparations:
        print(f"Preparing {name}...")
        path = prepare_fn(force=args.force)
        frame = load_fn()
        summary = summarize_frame(name, path, frame)
        summaries.append(summary)
        print(
            f"  rows={summary['rows']:,} attack={summary['attack_rows']:,} "
            f"benign={summary['benign_rows']:,} path={path}"
        )

    summary_path = PROCESSED_DIR / "external_dataset_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    print(f"Saved external dataset summary: {summary_path}")


if __name__ == "__main__":
    main()
