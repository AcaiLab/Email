import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from llmail_research.pipeline import run_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LLMail and external detector datasets.")
    parser.add_argument("--force", action="store_true", help="Rebuild cached parquet files.")
    parser.add_argument("--sample-rows", type=int, default=None, help="Optional LLMail sample size for smoke tests.")
    args = parser.parse_args()

    prepare_args = []
    if args.force:
        prepare_args.append("--force")
    if args.sample_rows:
        prepare_args.extend(["--sample-rows", str(args.sample_rows)])
    run_experiment("prepare_datasets", prepare_args)


if __name__ == "__main__":
    main()
