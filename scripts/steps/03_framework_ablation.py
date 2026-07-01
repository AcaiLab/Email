import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from llmail_research.pipeline import run_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final framework ablation.")
    parser.add_argument("--quick", action="store_true", help="Use capped test sets for a smoke run.")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-stage-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None, help="Use 0 for no cap.")
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--max-fpr", type=float, default=0.01)
    args = parser.parse_args()

    max_train_rows = args.max_train_rows or (20_000 if args.quick else 60_000)
    max_stage_rows = args.max_stage_rows or (20_000 if args.quick else 70_000)
    max_test_rows = args.max_test_rows
    if max_test_rows is None:
        max_test_rows = 2_000 if args.quick else 0
    max_features = args.max_features or (20_000 if args.quick else 60_000)

    run_experiment(
        "framework_ablation",
        [
            "--max-train-rows",
            str(max_train_rows),
            "--max-stage-rows",
            str(max_stage_rows),
            "--max-test-rows",
            str(max_test_rows),
            "--max-features",
            str(max_features),
            "--max-fpr",
            str(args.max_fpr),
        ],
    )


if __name__ == "__main__":
    main()
