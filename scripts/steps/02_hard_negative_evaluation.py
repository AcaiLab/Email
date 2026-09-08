import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from llmail_research.pipeline import run_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run transfer and NotInject hard-negative experiments.")
    parser.add_argument("--quick", action="store_true", help="Use smaller caps for a smoke run.")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-meta-rows", type=int, default=None)
    parser.add_argument("--max-threshold-rows", type=int, default=None)
    parser.add_argument("--max-stage-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--max-fpr", type=float, default=0.01)
    args = parser.parse_args()

    max_train_rows = args.max_train_rows or (20_000 if args.quick else 80_000)
    max_meta_rows = args.max_meta_rows or (4_000 if args.quick else 10_000)
    max_threshold_rows = args.max_threshold_rows or (500 if args.quick else 2_000)
    max_stage_rows = args.max_stage_rows or (20_000 if args.quick else 70_000)
    max_test_rows = args.max_test_rows or (500 if args.quick else 2_000)
    max_features = args.max_features or (20_000 if args.quick else 60_000)

    run_experiment(
        "cross_dataset_transfer",
        [
            "--max-train-rows",
            str(max_train_rows),
            "--max-test-rows",
            str(max_test_rows),
            "--max-features",
            str(max_features),
        ],
    )
    run_experiment(
        "hard_negative_training",
        [
            "--max-train-rows",
            str(max_train_rows),
            "--max-meta-rows",
            str(max_meta_rows),
            "--max-threshold-rows",
            str(max_threshold_rows),
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
