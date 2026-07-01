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
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--max-features", type=int, default=None)
    args = parser.parse_args()

    max_train_rows = args.max_train_rows or (20_000 if args.quick else 80_000)
    max_test_rows = args.max_test_rows or (20_000 if args.quick else 80_000)
    max_features = args.max_features or (10_000 if args.quick else 30_000)

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
        ["--max-train-rows", str(max_train_rows), "--max-features", str(max_features)],
    )


if __name__ == "__main__":
    main()
