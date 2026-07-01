import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from llmail_research.pipeline import run_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage-motivation experiments.")
    parser.add_argument("--quick", action="store_true", help="Use smaller caps for a smoke run.")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-features", type=int, default=None)
    args = parser.parse_args()

    max_rows = args.max_rows or (20_000 if args.quick else 80_000)
    max_features = args.max_features or (10_000 if args.quick else 30_000)

    run_experiment("dataset_summary")
    for stage in ["random", "phase"]:
        run_experiment(
            "stage_transfer",
            ["--stage", stage, "--max-rows", str(max_rows), "--max-features", str(max_features)],
        )
    run_experiment(
        "phase_diagnostics",
        ["--max-train-rows", str(max_rows), "--max-features", str(max_features)],
    )
    run_experiment(
        "stage_transitions",
        ["--mode", "all", "--max-rows", str(max_rows), "--max-features", str(max_features)],
    )


if __name__ == "__main__":
    main()
