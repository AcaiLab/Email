import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from llmail_research.pipeline import run_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run optional OpenAI LLM-judge baseline.")
    parser.add_argument("--model", default=os.environ.get("OPENAI_LLM_JUDGE_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--quick", action="store_true", help="Run a small smoke test.")
    parser.add_argument("--full", action="store_true", help="Use full available test/calibration sets.")
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--max-calibration-rows", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-fpr", type=float, default=0.01)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Export it before running this optional step.")

    if args.full:
        max_test_rows = args.max_test_rows or 10_000_000
        max_calibration_rows = args.max_calibration_rows or 10_000_000
    elif args.quick:
        max_test_rows = args.max_test_rows or 30
        max_calibration_rows = args.max_calibration_rows or 30
    else:
        max_test_rows = args.max_test_rows or 200
        max_calibration_rows = args.max_calibration_rows or 113

    run_experiment(
        "openai_judge",
        [
            "--model",
            args.model,
            "--max-test-rows",
            str(max_test_rows),
            "--max-calibration-rows",
            str(max_calibration_rows),
            "--max-fpr",
            str(args.max_fpr),
            "--workers",
            str(args.workers),
        ],
    )


if __name__ == "__main__":
    main()
