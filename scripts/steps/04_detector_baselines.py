import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from llmail_research.pipeline import run_experiment


DEFAULT_MODELS = [
    "ProtectAI-DeBERTa-v1",
    "ProtectAI-DeBERTa-v2",
    "deepset-DeBERTa-injection",
    "TestSavant-tiny",
    "Arkaean-DistilBERT",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run published detector baselines.")
    parser.add_argument("--quick", action="store_true", help="Run fewer test rows.")
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-fpr", type=float, default=0.01)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    args = parser.parse_args()

    max_test_rows = args.max_test_rows or (500 if args.quick else 2_000)
    run_experiment(
        "detector_baselines",
        [
            "--max-test-rows",
            str(max_test_rows),
            "--batch-size",
            str(args.batch_size),
            "--max-length",
            str(args.max_length),
            "--max-fpr",
            str(args.max_fpr),
            "--models",
            *args.models,
        ],
    )


if __name__ == "__main__":
    main()
