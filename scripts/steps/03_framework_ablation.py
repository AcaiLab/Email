import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from llmail_research.pipeline import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run final framework ablation.")
    parser.add_argument("--quick", action="store_true", help="Run capped test sets.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-meta-rows", type=int, default=None)
    parser.add_argument("--max-threshold-rows", type=int, default=None)
    parser.add_argument("--max-stage-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--max-fpr", type=float, default=0.01)
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    parser.add_argument("--padding-sample-rows", type=int, default=None)
    parser.add_argument("--chunk-words", type=int, default=256)
    args = parser.parse_args()

    seeds = args.seeds or ([2026] if args.quick else [2026, 2027, 2028, 2029, 2030])
    max_train_rows = args.max_train_rows or (20_000 if args.quick else 60_000)
    max_meta_rows = args.max_meta_rows or (4_000 if args.quick else 10_000)
    max_threshold_rows = args.max_threshold_rows or (500 if args.quick else 2_000)
    max_stage_rows = args.max_stage_rows or (20_000 if args.quick else 70_000)
    max_test_rows = args.max_test_rows or (500 if args.quick else 2_000)
    max_features = args.max_features or (20_000 if args.quick else 60_000)
    bootstrap_samples = args.bootstrap_samples or (100 if args.quick else 1_000)
    padding_sample_rows = args.padding_sample_rows or (100 if args.quick else 500)

    run_experiment(
        "framework_ablation",
        [
            "--seeds",
            *[str(seed) for seed in seeds],
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
            "--bootstrap-samples",
            str(bootstrap_samples),
            "--padding-sample-rows",
            str(padding_sample_rows),
            "--chunk-words",
            str(args.chunk_words),
        ],
    )


if __name__ == "__main__":
    main()
