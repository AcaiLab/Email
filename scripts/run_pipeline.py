import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from llmail_research.pipeline import PROJECT_ROOT, run_step  # noqa: E402


STEPS = [
    ("00", "prepare datasets", PROJECT_ROOT / "scripts" / "steps" / "00_prepare_datasets.py"),
    ("01", "stage analysis", PROJECT_ROOT / "scripts" / "steps" / "01_stage_analysis.py"),
    ("02", "hard-negative evaluation", PROJECT_ROOT / "scripts" / "steps" / "02_hard_negative_evaluation.py"),
    ("03", "framework ablation", PROJECT_ROOT / "scripts" / "steps" / "03_framework_ablation.py"),
    ("04", "detector baselines", PROJECT_ROOT / "scripts" / "steps" / "04_detector_baselines.py"),
]
OPENAI_STEP = ("05", "OpenAI LLM judge", PROJECT_ROOT / "scripts" / "steps" / "05_openai_judge_optional.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the email prompt-injection experiment pipeline.")
    parser.add_argument("--quick", action="store_true", help="Use smaller caps for a smoke run.")
    parser.add_argument("--force-data", action="store_true", help="Rebuild cached dataset parquet files.")
    parser.add_argument("--skip-detectors", action="store_true", help="Skip published detector baselines.")
    parser.add_argument("--include-openai", action="store_true", help="Also run the paid OpenAI LLM judge step.")
    parser.add_argument("--start-at", choices=[step[0] for step in STEPS] + ["05"], default="00")
    args = parser.parse_args()

    selected_steps = list(STEPS)
    if args.skip_detectors:
        selected_steps = [step for step in selected_steps if step[0] != "04"]
    if args.include_openai:
        selected_steps.append(OPENAI_STEP)

    selected_steps = [step for step in selected_steps if step[0] >= args.start_at]
    for step_id, description, path in selected_steps:
        step_args: list[str] = []
        if args.quick:
            step_args.append("--quick")
        if step_id == "00" and args.force_data:
            step_args.append("--force")
        print(f"\n=== Step {step_id}: {description} ===", flush=True)
        run_step(path, step_args)

    print("\nPipeline finished.")
    print(f"Results: {PROJECT_ROOT / 'results'}")


if __name__ == "__main__":
    main()
