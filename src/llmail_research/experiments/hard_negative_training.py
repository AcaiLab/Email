from __future__ import annotations

import argparse
import time

import pandas as pd
from sklearn.linear_model import LogisticRegression

from llmail_research.config import RESULTS_DIR, SEED
from llmail_research.data import load_clean_dataset
from llmail_research.external_datasets import (
    load_bipia_binary,
    load_llmail_binary,
    load_neuralchemy_core,
    load_notinject,
    load_nvidia_agentic_ipi,
    load_promptshield,
    load_shieldlm,
)
from llmail_research.metrics import tune_threshold_f1
from llmail_research.eval_utils import (
    NOTINJECT_ROTATIONS,
    TARGET,
    build_matched_test_sets,
    strict_threshold_from_benign,
)

from llmail_research.experiments.framework_ablation import (
    FINAL_LINEAR,
    build_training_partitions,
    evaluate_one,
    feature_variants,
    score_features,
    summarize_one_run,
)
from llmail_research.experiments.framework_core import HybridDetector, TextDetector, fit_stage_models


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run leakage-free NotInject hard-negative ablation."
    )
    parser.add_argument("--max-train-rows", type=int, default=60_000)
    parser.add_argument("--max-meta-rows", type=int, default=10_000)
    parser.add_argument("--max-threshold-rows", type=int, default=2_000)
    parser.add_argument("--max-stage-rows", type=int, default=70_000)
    parser.add_argument("--max-test-rows", type=int, default=2_000)
    parser.add_argument("--max-features", type=int, default=60_000)
    parser.add_argument("--max-fpr", type=float, default=0.01)
    args = parser.parse_args()

    start = time.time()
    rotation = NOTINJECT_ROTATIONS[0]
    llmail = load_llmail_binary()
    llmail_clean = load_clean_dataset()
    bipia = load_bipia_binary()
    notinject = load_notinject()
    nvidia = load_nvidia_agentic_ipi()
    promptshield = load_promptshield()
    shieldlm = load_shieldlm()
    neuralchemy = load_neuralchemy_core()
    base, meta, threshold_val, calibration, stage_notinject = build_training_partitions(
        llmail,
        bipia,
        notinject,
        rotation,
        SEED,
        args.max_train_rows,
        args.max_meta_rows,
        args.max_threshold_rows,
    )
    test_sets = build_matched_test_sets(
        llmail,
        bipia,
        notinject,
        nvidia,
        promptshield,
        shieldlm,
        neuralchemy,
        rotation.test_split,
        args.max_test_rows,
        SEED,
    )
    all_results = []
    for condition, use_notinject in [("with_notinject_training", True), ("without_notinject_training", False)]:
        print(condition, flush=True)
        condition_base = base if use_notinject else base[~base["source_dataset"].eq("NotInject")].reset_index(drop=True)
        condition_meta = meta if use_notinject else meta[~meta["source_dataset"].eq("NotInject")].reset_index(drop=True)
        condition_stage_notinject = stage_notinject if use_notinject else stage_notinject.iloc[0:0].copy()
        detectors = {
            "word": TextDetector("word", args.max_features, random_state=SEED).fit(condition_base),
            "char": TextDetector("char", args.max_features, random_state=SEED).fit(condition_base),
            "word_char_rules": HybridDetector(args.max_features, random_state=SEED).fit(condition_base),
        }
        stage_models = fit_stage_models(
            llmail_clean[llmail_clean["phase"].eq("Phase1")].reset_index(drop=True),
            condition_stage_notinject,
            args.max_stage_rows,
            args.max_features // 2,
            random_state=SEED,
        )
        x_meta = score_features(condition_meta, detectors, stage_models)
        columns = feature_variants(list(x_meta.columns))[FINAL_LINEAR]
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
        model.fit(x_meta[columns], condition_meta[TARGET].astype(int))
        x_threshold = score_features(threshold_val, detectors, stage_models)
        review_threshold = tune_threshold_f1(
            threshold_val[TARGET].astype(int), model.predict_proba(x_threshold[columns])[:, 1]
        )[0]
        x_calibration = score_features(calibration, detectors, stage_models)
        strict_threshold = strict_threshold_from_benign(
            model.predict_proba(x_calibration[columns])[:, 1], args.max_fpr
        )[0]
        x_tests = {name: score_features(frame, detectors, stage_models) for name, frame in test_sets.items()}
        for policy, selected_threshold in [
            ("validation_f1", review_threshold),
            (f"notinject_fpr_{args.max_fpr:.2f}", strict_threshold),
        ]:
            rows, _ = evaluate_one(
                condition,
                policy,
                selected_threshold,
                test_sets,
                x_tests,
                model,
                columns,
                SEED,
                rotation,
            )
            all_results.extend(rows)

    detailed = pd.DataFrame(all_results)
    summary = summarize_one_run(detailed)
    output_dir = RESULTS_DIR / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(output_dir / "hard_negative_ablation_detailed.csv", index=False)
    summary.to_csv(output_dir / "hard_negative_ablation_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    print(f"Finished in {(time.time() - start) / 60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
