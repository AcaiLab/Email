import argparse
import numpy as np
import pandas as pd

from llmail_research.binary_modeling import TARGET, fit_detector, split_train_val
from llmail_research.config import RESULTS_DIR, SEED
from llmail_research.external_datasets import (
    load_bipia_binary,
    load_llmail_binary,
    load_notinject,
    prepare_bipia_binary,
    prepare_llmail_binary,
    prepare_notinject,
    sample_equal_classes,
)
from llmail_research.features import make_vectorizer
from llmail_research.metrics import metrics_from_scores, safe_auc, safe_average_precision, tune_threshold_f1
from llmail_research.modeling import get_scores


def evaluate_external(bundle: dict, test: pd.DataFrame, experiment: str, train_source: str, test_source: str) -> dict:
    scores = get_scores(bundle["model"], bundle["vectorizer"].transform(test["text"]))
    y_test = test[TARGET].astype(int).to_numpy()
    row = {
        "experiment": experiment,
        "target": TARGET,
        "model": "SGD_Logistic",
        "train_source": train_source,
        "test_source": test_source,
        "n_train": bundle["n_train"],
        "n_val": bundle["n_val"],
        "n_test": len(test),
        "n_features": bundle["n_features"],
        "train_seconds": bundle["train_seconds"],
        "val_threshold": bundle["threshold"],
        "val_best_f1": bundle["val_best_f1"],
        "test_positive_rate": float(np.mean(y_test)),
        "test_score_mean": float(np.mean(scores)),
        "test_score_p95": float(np.quantile(scores, 0.95)),
        "predicted_positive_rate": float(np.mean(scores >= bundle["threshold"])),
    }
    if len(np.unique(y_test)) >= 2:
        row.update(metrics_from_scores(y_test, scores, bundle["threshold"], prefix="test_"))
        oracle_threshold, oracle_f1 = tune_threshold_f1(y_test, scores)
        row["test_oracle_threshold"] = oracle_threshold
        row["test_oracle_f1"] = oracle_f1
        row["test_roc_auc_direct"] = safe_auc(y_test, scores)
        row["test_avg_precision_direct"] = safe_average_precision(y_test, scores)
    else:
        label = int(y_test[0]) if len(y_test) else 0
        row.update(
            {
                "test_threshold": bundle["threshold"],
                "test_accuracy": float(np.mean((scores >= bundle["threshold"]).astype(int) == label)),
                "test_precision": np.nan,
                "test_recall": np.nan,
                "test_f1": np.nan,
                "test_roc_auc": np.nan,
                "test_avg_precision": np.nan,
                "test_tn": int(np.sum(scores < bundle["threshold"])) if label == 0 else 0,
                "test_fp": int(np.sum(scores >= bundle["threshold"])) if label == 0 else 0,
                "test_fn": int(np.sum(scores < bundle["threshold"])) if label == 1 else 0,
                "test_tp": int(np.sum(scores >= bundle["threshold"])) if label == 1 else 0,
                "test_false_positive_rate": float(np.mean(scores >= bundle["threshold"])) if label == 0 else np.nan,
                "test_false_negative_rate": float(np.mean(scores < bundle["threshold"])) if label == 1 else np.nan,
            }
        )
    return row


def train_and_eval(train_frame: pd.DataFrame, test_frames: dict[str, pd.DataFrame], name: str, max_train_rows: int, max_features: int) -> list[dict]:
    train_frame = sample_equal_classes(train_frame, TARGET, max_rows=max_train_rows)
    train, val = split_train_val(train_frame)
    bundle = fit_detector(train, val, max_features=max_features)
    bundle["n_train"] = len(train)
    bundle["n_val"] = len(val)
    rows = []
    for test_name, test_frame in test_frames.items():
        rows.append(evaluate_external(bundle, test_frame, f"{name}_to_{test_name}", name, test_name))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cross-dataset prompt-injection detector transfer experiments.")
    parser.add_argument("--max-train-rows", type=int, default=60_000)
    parser.add_argument("--max-test-rows", type=int, default=80_000)
    parser.add_argument("--max-features", type=int, default=30_000)
    parser.add_argument("--force-data", action="store_true")
    args = parser.parse_args()

    print("Preparing normalized datasets...")
    prepare_llmail_binary(force=args.force_data)
    prepare_bipia_binary(force=args.force_data)
    prepare_notinject(force=args.force_data)

    llmail = load_llmail_binary()
    bipia = load_bipia_binary()
    notinject = load_notinject()

    bipia_train = bipia[bipia["split"].eq("train")].reset_index(drop=True)
    bipia_test = sample_equal_classes(bipia[bipia["split"].eq("test")].reset_index(drop=True), TARGET, args.max_test_rows)
    llmail_eval = sample_equal_classes(llmail, TARGET, args.max_test_rows)
    notinject_eval = notinject.reset_index(drop=True)

    combined_train = pd.concat([llmail, bipia_train], ignore_index=True, sort=False)
    rows = []
    rows.extend(
        train_and_eval(
            llmail,
            {"bipia_test": bipia_test, "notinject": notinject_eval},
            "llmail_binary",
            args.max_train_rows,
            args.max_features,
        )
    )
    rows.extend(
        train_and_eval(
            bipia_train,
            {"bipia_test": bipia_test, "llmail_binary": llmail_eval, "notinject": notinject_eval},
            "bipia_train",
            args.max_train_rows,
            args.max_features,
        )
    )
    rows.extend(
        train_and_eval(
            combined_train,
            {"bipia_test": bipia_test, "llmail_binary": llmail_eval, "notinject": notinject_eval},
            "llmail_plus_bipia",
            args.max_train_rows,
            args.max_features,
        )
    )

    result = pd.DataFrame(rows)
    output = RESULTS_DIR / "models" / "cross_dataset_results.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result[["experiment", "n_train", "n_test", "test_positive_rate", "test_f1", "test_roc_auc", "predicted_positive_rate"]].to_string(index=False))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
