import argparse

import numpy as np
import pandas as pd

from llmail_research.binary_modeling import TARGET, fit_detector, split_train_val
from llmail_research.config import RESULTS_DIR
from llmail_research.external_datasets import (
    load_bipia_binary,
    load_llmail_binary,
    load_notinject,
    sample_equal_classes,
)
from llmail_research.metrics import metrics_from_scores, tune_threshold_f1
from llmail_research.modeling import get_scores


def build_training_sets(llmail: pd.DataFrame, bipia_train: pd.DataFrame, notinject: pd.DataFrame) -> dict[str, pd.DataFrame]:
    llmail_attacks = llmail[llmail[TARGET].eq(1)]
    llmail_benign = llmail[llmail[TARGET].eq(0)]
    bipia_attacks = bipia_train[bipia_train[TARGET].eq(1)]
    bipia_benign = bipia_train[bipia_train[TARGET].eq(0)]
    notinject_benign = notinject.copy()

    return {
        "llmail_binary": pd.concat([llmail_attacks, llmail_benign], ignore_index=True, sort=False),
        "bipia_train": pd.concat([bipia_attacks, bipia_benign], ignore_index=True, sort=False),
        "llmail_plus_bipia": pd.concat([llmail_attacks, llmail_benign, bipia_attacks, bipia_benign], ignore_index=True, sort=False),
        "llmail_plus_bipia_plus_notinject": pd.concat(
            [llmail_attacks, llmail_benign, bipia_attacks, bipia_benign, notinject_benign],
            ignore_index=True,
            sort=False,
        ),
        "bipia_plus_notinject": pd.concat([bipia_attacks, bipia_benign, notinject_benign], ignore_index=True, sort=False),
    }


def evaluate(bundle: dict, test: pd.DataFrame, train_name: str, test_name: str) -> dict:
    scores = get_scores(bundle["model"], bundle["vectorizer"].transform(test["text"]))
    y_test = test[TARGET].astype(int).to_numpy()
    row = {
        "experiment": f"{train_name}_to_{test_name}",
        "train_source": train_name,
        "test_source": test_name,
        "target": TARGET,
        "n_train": bundle["n_train"],
        "n_val": bundle["n_val"],
        "n_test": len(test),
        "val_threshold": bundle["threshold"],
        "val_best_f1": bundle["val_best_f1"],
        "test_positive_rate": float(y_test.mean()),
        "predicted_positive_rate": float((scores >= bundle["threshold"]).mean()),
        "score_mean": float(scores.mean()),
    }
    if len(np.unique(y_test)) == 2:
        row.update(metrics_from_scores(y_test, scores, bundle["threshold"], prefix="test_"))
        oracle_threshold, oracle_f1 = tune_threshold_f1(y_test, scores)
        row["test_oracle_threshold"] = oracle_threshold
        row["test_oracle_f1"] = oracle_f1
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
                "test_false_positive_rate": float(np.mean(scores >= bundle["threshold"])) if label == 0 else np.nan,
            }
        )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Train binary detectors with and without NotInject hard negatives.")
    parser.add_argument("--max-train-rows", type=int, default=80_000)
    parser.add_argument("--max-features", type=int, default=30_000)
    args = parser.parse_args()

    llmail = load_llmail_binary()
    bipia = load_bipia_binary()
    notinject = load_notinject()
    bipia_train = bipia[bipia["split"].eq("train")].reset_index(drop=True)
    bipia_test = sample_equal_classes(bipia[bipia["split"].eq("test")].reset_index(drop=True), TARGET)
    llmail_eval = sample_equal_classes(llmail, TARGET)

    training_sets = build_training_sets(llmail, bipia_train, notinject)
    tests = {
        "bipia_test": bipia_test,
        "llmail_binary": llmail_eval,
        "notinject": notinject.reset_index(drop=True),
    }

    rows = []
    for train_name, train_frame in training_sets.items():
        sampled = sample_equal_classes(train_frame, TARGET, max_rows=args.max_train_rows)
        train, val = split_train_val(sampled)
        bundle = fit_detector(train, val, max_features=args.max_features)
        bundle["n_train"] = len(train)
        bundle["n_val"] = len(val)
        for test_name, test_frame in tests.items():
            rows.append(evaluate(bundle, test_frame, train_name, test_name))

    result = pd.DataFrame(rows)
    output = RESULTS_DIR / "models" / "hard_negative_training_results.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result[["experiment", "n_train", "n_test", "test_f1", "test_roc_auc", "test_false_positive_rate", "predicted_positive_rate"]].to_string(index=False))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
