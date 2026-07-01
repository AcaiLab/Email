import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

from llmail_research.config import RESULTS_DIR, SEED, TARGET_COLUMNS
from llmail_research.data import load_clean_dataset
from llmail_research.features import make_vectorizer
from llmail_research.metrics import metrics_from_scores, tune_threshold_f1
from llmail_research.modeling import get_scores, prepare_modeling_frame


def sample_phase1(frame: pd.DataFrame, target: str, max_train_rows: int) -> pd.DataFrame:
    phase1 = frame[frame["phase"].eq("Phase1")].reset_index(drop=True)
    if len(phase1) <= max_train_rows:
        return phase1
    pieces = []
    for _, group in phase1.groupby(target):
        n = max(1, int(max_train_rows * len(group) / len(phase1)))
        pieces.append(group.sample(min(len(group), n), random_state=SEED))
    return pd.concat(pieces, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)


def capture_at_top_percent(y_true, scores, percent: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    k = max(1, int(np.ceil(len(scores) * percent / 100)))
    order = np.argsort(scores)[::-1][:k]
    positives = y_true.sum()
    captured = int(y_true[order].sum())
    return {
        f"top_{str(percent).replace('.', '_')}_percent_k": int(k),
        f"top_{str(percent).replace('.', '_')}_percent_precision": float(captured / k),
        f"top_{str(percent).replace('.', '_')}_percent_recall": float(captured / positives) if positives else 0.0,
        f"top_{str(percent).replace('.', '_')}_percent_captured": captured,
    }


def run_target(data: pd.DataFrame, target: str, max_train_rows: int, max_features: int) -> dict:
    model_df = prepare_modeling_frame(data, target, max_rows=None)
    phase1 = sample_phase1(model_df, target, max_train_rows)
    phase2 = model_df[model_df["phase"].eq("Phase2")].reset_index(drop=True)

    y = phase1[target].astype(int)
    train, val = train_test_split(
        phase1,
        test_size=0.20,
        random_state=SEED,
        stratify=y if y.nunique() == 2 and y.value_counts().min() >= 2 else None,
    )

    vectorizer = make_vectorizer(max_features=max_features)
    x_train = vectorizer.fit_transform(train["text"])
    x_val = vectorizer.transform(val["text"])
    x_test = vectorizer.transform(phase2["text"])

    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        max_iter=40,
        tol=1e-3,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED,
    )
    classifier.fit(x_train, train[target].astype(int).to_numpy())
    val_scores = get_scores(classifier, x_val)
    test_scores = get_scores(classifier, x_test)

    val_threshold, val_best_f1 = tune_threshold_f1(val[target], val_scores)
    oracle_threshold, oracle_best_f1 = tune_threshold_f1(phase2[target], test_scores)

    row = {
        "target": target,
        "model": "SGD_Logistic",
        "n_train": len(train),
        "n_val": len(val),
        "n_test_phase2": len(phase2),
        "phase2_prevalence": float(phase2[target].mean()),
        "val_threshold": val_threshold,
        "val_best_f1": val_best_f1,
        "phase2_oracle_threshold": oracle_threshold,
        "phase2_oracle_best_f1": oracle_best_f1,
    }
    row.update(metrics_from_scores(phase2[target], test_scores, val_threshold, prefix="phase2_fixed_"))
    row.update(metrics_from_scores(phase2[target], test_scores, oracle_threshold, prefix="phase2_oracle_"))
    for percent in [0.1, 0.5, 1.0, 5.0]:
        row.update(capture_at_top_percent(phase2[target], test_scores, percent))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Phase 1 to Phase 2 detector transfer and threshold drift.")
    parser.add_argument("--max-train-rows", type=int, default=80_000)
    parser.add_argument("--max-features", type=int, default=30_000)
    args = parser.parse_args()

    data = load_clean_dataset()
    rows = [run_target(data, target, args.max_train_rows, args.max_features) for target in TARGET_COLUMNS]
    result = pd.DataFrame(rows)
    output = RESULTS_DIR / "diagnostics" / "phase_transfer_diagnostics.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
