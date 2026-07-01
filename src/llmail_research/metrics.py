import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def safe_auc(y_true, scores) -> float:
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def safe_average_precision(y_true, scores) -> float:
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def tune_threshold_f1(y_true, scores) -> tuple[float, float]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    if len(np.unique(y_true)) < 2:
        return 0.5, float("nan")
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5, float("nan")
    f1_values = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = int(np.nanargmax(f1_values))
    return float(thresholds[best_idx]), float(f1_values[best_idx])


def metrics_from_scores(y_true, scores, threshold: float, prefix: str = "") -> dict:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        f"{prefix}threshold": threshold,
        f"{prefix}accuracy": float(accuracy_score(y_true, pred)),
        f"{prefix}precision": float(precision_score(y_true, pred, zero_division=0)),
        f"{prefix}recall": float(recall_score(y_true, pred, zero_division=0)),
        f"{prefix}f1": float(f1_score(y_true, pred, zero_division=0)),
        f"{prefix}roc_auc": safe_auc(y_true, scores),
        f"{prefix}avg_precision": safe_average_precision(y_true, scores),
        f"{prefix}tn": int(tn),
        f"{prefix}fp": int(fp),
        f"{prefix}fn": int(fn),
        f"{prefix}tp": int(tp),
    }
