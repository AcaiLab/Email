import numpy as np
import pandas as pd


def capture_at_top_percent(y_true, scores, percent: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    k = max(1, int(np.ceil(len(scores) * percent / 100)))
    order = np.argsort(scores)[::-1][:k]
    positives = int(y_true.sum())
    captured = int(y_true[order].sum())
    key = str(percent).replace(".", "_")
    return {
        f"top_{key}_percent_k": int(k),
        f"top_{key}_percent_precision": float(captured / k),
        f"top_{key}_percent_recall": float(captured / positives) if positives else 0.0,
        f"top_{key}_percent_captured": captured,
    }


def review_budget_rows(y_true, scores, budgets=(0.1, 0.5, 1.0, 5.0, 10.0)) -> list[dict]:
    rows = []
    for budget in budgets:
        row = {"review_budget_percent": budget}
        row.update(capture_at_top_percent(y_true, scores, budget))
        rows.append(row)
    return rows


def threshold_sweep_rows(y_true, scores, thresholds=None) -> list[dict]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    if thresholds is None:
        thresholds = np.unique(np.quantile(scores, np.linspace(0, 1, 101)))
    rows = []
    for threshold in thresholds:
        pred = scores >= threshold
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        rows.append(
            {
                "threshold": float(threshold),
                "predicted_positive_rate": float(pred.mean()),
                "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
                "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
                "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
                "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )
    return rows


def text_snippet(series: pd.Series, max_chars: int = 700) -> pd.Series:
    return series.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.slice(0, max_chars)
