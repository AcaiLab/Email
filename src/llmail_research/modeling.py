import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import MaxAbsScaler

from .config import SEED
from .features import build_heuristic_features, make_vectorizer
from .metrics import metrics_from_scores, tune_threshold_f1


def prepare_modeling_frame(data: pd.DataFrame, target_col: str, max_rows: int | None = None) -> pd.DataFrame:
    columns = ["text", target_col, "phase", "scenario_key", "scenario_group", "attack_chain_stage"]
    model_df = data[[col for col in columns if col in data.columns]].dropna(subset=["text", target_col]).copy()
    model_df[target_col] = model_df[target_col].astype(int)
    if max_rows is not None and len(model_df) > max_rows:
        pieces = []
        for _, group in model_df.groupby(target_col):
            n = max(1, int(max_rows * len(group) / len(model_df)))
            pieces.append(group.sample(min(len(group), n), random_state=SEED))
        model_df = pd.concat(pieces, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return model_df


def split_random(model_df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = model_df[target_col].astype(int)
    stratify = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None
    train_val, test = train_test_split(model_df, test_size=0.20, random_state=SEED, stratify=stratify)
    y_train_val = train_val[target_col].astype(int)
    stratify_tv = y_train_val if y_train_val.nunique() == 2 and y_train_val.value_counts().min() >= 2 else None
    train, val = train_test_split(train_val, test_size=0.20, random_state=SEED, stratify=stratify_tv)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def make_models() -> dict:
    return {
        "SGD_Logistic": SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-5,
            max_iter=30,
            tol=1e-3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=SEED,
        ),
        "ComplementNB": ComplementNB(alpha=0.1),
    }


def get_scores(model, matrix):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(matrix)
        return probabilities[:, 1] if probabilities.ndim == 2 and probabilities.shape[1] > 1 else probabilities.ravel()
    raw = model.decision_function(matrix)
    raw = np.asarray(raw, dtype=float)
    return 1 / (1 + np.exp(-np.clip(raw, -30, 30)))


def featurize(train, val, test, *, hybrid: bool = False, max_features: int = 30_000):
    vectorizer = make_vectorizer(max_features=max_features)
    x_train = vectorizer.fit_transform(train["text"])
    x_val = vectorizer.transform(val["text"])
    x_test = vectorizer.transform(test["text"])
    extra = {"vectorizer_features": x_train.shape[1], "heuristic_features": 0}
    if not hybrid:
        return x_train, x_val, x_test, extra

    scaler = MaxAbsScaler()
    h_train = scaler.fit_transform(build_heuristic_features(train["text"]))
    h_val = scaler.transform(build_heuristic_features(val["text"]))
    h_test = scaler.transform(build_heuristic_features(test["text"]))
    extra["heuristic_features"] = h_train.shape[1]
    return hstack([x_train, h_train]), hstack([x_val, h_val]), hstack([x_test, h_test]), extra


def run_split_experiment(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    experiment_name: str,
    *,
    hybrid: bool = False,
    models_to_use: list[str] | None = None,
    max_features: int = 30_000,
) -> pd.DataFrame:
    if train[target_col].nunique() < 2 or val[target_col].nunique() < 2 or test[target_col].nunique() < 2:
        return pd.DataFrame()

    x_train, x_val, x_test, feature_info = featurize(train, val, test, hybrid=hybrid, max_features=max_features)
    y_train = train[target_col].astype(int).to_numpy()
    y_val = val[target_col].astype(int).to_numpy()
    y_test = test[target_col].astype(int).to_numpy()
    models = make_models()
    if models_to_use:
        models = {name: model for name, model in models.items() if name in models_to_use}

    rows = []
    for model_name, model in models.items():
        start = time.time()
        model.fit(x_train, y_train)
        val_scores = get_scores(model, x_val)
        threshold, val_best_f1 = tune_threshold_f1(y_val, val_scores)
        test_scores = get_scores(model, x_test)
        row = {
            "experiment": experiment_name,
            "target": target_col,
            "model": model_name,
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "n_features": x_train.shape[1],
            "text_features": feature_info["vectorizer_features"],
            "heuristic_features": feature_info["heuristic_features"],
            "train_seconds": round(time.time() - start, 3),
            "val_best_f1": val_best_f1,
        }
        row.update(metrics_from_scores(y_val, val_scores, threshold, prefix="val_"))
        row.update(metrics_from_scores(y_test, test_scores, threshold, prefix="test_"))
        rows.append(row)
    return pd.DataFrame(rows)


def save_results(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
