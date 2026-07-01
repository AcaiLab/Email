import time

import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

from .config import SEED
from .features import make_vectorizer
from .metrics import tune_threshold_f1
from .modeling import get_scores

TARGET = "label_attack"


def split_train_val(frame: pd.DataFrame, target: str = TARGET) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = frame[target].astype(int)
    train, val = train_test_split(
        frame,
        test_size=0.20,
        random_state=SEED,
        stratify=y if y.nunique() == 2 and y.value_counts().min() >= 2 else None,
    )
    return train.reset_index(drop=True), val.reset_index(drop=True)


def fit_detector(train: pd.DataFrame, val: pd.DataFrame, max_features: int, target: str = TARGET) -> dict:
    vectorizer = make_vectorizer(max_features=max_features)
    x_train = vectorizer.fit_transform(train["text"])
    x_val = vectorizer.transform(val["text"])
    y_train = train[target].astype(int).to_numpy()
    y_val = val[target].astype(int).to_numpy()
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        max_iter=40,
        tol=1e-3,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED,
    )
    start = time.time()
    model.fit(x_train, y_train)
    val_scores = get_scores(model, x_val)
    threshold, val_best_f1 = tune_threshold_f1(y_val, val_scores)
    return {
        "model": model,
        "vectorizer": vectorizer,
        "threshold": threshold,
        "val_best_f1": val_best_f1,
        "train_seconds": round(time.time() - start, 3),
        "n_features": x_train.shape[1],
        "val_scores": val_scores,
    }
