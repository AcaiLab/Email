from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.model_selection import train_test_split

from llmail_research.config import SEED


TARGET = "label_attack"
NOTINJECT_SPLITS = ("NotInject_one", "NotInject_two", "NotInject_three")


@dataclass(frozen=True)
class NotInjectRotation:
    name: str
    train_split: str
    calibration_split: str
    test_split: str


NOTINJECT_ROTATIONS = (
    NotInjectRotation("rotation_1", "NotInject_one", "NotInject_two", "NotInject_three"),
    NotInjectRotation("rotation_2", "NotInject_two", "NotInject_three", "NotInject_one"),
    NotInjectRotation("rotation_3", "NotInject_three", "NotInject_one", "NotInject_two"),
)


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


def sample_equal_classes_seeded(
    frame: pd.DataFrame,
    max_rows: int | None,
    random_state: int = SEED,
    label_column: str = TARGET,
) -> pd.DataFrame:
    """Return a deterministic, class-balanced sample without replacing rows."""
    frame = frame.dropna(subset=[label_column]).copy()
    groups = [group for _, group in frame.groupby(label_column)]
    if len(groups) < 2:
        if max_rows and len(frame) > max_rows:
            return frame.sample(max_rows, random_state=random_state).reset_index(drop=True)
        return frame.sample(frac=1, random_state=random_state).reset_index(drop=True)
    per_class = min(len(group) for group in groups)
    if max_rows:
        per_class = min(per_class, max(1, max_rows // len(groups)))
    sampled = [group.sample(per_class, random_state=random_state) for group in groups]
    return pd.concat(sampled, ignore_index=True).sample(frac=1, random_state=random_state).reset_index(drop=True)


def add_bipia_context_group(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["_original_order"] = np.arange(len(out))
    group_ids = pd.Series(index=out.index, dtype="object")
    for domain, indices in out.groupby("domain", sort=False).groups.items():
        domain_rows = out.loc[indices].sort_values("_original_order")
        clean_count = int(domain_rows[TARGET].eq(0).sum())
        if clean_count == 0 or len(domain_rows) % clean_count:
            raise ValueError(f"Cannot recover BIPIA groups for domain={domain!r}")
        block_size = len(domain_rows) // clean_count
        local_groups = np.arange(len(domain_rows)) // block_size
        group_ids.loc[domain_rows.index] = [f"{domain}:{value}" for value in local_groups]
    out["bipia_context_group"] = group_ids
    return out.drop(columns=["_original_order"])


def split_bipia_train_by_context(
    frame: pd.DataFrame,
    random_state: int,
    base_fraction: float = 0.60,
    meta_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouped = add_bipia_context_group(frame.reset_index(drop=True))
    group_table = grouped[["bipia_context_group", "domain"]].drop_duplicates()
    base_groups: list[str] = []
    meta_groups: list[str] = []
    threshold_groups: list[str] = []
    for offset, (_, domain_groups) in enumerate(group_table.groupby("domain", sort=True)):
        ids = domain_groups["bipia_context_group"].to_numpy()
        if len(ids) < 3:
            raise ValueError("Each BIPIA domain needs at least three contexts")
        base, rest = train_test_split(ids, train_size=base_fraction, random_state=random_state + offset)
        relative_meta = meta_fraction / (1.0 - base_fraction)
        meta, threshold = train_test_split(
            rest,
            train_size=relative_meta,
            random_state=random_state + 100 + offset,
        )
        base_groups.extend(base.tolist())
        meta_groups.extend(meta.tolist())
        threshold_groups.extend(threshold.tolist())

    def select(group_ids: list[str]) -> pd.DataFrame:
        return grouped[grouped["bipia_context_group"].isin(group_ids)].reset_index(drop=True)

    base = select(base_groups)
    meta = select(meta_groups)
    threshold = select(threshold_groups)
    partitions = [set(part["bipia_context_group"]) for part in (base, meta, threshold)]
    if any(partitions[i] & partitions[j] for i in range(3) for j in range(i + 1, 3)):
        raise AssertionError("BIPIA context partitions overlap")
    if len(set.union(*partitions)) != group_table["bipia_context_group"].nunique():
        raise AssertionError("BIPIA context split lost a group")
    return base, meta, threshold


def split_notinject_training(frame: pd.DataFrame, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    base, meta = train_test_split(frame, train_size=0.60, random_state=random_state, shuffle=True)
    return base.reset_index(drop=True), meta.reset_index(drop=True)


def strict_threshold_from_benign(scores: np.ndarray, max_fpr: float) -> tuple[float, int, float]:
    # select cut off
    scores = np.asarray(scores, dtype=float)
    if not len(scores):
        return 1.01, 0, 0.0
    allowed = int(np.floor(max_fpr * len(scores)))
    candidates = np.r_[np.nextafter(np.unique(scores), np.inf), np.unique(scores), np.inf]
    valid = []
    for threshold in candidates:
        false_positives = int((scores >= threshold).sum())
        if false_positives <= allowed:
            valid.append((float(threshold), false_positives))
    threshold, false_positives = min(valid, key=lambda item: item[0])
    return threshold, false_positives, false_positives / len(scores)


def clopper_pearson_interval(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    # confidence interval
    if trials <= 0:
        return float("nan"), float("nan")
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return lower, upper


def build_matched_test_sets(
    llmail: pd.DataFrame,
    bipia: pd.DataFrame,
    notinject: pd.DataFrame,
    nvidia: pd.DataFrame,
    promptshield: pd.DataFrame,
    shieldlm: pd.DataFrame,
    neuralchemy: pd.DataFrame,
    notinject_test_split: str,
    max_test_rows: int = 2_000,
    random_state: int = SEED,
) -> dict[str, pd.DataFrame]:
    llmail_benign = llmail[llmail["source_dataset"].eq("LLMail_FP")].copy()
    llmail_phase2_attack = llmail[
        llmail["source_dataset"].eq("LLMail") & llmail["phase"].eq("Phase2")
    ].copy()
    maximum = max_test_rows if max_test_rows and max_test_rows > 0 else 2 * min(
        len(llmail_benign), len(llmail_phase2_attack)
    )
    attack_count = min(len(llmail_benign), len(llmail_phase2_attack), maximum // 2)
    llmail_test = pd.concat(
        [
            llmail_benign.sample(attack_count, random_state=random_state),
            llmail_phase2_attack.sample(attack_count, random_state=random_state),
        ],
        ignore_index=True,
        sort=False,
    ).sample(frac=1, random_state=random_state).reset_index(drop=True)
    phase1_normalized = set(
        llmail.loc[llmail["phase"].eq("Phase1"), "text"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    test_normalized = (
        llmail_test["text"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    llmail_test = llmail_test[
        ~(llmail_test["source_dataset"].eq("LLMail") & test_normalized.isin(phase1_normalized))
    ]
    sample_cap = max_test_rows if max_test_rows and max_test_rows > 0 else None
    llmail_test = sample_equal_classes_seeded(llmail_test, sample_cap, random_state)

    return {
        "llmail_phase2_binary": llmail_test,
        "bipia_test": sample_equal_classes_seeded(
            bipia[bipia["split"].eq("test")], sample_cap, random_state
        ),
        "notinject": notinject[notinject["split"].eq(notinject_test_split)].reset_index(drop=True),
        "nvidia_agentic_ipi": nvidia.reset_index(drop=True),
        "promptshield_test": sample_equal_classes_seeded(
            promptshield[promptshield["split"].eq("test")], sample_cap, random_state
        ),
        "shieldlm_test": sample_equal_classes_seeded(
            shieldlm[shieldlm["split"].eq("test")], sample_cap, random_state
        ),
        "neuralchemy_core_test": sample_equal_classes_seeded(
            neuralchemy[neuralchemy["split"].eq("test")], sample_cap, random_state
        ),
    }
