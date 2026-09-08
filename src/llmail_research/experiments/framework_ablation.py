import argparse
import json
import re
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

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
    sample_equal_classes,
)
from llmail_research.features import build_heuristic_features
from llmail_research.metrics import metrics_from_scores, tune_threshold_f1
from llmail_research.eval_utils import (
    NOTINJECT_ROTATIONS,
    build_matched_test_sets,
    clopper_pearson_interval,
    sample_equal_classes_seeded,
    split_bipia_train_by_context,
    split_notinject_training,
    strict_threshold_from_benign,
)

from llmail_research.experiments.action_features import build_action_verifier_features
from llmail_research.experiments.framework_core import (
    TARGET,
    STAGE_TARGETS,
    HybridDetector,
    TextDetector,
    choose_threshold_under_fpr,
    fit_stage_models,
    stack_features,
)


SEND_PATTERN = r"\b(?:send|sent|sending|forward|email|mail|api[_ -]?call|tool call|execute_userquery|send_email)\b"
EMAIL_PATTERN = r"[A-Za-z0-9._%+-]+\s*@\s*[A-Za-z0-9.-]+\s*\.\s*[A-Za-z]{2,}"
BOUNDARY_PATTERN = r"<<|>>|<\|user\|>|<\|assistant\|>|<\|system\|>|end_context|start_prompt|new_user_session|execute_userquery|```|<system>|</system>|BEGIN|END|###|---"
RAW_STAGE_COLUMNS = [f"stage_{target}_score" for target in STAGE_TARGETS]
DERIVED_STAGE_COLUMNS = ["stage_max_score", "unsafe_tool_chain_score"]
FINAL_LINEAR = "selective_stage_plus_action_context"
FINAL_NONLINEAR = "selective_stage_plus_action_context_hgb"


def cap_frame(frame: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if not max_rows or len(frame) <= max_rows:
        return frame.reset_index(drop=True)
    if TARGET in frame.columns and frame[TARGET].nunique() == 2:
        return sample_equal_classes(frame, TARGET, max_rows).reset_index(drop=True)
    return frame.sample(max_rows, random_state=SEED).reset_index(drop=True)


def evaluate_scores(
    method: str,
    thresholds: dict[str, float],
    test_frames: dict[str, pd.DataFrame],
    test_scores: dict[str, np.ndarray],
) -> list[dict]:
    rows = []
    for test_name, frame in test_frames.items():
        scores = test_scores[test_name]
        for threshold_policy, threshold in thresholds.items():
            row = {
                "method": method,
                "threshold_policy": threshold_policy,
                "test_source": test_name,
                "threshold": float(threshold),
                "n_test": len(frame),
                "test_positive_rate": float(frame[TARGET].mean()),
                "predicted_positive_rate": float((scores >= threshold).mean()),
            }
            if frame[TARGET].nunique() == 2:
                row.update(metrics_from_scores(frame[TARGET], scores, threshold, prefix="test_"))
            else:
                label = int(frame[TARGET].iloc[0]) if len(frame) else 0
                if label == 0:
                    row["test_false_positive_rate"] = float((scores >= threshold).mean())
                else:
                    row["test_false_negative_rate"] = float((scores < threshold).mean())
                    row["test_recall"] = float((scores >= threshold).mean())
            rows.append(row)
    return rows


def value_for(group: pd.DataFrame, test_source: str, column: str) -> float:
    value = group.loc[group["test_source"].eq(test_source), column]
    return float(value.iloc[0]) if len(value) else np.nan


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, policy), group in results.groupby(["method", "threshold_policy"]):
        binary = group[group["test_f1"].notna()].copy()
        benign = group[group["test_source"].eq("notinject")]
        nvidia = group[group["test_source"].eq("nvidia_agentic_ipi")]
        rows.append(
            {
                "method": method,
                "threshold_policy": policy,
                "mean_binary_f1": float(binary["test_f1"].mean()),
                "mean_binary_recall": float(binary["test_recall"].mean()),
                "notinject_fpr": float(benign["test_false_positive_rate"].iloc[0]) if len(benign) else np.nan,
                "nvidia_recall": float(1 - nvidia["test_false_negative_rate"].iloc[0]) if len(nvidia) else np.nan,
                "promptshield_f1": value_for(group, "promptshield_test", "test_f1"),
                "shieldlm_f1": value_for(group, "shieldlm_test", "test_f1"),
                "neuralchemy_f1": value_for(group, "neuralchemy_core_test", "test_f1"),
                "bipia_f1": value_for(group, "bipia_test", "test_f1"),
                "llmail_f1": value_for(group, "llmail_binary", "test_f1"),
            }
        )
    return pd.DataFrame(rows).sort_values(["threshold_policy", "mean_binary_f1"], ascending=[True, False])


def has_regex(series: pd.Series, pattern: str) -> pd.Series:
    return series.fillna("").astype(str).str.contains(pattern, flags=re.IGNORECASE, regex=True).astype(float)


def count_regex(series: pd.Series, pattern: str) -> pd.Series:
    return series.fillna("").astype(str).str.count(pattern, flags=re.IGNORECASE).astype(float)


def get_series(frame: pd.DataFrame, column: str, default: str = "") -> pd.Series:
    value = frame[column] if column in frame.columns else default
    if isinstance(value, pd.Series):
        return value.fillna("").astype(str)
    return pd.Series([value] * len(frame), index=frame.index).fillna("").astype(str)


def extract_untrusted_segment(frame: pd.DataFrame) -> pd.Series:
    """Extract the portion most likely to be untrusted external/email content."""
    text = get_series(frame, "text")
    body = get_series(frame, "body_text")
    external = text.str.extract(r"External content:\s*(.*)", flags=re.IGNORECASE | re.DOTALL, expand=False)
    llmail_body = text.str.extract(r"Body:\s*(.*)", flags=re.IGNORECASE | re.DOTALL, expand=False)
    return external.fillna("").where(external.fillna("").str.len() > 0, body).where(
        lambda s: s.str.len() > 0,
        llmail_body.fillna("").where(llmail_body.fillna("").str.len() > 0, text),
    )


def build_intent_boundary_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Lightweight user-intent, tool-argument, and untrusted-boundary features.

    These are deployable proxy checks for the proposed framework additions. They
    do not require an LLM judge or live tool-call schema.
    """
    text = get_series(frame, "text")
    user = get_series(frame, "user_intent", "Please summarize the two most recent emails.")
    untrusted = extract_untrusted_segment(frame)

    text_rules = build_heuristic_features(text)
    user_rules = build_heuristic_features(user)
    untrusted_rules = build_heuristic_features(untrusted)

    features = pd.DataFrame(index=frame.index)
    features["user_has_send_action"] = has_regex(user, SEND_PATTERN)
    features["text_has_send_action_proxy"] = has_regex(text, SEND_PATTERN)
    features["untrusted_has_send_action"] = has_regex(untrusted, SEND_PATTERN)
    features["user_email_count"] = count_regex(user, EMAIL_PATTERN)
    features["text_email_count_proxy"] = count_regex(text, EMAIL_PATTERN)
    features["untrusted_email_count"] = count_regex(untrusted, EMAIL_PATTERN)
    features["untrusted_boundary_marker_count"] = count_regex(untrusted, BOUNDARY_PATTERN)
    features["text_boundary_marker_count_proxy"] = count_regex(text, BOUNDARY_PATTERN)
    features["external_content_marker_present"] = has_regex(text, r"External content:")
    features["email_body_marker_present"] = has_regex(text, r"\bBody:")
    features["untrusted_instruction_score"] = (
        untrusted_rules["group_instruction_override_present"]
        + untrusted_rules["group_role_or_authority_present"]
        + untrusted_rules["group_delimiter_or_formatting_present"]
    ).clip(upper=1)
    features["user_instruction_score"] = (
        user_rules["group_instruction_override_present"]
        + user_rules["group_role_or_authority_present"]
        + user_rules["group_delimiter_or_formatting_present"]
    ).clip(upper=1)
    features["untrusted_tool_score"] = (
        untrusted_rules["group_tool_action_present"]
        + untrusted_rules["group_target_artifacts_present"]
        + (features["untrusted_email_count"] > 0).astype(float)
    ).clip(upper=1)
    features["user_tool_score"] = (
        user_rules["group_tool_action_present"]
        + user_rules["group_target_artifacts_present"]
        + (features["user_email_count"] > 0).astype(float)
    ).clip(upper=1)
    features["untrusted_exfil_score"] = (
        untrusted_rules["group_exfiltration_present"]
        + untrusted_rules["group_secrecy_or_stealth_present"]
    ).clip(upper=1)
    features["intent_tool_conflict"] = (
        features["untrusted_tool_score"].gt(0)
        & features["user_tool_score"].eq(0)
    ).astype(float)
    features["intent_send_conflict"] = (
        features["untrusted_has_send_action"].eq(1)
        & features["user_has_send_action"].eq(0)
    ).astype(float)
    features["recipient_introduced_by_untrusted"] = (
        features["untrusted_email_count"].gt(features["user_email_count"])
        & features["user_has_send_action"].eq(0)
    ).astype(float)
    features["boundary_control_risk"] = (
        features["untrusted_boundary_marker_count"].gt(0)
        & features["user_instruction_score"].eq(0)
    ).astype(float)
    features["tool_argument_risk_proxy"] = (
        0.35 * features["intent_send_conflict"]
        + 0.30 * features["recipient_introduced_by_untrusted"]
        + 0.20 * features["untrusted_exfil_score"]
        + 0.15 * features["boundary_control_risk"]
    ).clip(upper=1)
    features["untrusted_vs_user_instruction_delta"] = (
        features["untrusted_instruction_score"] - features["user_instruction_score"]
    ).clip(lower=0)
    return features.replace([np.inf, -np.inf], 0).fillna(0)


def load_experiment_data(max_test_rows: int | None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    llmail = load_llmail_binary()
    llmail_clean = load_clean_dataset()
    bipia = load_bipia_binary()
    notinject = load_notinject()
    nvidia = load_nvidia_agentic_ipi()
    promptshield = load_promptshield()
    shieldlm = load_shieldlm()
    neuralchemy = load_neuralchemy_core()
    test_sets = {
        "llmail_binary": cap_frame(llmail, max_test_rows),
        "bipia_test": cap_frame(bipia[bipia["split"].eq("test")], max_test_rows),
        "notinject": cap_frame(notinject, max_test_rows),
        "nvidia_agentic_ipi": cap_frame(nvidia, max_test_rows),
        "promptshield_test": cap_frame(promptshield[promptshield["split"].eq("test")], max_test_rows),
        "shieldlm_test": cap_frame(shieldlm[shieldlm["split"].eq("test")], max_test_rows),
        "neuralchemy_core_test": cap_frame(neuralchemy[neuralchemy["split"].eq("test")], max_test_rows),
    }
    train_pool = pd.concat([llmail, bipia[bipia["split"].eq("train")], notinject], ignore_index=True, sort=False)
    return llmail_clean, train_pool, test_sets


def augment_features(frame: pd.DataFrame, chain_features: pd.DataFrame) -> pd.DataFrame:
    frame_for_action = frame.copy()
    if "output_text" not in frame_for_action.columns:
        frame_for_action["output_text"] = ""
    if "output" not in frame_for_action.columns:
        frame_for_action["output"] = ""
    action = build_action_verifier_features(frame_for_action).add_prefix("action_")
    boundary = build_intent_boundary_features(frame_for_action).add_prefix("ctx_")
    out = pd.concat([chain_features, action, boundary], axis=1)
    selected_raw = [
        "stage_defense_bypassed_score",
        "stage_tool_called_score",
        "stage_destination_correct_score",
        "stage_content_correct_score",
    ]
    present = [column for column in selected_raw if column in out.columns]
    if present:
        out["selected_stage_max_score"] = out[present].max(axis=1)
    else:
        out["selected_stage_max_score"] = 0.0
    out["selected_tool_chain_score"] = (
        0.35 * out.get("stage_tool_called_score", 0)
        + 0.25 * out.get("stage_destination_correct_score", 0)
        + 0.25 * out.get("stage_content_correct_score", 0)
        + 0.15 * out.get("rule_tool_action", 0)
    ).clip(upper=1)
    out["intent_weighted_chain_score"] = (
        0.45 * out["selected_tool_chain_score"]
        + 0.35 * out["ctx_tool_argument_risk_proxy"]
        + 0.20 * out["ctx_untrusted_vs_user_instruction_delta"]
    ).clip(upper=1)
    return out.replace([np.inf, -np.inf], 0).fillna(0)


def feature_variants(columns: list[str]) -> dict[str, list[str]]:
    raw_stage = [column for column in RAW_STAGE_COLUMNS if column in columns]
    derived_stage = [column for column in DERIVED_STAGE_COLUMNS if column in columns]
    base = [column for column in columns if column not in set(raw_stage + derived_stage)]
    action = [column for column in columns if column.startswith("action_")]
    ctx = [column for column in columns if column.startswith("ctx_")]
    proposed_scores = [
        column
        for column in ["selected_stage_max_score", "selected_tool_chain_score", "intent_weighted_chain_score"]
        if column in columns
    ]
    non_extension_base = [
        column
        for column in base
        if not column.startswith("action_")
        and not column.startswith("ctx_")
        and column not in set(proposed_scores)
    ]
    selected_stage = [
        column
        for column in [
            "stage_defense_bypassed_score",
            "stage_tool_called_score",
            "stage_destination_correct_score",
            "stage_content_correct_score",
            "selected_stage_max_score",
            "selected_tool_chain_score",
        ]
        if column in columns
    ]
    all_chain = non_extension_base + raw_stage + derived_stage
    variants = {
        "text_rules_only": non_extension_base,
        "all_stage_framework": all_chain,
        "selective_stages_no_full_success": non_extension_base + selected_stage,
        "action_context_no_stage": non_extension_base + action + ctx,
        "selective_stage_plus_action_context": non_extension_base + selected_stage + action + ctx + proposed_scores,
        "all_stages_plus_action_context": all_chain + action + ctx + proposed_scores,
    }
    return {name: list(dict.fromkeys(cols)) for name, cols in variants.items()}


def add_delta_columns(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    for policy in out["threshold_policy"].unique():
        mask = out["threshold_policy"].eq(policy)
        base = out.loc[mask & out["method"].eq("all_stage_framework"), "mean_binary_f1"]
        text = out.loc[mask & out["method"].eq("text_rules_only"), "mean_binary_f1"]
        if len(base):
            out.loc[mask, "delta_vs_all_stage_f1"] = out.loc[mask, "mean_binary_f1"] - float(base.iloc[0])
        if len(text):
            out.loc[mask, "delta_vs_text_rules_f1"] = out.loc[mask, "mean_binary_f1"] - float(text.iloc[0])
    return out


def fit_meta(kind: str, features: pd.DataFrame, labels: np.ndarray, seed: int):
    if kind == "linear":
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        model.fit(features, labels)
        return model
    if kind == "hgb":
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=150,
            max_depth=3,
            min_samples_leaf=15,
            l2_regularization=1.0,
            random_state=seed,
        )
        model.fit(features, labels, sample_weight=compute_sample_weight("balanced", labels))
        return model
    raise ValueError(kind)


def build_training_partitions(
    llmail: pd.DataFrame,
    bipia: pd.DataFrame,
    notinject: pd.DataFrame,
    rotation,
    seed: int,
    max_train_rows: int,
    max_meta_rows: int,
    max_threshold_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bipia_train = bipia[bipia["split"].eq("train")].reset_index(drop=True)
    bipia_base, bipia_meta, bipia_threshold = split_bipia_train_by_context(bipia_train, seed)
    notinject_train = notinject[notinject["split"].eq(rotation.train_split)].reset_index(drop=True)
    notinject_base, notinject_meta = split_notinject_training(notinject_train, seed)

    llmail_phase1_attack = llmail[
        llmail["source_dataset"].eq("LLMail") & llmail["phase"].eq("Phase1")
    ].reset_index(drop=True)
    base_pool = pd.concat(
        [llmail_phase1_attack, bipia_base, notinject_base],
        ignore_index=True,
        sort=False,
    )
    meta_pool = pd.concat([bipia_meta, notinject_meta], ignore_index=True, sort=False)
    base_train = sample_equal_classes_seeded(base_pool, max_train_rows, seed)
    meta_train = sample_equal_classes_seeded(meta_pool, max_meta_rows, seed)
    threshold_validation = sample_equal_classes_seeded(bipia_threshold, max_threshold_rows, seed)
    calibration = notinject[notinject["split"].eq(rotation.calibration_split)].reset_index(drop=True)
    return base_train, meta_train, threshold_validation, calibration, notinject_base


def score_features(
    frame: pd.DataFrame,
    detectors: dict,
    stage_models: dict,
) -> pd.DataFrame:
    return augment_features(frame, stack_features(frame, detectors, stage_models))


def evaluate_one(
    method: str,
    policy: str,
    threshold: float,
    test_sets: dict[str, pd.DataFrame],
    feature_sets: dict[str, pd.DataFrame],
    model,
    columns: list[str],
    seed: int,
    rotation,
) -> tuple[list[dict], list[pd.DataFrame]]:
    result_rows = []
    prediction_frames = []
    for source, frame in test_sets.items():
        scores = model.predict_proba(feature_sets[source][columns])[:, 1]
        labels = frame[TARGET].astype(int).to_numpy()
        predictions = (scores >= threshold).astype(int)
        row = {
            "method": method,
            "threshold_policy": policy,
            "test_source": source,
            "threshold": float(threshold),
            "n_test": len(frame),
            "test_positive_rate": float(labels.mean()),
            "predicted_positive_rate": float(predictions.mean()),
            "seed": seed,
            "rotation": rotation.name,
            "notinject_train_split": rotation.train_split,
            "notinject_calibration_split": rotation.calibration_split,
            "notinject_test_split": rotation.test_split,
        }
        if len(np.unique(labels)) == 2:
            row.update(metrics_from_scores(labels, scores, threshold, prefix="test_"))
        elif len(labels) and labels[0] == 0:
            row["test_false_positive_rate"] = float(predictions.mean())
            row["test_fp"] = int(predictions.sum())
            row["test_tn"] = int(len(labels) - predictions.sum())
        elif len(labels):
            row["test_false_negative_rate"] = float(1 - predictions.mean())
            row["test_recall"] = float(predictions.mean())
            row["test_fn"] = int(len(labels) - predictions.sum())
            row["test_tp"] = int(predictions.sum())
        result_rows.append(row)

        prediction_frames.append(
            pd.DataFrame(
                {
                    "method": method,
                    "threshold_policy": policy,
                    "test_source": source,
                    "seed": seed,
                    "rotation": rotation.name,
                    "row_id": np.arange(len(frame)),
                    "label": labels,
                    "score": scores,
                    "prediction": predictions,
                }
            )
        )
    return result_rows, prediction_frames


def summarize_one_run(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, policy, seed, rotation), group in results.groupby(
        ["method", "threshold_policy", "seed", "rotation"]
    ):
        binary = group[group["test_f1"].notna()]
        notinject = group[group["test_source"].eq("notinject")]
        nvidia = group[group["test_source"].eq("nvidia_agentic_ipi")]

        def value(source: str, column: str) -> float:
            selected = group.loc[group["test_source"].eq(source), column]
            return float(selected.iloc[0]) if len(selected) else float("nan")

        rows.append(
            {
                "method": method,
                "threshold_policy": policy,
                "seed": seed,
                "rotation": rotation,
                "mean_binary_f1": float(binary["test_f1"].mean()),
                "mean_binary_precision": float(binary["test_precision"].mean()),
                "mean_binary_recall": float(binary["test_recall"].mean()),
                "notinject_fpr": float(notinject["test_false_positive_rate"].iloc[0]),
                "notinject_fp": int(notinject["test_fp"].iloc[0]),
                "notinject_n": int(notinject["n_test"].iloc[0]),
                "nvidia_recall": float(nvidia["test_recall"].iloc[0]),
                "bipia_f1": value("bipia_test", "test_f1"),
                "llmail_f1": value("llmail_phase2_binary", "test_f1"),
                "promptshield_f1": value("promptshield_test", "test_f1"),
                "shieldlm_f1": value("shieldlm_test", "test_f1"),
                "neuralchemy_f1": value("neuralchemy_core_test", "test_f1"),
            }
        )
    return pd.DataFrame(rows)


def aggregate_runs(run_summary: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "mean_binary_f1",
        "mean_binary_precision",
        "mean_binary_recall",
        "notinject_fpr",
        "nvidia_recall",
        "bipia_f1",
        "llmail_f1",
        "promptshield_f1",
        "shieldlm_f1",
        "neuralchemy_f1",
    ]
    rows = []
    for (method, policy), group in run_summary.groupby(["method", "threshold_policy"]):
        row = {"method": method, "threshold_policy": policy, "n_runs": len(group)}
        for column in metric_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["threshold_policy", "mean_binary_f1_mean"], ascending=[True, False]
    )


def notinject_intervals(run_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, policy, seed), group in run_summary.groupby(["method", "threshold_policy", "seed"]):
        false_positives = int(group["notinject_fp"].sum())
        trials = int(group["notinject_n"].sum())
        lower, upper = clopper_pearson_interval(false_positives, trials)
        rows.append(
            {
                "method": method,
                "threshold_policy": policy,
                "seed": seed,
                "false_positives": false_positives,
                "n_unique_notinject": trials,
                "fpr": false_positives / trials,
                "fpr_ci95_lower": lower,
                "fpr_ci95_upper": upper,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_intervals(
    predictions: pd.DataFrame,
    primary_seed: int,
    primary_rotation: str,
    n_bootstrap: int,
) -> pd.DataFrame:
    selected = predictions[
        predictions["seed"].eq(primary_seed)
        & predictions["rotation"].eq(primary_rotation)
        & predictions["test_source"].isin(
            ["bipia_test", "llmail_phase2_binary", "promptshield_test", "shieldlm_test", "neuralchemy_core_test"]
        )
    ]
    rng = np.random.default_rng(primary_seed)
    rows = []
    for (method, policy, source), group in selected.groupby(
        ["method", "threshold_policy", "test_source"]
    ):
        labels = group["label"].to_numpy()
        preds = group["prediction"].to_numpy()
        values = []
        for _ in range(n_bootstrap):
            indices = rng.integers(0, len(group), len(group))
            values.append(f1_score(labels[indices], preds[indices], zero_division=0))
        low, high = np.quantile(values, [0.025, 0.975])
        rows.append(
            {
                "method": method,
                "threshold_policy": policy,
                "test_source": source,
                "f1": f1_score(labels, preds, zero_division=0),
                "f1_bootstrap_ci95_lower": float(low),
                "f1_bootstrap_ci95_upper": float(high),
                "n_test": len(group),
                "n_bootstrap": n_bootstrap,
            }
        )
    return pd.DataFrame(rows)


def padded_copy(text: str, benign_texts: list[str], ratio: int, rng: np.random.Generator, position: str) -> str:
    if ratio == 0:
        return text
    target_words = max(1, len(str(text).split()) * ratio)
    words: list[str] = []
    while len(words) < target_words:
        words.extend(benign_texts[int(rng.integers(0, len(benign_texts)))].split())
    padding = " ".join(words[:target_words])
    return f"{padding}\n{text}" if position == "prefix" else f"{text}\n{padding}"


def run_padding_stress(
    artifacts: dict,
    llmail: pd.DataFrame,
    bipia: pd.DataFrame,
    seed: int,
    sample_rows: int,
) -> pd.DataFrame:
    llmail_pool = llmail[
        llmail["source_dataset"].eq("LLMail") & llmail["phase"].eq("Phase2")
    ]
    bipia_pool = bipia[
        bipia["split"].eq("test") & bipia["domain"].eq("email") & bipia[TARGET].eq(1)
    ]
    per_source = max(1, sample_rows // 2)
    llmail_attacks = llmail_pool.sample(min(per_source, len(llmail_pool)), random_state=seed)
    bipia_attacks = bipia_pool.sample(min(per_source, len(bipia_pool)), random_state=seed)
    attacks = pd.concat([llmail_attacks, bipia_attacks], ignore_index=True, sort=False)
    attacks["user_intent"] = attacks["user_intent"].fillna("Summarize the retrieved email.")
    benign_texts = llmail[llmail["source_dataset"].eq("LLMail_FP")]["text"].fillna("").astype(str).tolist()
    rows = []
    rng = np.random.default_rng(seed)
    for ratio in [0, 1, 2, 5, 10]:
        positions = ["none"] if ratio == 0 else ["prefix", "suffix"]
        for position in positions:
            padded = attacks.copy()
            padded["text"] = [
                padded_copy(text, benign_texts, ratio, rng, position)
                for text in padded["text"].fillna("").astype(str)
            ]
            features = score_features(padded, artifacts["detectors"], artifacts["stage_models"])
            for method in ["text_rules_only", FINAL_LINEAR, FINAL_NONLINEAR]:
                model = artifacts["models"][method]
                columns = artifacts["columns"][method]
                threshold = artifacts["strict_thresholds"][method]
                scores = model.predict_proba(features[columns])[:, 1]
                rows.append(
                    {
                        "method": method,
                        "padding_ratio": ratio,
                        "position": position,
                        "n_attacks": len(padded),
                        "strict_threshold": threshold,
                        "recall": float((scores >= threshold).mean()),
                        "mean_score": float(scores.mean()),
                        "median_score": float(np.median(scores)),
                    }
                )
    return pd.DataFrame(rows)


def run_literal_rename_stress(artifacts: dict, llmail: pd.DataFrame) -> pd.DataFrame:
    attack_pool = llmail[
        llmail["source_dataset"].eq("LLMail") & llmail["phase"].eq("Phase2")
    ]
    attacks = attack_pool.sample(min(500, len(attack_pool)), random_state=SEED).reset_index(drop=True)
    renamed = attacks.copy()
    renamed["text"] = (
        renamed["text"]
        .fillna("")
        .astype(str)
        .str.replace("contact@contact.com", "finance-team@example.org", case=False, regex=False)
        .str.replace("confirmation", "authorization receipt", case=False, regex=False)
        .str.replace("send_email_xtbiu", "dispatch_message", case=False, regex=False)
    )
    rows = []
    for version, frame in [("original", attacks), ("renamed", renamed)]:
        features = score_features(frame, artifacts["detectors"], artifacts["stage_models"])
        for method in ["text_rules_only", FINAL_LINEAR, FINAL_NONLINEAR]:
            model = artifacts["models"][method]
            columns = artifacts["columns"][method]
            threshold = artifacts["strict_thresholds"][method]
            scores = model.predict_proba(features[columns])[:, 1]
            rows.append(
                {
                    "method": method,
                    "version": version,
                    "n_attacks": len(frame),
                    "strict_threshold": threshold,
                    "recall": float((scores >= threshold).mean()),
                    "mean_score": float(scores.mean()),
                }
            )
    return pd.DataFrame(rows)


def negative_word_vocabulary(detector: TextDetector, limit: int = 100) -> list[str]:
    vocabulary = detector.vectorizer.get_feature_names_out()
    weights = detector.model.coef_[0]
    order = np.argsort(weights)
    blocked = re.compile(
        r"ignore|instruction|system|prompt|tool|send|email|secret|password|token|execute|recipient|body|subject"
    )
    words = []
    for index in order:
        token = str(vocabulary[index])
        if re.fullmatch(r"[a-z]{3,15}", token) and not blocked.search(token):
            words.append(token)
        if len(words) == limit:
            break
    return words or ["meeting", "schedule", "project", "update"]


def pad_with_fixed_words(text: str, words: list[str], ratio: int, position: str) -> str:
    if ratio == 0:
        return text
    target_words = max(1, len(str(text).split()) * ratio)
    repeats = (target_words + len(words) - 1) // len(words)
    padding = " ".join((words * repeats)[:target_words])
    return f"{padding}\n{text}" if position == "prefix" else f"{text}\n{padding}"


def chunk_frame(frame: pd.DataFrame, chunk_words: int) -> pd.DataFrame:
    """Split each example into fixed-size word chunks for max-risk aggregation."""
    rows = []
    for attack_id, (_, row) in enumerate(frame.iterrows()):
        words = str(row.get("text", "") or "").split()
        chunks = [words[start : start + chunk_words] for start in range(0, len(words), chunk_words)] or [[]]
        for chunk_id, chunk in enumerate(chunks):
            item = row.to_dict()
            item["text"] = " ".join(chunk)
            item["_attack_id"] = attack_id
            item["_chunk_id"] = chunk_id
            rows.append(item)
    return pd.DataFrame(rows)



def run_adaptive_padding_stress(
    artifacts: dict,
    llmail: pd.DataFrame,
    bipia: pd.DataFrame,
    seed: int,
    sample_rows: int,
    chunk_words: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate random and detector-aware padding, with optional chunk aggregation."""
    llmail_pool = llmail[
        llmail["source_dataset"].eq("LLMail") & llmail["phase"].eq("Phase2")
    ]
    bipia_pool = bipia[
        bipia["split"].eq("test") & bipia["domain"].eq("email") & bipia[TARGET].eq(1)
    ]
    per_source = max(1, sample_rows // 2)
    attacks = pd.concat(
        [
            llmail_pool.sample(min(per_source, len(llmail_pool)), random_state=seed),
            bipia_pool.sample(min(per_source, len(bipia_pool)), random_state=seed),
        ],
        ignore_index=True,
        sort=False,
    )
    attacks["user_intent"] = attacks["user_intent"].fillna("Summarize the retrieved email.")

    benign = llmail[llmail["source_dataset"].eq("LLMail_FP")].reset_index(drop=True)
    benign_features = score_features(benign, artifacts["detectors"], artifacts["stage_models"])
    final_columns = artifacts["columns"][FINAL_LINEAR]
    final_scores = artifacts["models"][FINAL_LINEAR].predict_proba(
        benign_features[final_columns]
    )[:, 1]
    lowest_risk_texts = (
        benign.iloc[np.argsort(final_scores)[:20]]["text"].fillna("").astype(str).tolist()
    )
    random_benign_texts = benign["text"].fillna("").astype(str).tolist()
    negative_words = negative_word_vocabulary(artifacts["detectors"]["word"])

    rng = np.random.default_rng(seed)
    rows = []
    for strategy in ["random_benign_email", "lowest_risk_benign_email", "negative_weight_words"]:
        for ratio in [0, 1, 2, 5, 10]:
            positions = ["none"] if ratio == 0 else ["prefix", "suffix"]
            for position in positions:
                padded = attacks.copy()
                if ratio:
                    if strategy == "random_benign_email":
                        padded["text"] = [
                            padded_copy(text, random_benign_texts, ratio, rng, position)
                            for text in padded["text"].fillna("").astype(str)
                        ]
                    elif strategy == "lowest_risk_benign_email":
                        padded["text"] = [
                            padded_copy(text, lowest_risk_texts, ratio, rng, position)
                            for text in padded["text"].fillna("").astype(str)
                        ]
                    else:
                        padded["text"] = [
                            pad_with_fixed_words(text, negative_words, ratio, position)
                            for text in padded["text"].fillna("").astype(str)
                        ]
                features = score_features(padded, artifacts["detectors"], artifacts["stage_models"])
                for method in ["text_rules_only", FINAL_LINEAR]:
                    model = artifacts["models"][method]
                    columns = artifacts["columns"][method]
                    threshold = artifacts["strict_thresholds"][method]
                    scores = model.predict_proba(features[columns])[:, 1]
                    rows.append(
                        {
                            "method": method,
                            "strategy": strategy,
                            "padding_ratio": ratio,
                            "position": position,
                            "n_attacks": len(padded),
                            "strict_threshold": threshold,
                            "aggregation": "full_document",
                            "recall": float((scores >= threshold).mean()),
                            "mean_score": float(scores.mean()),
                            "median_score": float(np.median(scores)),
                        }
                    )
                if strategy == "negative_weight_words":
                    chunks = chunk_frame(padded, chunk_words)
                    chunk_features = score_features(
                        chunks, artifacts["detectors"], artifacts["stage_models"]
                    )
                    for method in ["text_rules_only", FINAL_LINEAR]:
                        model = artifacts["models"][method]
                        columns = artifacts["columns"][method]
                        threshold = artifacts["strict_thresholds"][method]
                        chunk_scores = model.predict_proba(chunk_features[columns])[:, 1]
                        max_scores = (
                            pd.Series(chunk_scores)
                            .groupby(chunks["_attack_id"].to_numpy())
                            .max()
                            .to_numpy()
                        )
                        rows.append(
                            {
                                "method": method,
                                "strategy": strategy,
                                "padding_ratio": ratio,
                                "position": position,
                                "n_attacks": len(padded),
                                "strict_threshold": threshold,
                                "aggregation": f"chunk_max_{chunk_words}_words",
                                "recall": float((max_scores >= threshold).mean()),
                                "mean_score": float(max_scores.mean()),
                                "median_score": float(np.median(max_scores)),
                            }
                        )
    vocabulary = pd.DataFrame(
        {
            "rank": np.arange(1, len(negative_words) + 1),
            "negative_weight_word": negative_words,
        }
    )
    return pd.DataFrame(rows), vocabulary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-resistant framework evaluation.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028, 2029, 2030])
    parser.add_argument("--max-train-rows", type=int, default=60_000)
    parser.add_argument("--max-meta-rows", type=int, default=10_000)
    parser.add_argument("--max-threshold-rows", type=int, default=2_000)
    parser.add_argument("--max-stage-rows", type=int, default=70_000)
    parser.add_argument("--max-test-rows", type=int, default=2_000)
    parser.add_argument("--max-features", type=int, default=60_000)
    parser.add_argument("--max-fpr", type=float, default=0.01)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--padding-sample-rows", type=int, default=500)
    parser.add_argument("--chunk-words", type=int, default=256)
    args = parser.parse_args()

    start = time.time()
    output_dir = RESULTS_DIR / "framework"
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Loading datasets...", flush=True)
    llmail = load_llmail_binary()
    llmail_clean = load_clean_dataset()
    bipia = load_bipia_binary()
    notinject = load_notinject()
    nvidia = load_nvidia_agentic_ipi()
    promptshield = load_promptshield()
    shieldlm = load_shieldlm()
    neuralchemy = load_neuralchemy_core()

    all_results: list[dict] = []
    all_predictions: list[pd.DataFrame] = []
    diagnostics: list[dict] = []
    primary_artifacts = None

    run_plan = [(args.seeds[0], rotation) for rotation in NOTINJECT_ROTATIONS]
    run_plan.extend((seed, NOTINJECT_ROTATIONS[0]) for seed in args.seeds[1:])
    for seed, rotation in run_plan:
            run_start = time.time()
            print(f"\n[{rotation.name} / seed {seed}] preparing disjoint partitions", flush=True)
            base_train, meta_train, threshold_val, calibration, stage_notinject = build_training_partitions(
                llmail,
                bipia,
                notinject,
                rotation,
                seed,
                args.max_train_rows,
                args.max_meta_rows,
                args.max_threshold_rows,
            )
            print(
                f"  base={len(base_train)} meta={len(meta_train)} threshold_val={len(threshold_val)} "
                f"notinject_cal={len(calibration)}",
                flush=True,
            )

            detectors = {
                "word": TextDetector("word", args.max_features, random_state=seed).fit(base_train),
                "char": TextDetector("char", args.max_features, random_state=seed).fit(base_train),
                "word_char_rules": HybridDetector(args.max_features, random_state=seed).fit(base_train),
            }
            stage_training = llmail_clean[llmail_clean["phase"].eq("Phase1")].reset_index(drop=True)
            stage_models = fit_stage_models(
                stage_training,
                stage_notinject,
                max_rows=args.max_stage_rows,
                max_features=args.max_features // 2,
                random_state=seed,
            )

            x_meta = score_features(meta_train, detectors, stage_models)
            x_threshold = score_features(threshold_val, detectors, stage_models)
            x_calibration = score_features(calibration, detectors, stage_models)
            variants = feature_variants(list(x_meta.columns))
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
            x_tests = {name: score_features(frame, detectors, stage_models) for name, frame in test_sets.items()}
            labels_meta = meta_train[TARGET].astype(int).to_numpy()
            labels_threshold = threshold_val[TARGET].astype(int).to_numpy()
            run_models = {}
            run_columns = {}
            strict_thresholds = {}

            for method, columns in variants.items():
                model = fit_meta("linear", x_meta[columns], labels_meta, seed)
                run_models[method] = model
                run_columns[method] = columns
                review_scores = model.predict_proba(x_threshold[columns])[:, 1]
                review_threshold, validation_f1 = tune_threshold_f1(labels_threshold, review_scores)
                calibration_scores = model.predict_proba(x_calibration[columns])[:, 1]
                strict_threshold, calibration_fp, calibration_fpr = strict_threshold_from_benign(
                    calibration_scores, args.max_fpr
                )
                strict_thresholds[method] = strict_threshold
                diagnostics.extend(
                    [
                        {
                            "method": method,
                            "threshold_policy": "validation_f1",
                            "threshold": review_threshold,
                            "validation_f1": validation_f1,
                            "calibration_fp": np.nan,
                            "calibration_fpr": np.nan,
                            "seed": seed,
                            "rotation": rotation.name,
                        },
                        {
                            "method": method,
                            "threshold_policy": f"notinject_fpr_{args.max_fpr:.2f}",
                            "threshold": strict_threshold,
                            "validation_f1": np.nan,
                            "calibration_fp": calibration_fp,
                            "calibration_fpr": calibration_fpr,
                            "seed": seed,
                            "rotation": rotation.name,
                        },
                    ]
                )
                for policy, threshold in [
                    ("validation_f1", review_threshold),
                    (f"notinject_fpr_{args.max_fpr:.2f}", strict_threshold),
                ]:
                    rows, predictions = evaluate_one(
                        method, policy, threshold, test_sets, x_tests, model, columns, seed, rotation
                    )
                    all_results.extend(rows)
                    all_predictions.extend(predictions)

            final_columns = variants[FINAL_LINEAR]
            nonlinear = fit_meta("hgb", x_meta[final_columns], labels_meta, seed)
            run_models[FINAL_NONLINEAR] = nonlinear
            run_columns[FINAL_NONLINEAR] = final_columns
            review_scores = nonlinear.predict_proba(x_threshold[final_columns])[:, 1]
            review_threshold, validation_f1 = tune_threshold_f1(labels_threshold, review_scores)
            calibration_scores = nonlinear.predict_proba(x_calibration[final_columns])[:, 1]
            strict_threshold, calibration_fp, calibration_fpr = strict_threshold_from_benign(
                calibration_scores, args.max_fpr
            )
            strict_thresholds[FINAL_NONLINEAR] = strict_threshold
            diagnostics.extend(
                [
                    {
                        "method": FINAL_NONLINEAR,
                        "threshold_policy": "validation_f1",
                        "threshold": review_threshold,
                        "validation_f1": validation_f1,
                        "calibration_fp": np.nan,
                        "calibration_fpr": np.nan,
                        "seed": seed,
                        "rotation": rotation.name,
                    },
                    {
                        "method": FINAL_NONLINEAR,
                        "threshold_policy": f"notinject_fpr_{args.max_fpr:.2f}",
                        "threshold": strict_threshold,
                        "validation_f1": np.nan,
                        "calibration_fp": calibration_fp,
                        "calibration_fpr": calibration_fpr,
                        "seed": seed,
                        "rotation": rotation.name,
                    },
                ]
            )
            for policy, threshold in [
                ("validation_f1", review_threshold),
                (f"notinject_fpr_{args.max_fpr:.2f}", strict_threshold),
            ]:
                rows, predictions = evaluate_one(
                    FINAL_NONLINEAR,
                    policy,
                    threshold,
                    test_sets,
                    x_tests,
                    nonlinear,
                    final_columns,
                    seed,
                    rotation,
                )
                all_results.extend(rows)
                all_predictions.extend(predictions)

            if seed == args.seeds[0] and rotation == NOTINJECT_ROTATIONS[0]:
                primary_artifacts = {
                    "detectors": detectors,
                    "stage_models": stage_models,
                    "models": run_models,
                    "columns": run_columns,
                    "strict_thresholds": strict_thresholds,
                }
            print(f"  finished in {(time.time() - run_start) / 60:.1f} minutes", flush=True)

    results = pd.DataFrame(all_results)
    predictions = pd.concat(all_predictions, ignore_index=True)
    run_summary = summarize_one_run(results)
    aggregate = aggregate_runs(run_summary)
    crossfit = aggregate_runs(run_summary[run_summary["seed"].eq(args.seeds[0])])
    seed_variance = aggregate_runs(
        run_summary[run_summary["rotation"].eq(NOTINJECT_ROTATIONS[0].name)]
    )
    notinject_ci = notinject_intervals(run_summary)
    bootstrap = bootstrap_intervals(
        predictions,
        args.seeds[0],
        NOTINJECT_ROTATIONS[0].name,
        args.bootstrap_samples,
    )
    if primary_artifacts is None:
        raise RuntimeError("Primary experiment artifacts were not retained")
    padding = run_padding_stress(primary_artifacts, llmail, bipia, args.seeds[0], args.padding_sample_rows)
    rename = run_literal_rename_stress(primary_artifacts, llmail)
    adaptive_padding, negative_words = run_adaptive_padding_stress(
        primary_artifacts,
        llmail,
        bipia,
        args.seeds[0],
        args.padding_sample_rows,
        args.chunk_words,
    )

    outputs = {
        "evaluation_results.csv": results,
        "evaluation_predictions.csv": predictions,
        "evaluation_run_summary.csv": run_summary,
        "evaluation_aggregate_summary.csv": aggregate,
        "evaluation_crossfit_summary.csv": crossfit,
        "evaluation_seed_variance_summary.csv": seed_variance,
        "evaluation_threshold_diagnostics.csv": pd.DataFrame(diagnostics),
        "notinject_confidence_intervals.csv": notinject_ci,
        "bootstrap_confidence_intervals.csv": bootstrap,
        "padding_stress.csv": padding,
        "literal_rename_stress.csv": rename,
        "adaptive_padding_stress.csv": adaptive_padding,
        "adaptive_padding_negative_words.csv": negative_words,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False)
    settings = vars(args) | {
        "notinject_rotations": [rotation.__dict__ for rotation in NOTINJECT_ROTATIONS],
        "run_plan": [{"seed": seed, "rotation": rotation.name} for seed, rotation in run_plan],
    }
    (output_dir / "evaluation_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print("\nAggregate summary", flush=True)
    print(aggregate.to_string(index=False), flush=True)
    print(f"Saved framework evaluation results to {output_dir}", flush=True)
    print(f"Finished in {(time.time() - start) / 60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
