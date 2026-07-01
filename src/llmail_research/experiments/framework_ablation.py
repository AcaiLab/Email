import argparse
import re
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from llmail_research.config import RESULTS_DIR, SEED, TARGET_DESTINATION
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
TARGET_EMAIL_PATTERN = re.escape(TARGET_DESTINATION).replace("@", r"\s*@\s*").replace(r"\.", r"\s*\.\s*")
RAW_STAGE_COLUMNS = [f"stage_{target}_score" for target in STAGE_TARGETS]
DERIVED_STAGE_COLUMNS = ["stage_max_score", "unsafe_tool_chain_score"]


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
    features["untrusted_has_target_email"] = has_regex(untrusted, TARGET_EMAIL_PATTERN)
    features["text_has_target_email_proxy"] = has_regex(text, TARGET_EMAIL_PATTERN)
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
        0.30 * features["intent_send_conflict"]
        + 0.25 * features["recipient_introduced_by_untrusted"]
        + 0.20 * features["untrusted_exfil_score"]
        + 0.15 * features["boundary_control_risk"]
        + 0.10 * features["untrusted_has_target_email"]
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate proposed attack-chain framework components.")
    parser.add_argument("--max-train-rows", type=int, default=60_000)
    parser.add_argument("--max-stage-rows", type=int, default=70_000)
    parser.add_argument("--max-test-rows", type=int, default=2_000)
    parser.add_argument("--max-features", type=int, default=60_000)
    parser.add_argument("--max-fpr", type=float, default=0.01)
    args = parser.parse_args()

    start = time.time()
    output_dir = RESULTS_DIR / "framework"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    llmail_clean, train_pool, test_sets = load_experiment_data(args.max_test_rows)
    sampled_train = sample_equal_classes(train_pool, TARGET, args.max_train_rows)
    train, policy_val = train_test_split(
        sampled_train,
        test_size=0.25,
        random_state=SEED,
        stratify=sampled_train[TARGET].astype(int),
    )
    train = train.reset_index(drop=True)
    policy_val = policy_val.reset_index(drop=True)

    print("Fitting base detectors...")
    word_char = HybridDetector(max_features=args.max_features).fit(train)
    word = TextDetector("word", max_features=args.max_features).fit(train)
    char = TextDetector("char", max_features=args.max_features).fit(train)
    detectors = {"word": word, "char": char, "word_char_rules": word_char}

    print("Fitting LLMail stage verifier models...")
    stage_models = fit_stage_models(llmail_clean, load_notinject(), max_rows=args.max_stage_rows, max_features=args.max_features // 2)

    print("Building validation features...")
    x_val_chain = stack_features(policy_val, detectors, stage_models)
    x_val_all = augment_features(policy_val, x_val_chain)
    variants = feature_variants(list(x_val_all.columns))
    y_val = policy_val[TARGET].astype(int).to_numpy()
    benign_mask = policy_val["source_dataset"].eq("NotInject").to_numpy()

    print("Building test features...")
    x_test_all = {}
    for name, frame in test_sets.items():
        x_test_all[name] = augment_features(frame, stack_features(frame, detectors, stage_models))

    rows = []
    diagnostic_rows = []
    policy_name = f"notinject_fpr_{args.max_fpr:.2f}"
    for method, columns in variants.items():
        print(f"Fitting meta policy: {method} ({len(columns)} features)")
        meta = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
        meta.fit(x_val_all[columns], y_val)
        val_scores = meta.predict_proba(x_val_all[columns])[:, 1]
        f1_threshold, f1_val = tune_threshold_f1(y_val, val_scores)
        calibrated_threshold, calibrated_recall, calibrated_fpr = choose_threshold_under_fpr(
            y_val,
            val_scores,
            benign_mask,
            args.max_fpr,
        )
        thresholds = {
            "val_f1": f1_threshold,
            policy_name: calibrated_threshold,
        }
        diagnostic_rows.extend(
            [
                {
                    "method": method,
                    "threshold_policy": "val_f1",
                    "threshold": f1_threshold,
                    "val_f1": f1_val,
                    "notinject_val_recall": np.nan,
                    "notinject_val_fpr": np.nan,
                    "n_features": len(columns),
                    "features": ",".join(columns),
                },
                {
                    "method": method,
                    "threshold_policy": policy_name,
                    "threshold": calibrated_threshold,
                    "val_f1": np.nan,
                    "notinject_val_recall": calibrated_recall,
                    "notinject_val_fpr": calibrated_fpr,
                    "n_features": len(columns),
                    "features": ",".join(columns),
                },
            ]
        )
        test_scores = {
            name: meta.predict_proba(features[columns])[:, 1]
            for name, features in x_test_all.items()
        }
        rows.extend(evaluate_scores(method, thresholds, test_sets, test_scores))

    results = pd.DataFrame(rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    summary = add_delta_columns(summarize(results))

    results_path = output_dir / "component_extension_results.csv"
    summary_path = output_dir / "component_extension_summary.csv"
    diagnostics_path = output_dir / "component_extension_diagnostics.csv"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)

    print(f"Saved detailed results to {results_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved diagnostics to {diagnostics_path}")
    print(f"Finished in {(time.time() - start) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
