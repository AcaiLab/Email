import argparse
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

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
from llmail_research.modeling import get_scores

TARGET = "label_attack"
STAGE_TARGETS = ["defense_bypassed", "tool_called", "destination_correct", "content_correct", "full_success"]


def vectorizer_for(mode: str, max_features: int) -> TfidfVectorizer:
    if mode == "word":
        return TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    if mode == "char":
        return TfidfVectorizer(max_features=max_features, analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)
    raise ValueError(f"Unsupported mode: {mode}")


def fit_sgd(x_train, y_train) -> SGDClassifier:
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        max_iter=50,
        tol=1e-3,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED,
    )
    model.fit(x_train, y_train)
    return model


class TextDetector:
    def __init__(self, mode: str, max_features: int):
        self.mode = mode
        self.vectorizer = vectorizer_for(mode, max_features)
        self.model = None

    def fit(self, frame: pd.DataFrame, target: str = TARGET) -> "TextDetector":
        x = self.vectorizer.fit_transform(frame["text"].fillna("").astype(str))
        self.model = fit_sgd(x, frame[target].astype(int).to_numpy())
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        x = self.vectorizer.transform(frame["text"].fillna("").astype(str))
        return get_scores(self.model, x)


class HybridDetector:
    def __init__(self, max_features: int):
        self.word = vectorizer_for("word", max_features // 2)
        self.char = vectorizer_for("char", max_features // 2)
        self.scaler = MaxAbsScaler()
        self.model = None

    def _fit_matrix(self, frame: pd.DataFrame):
        text = frame["text"].fillna("").astype(str)
        x_word = self.word.fit_transform(text)
        x_char = self.char.fit_transform(text)
        x_rules = self.scaler.fit_transform(build_heuristic_features(text))
        return hstack([x_word, x_char, x_rules])

    def _matrix(self, frame: pd.DataFrame):
        text = frame["text"].fillna("").astype(str)
        x_word = self.word.transform(text)
        x_char = self.char.transform(text)
        x_rules = self.scaler.transform(build_heuristic_features(text))
        return hstack([x_word, x_char, x_rules])

    def fit(self, frame: pd.DataFrame, target: str = TARGET) -> "HybridDetector":
        x = self._fit_matrix(frame)
        self.model = fit_sgd(x, frame[target].astype(int).to_numpy())
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        return get_scores(self.model, self._matrix(frame))


def balanced_sample(frame: pd.DataFrame, target: str, max_rows: int) -> pd.DataFrame:
    frame = frame.dropna(subset=["text", target]).copy()
    frame[target] = frame[target].astype(int)
    groups = [group for _, group in frame.groupby(target)]
    if len(groups) < 2:
        return frame.sample(min(len(frame), max_rows), random_state=SEED).reset_index(drop=True)
    per_class = min(min(len(group) for group in groups), max(1, max_rows // 2))
    sampled = [group.sample(per_class, random_state=SEED) for group in groups]
    return pd.concat(sampled, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)


def build_external_stage_like_frame(
    bipia_train: pd.DataFrame,
    promptshield_train: pd.DataFrame,
    shieldlm_train: pd.DataFrame,
    neuralchemy_train: pd.DataFrame,
    max_rows: int,
) -> pd.DataFrame:
    """Approximate stage labels from non-LLMail training datasets.

    These are intentionally noisy auxiliary labels, not claimed ground truth.
    NVIDIA is excluded because it has only a train split and is evaluated as a
    held-out attack-only benchmark below.
    """
    frames = []
    keep = [bipia_train, promptshield_train, shieldlm_train, neuralchemy_train]
    for frame in keep:
        cols = [col for col in ["source_dataset", "text", "label_attack", "domain", "label_category", "category"] if col in frame.columns]
        frames.append(frame[cols].copy())
    external = pd.concat(frames, ignore_index=True, sort=False)
    external = external.dropna(subset=["text", "label_attack"]).copy()
    if len(external) > max_rows:
        pieces = []
        for _, group in external.groupby("label_attack"):
            n = min(len(group), max(1, int(max_rows * len(group) / len(external))))
            pieces.append(group.sample(n, random_state=SEED))
        external = pd.concat(pieces, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
    rules = rule_scores(external)
    attack = external["label_attack"].astype(int).to_numpy()
    category_text = (
        external.get("domain", "").fillna("").astype(str)
        + " "
        + external.get("label_category", "").fillna("").astype(str)
        + " "
        + external.get("category", "").fillna("").astype(str)
    ).str.lower()

    toolish_category = category_text.str.contains("indirect|agent|rag|tool|system|training|extraction|smuggling", regex=True)
    destination_like = (rules["rule_tool_action"].to_numpy() > 0) | external["text"].fillna("").astype(str).str.contains(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", case=False, regex=True
    ).to_numpy()
    content_like = (rules["rule_exfiltration"].to_numpy() > 0) | category_text.str.contains(
        "exfil|training|extraction|rag|system|data|secret", regex=True
    ).to_numpy()

    out = external[["text"]].copy()
    out["defense_bypassed"] = attack
    out["tool_called"] = (attack & ((rules["rule_tool_action"].to_numpy() > 0) | toolish_category.to_numpy())).astype(int)
    out["destination_correct"] = (attack & destination_like).astype(int)
    out["content_correct"] = (attack & content_like).astype(int)
    out["full_success"] = (out["tool_called"].to_numpy() & (out["destination_correct"].to_numpy() | out["content_correct"].to_numpy())).astype(int)
    return out.reset_index(drop=True)


def fit_stage_models(
    llmail_clean: pd.DataFrame,
    notinject: pd.DataFrame,
    max_rows: int,
    max_features: int,
    external_stage_like: pd.DataFrame | None = None,
) -> dict[str, TextDetector]:
    models = {}
    for target in STAGE_TARGETS:
        benign = notinject[["text"]].copy()
        positives_and_negatives = llmail_clean[["text", target]].copy()
        benign[target] = 0
        pieces = [positives_and_negatives, benign[["text", target]]]
        if external_stage_like is not None and target in external_stage_like.columns:
            pieces.append(external_stage_like[["text", target]])
        stage_frame = pd.concat(pieces, ignore_index=True, sort=False)
        sampled = balanced_sample(stage_frame, target, max_rows=max_rows)
        models[target] = TextDetector("char", max_features=max_features).fit(sampled, target=target)
    return models


def rule_scores(frame: pd.DataFrame) -> pd.DataFrame:
    heuristics = build_heuristic_features(frame["text"].fillna("").astype(str))
    score = pd.DataFrame(index=frame.index)
    score["rule_instruction"] = (
        heuristics["group_instruction_override_present"]
        + heuristics["group_role_or_authority_present"]
        + heuristics["group_delimiter_or_formatting_present"]
    ).clip(upper=1)
    score["rule_tool_action"] = (
        heuristics["group_tool_action_present"]
        + heuristics["group_target_artifacts_present"]
        + (heuristics["email_address_count"] > 0).astype(float)
    ).clip(upper=1)
    score["rule_exfiltration"] = (
        heuristics["group_exfiltration_present"] + heuristics["group_secrecy_or_stealth_present"]
    ).clip(upper=1)
    score["rule_obfuscation"] = heuristics["group_obfuscation_present"].clip(upper=1)
    score["rule_any"] = score.max(axis=1)
    score["rule_chain"] = (
        0.30 * score["rule_instruction"]
        + 0.35 * score["rule_tool_action"]
        + 0.25 * score["rule_exfiltration"]
        + 0.10 * score["rule_obfuscation"]
    ).clip(upper=1)
    return score


def stack_features(frame: pd.DataFrame, detectors: dict, stage_models: dict[str, TextDetector]) -> pd.DataFrame:
    features = pd.DataFrame(index=frame.index)
    for name, detector in detectors.items():
        features[f"{name}_score"] = detector.score(frame)
    for target, model in stage_models.items():
        features[f"stage_{target}_score"] = model.score(frame)
    rules = rule_scores(frame)
    features = pd.concat([features, rules], axis=1)
    features["stage_max_score"] = features[[f"stage_{target}_score" for target in STAGE_TARGETS]].max(axis=1)
    features["unsafe_tool_chain_score"] = (
        0.35 * features["stage_tool_called_score"]
        + 0.25 * features["stage_destination_correct_score"]
        + 0.25 * features["stage_content_correct_score"]
        + 0.15 * features["rule_tool_action"]
    ).clip(upper=1)
    return features.replace([np.inf, -np.inf], 0).fillna(0)


def choose_threshold_under_fpr(y_true, scores, benign_mask, max_fpr: float) -> tuple[float, float, float]:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    benign_mask = np.asarray(benign_mask).astype(bool)
    thresholds = np.unique(np.quantile(scores, np.linspace(0, 1, 501)))
    thresholds = np.r_[thresholds, 1.01]
    best = (1.01, -1.0, 0.0)
    for threshold in thresholds:
        pred = scores >= threshold
        benign_count = int(benign_mask.sum())
        fpr = float(pred[benign_mask].mean()) if benign_count else 0.0
        recall = float(pred[y_true == 1].mean()) if (y_true == 1).any() else 0.0
        if fpr <= max_fpr and recall > best[1]:
            best = (float(threshold), recall, fpr)
    if best[1] < 0:
        return 1.01, 0.0, 0.0
    return best


def evaluate_frame(method: str, score_fn, thresholds: dict[str, float], test_sets: dict[str, pd.DataFrame]) -> list[dict]:
    rows = []
    for test_name, test_frame in test_sets.items():
        scores = score_fn(test_frame)
        for threshold_name, threshold in thresholds.items():
            row = {
                "method": method,
                "threshold_policy": threshold_name,
                "test_source": test_name,
                "threshold": threshold,
                "n_test": len(test_frame),
                "test_positive_rate": float(test_frame[TARGET].mean()),
                "predicted_positive_rate": float((scores >= threshold).mean()),
            }
            if test_frame[TARGET].nunique() == 2:
                row.update(metrics_from_scores(test_frame[TARGET], scores, threshold, prefix="test_"))
            else:
                label = int(test_frame[TARGET].iloc[0]) if len(test_frame) else 0
                if label == 0:
                    row["test_false_positive_rate"] = float((scores >= threshold).mean())
                else:
                    row["test_false_negative_rate"] = float((scores < threshold).mean())
                    row["test_recall"] = float((scores >= threshold).mean())
            rows.append(row)
    return rows


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


def value_for(group: pd.DataFrame, test_source: str, column: str) -> float:
    value = group.loc[group["test_source"].eq(test_source), column]
    return float(value.iloc[0]) if len(value) else np.nan


def main() -> None:
    parser = argparse.ArgumentParser(description="Run attack-chain framework experiments.")
    parser.add_argument("--max-train-rows", type=int, default=60_000)
    parser.add_argument("--max-stage-rows", type=int, default=70_000)
    parser.add_argument("--max-test-rows", type=int, default=20_000)
    parser.add_argument("--max-features", type=int, default=60_000)
    parser.add_argument("--max-fpr", type=float, default=0.01)
    args = parser.parse_args()

    start = time.time()
    llmail = load_llmail_binary()
    llmail_clean = load_clean_dataset()
    bipia = load_bipia_binary()
    notinject = load_notinject()
    nvidia = load_nvidia_agentic_ipi()
    promptshield = load_promptshield()
    shieldlm = load_shieldlm()
    neuralchemy = load_neuralchemy_core()

    bipia_train = bipia[bipia["split"].eq("train")].reset_index(drop=True)
    promptshield_train = promptshield[promptshield["split"].eq("train")].reset_index(drop=True)
    shieldlm_train = shieldlm[shieldlm["split"].eq("train")].reset_index(drop=True)
    neuralchemy_train = neuralchemy[neuralchemy["split"].eq("train")].reset_index(drop=True)
    train_pool = pd.concat([llmail, bipia_train, notinject], ignore_index=True, sort=False)
    sampled_train = sample_equal_classes(train_pool, TARGET, args.max_train_rows)
    train, policy_val = train_test_split(
        sampled_train,
        test_size=0.25,
        random_state=SEED,
        stratify=sampled_train[TARGET].astype(int),
    )
    train = train.reset_index(drop=True)
    policy_val = policy_val.reset_index(drop=True)

    print("Fitting baseline detectors...")
    word_char = HybridDetector(max_features=args.max_features).fit(train)
    word = TextDetector("word", max_features=args.max_features).fit(train)
    char = TextDetector("char", max_features=args.max_features).fit(train)

    print("Fitting stage verifier models...")
    stage_models = fit_stage_models(llmail_clean, notinject, max_rows=args.max_stage_rows, max_features=args.max_features // 2)
    external_stage_like = build_external_stage_like_frame(
        bipia_train,
        promptshield_train,
        shieldlm_train,
        neuralchemy_train,
        max_rows=args.max_stage_rows,
    )
    print("Fitting externally enriched stage verifier models...")
    enriched_stage_models = fit_stage_models(
        llmail_clean,
        notinject,
        max_rows=args.max_stage_rows,
        max_features=args.max_features // 2,
        external_stage_like=external_stage_like,
    )

    detectors = {"word": word, "char": char, "word_char_rules": word_char}
    print("Fitting attack-chain meta policy...")
    x_meta = stack_features(policy_val, detectors, stage_models)
    meta = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    meta.fit(x_meta, policy_val[TARGET].astype(int))
    framework_val_scores = meta.predict_proba(x_meta)[:, 1]
    x_enriched_meta = stack_features(policy_val, detectors, enriched_stage_models)
    enriched_meta = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    enriched_meta.fit(x_enriched_meta, policy_val[TARGET].astype(int))
    enriched_val_scores = enriched_meta.predict_proba(x_enriched_meta)[:, 1]

    word_char_val_scores = word_char.score(policy_val)
    f1_threshold, f1_val = tune_threshold_f1(policy_val[TARGET], framework_val_scores)
    enriched_f1_threshold, enriched_f1_val = tune_threshold_f1(policy_val[TARGET], enriched_val_scores)
    baseline_f1_threshold, baseline_f1_val = tune_threshold_f1(policy_val[TARGET], word_char_val_scores)
    benign_mask = policy_val["source_dataset"].eq("NotInject").to_numpy()
    calibrated_threshold, calibrated_recall, calibrated_fpr = choose_threshold_under_fpr(
        policy_val[TARGET], framework_val_scores, benign_mask, args.max_fpr
    )
    enriched_calibrated_threshold, enriched_calibrated_recall, enriched_calibrated_fpr = choose_threshold_under_fpr(
        policy_val[TARGET], enriched_val_scores, benign_mask, args.max_fpr
    )
    baseline_calibrated_threshold, baseline_calibrated_recall, baseline_calibrated_fpr = choose_threshold_under_fpr(
        policy_val[TARGET], word_char_val_scores, benign_mask, args.max_fpr
    )

    test_sets = {
        "bipia_test": sample_equal_classes(bipia[bipia["split"].eq("test")].reset_index(drop=True), TARGET, args.max_test_rows),
        "llmail_binary": sample_equal_classes(llmail, TARGET, args.max_test_rows),
        "notinject": notinject.reset_index(drop=True),
        "nvidia_agentic_ipi": nvidia.reset_index(drop=True),
        "promptshield_test": sample_equal_classes(promptshield[promptshield["split"].eq("test")].reset_index(drop=True), TARGET, args.max_test_rows),
        "shieldlm_test": sample_equal_classes(shieldlm[shieldlm["split"].eq("test")].reset_index(drop=True), TARGET, args.max_test_rows),
        "neuralchemy_core_test": sample_equal_classes(neuralchemy[neuralchemy["split"].eq("test")].reset_index(drop=True), TARGET, args.max_test_rows),
    }

    rows = []
    rows.extend(
        evaluate_frame(
            "baseline_word_char_rules",
            lambda frame: word_char.score(frame),
            {
                "val_f1": baseline_f1_threshold,
                f"notinject_fpr_{args.max_fpr:.2f}": baseline_calibrated_threshold,
            },
            test_sets,
        )
    )
    rows.extend(
        evaluate_frame(
            "attack_chain_framework",
            lambda frame: meta.predict_proba(stack_features(frame, detectors, stage_models))[:, 1],
            {
                "val_f1": f1_threshold,
                f"notinject_fpr_{args.max_fpr:.2f}": calibrated_threshold,
            },
            test_sets,
        )
    )
    rows.extend(
        evaluate_frame(
            "attack_chain_framework_external_stage",
            lambda frame: enriched_meta.predict_proba(stack_features(frame, detectors, enriched_stage_models))[:, 1],
            {
                "val_f1": enriched_f1_threshold,
                f"notinject_fpr_{args.max_fpr:.2f}": enriched_calibrated_threshold,
            },
            test_sets,
        )
    )
    results = pd.DataFrame(rows)
    output = RESULTS_DIR / "framework" / "attack_chain_framework_results.csv"
    summary_output = RESULTS_DIR / "framework" / "attack_chain_framework_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    summary = summarize(results)
    summary.to_csv(summary_output, index=False)

    diagnostics = pd.DataFrame(
        [
            {
                "method": "baseline_word_char_rules",
                "val_f1_threshold": baseline_f1_threshold,
                "val_best_f1": baseline_f1_val,
                "calibrated_threshold": baseline_calibrated_threshold,
                "calibrated_val_attack_recall": baseline_calibrated_recall,
                "calibrated_val_notinject_fpr": baseline_calibrated_fpr,
            },
            {
                "method": "attack_chain_framework",
                "val_f1_threshold": f1_threshold,
                "val_best_f1": f1_val,
                "calibrated_threshold": calibrated_threshold,
                "calibrated_val_attack_recall": calibrated_recall,
                "calibrated_val_notinject_fpr": calibrated_fpr,
            },
            {
                "method": "attack_chain_framework_external_stage",
                "val_f1_threshold": enriched_f1_threshold,
                "val_best_f1": enriched_f1_val,
                "calibrated_threshold": enriched_calibrated_threshold,
                "calibrated_val_attack_recall": enriched_calibrated_recall,
                "calibrated_val_notinject_fpr": enriched_calibrated_fpr,
            },
        ]
    )
    diagnostics_output = RESULTS_DIR / "framework" / "attack_chain_framework_diagnostics.csv"
    diagnostics.to_csv(diagnostics_output, index=False)
    print(summary.to_string(index=False))
    print(diagnostics.to_string(index=False))
    print(f"Saved: {output}")
    print(f"Saved: {summary_output}")
    print(f"Saved: {diagnostics_output}")
    print(f"Elapsed seconds: {time.time() - start:.1f}")


if __name__ == "__main__":
    main()
