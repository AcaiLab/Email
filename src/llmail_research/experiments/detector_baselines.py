import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from llmail_research.config import RESULTS_DIR, SEED
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
from llmail_research.eval_utils import (
    NOTINJECT_ROTATIONS,
    build_matched_test_sets,
    clopper_pearson_interval,
    strict_threshold_from_benign,
)
from llmail_research.metrics import metrics_from_scores

TARGET = "label_attack"

MODEL_SPECS = [
    {
        "method": "ProtectAI-DeBERTa-v1",
        "model_id": "protectai/deberta-v3-base-prompt-injection",
        "positive_labels": ["INJECTION", "MALICIOUS", "LABEL_1"],
    },
    {
        "method": "ProtectAI-DeBERTa-v2",
        "model_id": "ProtectAI/deberta-v3-base-prompt-injection-v2",
        "positive_labels": ["INJECTION", "MALICIOUS", "LABEL_1"],
    },
    {
        "method": "deepset-DeBERTa-injection",
        "model_id": "deepset/deberta-v3-base-injection",
        "positive_labels": ["INJECTION", "MALICIOUS", "LABEL_1"],
    },
    {
        "method": "TestSavant-tiny",
        "model_id": "testsavantai/prompt-injection-defender-tiny-v0",
        "positive_labels": ["INJECTION", "MALICIOUS", "LABEL_1"],
    },
    {
        "method": "Arkaean-DistilBERT",
        "model_id": "arkaean/promptguard-distilbert",
        "positive_labels": ["INJECTION", "MALICIOUS", "LABEL_1"],
    },
]

INACCESSIBLE_MODEL_SPECS = [
    {"method": "Meta-Prompt-Guard-86M", "model_id": "meta-llama/Prompt-Guard-86M", "reason": "gated"},
    {"method": "ProtectAI-DeBERTa-small-v2", "model_id": "ProtectAI/deberta-v3-small-prompt-injection-v2", "reason": "gated"},
    {"method": "CodeIntegrity-PromptGuard", "model_id": "codeintegrity-ai/promptguard", "reason": "gated"},
]


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def positive_index(config, positive_labels: list[str]) -> int:
    labels = {int(idx): str(label).upper() for idx, label in config.id2label.items()}
    for idx, label in labels.items():
        if label in {item.upper() for item in positive_labels}:
            return idx
    if len(labels) == 2:
        return 1
    raise ValueError(f"Cannot infer positive label from {labels}")


def batched_scores(model, tokenizer, texts: list[str], pos_idx: int, device: torch.device, batch_size: int, max_length: int) -> np.ndarray:
    scores = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)[:, pos_idx]
            scores.extend(probs.detach().cpu().numpy().tolist())
    return np.asarray(scores, dtype=float)


def metrics_binary(y_true, scores, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out = {
        "threshold": float(threshold),
        "accuracy": float((pred == y_true).mean()),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
        out["avg_precision"] = float(average_precision_score(y_true, scores))
    return out


def threshold_from_benign(scores: np.ndarray, max_fpr: float) -> float:
    if len(scores) == 0:
        return 1.01
    # Minimum threshold that keeps predicted-positive rate <= max_fpr.
    return float(np.quantile(scores, 1 - max_fpr, method="higher"))


def evaluate(method, policy, threshold, source, frame, scores, rotation) -> dict:
    labels = frame[TARGET].astype(int).to_numpy()
    predictions = (scores >= threshold).astype(int)
    row = {
        "method": method,
        "training_adaptation": "frozen pretrained weights; threshold calibration only",
        "threshold_policy": policy,
        "threshold": float(threshold),
        "rotation": rotation.name,
        "notinject_calibration_split": rotation.calibration_split,
        "notinject_test_split": rotation.test_split,
        "test_source": source,
        "n_test": len(frame),
        "test_positive_rate": float(labels.mean()) if len(labels) else float("nan"),
        "predicted_positive_rate": float(predictions.mean()),
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
    return row


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, policy, rotation), group in results.groupby(["method", "threshold_policy", "rotation"]):
        binary = group[group["test_f1"].notna()]

        def value(source, column):
            selected = group.loc[group["test_source"].eq(source), column]
            return float(selected.iloc[0]) if len(selected) else float("nan")

        rows.append(
            {
                "method": method,
                "threshold_policy": policy,
                "rotation": rotation,
                "mean_binary_f1": float(binary["test_f1"].mean()),
                "mean_binary_precision": float(binary["test_precision"].mean()),
                "mean_binary_recall": float(binary["test_recall"].mean()),
                "notinject_fpr": value("notinject", "test_false_positive_rate"),
                "notinject_fp": int(value("notinject", "test_fp")),
                "notinject_n": int(value("notinject", "n_test")),
                "nvidia_recall": value("nvidia_agentic_ipi", "test_recall"),
                "bipia_f1": value("bipia_test", "test_f1"),
                "llmail_f1": value("llmail_phase2_binary", "test_f1"),
                "promptshield_f1": value("promptshield_test", "test_f1"),
                "shieldlm_f1": value("shieldlm_test", "test_f1"),
                "neuralchemy_f1": value("neuralchemy_core_test", "test_f1"),
            }
        )
    return pd.DataFrame(rows)


def aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
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
    for (method, policy), group in summary.groupby(["method", "threshold_policy"]):
        row = {"method": method, "threshold_policy": policy, "n_rotations": len(group)}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        false_positives = int(group["notinject_fp"].sum())
        trials = int(group["notinject_n"].sum())
        low, high = clopper_pearson_interval(false_positives, trials)
        row["notinject_fp"] = false_positives
        row["notinject_n"] = trials
        row["notinject_fpr_ci95_lower"] = low
        row["notinject_fpr_ci95_upper"] = high
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["threshold_policy", "mean_binary_f1_mean"], ascending=[True, False]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matched published-detector comparison.")
    parser.add_argument("--max-test-rows", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-fpr", type=float, default=0.01)
    parser.add_argument("--models", nargs="*", default=[spec["method"] for spec in MODEL_SPECS])
    args = parser.parse_args()

    start = time.time()
    output_dir = RESULTS_DIR / "published_detectors"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device()
    print(f"Using device: {device}", flush=True)

    llmail = load_llmail_binary()
    bipia = load_bipia_binary()
    notinject = load_notinject()
    nvidia = load_nvidia_agentic_ipi()
    promptshield = load_promptshield()
    shieldlm = load_shieldlm()
    neuralchemy = load_neuralchemy_core()
    common_sets = build_matched_test_sets(
        llmail,
        bipia,
        notinject,
        nvidia,
        promptshield,
        shieldlm,
        neuralchemy,
        NOTINJECT_ROTATIONS[0].test_split,
        args.max_test_rows,
        SEED,
    )
    common_sets.pop("notinject")
    selected_specs = [spec for spec in MODEL_SPECS if spec["method"] in set(args.models)]
    results = []
    audit = []

    for spec in selected_specs:
        model_start = time.time()
        print(f"Loading {spec['method']} ({spec['model_id']})", flush=True)
        try:
            config = AutoConfig.from_pretrained(spec["model_id"])
            tokenizer = AutoTokenizer.from_pretrained(spec["model_id"])
            model = AutoModelForSequenceClassification.from_pretrained(spec["model_id"])
            model.to(device)
            positive_label = positive_index(config, spec["positive_labels"])
            score_cache = {}
            for source, frame in common_sets.items():
                print(f"  scoring {source}: {len(frame)}", flush=True)
                score_cache[source] = batched_scores(
                    model,
                    tokenizer,
                    frame["text"].fillna("").astype(str).tolist(),
                    positive_label,
                    device,
                    args.batch_size,
                    args.max_length,
                )
            for split in sorted(notinject["split"].unique()):
                frame = notinject[notinject["split"].eq(split)].reset_index(drop=True)
                score_cache[split] = batched_scores(
                    model,
                    tokenizer,
                    frame["text"].fillna("").astype(str).tolist(),
                    positive_label,
                    device,
                    args.batch_size,
                    args.max_length,
                )

            for rotation in NOTINJECT_ROTATIONS:
                calibration_scores = score_cache[rotation.calibration_split]
                strict_threshold, _, _ = strict_threshold_from_benign(calibration_scores, args.max_fpr)
                thresholds = {"default_0.50": 0.5, f"notinject_fpr_{args.max_fpr:.2f}": strict_threshold}
                for policy, threshold in thresholds.items():
                    for source, frame in common_sets.items():
                        results.append(
                            evaluate(spec["method"], policy, threshold, source, frame, score_cache[source], rotation)
                        )
                    ni_test = notinject[notinject["split"].eq(rotation.test_split)].reset_index(drop=True)
                    results.append(
                        evaluate(
                            spec["method"],
                            policy,
                            threshold,
                            "notinject",
                            ni_test,
                            score_cache[rotation.test_split],
                            rotation,
                        )
                    )
            audit.append(
                {
                    **spec,
                    "status": "ok",
                    "seconds": round(time.time() - model_start, 1),
                    "positive_index": positive_label,
                    "labels": config.id2label,
                }
            )
            del model
            if device.type == "mps":
                torch.mps.empty_cache()
        except Exception as exc:
            audit.append({**spec, "status": "failed", "error": repr(exc)})
            print(f"FAILED {spec['method']}: {exc}", flush=True)

    for spec in INACCESSIBLE_MODEL_SPECS:
        audit.append({**spec, "status": "not_run", "error": spec["reason"]})

    detailed = pd.DataFrame(results)
    per_rotation = summarize(detailed) if len(detailed) else pd.DataFrame()
    summary = aggregate(per_rotation) if len(per_rotation) else pd.DataFrame()
    detailed.to_csv(output_dir / "published_detector_results.csv", index=False)
    per_rotation.to_csv(output_dir / "published_detector_rotation_summary.csv", index=False)
    summary.to_csv(output_dir / "published_detector_summary.csv", index=False)
    (output_dir / "published_detector_audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"Saved matched published-detector results to {output_dir}", flush=True)
    print(f"Finished in {(time.time() - start) / 60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
