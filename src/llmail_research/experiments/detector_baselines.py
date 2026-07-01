import argparse
import json
import time
from pathlib import Path

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


def test_sets(max_test_rows: int) -> dict[str, pd.DataFrame]:
    llmail = load_llmail_binary()
    bipia = load_bipia_binary()
    notinject = load_notinject()
    nvidia = load_nvidia_agentic_ipi()
    promptshield = load_promptshield()
    shieldlm = load_shieldlm()
    neuralchemy = load_neuralchemy_core()
    heldout_notinject = notinject[notinject["split"].isin(["NotInject_two", "NotInject_three"])].reset_index(drop=True)
    calibrate_notinject = notinject[notinject["split"].eq("NotInject_one")].reset_index(drop=True)
    return {
        "notinject_calibration": calibrate_notinject,
        "bipia_test": sample_equal_classes(bipia[bipia["split"].eq("test")].reset_index(drop=True), TARGET, max_test_rows),
        "llmail_binary": sample_equal_classes(llmail, TARGET, max_test_rows),
        "notinject": heldout_notinject,
        "nvidia_agentic_ipi": nvidia.reset_index(drop=True),
        "promptshield_test": sample_equal_classes(promptshield[promptshield["split"].eq("test")].reset_index(drop=True), TARGET, max_test_rows),
        "shieldlm_test": sample_equal_classes(shieldlm[shieldlm["split"].eq("test")].reset_index(drop=True), TARGET, max_test_rows),
        "neuralchemy_core_test": sample_equal_classes(neuralchemy[neuralchemy["split"].eq("test")].reset_index(drop=True), TARGET, max_test_rows),
    }


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for (method, policy), group in rows.groupby(["method", "threshold_policy"]):
        binary = group[group["test_f1"].notna()]
        benign = group[group["test_source"].eq("notinject")]
        nvidia = group[group["test_source"].eq("nvidia_agentic_ipi")]
        summary_rows.append(
            {
                "method": method,
                "threshold_policy": policy,
                "mean_binary_f1": float(binary["test_f1"].mean()),
                "mean_binary_recall": float(binary["test_recall"].mean()),
                "notinject_fpr": float(benign["test_false_positive_rate"].iloc[0]) if len(benign) else np.nan,
                "nvidia_recall": float(1 - nvidia["test_false_negative_rate"].iloc[0]) if len(nvidia) else np.nan,
                "bipia_f1": value_for(group, "bipia_test", "test_f1"),
                "llmail_f1": value_for(group, "llmail_binary", "test_f1"),
                "promptshield_f1": value_for(group, "promptshield_test", "test_f1"),
                "shieldlm_f1": value_for(group, "shieldlm_test", "test_f1"),
                "neuralchemy_f1": value_for(group, "neuralchemy_core_test", "test_f1"),
            }
        )
    return pd.DataFrame(summary_rows).sort_values(["threshold_policy", "mean_binary_f1"], ascending=[True, False])


def value_for(group: pd.DataFrame, source: str, column: str) -> float:
    values = group.loc[group["test_source"].eq(source), column]
    return float(values.iloc[0]) if len(values) else np.nan


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate published prompt-injection detectors.")
    parser.add_argument("--max-test-rows", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-fpr", type=float, default=0.01)
    parser.add_argument("--models", nargs="*", default=[spec["method"] for spec in MODEL_SPECS])
    args = parser.parse_args()

    device = choose_device()
    print(f"Using device: {device}")
    sets = test_sets(args.max_test_rows)
    selected = [spec for spec in MODEL_SPECS if spec["method"] in set(args.models)]
    rows = []
    audit = []
    for spec in selected:
        start = time.time()
        print(f"Loading {spec['method']} ({spec['model_id']})")
        try:
            config = AutoConfig.from_pretrained(spec["model_id"])
            tokenizer = AutoTokenizer.from_pretrained(spec["model_id"])
            model = AutoModelForSequenceClassification.from_pretrained(spec["model_id"])
            model.to(device)
            pos_idx = positive_index(config, spec["positive_labels"])
            calibration_scores = batched_scores(
                model,
                tokenizer,
                sets["notinject_calibration"]["text"].fillna("").astype(str).tolist(),
                pos_idx,
                device,
                args.batch_size,
                args.max_length,
            )
            calibrated_threshold = threshold_from_benign(calibration_scores, args.max_fpr)
            thresholds = {"default_0.50": 0.5, f"notinject_calib_{args.max_fpr:.2f}": calibrated_threshold}
            for source, frame in sets.items():
                if source == "notinject_calibration":
                    continue
                print(f"  scoring {source}: {len(frame)} rows")
                scores = batched_scores(
                    model,
                    tokenizer,
                    frame["text"].fillna("").astype(str).tolist(),
                    pos_idx,
                    device,
                    args.batch_size,
                    args.max_length,
                )
                for policy, threshold in thresholds.items():
                    row = {
                        "method": spec["method"],
                        "model_id": spec["model_id"],
                        "threshold_policy": policy,
                        "test_source": source,
                        "n_test": len(frame),
                        "test_positive_rate": float(frame[TARGET].mean()),
                        "predicted_positive_rate": float((scores >= threshold).mean()),
                    }
                    if frame[TARGET].nunique() == 2:
                        metrics = metrics_binary(frame[TARGET], scores, threshold)
                        row.update({f"test_{key}": value for key, value in metrics.items()})
                    else:
                        label = int(frame[TARGET].iloc[0]) if len(frame) else 0
                        if label == 0:
                            row["test_false_positive_rate"] = float((scores >= threshold).mean())
                        else:
                            row["test_false_negative_rate"] = float((scores < threshold).mean())
                            row["test_recall"] = float((scores >= threshold).mean())
                    rows.append(row)
            audit.append({**spec, "status": "ok", "seconds": round(time.time() - start, 1), "positive_index": pos_idx, "labels": config.id2label})
            del model
            if device.type == "mps":
                torch.mps.empty_cache()
        except Exception as exc:
            audit.append({**spec, "status": "failed", "error": repr(exc)})
            print(f"FAILED {spec['method']}: {exc}")

    for spec in INACCESSIBLE_MODEL_SPECS:
        audit.append({**spec, "status": "not_run", "error": spec["reason"]})

    results = pd.DataFrame(rows)
    output = RESULTS_DIR / "published_detectors" / "published_detector_results.csv"
    summary_output = RESULTS_DIR / "published_detectors" / "published_detector_summary.csv"
    audit_output = RESULTS_DIR / "published_detectors" / "published_detector_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    summary = summarize(results) if len(results) else pd.DataFrame()
    summary.to_csv(summary_output, index=False)
    audit_output.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Saved: {output}")
    print(f"Saved: {summary_output}")
    print(f"Saved: {audit_output}")


if __name__ == "__main__":
    main()
