import json
import subprocess
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from .config import PROCESSED_DIR, RAW_DIR, REPO_ID, SEED
from .data import load_clean_dataset

BIPIA_REPO = "https://github.com/microsoft/BIPIA.git"
BIPIA_DIR = RAW_DIR / "BIPIA"
LLMAIL_BINARY_PATH = PROCESSED_DIR / "llmail_binary_attack_benign.parquet"
BIPIA_BINARY_PATH = PROCESSED_DIR / "bipia_binary_generated.parquet"
NOTINJECT_PATH = PROCESSED_DIR / "notinject_benign.parquet"
NVIDIA_AGENTIC_IPI_PATH = PROCESSED_DIR / "nvidia_agentic_ipi.parquet"
PROMPTSHIELD_PATH = PROCESSED_DIR / "promptshield_binary.parquet"
SHIELDLM_PATH = PROCESSED_DIR / "shieldlm_prompt_injection.parquet"
NEURALCHEMY_CORE_PATH = PROCESSED_DIR / "neuralchemy_prompt_injection_core.parquet"


def ensure_bipia_repo() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if BIPIA_DIR.exists():
        subprocess.run(["git", "-C", str(BIPIA_DIR), "pull", "--ff-only"], check=False)
    else:
        subprocess.run(["git", "clone", "--depth", "1", BIPIA_REPO, str(BIPIA_DIR)], check=True)
    return BIPIA_DIR


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_attack_templates(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    flat = {}
    for attack_name, prompts in data.items():
        for idx, prompt in enumerate(prompts):
            flat[f"{attack_name}-{idx}"] = prompt
    return flat


def insert_attack(context: str, attack: str, position: str) -> str:
    context = str(context)
    if position == "start":
        return f"{attack}\n{context}"
    if position == "end":
        return f"{context}\n{attack}"
    raise ValueError(f"Unsupported attack insertion position: {position}")


def context_to_text(question: str, context: str, domain: str) -> str:
    return f"Task domain: {domain}\nUser request: {question}\nExternal content:\n{context}"


def build_bipia_split(repo_dir: Path, split: str, positions: tuple[str, ...]) -> pd.DataFrame:
    benchmark_dir = repo_dir / "benchmark"
    text_attacks = read_attack_templates(benchmark_dir / f"text_attack_{split}.json")
    code_attacks = read_attack_templates(benchmark_dir / f"code_attack_{split}.json")
    task_specs = [
        ("email", benchmark_dir / "email" / f"{split}.jsonl", text_attacks),
        ("table", benchmark_dir / "table" / f"{split}.jsonl", text_attacks),
        ("code", benchmark_dir / "code" / f"{split}.jsonl", code_attacks),
    ]
    rows = []
    for domain, context_path, attacks in task_specs:
        for idx, sample in enumerate(read_jsonl(context_path)):
            question = sample.get("question") or "Fix the code using the external context."
            context = sample.get("context")
            if isinstance(context, list):
                context = "\n".join(context)
            if context is None:
                context = "\n".join(sample.get("code", []))
            rows.append(
                {
                    "source_dataset": "BIPIA",
                    "split": split,
                    "domain": domain,
                    "text": context_to_text(question, context, domain),
                    "user_intent": question,
                    "label_attack": 0,
                    "attack_name": "none",
                    "position": "clean",
                }
            )
            for attack_name, attack in attacks.items():
                for position in positions:
                    poisoned = insert_attack(context, attack, position)
                    rows.append(
                        {
                            "source_dataset": "BIPIA",
                            "split": split,
                            "domain": domain,
                            "text": context_to_text(question, poisoned, domain),
                            "user_intent": question,
                            "label_attack": 1,
                            "attack_name": attack_name,
                            "position": position,
                        }
                    )
    return pd.DataFrame(rows)


def prepare_bipia_binary(force: bool = False, positions: tuple[str, ...] = ("start", "end")) -> Path:
    if BIPIA_BINARY_PATH.exists() and not force:
        return BIPIA_BINARY_PATH
    repo_dir = ensure_bipia_repo()
    frame = pd.concat(
        [
            build_bipia_split(repo_dir, "train", positions),
            build_bipia_split(repo_dir, "test", positions),
        ],
        ignore_index=True,
    )
    BIPIA_BINARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(BIPIA_BINARY_PATH, index=False)
    return BIPIA_BINARY_PATH


def load_bipia_binary(force: bool = False) -> pd.DataFrame:
    return pd.read_parquet(prepare_bipia_binary(force=force))


def load_llmail_false_positive_emails() -> pd.DataFrame:
    path = hf_hub_download(REPO_ID, "data/emails_for_fp_tests.json", repo_type="dataset")
    emails = json.loads(Path(path).read_text(encoding="utf-8"))
    return pd.DataFrame(
        {
            "source_dataset": "LLMail_FP",
            "split": "fp_benign",
            "domain": "email",
            "text": emails,
            "user_intent": "Benign email content used for false-positive testing.",
            "label_attack": 0,
        }
    )


def prepare_llmail_binary(force: bool = False, max_attack_rows: int | None = 40_000) -> Path:
    if LLMAIL_BINARY_PATH.exists() and not force:
        return LLMAIL_BINARY_PATH
    llmail = load_clean_dataset()
    attack = llmail[["text", "phase", "scenario_key", "scenario_group", "attack_chain_stage"]].copy()
    if max_attack_rows is not None and len(attack) > max_attack_rows:
        attack = attack.sample(max_attack_rows, random_state=SEED).reset_index(drop=True)
    attack["source_dataset"] = "LLMail"
    attack["split"] = attack["phase"]
    attack["domain"] = "email"
    attack["user_intent"] = "Summarize recent emails without following untrusted email instructions."
    attack["label_attack"] = 1
    benign = load_llmail_false_positive_emails()
    frame = pd.concat([attack, benign], ignore_index=True, sort=False)
    LLMAIL_BINARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(LLMAIL_BINARY_PATH, index=False)
    return LLMAIL_BINARY_PATH


def load_llmail_binary(force: bool = False, max_attack_rows: int | None = 40_000) -> pd.DataFrame:
    return pd.read_parquet(prepare_llmail_binary(force=force, max_attack_rows=max_attack_rows))


def prepare_notinject(force: bool = False) -> Path:
    if NOTINJECT_PATH.exists() and not force:
        return NOTINJECT_PATH
    dataset = load_dataset("leolee99/NotInject")
    rows = []
    for split_name, split in dataset.items():
        frame = split.to_pandas()
        frame["source_dataset"] = "NotInject"
        frame["split"] = split_name
        frame["domain"] = "hard_negative"
        frame["text"] = frame["prompt"].fillna("").astype(str)
        frame["user_intent"] = "Benign prompt containing suspicious trigger words."
        frame["label_attack"] = 0
        rows.append(frame[["source_dataset", "split", "domain", "text", "user_intent", "label_attack", "category", "word_list"]])
    result = pd.concat(rows, ignore_index=True)
    NOTINJECT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(NOTINJECT_PATH, index=False)
    return NOTINJECT_PATH


def load_notinject(force: bool = False) -> pd.DataFrame:
    return pd.read_parquet(prepare_notinject(force=force))


def compact_jsonish(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def prepare_nvidia_agentic_ipi(force: bool = False) -> Path:
    if NVIDIA_AGENTIC_IPI_PATH.exists() and not force:
        return NVIDIA_AGENTIC_IPI_PATH
    dataset = load_dataset("nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1", split="train")
    frame = dataset.to_pandas()
    frame["source_dataset"] = "NVIDIA_Agentic_IPI"
    frame["split"] = "train"
    frame["user_intent"] = frame["responses_create_params"].map(compact_jsonish)
    frame["text"] = frame.apply(
        lambda row: "\n".join(
            [
                f"Domain: {row.get('domain', '')}",
                f"Attack category: {row.get('attack_category', '')}",
                f"Target tool: {row.get('target_tool', '')}",
                f"Injection vector: {row.get('injection_vector', '')}",
                f"Agent task and messages:\n{compact_jsonish(row.get('responses_create_params'))}",
                f"Tool-returned environment:\n{compact_jsonish(row.get('environment'))}",
                f"Hidden injection:\n{compact_jsonish(row.get('injection'))}",
            ]
        ),
        axis=1,
    )
    frame["label_attack"] = 1
    frame["domain"] = frame["domain"].fillna("agentic_tool_use").astype(str)
    keep = [
        "source_dataset",
        "split",
        "domain",
        "text",
        "user_intent",
        "label_attack",
        "attack_category",
        "target_tool",
        "injection_vector",
        "id",
    ]
    NVIDIA_AGENTIC_IPI_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame[keep].to_parquet(NVIDIA_AGENTIC_IPI_PATH, index=False)
    return NVIDIA_AGENTIC_IPI_PATH


def load_nvidia_agentic_ipi(force: bool = False) -> pd.DataFrame:
    return pd.read_parquet(prepare_nvidia_agentic_ipi(force=force))


def prepare_promptshield(force: bool = False) -> Path:
    if PROMPTSHIELD_PATH.exists() and not force:
        return PROMPTSHIELD_PATH
    dataset = load_dataset("hendzh/PromptShield")
    rows = []
    for split_name, split in dataset.items():
        frame = split.to_pandas()
        frame["source_dataset"] = "PromptShield"
        frame["split"] = split_name
        frame["domain"] = "prompt_injection_detection"
        frame["text"] = frame["prompt"].fillna("").astype(str)
        frame["user_intent"] = "PromptShield binary prompt-injection benchmark row."
        frame["label_attack"] = frame["label"].astype(int)
        rows.append(frame[["source_dataset", "split", "domain", "text", "user_intent", "label_attack"]])
    result = pd.concat(rows, ignore_index=True)
    PROMPTSHIELD_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(PROMPTSHIELD_PATH, index=False)
    return PROMPTSHIELD_PATH


def load_promptshield(force: bool = False) -> pd.DataFrame:
    return pd.read_parquet(prepare_promptshield(force=force))


def prepare_shieldlm(force: bool = False) -> Path:
    if SHIELDLM_PATH.exists() and not force:
        return SHIELDLM_PATH
    dataset = load_dataset("Abdennebi/shieldlm-prompt-injection")
    rows = []
    for split_name, split in dataset.items():
        frame = split.to_pandas()
        frame["source_dataset"] = "ShieldLM"
        frame["split"] = split_name
        frame["domain"] = frame["label_category"].fillna("unknown").astype(str)
        frame["text"] = frame["text"].fillna("").astype(str)
        frame["user_intent"] = "ShieldLM unified prompt-injection detection row."
        frame["label_attack"] = frame["label_binary"].astype(int)
        rows.append(
            frame[
                [
                    "source_dataset",
                    "split",
                    "domain",
                    "text",
                    "user_intent",
                    "label_attack",
                    "label_category",
                    "label_intent",
                    "source",
                    "language",
                ]
            ]
        )
    result = pd.concat(rows, ignore_index=True)
    SHIELDLM_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(SHIELDLM_PATH, index=False)
    return SHIELDLM_PATH


def load_shieldlm(force: bool = False) -> pd.DataFrame:
    return pd.read_parquet(prepare_shieldlm(force=force))


def prepare_neuralchemy_core(force: bool = False) -> Path:
    if NEURALCHEMY_CORE_PATH.exists() and not force:
        return NEURALCHEMY_CORE_PATH
    dataset = load_dataset("neuralchemy/Prompt-injection-dataset", "core")
    rows = []
    for split_name, split in dataset.items():
        frame = split.to_pandas()
        frame["source_dataset"] = "Neuralchemy_PI_Core"
        frame["split"] = split_name
        frame["domain"] = frame["category"].fillna("unknown").astype(str)
        frame["text"] = frame["text"].fillna("").astype(str)
        frame["user_intent"] = "Neuralchemy prompt-injection and jailbreak detector benchmark row."
        frame["label_attack"] = frame["label"].astype(int)
        rows.append(
            frame[
                [
                    "source_dataset",
                    "split",
                    "domain",
                    "text",
                    "user_intent",
                    "label_attack",
                    "category",
                    "source",
                    "severity",
                    "group_id",
                    "augmented",
                    "tags",
                ]
            ]
        )
    result = pd.concat(rows, ignore_index=True)
    NEURALCHEMY_CORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(NEURALCHEMY_CORE_PATH, index=False)
    return NEURALCHEMY_CORE_PATH


def load_neuralchemy_core(force: bool = False) -> pd.DataFrame:
    return pd.read_parquet(prepare_neuralchemy_core(force=force))


def sample_balanced(frame: pd.DataFrame, label_col: str = "label_attack", max_rows: int | None = None) -> pd.DataFrame:
    if max_rows is None or len(frame) <= max_rows:
        return frame.sample(frac=1, random_state=SEED).reset_index(drop=True)
    pieces = []
    for _, group in frame.groupby(label_col):
        n = max(1, int(max_rows * len(group) / len(frame)))
        pieces.append(group.sample(min(len(group), n), random_state=SEED))
    return pd.concat(pieces, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)


def sample_equal_classes(frame: pd.DataFrame, label_col: str = "label_attack", max_rows: int | None = None) -> pd.DataFrame:
    groups = [group for _, group in frame.groupby(label_col)]
    if len(groups) < 2:
        return frame.sample(frac=1, random_state=SEED).reset_index(drop=True)
    per_class = min(len(group) for group in groups)
    if max_rows is not None:
        per_class = min(per_class, max(1, max_rows // len(groups)))
    pieces = [group.sample(per_class, random_state=SEED) for group in groups]
    return pd.concat(pieces, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
