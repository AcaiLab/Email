import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


SECURITY_PATTERN_GROUPS = {
    "instruction_override": [
        r"ignore (all )?(previous|prior|above) instructions",
        r"disregard (all )?(previous|prior|above) instructions",
        r"forget (all )?(previous|prior|above) instructions",
        r"override (the )?(system|developer|user) instructions",
    ],
    "role_or_authority": [
        r"you are now",
        r"act as",
        r"pretend to be",
        r"developer mode",
        r"system prompt",
        r"highest priority",
        r"administrator",
    ],
    "tool_action": [
        r"api_call",
        r"tool call",
        r"call the tool",
        r"send (an )?email",
        r"send_email",
        r"execute",
        r"function call",
        r"invoke",
    ],
    "exfiltration": [
        r"exfiltrat",
        r"secret",
        r"confidential",
        r"private",
        r"password",
        r"token",
        r"credential",
        r"leak",
        r"send.*secret",
    ],
    "target_artifacts": [
        r"\brecipient\s*[:=]",
        r"\bdestination\s*[:=]",
        r"\bbody\s*[:=]",
        r"\bmessage\s*[:=]",
        r"\bto\s*:\s*[A-Za-z0-9._%+-]+\s*@",
        r"\bsubject\s*[:=]",
    ],
    "obfuscation": [
        r"base64",
        r"rot13",
        r"encoded",
        r"decode",
        r"unicode",
        r"hidden",
        r"invisible",
        r"white text",
    ],
    "delimiter_or_formatting": [
        r"```",
        r"<system>",
        r"</system>",
        r"###",
        r"---",
        r"BEGIN",
        r"END",
        r"xml",
        r"json",
    ],
    "secrecy_or_stealth": [
        r"do not tell",
        r"do not mention",
        r"without telling",
        r"silently",
        r"secretly",
        r"hide this",
        r"do not reveal",
    ],
    "urgency_social": [
        r"urgent",
        r"immediately",
        r"important",
        r"asap",
        r"required",
        r"must",
        r"priority",
    ],
}


def make_vectorizer(max_features: int = 30_000) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.98,
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,
        dtype=np.float32,
    )


def count_regex_group(text_series: pd.Series, patterns: list[str]) -> pd.Series:
    combined = "|".join(f"(?:{pattern})" for pattern in patterns)
    return text_series.str.count(combined, flags=re.IGNORECASE).fillna(0).astype(np.float32)


def build_heuristic_features(text_series: pd.Series) -> pd.DataFrame:
    series = text_series.fillna("").astype(str)
    features = pd.DataFrame(index=series.index)
    for group_name, patterns in SECURITY_PATTERN_GROUPS.items():
        features[f"group_{group_name}_count"] = count_regex_group(series, patterns)
        features[f"group_{group_name}_present"] = (features[f"group_{group_name}_count"] > 0).astype(np.float32)

    features["char_len"] = series.str.len().astype(np.float32)
    features["word_count"] = series.str.split().map(len).astype(np.float32)
    features["line_count"] = (series.str.count(r"\n") + 1).astype(np.float32)
    features["exclamation_count"] = series.str.count(r"!").astype(np.float32)
    features["question_count"] = series.str.count(r"\?").astype(np.float32)
    features["quote_count"] = series.str.count(r"['\"]").astype(np.float32)
    features["url_count"] = series.str.count(r"https?://|www\.").astype(np.float32)
    features["email_address_count"] = series.str.count(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}").astype(np.float32)
    features["digit_count"] = series.str.count(r"\d").astype(np.float32)
    features["uppercase_count"] = series.map(lambda value: sum(1 for char in value if char.isupper())).astype(np.float32)
    features["uppercase_ratio"] = features["uppercase_count"] / features["char_len"].clip(lower=1)
    features["digit_ratio"] = features["digit_count"] / features["char_len"].clip(lower=1)
    return features.replace([np.inf, -np.inf], 0).fillna(0)
