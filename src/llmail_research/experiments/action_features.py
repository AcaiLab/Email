import re

import numpy as np
import pandas as pd

from llmail_research.config import USER_QUERY


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").lower()).strip()


def count_regex(series: pd.Series, pattern: str) -> pd.Series:
    return series.fillna("").astype(str).str.contains(pattern, flags=re.IGNORECASE, regex=True).astype(float)


def has_regex(series: pd.Series, pattern: str) -> pd.Series:
    return (count_regex(series, pattern) > 0).astype(float)


def extract_email_sets(series: pd.Series) -> pd.Series:
    pattern = r"[A-Za-z0-9._%+-]+\s*@\s*[A-Za-z0-9.-]+\s*\.\s*[A-Za-z]{2,}"
    return series.fillna("").astype(str).map(
        lambda value: {
            re.sub(r"\s+", "", match).lower()
            for match in re.findall(pattern, value, flags=re.IGNORECASE)
        }
    )


def normalized_tokens(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).map(
        lambda value: set(re.findall(r"[a-z0-9]{3,}", normalize(value)))
    )


def build_action_verifier_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare untrusted email text, user intent, and the proposed output."""
    text = frame["text"].fillna("").astype(str)
    output = frame.get("output_text", frame.get("output", ""))
    if not isinstance(output, pd.Series):
        output = pd.Series([output] * len(frame), index=frame.index)
    output = output.fillna("").astype(str)
    user = frame.get("user_intent", USER_QUERY)
    if not isinstance(user, pd.Series):
        user = pd.Series([user] * len(frame), index=frame.index)
    user = user.fillna("").astype(str)

    combined = text + "\n" + output
    email_pattern = r"[A-Za-z0-9._%+-]+\s*@\s*[A-Za-z0-9.-]+\s*\.\s*[A-Za-z]{2,}"
    send_pattern = r"\b(?:send|sent|sending|forward|email|mail|api[_ -]?call|tool call|execute_userquery|send_email)\b"
    body_pattern = r"\b(?:body|content|message|payload)\s*[:=]"
    boundary_pattern = r"<<|>>|<\|user\|>|<\|assistant\|>|end_context|start_prompt|new_user_session|execute_userquery|```"

    text_emails = extract_email_sets(text)
    output_emails = extract_email_sets(output)
    user_emails = extract_email_sets(user)
    text_tokens = normalized_tokens(text)
    output_tokens = normalized_tokens(output)
    user_tokens = normalized_tokens(user)

    features = pd.DataFrame(index=frame.index)
    features["text_has_send_action"] = has_regex(text, send_pattern)
    features["output_has_send_action"] = has_regex(output, send_pattern)
    features["combined_has_send_action"] = has_regex(combined, send_pattern)
    features["text_email_count"] = count_regex(text, email_pattern)
    features["output_email_count"] = count_regex(output, email_pattern)
    features["text_has_body_language"] = has_regex(text, body_pattern)
    features["output_has_body_language"] = has_regex(output, body_pattern)
    features["text_boundary_markers"] = count_regex(text, boundary_pattern)
    features["output_boundary_markers"] = count_regex(output, boundary_pattern)
    features["user_mentions_send"] = has_regex(user, send_pattern)
    features["recipient_from_untrusted_not_user"] = (
        pd.Series(
            [
                bool((output_set & text_set) - user_set)
                for output_set, text_set, user_set in zip(output_emails, text_emails, user_emails)
            ],
            index=frame.index,
        )
    ).astype(float)
    features["content_from_untrusted_not_user"] = (
        pd.Series(
            [
                len((output_set & text_set) - user_set) >= 2
                for output_set, text_set, user_set in zip(output_tokens, text_tokens, user_tokens)
            ],
            index=frame.index,
        )
        & features["output_has_send_action"].eq(1)
    ).astype(float)
    features["unsafe_action_alignment"] = (
        0.30 * features["output_has_send_action"]
        + 0.25 * features["recipient_from_untrusted_not_user"]
        + 0.25 * features["content_from_untrusted_not_user"]
        + 0.20 * (features["output_boundary_markers"] > 0).astype(float)
    )
    features["output_to_text_len_ratio"] = output.str.len().astype(float) / text.str.len().clip(lower=1).astype(float)
    return features.replace([np.inf, -np.inf], 0).fillna(0)
