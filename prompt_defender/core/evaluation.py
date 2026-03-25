from __future__ import annotations

import re
from typing import Any

import torch


CANONICAL_LABELS = {
    "safe": "Safe",
    "unsafe": "Unsafe",
    "controversial": "Controversial",
}
UNPARSED_LABEL = "Unparsed"
SAFETY_PATTERN = re.compile(
    r'(?i)(?:^|[\{\[\(,\n])\s*"?safety"?\s*[:=]\s*"?'
    r"(safe|unsafe|controversial)\b"
)
BARE_SAFETY_PATTERN = re.compile(r'(?i)^\s*"?\s*(safe|unsafe|controversial)\b')
RAW_PREDICTION_ORDER = ["Safe", "Controversial", "Unsafe", UNPARSED_LABEL]


def detect_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend and mps_backend.is_available():
        return "mps"
    return "cpu"


def canonicalize_label(label: Any) -> str:
    if label is None:
        return UNPARSED_LABEL

    normalized = str(label).strip().lower()
    if not normalized:
        return UNPARSED_LABEL
    return CANONICAL_LABELS.get(normalized, str(label).strip())


def normalize_generated_safety_text(text: str) -> str:
    normalized = (text or "").strip()
    if not normalized:
        return ""
    if SAFETY_PATTERN.search(normalized):
        return normalized
    if BARE_SAFETY_PATTERN.search(normalized):
        return f"Safety: {normalized}"
    return normalized


def parse_safety_label(text: str) -> str:
    match = SAFETY_PATTERN.search(normalize_generated_safety_text(text))
    if not match:
        return UNPARSED_LABEL
    return canonicalize_label(match.group(1))
