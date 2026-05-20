#!/usr/bin/env python3
"""
Merge and clean annotated/test datasets.

For each record keeps only:
  - dataset     : source dataset name
  - instruction : user message (first user turn)
  - label       : safety label (normalised to lowercase)
  - category    : query category / unsafe type

Outputs:
  datasets/train_merged_clean.jsonl
  datasets/val_merged_clean.jsonl
  datasets/test_merged_clean.jsonl
"""

from __future__ import annotations

import json
import re
from pathlib import Path

DATASETS_DIR = Path("datasets")
TRAIN_OUT = DATASETS_DIR / "train_merged_clean.jsonl"
VAL_OUT   = DATASETS_DIR / "val_merged_clean.jsonl"
TEST_OUT  = DATASETS_DIR / "test_merged_clean.jsonl"


def dataset_name(path: Path) -> str:
    name = path.stem
    name = re.sub(r"^(train|val)_", "", name)
    name = re.sub(r"_annotated$", "", name)
    return name


# ── annotated JSONL (train / val) ──────────────────────────────────────────

def clean_annotated_record(raw: dict, source: str) -> dict | None:
    messages = raw.get("messages", [])
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        return None
    return {
        "dataset": source,
        "instruction": user_msgs[0]["content"],
        "label": raw.get("query_safety", ""),
        "category": raw.get("query_category", "None"),
    }


def process_annotated_files(paths: list[Path], out_path: Path) -> None:
    total = skipped = 0
    with out_path.open("w", encoding="utf-8") as f_out:
        for path in paths:
            source = dataset_name(path)
            with path.open(encoding="utf-8") as f_in:
                for line in f_in:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1
                        continue
                    rec = clean_annotated_record(raw, source)
                    if rec is None:
                        skipped += 1
                        continue
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total += 1
    print(f"{out_path}: {total} records written, {skipped} skipped")


# ── test JSON (list of records) ─────────────────────────────────────────────

def clean_test_record(raw: dict, fallback_source: str) -> dict | None:
    messages = raw.get("message", [])
    user_msgs = [m for m in messages if m.get("role") == "user"]
    if not user_msgs:
        return None
    label = str(raw.get("label", "")).strip().lower()   # 'Safe'/'Unsafe' → 'safe'/'unsafe'
    source = raw.get("source") or fallback_source
    # Normalise source: strip path prefixes like 'test-expand/foo.txt' → 'test-expand'
    if "/" in source:
        source = source.split("/")[0]
    return {
        "dataset": source,
        "instruction": user_msgs[0]["content"],
        "label": label,
        "category": raw.get("unsafe_type") or "None",
    }


def process_test_files(paths: list[Path], out_path: Path) -> None:
    total = skipped = 0
    with out_path.open("w", encoding="utf-8") as f_out:
        for path in paths:
            fallback = re.sub(r"_test$", "", path.stem)
            with path.open(encoding="utf-8") as f_in:
                data = json.load(f_in)
            for raw in data:
                rec = clean_test_record(raw, fallback)
                if rec is None:
                    skipped += 1
                    continue
                f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += 1
    print(f"{out_path}: {total} records written, {skipped} skipped")


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    all_annotated = sorted(DATASETS_DIR.glob("*_annotated.jsonl"))
    train_files = [p for p in all_annotated if p.name.startswith("train_")]
    val_files   = [p for p in all_annotated if p.name.startswith("val_")]

    # Individual test files first, then the merged test.json last (for dedup)
    individual_tests = sorted(DATASETS_DIR.glob("*_test.json"))
    test_files = individual_tests + [DATASETS_DIR / "test.json"]

    print("Train files:")
    for p in train_files:
        print(f"  {p.name}")
    print("Val files:")
    for p in val_files:
        print(f"  {p.name}")
    print("Test files:")
    for p in test_files:
        print(f"  {p.name}")
    print()

    process_annotated_files(train_files, TRAIN_OUT)
    process_annotated_files(val_files, VAL_OUT)
    process_test_files(test_files, TEST_OUT)


if __name__ == "__main__":
    main()
