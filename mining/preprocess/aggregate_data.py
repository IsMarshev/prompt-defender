#!/usr/bin/env python3
"""
Merge dataset JSONL files into two files: train and val.

Split is detected from file name:
  - if name contains val/valid/validation/dev -> val
  - else if name contains train -> train
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

VAL_TOKENS = {"val", "valid", "validation", "dev"}


def detect_split(path: Path) -> str | None:
    tokens = [t for t in re.split(r"[^a-z0-9]+", path.stem.lower()) if t]
    if any(token in VAL_TOKENS for token in tokens):
        return "val"
    if "train" in tokens:
        return "train"
    return None


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_num} invalid JSON: {exc.msg}") from exc


def merge_jsonl(files: list[Path], output_path: Path) -> tuple[int, list[tuple[Path, int]]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    per_file: list[tuple[Path, int]] = []

    with output_path.open("w", encoding="utf-8") as out:
        for path in files:
            written = 0
            for obj in iter_jsonl(path):
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                written += 1
                total += 1
            per_file.append((path, written))

    return total, per_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate datasets/*.jsonl into separate train and val files."
    )
    parser.add_argument("--datasets-dir", default="datasets", help="Directory with source JSONL files")
    parser.add_argument("--pattern", default="*.jsonl", help="Glob pattern inside datasets-dir")
    parser.add_argument(
        "--train-output",
        default="datasets/train_merged.jsonl",
        help="Output JSONL for train split",
    )
    parser.add_argument(
        "--val-output",
        default="datasets/val_merged.jsonl",
        help="Output JSONL for val split",
    )
    args = parser.parse_args()

    datasets_dir = Path(args.datasets_dir)
    train_output = Path(args.train_output)
    val_output = Path(args.val_output)

    if not datasets_dir.exists() or not datasets_dir.is_dir():
        print(f"Datasets directory not found: {datasets_dir}", file=sys.stderr)
        return 1

    files = sorted(p for p in datasets_dir.glob(args.pattern) if p.is_file())
    if not files:
        print(f"No files found: {datasets_dir}/{args.pattern}", file=sys.stderr)
        return 1

    excluded = {train_output.resolve(), val_output.resolve()}
    train_files: list[Path] = []
    val_files: list[Path] = []
    skipped: list[Path] = []

    for path in files:
        if path.resolve() in excluded:
            continue
        split = detect_split(path)
        if split == "train":
            train_files.append(path)
        elif split == "val":
            val_files.append(path)
        else:
            skipped.append(path)

    if not train_files and not val_files:
        print("No train/val files matched by name tokens.", file=sys.stderr)
        return 1

    train_total, train_stats = merge_jsonl(train_files, train_output)
    val_total, val_stats = merge_jsonl(val_files, val_output)

    print("\nTrain sources:")
    for path, cnt in train_stats:
        print(f"  {path} -> {cnt}")

    print("\nVal sources:")
    for path, cnt in val_stats:
        print(f"  {path} -> {cnt}")

    if skipped:
        print("\nSkipped (unknown split):")
        for path in skipped:
            print(f"  {path}")

    print(f"\nTrain total: {train_total} -> {train_output}")
    print(f"Val total: {val_total} -> {val_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
