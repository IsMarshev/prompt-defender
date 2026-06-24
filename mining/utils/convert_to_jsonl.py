#!/usr/bin/env python3
"""
Convert data files (.csv / .parquet / .json) to JSONL.

Examples:
    python convert_to_jsonl.py --input data/toxicchat/toxic-chat_annotation_train.csv \
        --output data/toxicchat/toxic-chat_annotation_train.jsonl

    python convert_to_jsonl.py --input data/wildguard/wildguard_train.parquet \
        --output data/wildguard/wildguard_train.jsonl

    python convert_to_jsonl.py --input data/aegis/test.json \
        --output data/aegis/test.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import date, datetime, time
from pathlib import Path
from typing import Iterable


def _normalize_value(value):
    if value is None:
        return None

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, (date, datetime, time)):
        return value.isoformat()

    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()

    if isinstance(value, dict):
        return {str(k): _normalize_value(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_normalize_value(v) for v in value]

    if hasattr(value, "item"):
        try:
            return _normalize_value(value.item())
        except Exception:
            pass

    return str(value)


def _iter_csv(path: Path, delimiter: str, encoding: str) -> Iterable[dict]:
    with path.open("r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            yield row


def _iter_parquet(path: Path) -> Iterable[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Parquet support requires pyarrow. Install it with: pip install pyarrow"
        ) from exc

    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches():
        for row in batch.to_pylist():
            yield row


def _iter_json(path: Path, encoding: str) -> Iterable[object]:
    with path.open("r", encoding=encoding) as f:
        data = json.load(f)

    if isinstance(data, list):
        for row in data:
            yield row
        return

    yield data


def _write_jsonl(rows: Iterable[object], output_path: Path, limit: int | None) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with output_path.open("w", encoding="utf-8") as out:
        for row in rows:
            if limit is not None and count >= limit:
                break
            normalized = _normalize_value(row)
            out.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            count += 1

    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert .csv, .parquet or .json files to JSONL (one record per line)."
    )
    parser.add_argument("--input", required=True, help="Input .csv, .parquet or .json file")
    parser.add_argument("--output", required=True, help="Output .jsonl file")
    parser.add_argument("--delimiter", default=",", help="CSV delimiter (default: ',')")
    parser.add_argument("--encoding", default="utf-8", help="CSV file encoding")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: write only first N rows",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists() or not input_path.is_file():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        rows = _iter_csv(input_path, args.delimiter, args.encoding)
    elif suffix == ".parquet":
        try:
            rows = _iter_parquet(input_path)
        except RuntimeError as err:
            print(str(err), file=sys.stderr)
            return 1
    elif suffix == ".json":
        rows = _iter_json(input_path, args.encoding)
    else:
        print("Unsupported input type. Use .csv, .parquet or .json", file=sys.stderr)
        return 1

    converted = _write_jsonl(rows, output_path, args.limit)
    print(f"Converted {converted} rows: {input_path} -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
