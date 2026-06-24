#!/usr/bin/env python3
"""
Obfuscate instructions into Russian using OpenAI SDK.

Default input shape is project datasets JSONL where instruction lives in:
  messages.0.content

Example:
    python3 obfuscate_instructions_openai.py \
      --input datasets/train_merged.jsonl \
      --output datasets/train_merged_ru_obf.jsonl \
      --source-field messages.0.content \
      --target-field messages.0.content \
      --model gpt-4.1-mini
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from tqdm import tqdm

from openai import OpenAI


SYSTEM_PROMPT = (
    "Перефразируй инструкцию по-русски, сохраняя смысл, но делая формулировку менее прямой — "
    "используй эвфемизмы, меняй структуру предложений. "
    "Ничего не добавляй от себя. Верни только результат."
)


def parse_path(path: str) -> list[str | int]:
    result: list[str | int] = []
    for token in path.split("."):
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            result.append(int(token))
        else:
            result.append(token)
    if not result:
        raise ValueError(f"Invalid path: {path!r}")
    return result


def get_nested_value(obj: Any, path: list[str | int]) -> Any:
    cur = obj
    for key in path:
        if isinstance(key, int):
            if not isinstance(cur, list) or key >= len(cur):
                return None
            cur = cur[key]
        else:
            if not isinstance(cur, dict) or key not in cur:
                return None
            cur = cur[key]
    return cur


def set_nested_value(obj: Any, path: list[str | int], value: Any):
    cur = obj
    for idx, key in enumerate(path[:-1]):
        next_key = path[idx + 1]
        if isinstance(key, int):
            if not isinstance(cur, list):
                raise TypeError(f"Expected list on path segment {key}, got {type(cur).__name__}")
            if key >= len(cur):
                cur.extend([None] * (key - len(cur) + 1))
            if cur[key] is None:
                cur[key] = [] if isinstance(next_key, int) else {}
            cur = cur[key]
        else:
            if not isinstance(cur, dict):
                raise TypeError(f"Expected dict on path segment {key}, got {type(cur).__name__}")
            if key not in cur or cur[key] is None:
                cur[key] = [] if isinstance(next_key, int) else {}
            cur = cur[key]

    last_key = path[-1]
    if isinstance(last_key, int):
        if not isinstance(cur, list):
            raise TypeError(f"Expected list at final segment {last_key}, got {type(cur).__name__}")
        if last_key >= len(cur):
            cur.extend([None] * (last_key - len(cur) + 1))
        cur[last_key] = value
    else:
        if not isinstance(cur, dict):
            raise TypeError(f"Expected dict at final segment {last_key}, got {type(cur).__name__}")
        cur[last_key] = value


def obfuscate_instruction(
    client: OpenAI,
    text: str,
    model: str,
    max_retries: int,
    retry_delay: float,
) -> str:
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
            )
            content = (response.choices[0].message.content or "").strip()
            if content:
                return content
            raise RuntimeError("empty response")
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            sleep_seconds = retry_delay * attempt
            time.sleep(sleep_seconds)

    assert last_error is not None
    raise last_error


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Use OpenAI SDK to obfuscate instructions into Russian for JSONL records."
    )
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--source-field", default="messages.0.content", help="Path to source text")
    parser.add_argument("--target-field", default="messages.0.content", help="Path to write obfuscated text")
    parser.add_argument(
        "--save-original-field",
        default="original_instruction",
        help="Optional top-level field to keep original text; empty string disables",
    )
    parser.add_argument("--model", default="gpt-4.1-mini", help="OpenAI model name")
    parser.add_argument("--max-retries", type=int, default=5, help="Retries per record on API error")
    parser.add_argument("--retry-delay", type=float, default=1.5, help="Base delay in seconds")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N rows")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is not set", file=sys.stderr)
        return 1

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1

    source_path = parse_path(args.source_field)
    target_path = parse_path(args.target_field)
    client = OpenAI(base_url="https://bothub.chat/api/v2/openai/v1", api_key=api_key)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    converted = 0
    skipped = 0
    failed = 0

    with input_path.open("r", encoding="utf-8") as inp, output_path.open("w", encoding="utf-8") as out:
        lines = [l for l in inp if l.strip()]
        if args.limit is not None:
            lines = lines[: args.limit]

        for line_num, raw in enumerate(tqdm(lines), start=1):
            total += 1
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                failed += 1
                print(f"[line {line_num}] invalid json: {exc}", file=sys.stderr)
                continue

            source_text = get_nested_value(row, source_path)
            if not isinstance(source_text, str) or not source_text.strip():
                skipped += 1
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            try:
                obfuscated = obfuscate_instruction(
                    client=client,
                    text=source_text,
                    model=args.model,
                    max_retries=args.max_retries,
                    retry_delay=args.retry_delay,
                )
                if args.save_original_field:
                    row[args.save_original_field] = source_text
                set_nested_value(row, target_path, obfuscated)
                converted += 1
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
            except Exception as exc:
                failed += 1
                print(f"[line {line_num}] api error: {exc}", file=sys.stderr)

    print(
        f"Done: total={total} converted={converted} skipped={skipped} failed={failed} -> {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
