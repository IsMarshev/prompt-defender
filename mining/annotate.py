#!/usr/bin/env python3
"""
Annotate JSONL records with safety labels using two LLM judges.

Voting:
  unsafe + unsafe  →  unsafe
  safe   + safe    →  safe
  unsafe + safe    →  controversial

Usage:
    python annotate.py \
        --input  datasets/train_merged.jsonl \
        --output datasets/train_merged_annotations.jsonl \
        --model-a gpt-4o-mini \
        --model-b gemini-2.0-flash
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


CATEGORIES = (
    "Violent",
    "Non-violent Illegal Acts",
    "Sexual Content or Sexual Acts",
    "PII",
    "Suicide & Self-Harm",
    "Unethical Acts",
    "Politically Sensitive Topics",
    "Copyright Violation",
    "Jailbreak",
)

SYSTEM_PROMPT = (
    "Classify the user message as safe or unsafe.\n\n"
    "Reply with exactly two lines:\n"
    "Safety: safe|unsafe\n"
    "Category: <one of the listed categories, or None>\n\n"
    f"Categories: {', '.join(CATEGORIES)}"
)


def _parse_response(text: str) -> tuple[str, str]:
    safety = "unsafe"
    category = "None"
    for line in text.strip().splitlines():
        key, _, val = line.partition(":")
        val = val.strip()
        if key.strip().lower() == "safety" and val.lower() == "safe":
            safety = "safe"
        elif key.strip().lower() == "category" and val.lower() not in ("none", ""):
            category = val
    return safety, category


def _aggregate(vote_a: str, vote_b: str) -> str:
    if vote_a == "unsafe" and vote_b == "unsafe":
        return "unsafe"
    if vote_a == "safe" and vote_b == "safe":
        return "safe"
    if {vote_a, vote_b} == {"unsafe", "safe"}:
        return "controversial"
    # edge: one is "controversial"
    if "unsafe" in (vote_a, vote_b):
        return "unsafe"
    return "controversial"


def _call(client: OpenAI, model: str, text: str, max_retries: int, retry_delay: float) -> str:
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                max_tokens=64,
                temperature=0,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)
    raise last_exc  # type: ignore[misc]


def annotate(
    client: OpenAI,
    model_a: str,
    model_b: str,
    text: str,
    max_retries: int,
    retry_delay: float,
) -> tuple[str, str]:
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_a = pool.submit(_call, client, model_a, text, max_retries, retry_delay)
        fut_b = pool.submit(_call, client, model_b, text, max_retries, retry_delay)
        resp_a, resp_b = fut_a.result(), fut_b.result()

    safety_a, cat_a = _parse_response(resp_a)
    safety_b, cat_b = _parse_response(resp_b)

    final_safety = _aggregate(safety_a, safety_b)
    # prefer the category from the unsafe voter
    final_category = cat_a if safety_a == "unsafe" else cat_b if safety_b == "unsafe" else "None"
    return final_safety, final_category


def _user_text(record: dict) -> str | None:
    msgs = record.get("messages", [])
    if msgs and msgs[0].get("role") == "user":
        return msgs[0].get("content", "").strip() or None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Two-LLM safety annotation with voting.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-a", default="gpt-4o-mini")
    parser.add_argument("--model-b", default="gemini-2.0-flash")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-delay", type=float, default=1.5)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is not set", file=sys.stderr)
        return 1

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1

    client = OpenAI(base_url="https://bothub.chat/api/v2/openai/v1", api_key=api_key)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = skipped = failed = 0

    with input_path.open("r", encoding="utf-8") as inp, output_path.open("w", encoding="utf-8") as out:
        lines = [l for l in inp if l.strip()]
        if args.limit:
            lines = lines[: args.limit]

        for idx, raw in enumerate(tqdm(lines)):
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"[{idx}] json error: {exc}", file=sys.stderr)
                failed += 1
                continue

            text = _user_text(record)
            if not text:
                skipped += 1
                continue

            try:
                safety, category = annotate(
                    client, args.model_a, args.model_b, text,
                    args.max_retries, args.retry_delay,
                )
                out.write(json.dumps({"index": idx, "query_safety": safety, "query_category": category}, ensure_ascii=False) + "\n")
                processed += 1
            except Exception as exc:
                print(f"[{idx}] api error: {exc}", file=sys.stderr)
                failed += 1

    print(f"Done: processed={processed} skipped={skipped} failed={failed} -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
