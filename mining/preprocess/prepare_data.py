"""
Data preparation pipeline for Stream Qwen3Guard.

Converts public safety datasets into the unified JSONL format:
{
    "messages": [{"role": "user", "content": "..."}, ...],
    "query_safety": "safe|controversial|unsafe",
    "query_category": "<category>|None",
    "response_safety": "safe|controversial|unsafe",
    "response_category": "<category>|None",
    "boundary_token": null | int
}

Supported sources:
    beavertails, toxicchat, wildguard, aegis, custom (CSV/JSONL)

Usage:
    python prepare_data.py --source beavertails --input data/raw/ --output datasets/train.jsonl
    python prepare_data.py --source custom --input data/my_data.csv --output datasets/train.jsonl \
        --text_col prompt --label_col safety --category_col category
    python prepare_data.py --source wildguard \
        --input data/wildguard.jsonl --output datasets/train.jsonl --split 0.9
"""

import argparse
import csv
import json
import random
from pathlib import Path

VALID_CATEGORIES = {
    "Violent",
    "Non-violent Illegal Acts",
    "Sexual Content or Sexual Acts",
    "PII",
    "Suicide & Self-Harm",
    "Unethical Acts",
    "Politically Sensitive Topics",
    "Copyright Violation",
    "Jailbreak",
}

VALID_SAFETY = {"safe", "controversial", "unsafe"}

_CAT_ALIASES = {
    "violence": "Violent",
    "violent": "Violent",
    "violence_and_physical_harm": "Violent",
    "weapon": "Violent",
    "weapons": "Violent",
    "non-violent illegal acts": "Non-violent Illegal Acts",
    "illegal": "Non-violent Illegal Acts",
    "illegal_activity": "Non-violent Illegal Acts",
    "hacking": "Non-violent Illegal Acts",
    "drug": "Non-violent Illegal Acts",
    "drugs": "Non-violent Illegal Acts",
    "fraud": "Non-violent Illegal Acts",
    "theft": "Non-violent Illegal Acts",
    "sexual": "Sexual Content or Sexual Acts",
    "sexual content": "Sexual Content or Sexual Acts",
    "sexual_content": "Sexual Content or Sexual Acts",
    "sexual content or sexual acts": "Sexual Content or Sexual Acts",
    "pii": "PII",
    "privacy": "PII",
    "personal_information": "PII",
    "personally identifiable information": "PII",
    "suicide": "Suicide & Self-Harm",
    "self-harm": "Suicide & Self-Harm",
    "self_harm": "Suicide & Self-Harm",
    "suicide & self-harm": "Suicide & Self-Harm",
    "suicide_and_self_harm": "Suicide & Self-Harm",
    "unethical": "Unethical Acts",
    "unethical acts": "Unethical Acts",
    "hate": "Unethical Acts",
    "hate_speech": "Unethical Acts",
    "harassment": "Unethical Acts",
    "discrimination": "Unethical Acts",
    "bias": "Unethical Acts",
    "misinformation": "Unethical Acts",
    "offensive": "Unethical Acts",
    "political": "Politically Sensitive Topics",
    "politically sensitive topics": "Politically Sensitive Topics",
    "political_sensitive": "Politically Sensitive Topics",
    "copyright": "Copyright Violation",
    "copyright violation": "Copyright Violation",
    "copyright_violation": "Copyright Violation",
    "jailbreak": "Jailbreak",
    "prompt_injection": "Jailbreak",
    "prompt injection": "Jailbreak",
}


def normalize_category(raw: str | None) -> str:
    if not raw or raw.lower().strip() in ("none", "safe", "n/a", ""):
        return "None"
    key = raw.lower().strip()
    if key in _CAT_ALIASES:
        return _CAT_ALIASES[key]
    for cat in VALID_CATEGORIES:
        if cat.lower() == key:
            return cat
    return "None"


def normalize_safety(raw: str | None) -> str:
    if not raw:
        return "safe"
    val = raw.lower().strip()
    if val in ("unsafe", "1", "true", "harmful", "toxic", "bad"):
        return "unsafe"
    if val in ("controversial", "borderline", "ambiguous"):
        return "controversial"
    return "safe"


def make_record(
    prompt: str,
    response: str | None = None,
    query_safety: str = "safe",
    query_category: str = "None",
    response_safety: str = "safe",
    response_category: str = "None",
    boundary_token: int | None = None,
) -> dict:
    messages = [{"role": "user", "content": prompt}]
    if response:
        messages.append({"role": "assistant", "content": response})
    return {
        "messages": messages,
        "query_safety": query_safety,
        "query_category": query_category,
        "response_safety": response_safety,
        "response_category": response_category,
        "boundary_token": boundary_token,
    }


def parse_beavertails(path: str) -> list[dict]:
    records = []
    for line in _iter_jsonl(path):
        prompt = line.get("prompt", "")
        response = line.get("response", "")
        categories = line.get("category", {})
        if isinstance(categories, dict):
            active = [k for k, v in categories.items() if v]
            cat = normalize_category(active[0]) if active else "None"
        else:
            cat = normalize_category(str(categories))
        is_safe = line.get("is_safe", True)
        r_safety = "safe" if is_safe else "unsafe"
        records.append(make_record(
            prompt=prompt,
            response=response,
            query_safety="unsafe" if not is_safe else "safe",
            query_category=cat if not is_safe else "None",
            response_safety=r_safety,
            response_category=cat if not is_safe else "None",
        ))
    return records


def parse_toxicchat(path: str) -> list[dict]:
    records = []
    for line in _iter_jsonl(path):
        prompt = line.get("user_input", "")
        response = line.get("model_output", "")
        toxic = line.get("toxicity", 0)
        jailbreak = line.get("jailbreaking", 0)
        if "1" in jailbreak:
            q_safety, q_cat = "unsafe", "Jailbreak"
        elif int(toxic):
            q_safety, q_cat = "unsafe", "Unethical Acts"
        else:
            q_safety, q_cat = "safe", "None"
        records.append(make_record(
            prompt=prompt,
            response=response if response else None,
            query_safety=q_safety,
            query_category=q_cat,
        ))
    return records


def parse_wildguard(path: str) -> list[dict]:
    records = []
    for line in _iter_jsonl(path):
        prompt = line.get("prompt", "")
        response = line.get("response", None)
        q_safety = normalize_safety(line.get("prompt_harm_label", "safe"))
        r_safety = normalize_safety(line.get("response_harm_label", "safe"))
        cat = normalize_category(line.get("harm_category", None))
        records.append(make_record(
            prompt=prompt,
            response=response,
            query_safety=q_safety,
            query_category=cat if q_safety != "safe" else "None",
            response_safety=r_safety,
            response_category=cat if r_safety != "safe" else "None",
        ))
    return records


def parse_aegis(path: str) -> list[dict]:
    records = []
    for line in _iter_jsonl(path):
        prompt = line.get("text", line.get("prompt", ""))
        response = line.get("response_label", None)
        label = normalize_safety(line.get("prompt_label", "safe"))
        cat = normalize_category(line.get("violated_categories", None))
        records.append(make_record(
            prompt=prompt,
            response=response,
            query_safety=label,
            query_category=cat if label != "safe" else "None",
            response_safety=normalize_safety(line.get("response_label", "safe")) if response else "safe",
            response_category=cat if response and label != "safe" else "None",
        ))
    return records


def parse_custom(
    path: str,
    text_col: str,
    label_col: str,
    category_col: str | None,
    response_col: str | None,
    delimiter: str,
) -> list[dict]:
    records = []
    suffix = Path(path).suffix.lower()

    if suffix in (".jsonl", ".json"):
        for line in _iter_jsonl(path):
            prompt = line.get(text_col, "")
            label = normalize_safety(line.get(label_col, "safe"))
            cat = normalize_category(line.get(category_col, None)) if category_col else "None"
            response = line.get(response_col, None) if response_col else None
            records.append(make_record(
                prompt=prompt,
                response=response,
                query_safety=label,
                query_category=cat if label != "safe" else "None",
            ))
    else:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                prompt = row.get(text_col, "")
                label = normalize_safety(row.get(label_col, "safe"))
                cat = normalize_category(row.get(category_col)) if category_col else "None"
                response = row.get(response_col, None) if response_col else None
                records.append(make_record(
                    prompt=prompt,
                    response=response,
                    query_safety=label,
                    query_category=cat if label != "safe" else "None",
                ))
    return records


def _iter_jsonl(path: str):
    p = Path(path)
    files = sorted(p.glob("*.jsonl")) + sorted(p.glob("*.json")) if p.is_dir() else [p]
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            first_char = fh.read(1)
            fh.seek(0)
            if first_char == "[":
                yield from json.load(fh)
            else:
                for line in fh:
                    line = line.strip()
                    if line:
                        yield json.loads(line)


def write_jsonl(records: list[dict], path: str):
    """Write records to a JSONL file, creating parent directories as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records -> {path}")


def train_val_split(records: list[dict], ratio: float, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Split records into train/val by ratio."""
    rng = random.Random(seed)
    shuffled = records.copy()
    rng.shuffle(shuffled)
    n = int(len(shuffled) * ratio)
    return shuffled[:n], shuffled[n:]


def print_stats(records: list[dict], name: str = ""):
    total = len(records)
    safety_counts = {"safe": 0, "controversial": 0, "unsafe": 0}
    cat_counts: dict[str, int] = {}
    has_response = 0

    for r in records:
        safety_counts[r["query_safety"]] += 1
        cat = r["query_category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        if len(r["messages"]) > 1:
            has_response += 1

    sep = "─" * 50
    print(f"\n{sep}")
    print(f"  {name or 'Dataset'}: {total} samples")
    print(f"  With response: {has_response} ({has_response/total*100:.1f}%)")
    print(f"  Query safety:  safe={safety_counts['safe']}  "
          f"controversial={safety_counts['controversial']}  "
          f"unsafe={safety_counts['unsafe']}")
    print("  Categories:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat:40s} {cnt:>6d}  ({cnt/total*100:.1f}%)")
    print(f"{sep}\n")


PARSERS = {
    "beavertails": parse_beavertails,
    "toxicchat": parse_toxicchat,
    "wildguard": parse_wildguard,
    "aegis": parse_aegis,
}


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Convert safety datasets to Guard JSONL format")
    parser.add_argument("--source", required=True, choices=[*PARSERS.keys(), "custom"])
    parser.add_argument("--input", required=True, help="Input file or directory")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--split", type=float, default=None,
                        help="Train ratio (e.g. 0.9). Produces <output> and <output>.val.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--category_col", default=None)
    parser.add_argument("--response_col", default=None)
    parser.add_argument("--delimiter", default=",")
    args = parser.parse_args()

    if args.source == "custom":
        records = parse_custom(
            args.input, args.text_col, args.label_col,
            args.category_col, args.response_col, args.delimiter,
        )
    else:
        records = PARSERS[args.source](args.input)

    print_stats(records, name=f"{args.source} ({args.input})")

    if args.split:
        train, val = train_val_split(records, args.split, args.seed)
        out = Path(args.output)
        val_path = str(out.with_suffix("")) + "_val.jsonl"
        write_jsonl(train, args.output)
        write_jsonl(val, val_path)
        print_stats(train, "train")
        print_stats(val, "val")
    else:
        write_jsonl(records, args.output)


if __name__ == "__main__":
    main()
