#!/usr/bin/env python3
"""Convert raw test datasets from ./data/ into the standard datasets/test.json format.

Produces:
  datasets/aegis_test.json       (AEGIS Safety Dataset)
  datasets/beavertails_test.json (BeaverTails)
  datasets/toxicchat_test.json   (ToxicChat)
  datasets/wildguard_test.json   (WildGuard)
  datasets/jdd_test.json         (JailbreakDetection Dataset)

Usage:
    python prepare_test_datasets.py
    python prepare_test_datasets.py --only aegis wildguard
"""

import argparse
import csv
import json
from pathlib import Path

DATA_DIR = Path("data")
OUT_DIR = Path("datasets")


def make_row(uid: int, message: list, label: str, unsafe_type: str, source: str) -> dict:
    return {
        "unique_id": uid,
        "message": message,
        "unsafe_type": unsafe_type,
        "source": source,
        "label": label,
        "input_ids": None,
        "unsafe_end_index": None,
        "unsafe_start_index": None,
    }


# ---------------------------------------------------------------------------
# Per-dataset builders
# ---------------------------------------------------------------------------

def build_aegis() -> list[dict]:
    raw = json.loads((DATA_DIR / "aegis" / "test.json").read_text())
    out = []
    for i, row in enumerate(raw):
        prompt = row.get("prompt", "")
        response = row.get("response", "")
        label = "Unsafe" if row.get("prompt_label") == "unsafe" else "Safe"
        unsafe_type = row.get("violated_categories") or ""
        message = [{"role": "user", "content": prompt}]
        if response:
            message.append({"role": "assistant", "content": response})
        out.append(make_row(i, message, label, unsafe_type, "aegis"))
    return out


def build_beavertails() -> list[dict]:
    out = []
    with open(DATA_DIR / "beavertails" / "test.jsonl", encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            is_safe = row.get("is_safe", True)
            label = "Safe" if is_safe else "Unsafe"
            category = row.get("category") or {}
            active_cats = [k for k, v in category.items() if v]
            unsafe_type = ", ".join(active_cats) if active_cats else ""
            message = [{"role": "user", "content": row.get("prompt", "")}]
            response = row.get("response", "")
            if response:
                message.append({"role": "assistant", "content": response})
            out.append(make_row(i, message, label, unsafe_type, "beavertails"))
    return out


def build_toxicchat() -> list[dict]:
    out = []
    with open(
        DATA_DIR / "toxicchat" / "toxic-chat_annotation_test.csv",
        newline="", encoding="utf-8",
    ) as f:
        for i, row in enumerate(csv.DictReader(f)):
            toxicity = int(float(row.get("toxicity", 0) or 0))
            jailbreaking = int(float(row.get("jailbreaking", 0) or 0))
            label = "Unsafe" if toxicity == 1 else "Safe"
            unsafe_type = "Jailbreak" if jailbreaking else ""
            message = [{"role": "user", "content": row.get("user_input", "")}]
            out.append(make_row(i, message, label, unsafe_type, "toxicchat"))
    return out


def build_wildguard() -> list[dict]:
    import pandas as pd
    df = pd.read_parquet(DATA_DIR / "wildguard" / "wildguard_test.parquet")
    out = []
    for i, row in df.iterrows():
        prompt = str(row.get("prompt") or "")
        response = str(row.get("response") or "")
        label = "Unsafe" if row.get("prompt_harm_label") == "harmful" else "Safe"
        subcategory = str(row.get("subcategory") or "")
        unsafe_type = subcategory if label == "Unsafe" else ""
        message = [{"role": "user", "content": prompt}]
        if response and response != "nan":
            message.append({"role": "assistant", "content": response})
        out.append(make_row(i, message, label, unsafe_type, "wildguard"))
    return out


def build_jdd() -> list[dict]:
    import pyarrow.parquet as pq
    table = pq.read_table(str(DATA_DIR / "jdd" / "test-00000-of-00001.parquet"))
    texts = table["text"].to_pylist()
    labels_raw = table["label"].to_pylist()
    out = []
    for i, (text, lbl) in enumerate(zip(texts, labels_raw)):
        label = "Unsafe" if int(lbl) == 1 else "Safe"
        unsafe_type = "Jailbreak" if label == "Unsafe" else ""
        message = [{"role": "user", "content": str(text)}]
        out.append(make_row(i, message, label, unsafe_type, "jdd"))
    return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BUILDERS: dict[str, callable] = {
    "aegis_test":       build_aegis,
    "beavertails_test": build_beavertails,
    "toxicchat_test":   build_toxicchat,
    "wildguard_test":   build_wildguard,
    "jdd_test":         build_jdd,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", nargs="+", default=None,
        help="Build only these datasets (e.g. --only aegis_test wildguard_test).",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    targets = {k: v for k, v in BUILDERS.items()
               if args.only is None or k in args.only}

    for name, builder in targets.items():
        print(f"Building {name}...", end=" ", flush=True)
        try:
            data = builder()
            out_path = OUT_DIR / f"{name}.json"
            out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            safe = sum(1 for d in data if d["label"] == "Safe")
            unsafe = sum(1 for d in data if d["label"] == "Unsafe")
            print(f"{len(data):,} rows  Safe={safe:,}  Unsafe={unsafe:,}  →  {out_path}")
        except Exception as exc:
            print(f"FAILED: {exc}")


if __name__ == "__main__":
    main()
