#!/usr/bin/env python3
"""
Evaluate exported HF models from artifacts/ on one or more test datasets.

Each model × dataset is run independently; results are saved to
tests/{model_name}__{dataset}.json  (or tests/{model_name}.json for "test").
Labels are binary (Safe / Unsafe). "Controversial" is treated as positive.

Available datasets (all drawn from datasets/test_merged_clean.jsonl):
  test         all rows  (default)
  aegis        dataset == "aegis"
  beavertails  dataset == "beavertails"
  toxicchat    dataset == "toxicchat"
  wildguard    dataset == "wildguard"
  jdd          dataset == "jdd"

Usage:
    python run_tests.py                                    # all models, test set
    python run_tests.py --datasets all                     # all models, all sets
    python run_tests.py --datasets aegis wildguard         # specific datasets
    python run_tests.py --only qwen0.6B-guard --datasets aegis
    python run_tests.py --batch-size 8 --max-new-tokens 16
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from prompt_defender.core.dataset import SAFETY_PREFIX
from prompt_defender.core.evaluation import (
    UNPARSED_LABEL,
    canonicalize_label,
    detect_default_device,
    normalize_generated_safety_text,
    parse_safety_label,
)

ARTIFACTS_DIR = Path("artifacts")
MERGED_TEST = Path("datasets/test_merged_clean.jsonl")

# For binary evaluation: Unsafe (and Controversial) are "positive"
GOLD_POSITIVE = {"Unsafe"}
PRED_POSITIVE = {"Unsafe", "Controversial"}

# dataset name → value of the "dataset" field to filter on (None = all rows)
TEST_DATASETS: dict[str, str | None] = {
    "test":        None,
    "aegis":       "aegis",
    "beavertails": "beavertails",
    "toxicchat":   "toxicchat",
    "wildguard":   "wildguard",
    "jdd":         "jdd",
}

_CLASSIFIER_LABEL_MAP = {
    "safe":      "Safe",
    "injection": "Unsafe",
    "unsafe":    "Unsafe",
    "jailbreak": "Unsafe",
    "benign":    "Safe",
}


# ---------------------------------------------------------------------------
# Model type detection
# ---------------------------------------------------------------------------

def detect_model_type(model_dir: Path) -> str:
    cfg_path = model_dir / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        archs = cfg.get("architectures") or []
        if any("SequenceClassification" in a for a in archs):
            return "classifier"
    return "generative"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_generative(model_dir: Path, device: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, padding_side="left", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.eval()
    return model, tokenizer


def load_classifier(model_dir: Path, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.eval()
    cfg = json.loads((model_dir / "config.json").read_text())
    id2label = cfg.get("id2label", {})
    label_map = {
        int(k): _CLASSIFIER_LABEL_MAP.get(v.lower(), v.capitalize())
        for k, v in id2label.items()
    }
    return model, tokenizer, label_map


# ---------------------------------------------------------------------------
# Inference — generative
# ---------------------------------------------------------------------------

def build_gen_prompt(tokenizer, row: dict) -> str:
    messages = [{"role": "user", "content": row["instruction"]}]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
    except (TypeError, ValueError):
        # Fallback: no chat template available — use raw instruction
        text = row["instruction"]
    return text + SAFETY_PREFIX


def run_generative(model, tokenizer, rows, batch_size, max_new_tokens, device):
    prompt_texts = [build_gen_prompt(tokenizer, r) for r in rows]
    results: list[dict] = []

    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch_texts = prompt_texts[start : start + batch_size]
            batch_rows = rows[start : start + batch_size]

            inputs = tokenizer(
                batch_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=2048,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            generated = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
            new_ids = generated[:, inputs["input_ids"].shape[1]:]
            raw_outputs = tokenizer.batch_decode(new_ids, skip_special_tokens=True)

            for offset, raw_output in enumerate(raw_outputs):
                row = batch_rows[offset]
                normalized = normalize_generated_safety_text(raw_output)
                predicted = parse_safety_label(normalized)
                results.append({
                    "label": row.get("label"),
                    "category": row.get("category"),
                    "dataset": row.get("dataset"),
                    "guard_raw_output": raw_output,
                    "guard_predict": normalized,
                    "guard_predicted_label": predicted,
                    "guard_gold_label": canonicalize_label(row.get("label")),
                })

            done = min(start + batch_size, len(rows))
            if done % 500 == 0 or done == len(rows):
                print(f"  {done}/{len(rows)} done", file=sys.stderr)

    return results


# ---------------------------------------------------------------------------
# Inference — sequence classifier
# ---------------------------------------------------------------------------

def run_classifier(model, tokenizer, label_map, rows, batch_size, device):
    texts = [r["instruction"] for r in rows]
    results: list[dict] = []

    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch_texts = texts[start : start + batch_size]
            batch_rows = rows[start : start + batch_size]

            inputs = tokenizer(
                batch_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            pred_ids = logits.argmax(dim=-1).tolist()

            for offset, pred_id in enumerate(pred_ids):
                row = batch_rows[offset]
                predicted = label_map.get(pred_id, UNPARSED_LABEL)
                raw = f"label_id={pred_id} ({label_map.get(pred_id, '?')})"
                results.append({
                    "label": row.get("label"),
                    "category": row.get("category"),
                    "dataset": row.get("dataset"),
                    "guard_raw_output": raw,
                    "guard_predict": f"Safety: {predicted}",
                    "guard_predicted_label": predicted,
                    "guard_gold_label": canonicalize_label(row.get("label")),
                })

            done = min(start + batch_size, len(rows))
            if done % 500 == 0 or done == len(rows):
                print(f"  {done}/{len(rows)} done", file=sys.stderr)

    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: list[dict]) -> dict:
    tp = fp = tn = fn = 0
    pred_counts: Counter = Counter()
    gold_counts: Counter = Counter()

    for r in results:
        g = r["guard_gold_label"]
        p = r["guard_predicted_label"]
        gold_counts[g] += 1
        pred_counts[p] += 1
        gp = g in GOLD_POSITIVE
        pp = p in PRED_POSITIVE
        if gp and pp:
            tp += 1
        elif gp and not pp:
            fn += 1
        elif not gp and pp:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    accuracy = (tp + tn) / len(results) if results else 0.0
    parse_rate = 1.0 - pred_counts.get(UNPARSED_LABEL, 0) / len(results)

    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "accuracy": round(accuracy, 6),
        "parse_rate": round(parse_rate, 6),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "gold_counts": dict(gold_counts),
        "pred_counts": dict(pred_counts),
        "total": len(results),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Run only these model names (subfolder names in artifacts/).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["test"],
        metavar="DATASET",
        help=(
            "Which dataset(s) to evaluate on. "
            f"Choices: all, {', '.join(TEST_DATASETS)}. "
            "Default: test"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--device", default=detect_default_device())
    parser.add_argument("--out-dir", default="tests")
    return parser.parse_args()


def _resolve_datasets(requested: list[str]) -> dict[str, str | None]:
    if "all" in requested:
        return dict(TEST_DATASETS)
    unknown = [d for d in requested if d not in TEST_DATASETS]
    if unknown:
        print(
            f"Unknown dataset(s): {unknown}. "
            f"Available: {list(TEST_DATASETS)}",
            file=sys.stderr,
        )
        sys.exit(1)
    return {k: TEST_DATASETS[k] for k in requested}


def _load_dataset(dataset_filter: str | None) -> list[dict]:
    if not MERGED_TEST.exists():
        raise FileNotFoundError(
            f"{MERGED_TEST} not found."
        )
    rows = []
    with open(MERGED_TEST, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if dataset_filter is None or row.get("dataset") == dataset_filter:
                rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    device = args.device
    print(f"Device: {device}")

    datasets = _resolve_datasets(args.datasets)
    print(f"Datasets: {list(datasets)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dirs = sorted(d for d in ARTIFACTS_DIR.iterdir() if d.is_dir())
    if args.only:
        model_dirs = [d for d in model_dirs if d.name in args.only]

    if not model_dirs:
        print(f"No models found in {ARTIFACTS_DIR}/", file=sys.stderr)
        sys.exit(1)

    summary: list[dict] = []

    for model_dir in model_dirs:
        model_name = model_dir.name
        model_type = detect_model_type(model_dir)
        print(f"\n=== {model_name} [{model_type}] ===")

        label_map: dict = {}
        if model_type == "classifier":
            model, tokenizer, label_map = load_classifier(model_dir, device)
        else:
            model, tokenizer = load_generative(model_dir, device)

        for dataset_name, dataset_filter in datasets.items():
            print(f"  -- dataset: {dataset_name}")
            rows: list[dict] = _load_dataset(dataset_filter)
            print(f"     {len(rows):,} samples")

            if model_type == "classifier":
                results = run_classifier(
                    model, tokenizer, label_map, rows,
                    batch_size=args.batch_size, device=device,
                )
            else:
                results = run_generative(
                    model, tokenizer, rows,
                    batch_size=args.batch_size,
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                )

            metrics = compute_metrics(results)

            output = {
                "model_name": model_name,
                "model_dir": str(model_dir),
                "model_type": model_type,
                "dataset": dataset_name,
                "metrics": metrics,
                "predictions": results,
            }

            safe_name = model_name.replace(".", "_").replace(" ", "_")
            if dataset_name == "test":
                out_path = out_dir / f"{safe_name}.json"
            else:
                out_path = out_dir / f"{safe_name}__{dataset_name}.json"

            out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
            print(f"     Saved → {out_path}")
            print(
                f"     F1={metrics['f1']:.4f}  "
                f"P={metrics['precision']:.4f}  "
                f"R={metrics['recall']:.4f}  "
                f"Acc={metrics['accuracy']:.4f}  "
                f"ParseRate={metrics['parse_rate']:.4f}"
            )

            summary.append({
                "model": model_name,
                "dataset": dataset_name,
                **metrics,
            })

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n=== Summary ===")
    print(f"{'Model':35s}  {'Dataset':12s}  {'F1':>7}  {'P':>7}  {'R':>7}  {'Acc':>7}")
    print("-" * 80)
    for row in sorted(summary, key=lambda x: (x["dataset"], -x["f1"])):
        print(
            f"{row['model']:35s}  {row['dataset']:12s}  "
            f"{row['f1']:7.4f}  {row['precision']:7.4f}  "
            f"{row['recall']:7.4f}  {row['accuracy']:7.4f}"
        )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSummary saved → {summary_path}")


if __name__ == "__main__":
    main()
