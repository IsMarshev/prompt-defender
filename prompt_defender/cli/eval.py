import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt_defender.core.evaluation import (
    RAW_PREDICTION_ORDER,
    UNPARSED_LABEL,
    canonicalize_label,
    detect_default_device,
    normalize_generated_safety_text,
    parse_safety_label,
)
from prompt_defender.core.dataset import SAFETY_PREFIX
from prompt_defender.pipeline.experiment_utils import now_iso, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a generative guard model on thinking.jsonl-style data."
    )
    parser.add_argument("--model-path", default="guard_model", help="Path to exported HF model")
    parser.add_argument("--data-path", default="thinking.jsonl", help="Path to eval JSONL")
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional JSONL file with per-example predictions",
    )
    parser.add_argument(
        "--metrics-file",
        default=None,
        help="Optional JSON file with aggregate evaluation metrics",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=None,
        help="Optional tokenizer truncation length",
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only first N rows")
    parser.add_argument(
        "--device",
        default=detect_default_device(),
        help="cuda, cpu, mps, ...",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Model dtype",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling during generation",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--gold-positive-labels",
        nargs="+",
        default=["Unsafe"],
        help="Gold labels treated as positive for Recall/Precision/F1",
    )
    parser.add_argument(
        "--pred-positive-labels",
        nargs="+",
        default=["Unsafe", "Controversial"],
        help="Predicted labels treated as positive for Recall/Precision/F1",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Forward enable_thinking=False to chat template when supported",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N batches to stderr; set 0 to disable",
    )
    return parser.parse_args()

def load_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break

    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def get_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages = row.get("messages") or row.get("message") or []
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"Row is missing a non-empty messages/message list: {row}")
    return messages


def build_prompt_text(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, Any]],
    enable_thinking: bool,
) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )


def batched(items: list[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start:start + batch_size]


def make_generation_kwargs(args: argparse.Namespace, tokenizer: AutoTokenizer) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": args.do_sample,
    }
    if args.do_sample:
        kwargs.update(
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            }
        )
    return kwargs


def evaluate_rows(
    model,
    tokenizer,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    dtype_map = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    model_dtype = dtype_map[args.dtype]
    if model_dtype != "auto":
        model.to(dtype=model_dtype)

    tokenizer.padding_side = "left"
    generation_kwargs = make_generation_kwargs(args, tokenizer)
    results: list[dict[str, Any]] = []

    prompt_texts = [
        build_prompt_text(
            tokenizer=tokenizer,
            messages=get_messages(row),
            enable_thinking=not args.disable_thinking,
        ) + SAFETY_PREFIX
        for row in rows
    ]

    with torch.inference_mode():
        for batch_index, (start, batch_prompt_texts) in enumerate(
            batched(prompt_texts, args.batch_size),
            start=1,
        ):
            encode_kwargs: dict[str, Any] = {
                "return_tensors": "pt",
                "padding": True,
            }
            if args.max_input_length is not None:
                encode_kwargs["truncation"] = True
                encode_kwargs["max_length"] = args.max_input_length

            inputs = tokenizer(batch_prompt_texts, **encode_kwargs)
            inputs = {key: value.to(model.device) for key, value in inputs.items()}

            generated = model.generate(
                **inputs,
                **generation_kwargs,
            )
            generated_ids = generated[:, inputs["input_ids"].shape[1]:]
            batch_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for offset, raw_output in enumerate(batch_outputs):
                row = rows[start + offset]
                gold_label = canonicalize_label(row.get("label"))
                normalized_output = normalize_generated_safety_text(raw_output)
                predicted_label = parse_safety_label(normalized_output)
                result = dict(row)
                result["guard_predict"] = normalized_output
                result["guard_predicted_label"] = predicted_label
                result["guard_gold_label"] = gold_label
                results.append(result)

            if args.progress_every and batch_index % args.progress_every == 0:
                print(
                    f"Processed {min(start + len(batch_prompt_texts), len(rows))}/{len(rows)} rows",
                    file=sys.stderr,
                )

    return results


def compute_binary_metrics(
    results: list[dict[str, Any]],
    gold_positive_labels: set[str],
    pred_positive_labels: set[str],
) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    raw_confusion: dict[str, Counter[str]] = {}
    gold_counts: Counter[str] = Counter()
    pred_counts: Counter[str] = Counter()

    for row in results:
        gold_label = canonicalize_label(row.get("guard_gold_label") or row.get("label"))
        pred_label = canonicalize_label(row.get("guard_predicted_label"))

        gold_counts[gold_label] += 1
        pred_counts[pred_label] += 1
        raw_confusion.setdefault(gold_label, Counter())
        raw_confusion[gold_label][pred_label] += 1

        gold_positive = gold_label in gold_positive_labels
        pred_positive = pred_label in pred_positive_labels

        if gold_positive and pred_positive:
            tp += 1
        elif gold_positive and not pred_positive:
            fn += 1
        elif not gold_positive and pred_positive:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    parse_rate = 1.0 - (pred_counts[UNPARSED_LABEL] / len(results))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "parse_rate": parse_rate,
        "binary_confusion": {
            "labels": ["Negative", "Positive"],
            "rows": {
                "Negative": [tn, fp],
                "Positive": [fn, tp],
            },
        },
        "raw_confusion": raw_confusion,
        "gold_counts": gold_counts,
        "pred_counts": pred_counts,
        "total": len(results),
    }


def save_results(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_binary_confusion(metrics: dict[str, Any]) -> str:
    rows = metrics["binary_confusion"]["rows"]
    matrix = [
        ("True Safe", rows["Negative"][0], rows["Negative"][1]),
        ("True Unsafe", rows["Positive"][0], rows["Positive"][1]),
    ]
    header = f"{'':14} {'Pred Safe/Other':>16} {'Pred Unsafe/Controversial':>27}"
    lines = [header]
    for name, negative_count, positive_count in matrix:
        lines.append(f"{name:14} {negative_count:16d} {positive_count:27d}")
    return "\n".join(lines)


def format_raw_confusion(metrics: dict[str, Any]) -> str:
    gold_labels = sorted(metrics["raw_confusion"].keys())
    columns = RAW_PREDICTION_ORDER
    header = f"{'':14} " + " ".join(f"{name:>15}" for name in columns)
    lines = [header]
    for gold_label in gold_labels:
        counts = metrics["raw_confusion"][gold_label]
        line = f"{gold_label:14} " + " ".join(f"{counts.get(name, 0):15d}" for name in columns)
        lines.append(line)
    return "\n".join(lines)


def print_summary(
    metrics: dict[str, Any],
    gold_positive_labels: set[str],
    pred_positive_labels: set[str],
) -> None:
    print(f"Total: {metrics['total']}")
    print(f"Gold positive labels: {sorted(gold_positive_labels)}")
    print(f"Pred positive labels: {sorted(pred_positive_labels)}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Parse Rate: {metrics['parse_rate']:.4f}")
    print()
    print("Confusion Matrix (rows=true, cols=pred):")
    print(format_binary_confusion(metrics))
    print()
    print("Raw Parsed Labels (rows=true label, cols=predicted label):")
    print(format_raw_confusion(metrics))
    print()
    print("Gold label counts:", dict(metrics["gold_counts"]))
    print("Predicted label counts:", dict(metrics["pred_counts"]))


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()

    data_path = Path(args.data_path)
    model_path = Path(args.model_path)

    rows = load_rows(data_path, args.limit)
    gold_positive_labels = {canonicalize_label(label) for label in args.gold_positive_labels}
    pred_positive_labels = {canonicalize_label(label) for label in args.pred_positive_labels}

    dtype_map = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype_map[args.dtype],
        trust_remote_code=True,
    ).to(args.device)
    model.eval()

    results = evaluate_rows(model=model, tokenizer=tokenizer, rows=rows, args=args)

    if args.output_file:
        save_results(Path(args.output_file), results)

    metrics = compute_binary_metrics(
        results=results,
        gold_positive_labels=gold_positive_labels,
        pred_positive_labels=pred_positive_labels,
    )
    print_summary(
        metrics=metrics,
        gold_positive_labels=gold_positive_labels,
        pred_positive_labels=pred_positive_labels,
    )

    if args.metrics_file:
        save_json(
            Path(args.metrics_file),
            {
                "created_at": now_iso(),
                "model_path": str(model_path),
                "data_path": str(data_path),
                "output_file": args.output_file,
                "metrics": metrics,
                "gold_positive_labels": sorted(gold_positive_labels),
                "pred_positive_labels": sorted(pred_positive_labels),
                "args": vars(args),
            },
        )


if __name__ == "__main__":
    main()
