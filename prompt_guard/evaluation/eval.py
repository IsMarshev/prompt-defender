import argparse
import json
from pathlib import Path
from typing import Optional

from sklearn.metrics import f1_score, precision_score, recall_score

from prompt_guard.inference.classifier import PromptClassifier


def _load_benchmark(path: str) -> list[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def _binary_label(label: str, effective: bool = True) -> int:
    """Map label to binary: Unsafe=1, Safe=0. Controversial depends on mode."""
    return 1 if label == "Unsafe" else 0


def evaluate(
    classifier: PromptClassifier,
    benchmark_file: str,
    batch_size: int = 16,
) -> dict:
    samples = _load_benchmark(benchmark_file)
    texts = [s["text"] for s in samples]
    true_labels = [s["label"] for s in samples]

    # Run in batches
    predictions = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        predictions.extend(classifier.classify_batch(batch))

    y_pred = [1 if p["effective_label"] == "Unsafe" else 0 for p in predictions]
    y_true = [1 if lbl == "Unsafe" else 0 for lbl in true_labels]

    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def run_evaluation(
    model_path: str,
    benchmarks: list[tuple[str, str]],  # (name, path)
    device: str = "cuda",
    batch_size: int = 16,
):
    """
    Evaluate model on multiple benchmarks in both strict and loose modes.
    benchmarks: list of (benchmark_name, benchmark_file_path)
    """
    header = f"{'Benchmark':<20} | {'Mode':<6} | {'Precision':>9} | {'Recall':>6} | {'F1':>6}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    all_results = {}
    for mode in ("strict", "loose"):
        classifier = PromptClassifier(model_path=model_path, mode=mode, device=device)
        for bench_name, bench_path in benchmarks:
            metrics = evaluate(classifier, bench_path, batch_size=batch_size)
            key = f"{bench_name}_{mode}"
            all_results[key] = metrics
            print(
                f"{bench_name:<20} | {mode:<6} | "
                f"{metrics['precision']:>9.4f} | "
                f"{metrics['recall']:>6.4f} | "
                f"{metrics['f1']:>6.4f}"
            )

    print(sep)
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate PromptClassifier on benchmarks")
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument(
        "--benchmark",
        required=True,
        nargs="+",
        help="Benchmark JSONL files (optionally as name:path)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    benchmarks = []
    for b in args.benchmark:
        if ":" in b:
            name, path = b.split(":", 1)
        else:
            name = Path(b).stem
            path = b
        benchmarks.append((name, path))

    run_evaluation(
        model_path=args.model,
        benchmarks=benchmarks,
        device=args.device,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
