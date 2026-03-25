from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from experiment_utils import (
    apply_overrides,
    build_run_name,
    expand_matrix,
    load_json,
    load_yaml,
    now_iso,
    render_name_template,
    save_json,
)


REPO_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a grid of experiments and aggregate results."
    )
    parser.add_argument("--grid", default="grid.yaml", help="Grid definition YAML")
    parser.add_argument("--run-root", default=None, help="Optional override for run root")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def serialize_override_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_eval_dataset_args(grid_config: dict[str, Any]) -> list[str]:
    eval_config = grid_config.get("eval", {})
    datasets = eval_config.get("datasets", {})
    args: list[str] = []
    if isinstance(datasets, dict):
        for name, path in datasets.items():
            args.extend(["--eval-data", f"{name}={path}"])
    return args


def build_run_command(
    grid_config: dict[str, Any],
    *,
    python_executable: str,
    run_root: Path,
    run_name: str,
    overrides: dict[str, Any],
    cli_args: argparse.Namespace,
) -> list[str]:
    train_config = grid_config.get("train", {})
    eval_config = grid_config.get("eval", {})

    command = [
        python_executable,
        "run_experiment.py",
        "--config",
        str(grid_config.get("base_config", "config.yaml")),
        "--run-root",
        str(run_root),
        "--run-name",
        run_name,
        "--devices",
        str(train_config.get("devices", 1)),
        "--num-nodes",
        str(train_config.get("num_nodes", 1)),
        "--strategy",
        str(train_config.get("strategy", "auto")),
        "--logger",
        str(train_config.get("logger", "tensorboard")),
        "--eval-batch-size",
        str(eval_config.get("batch_size", 4)),
        "--eval-max-new-tokens",
        str(eval_config.get("max_new_tokens", 256)),
        "--eval-dtype",
        str(eval_config.get("dtype", "auto")),
    ]

    if "seed" in train_config and train_config["seed"] is not None:
        command.extend(["--seed", str(train_config["seed"])])
    if train_config.get("resume"):
        command.extend(["--resume", str(train_config["resume"])])
    if eval_config.get("max_input_length") is not None:
        command.extend(["--eval-max-input-length", str(eval_config["max_input_length"])])
    if eval_config.get("limit") is not None:
        command.extend(["--eval-limit", str(eval_config["limit"])])
    if eval_config.get("device"):
        command.extend(["--eval-device", str(eval_config["device"])])
    if eval_config.get("disable_thinking"):
        command.append("--disable-thinking")
    if eval_config.get("save_predictions"):
        command.append("--save-predictions")
    if cli_args.skip_existing:
        command.append("--skip-existing")
    if cli_args.dry_run:
        command.append("--dry-run")

    command.extend(build_eval_dataset_args(grid_config))

    for key, value in overrides.items():
        command.extend(["--set", f"{key}={serialize_override_value(value)}"])

    return command


def load_run_summary(run_root: Path, run_name: str) -> dict[str, Any] | None:
    summary_path = run_root / run_name / "experiment_summary.json"
    if summary_path.exists():
        return load_json(summary_path)
    return None


def summary_to_row(summary: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_name": summary.get("run_name"),
        "status": summary.get("status"),
        "created_at": summary.get("created_at"),
        "completed_at": summary.get("completed_at"),
        "error": summary.get("error"),
    }

    for key, value in overrides.items():
        row[f"override.{key}"] = value

    train_summary = summary.get("train") or {}
    if train_summary:
        row["train.best_score"] = train_summary.get("best_score")
        row["train.best_checkpoint"] = train_summary.get("best_checkpoint")

    evaluations = summary.get("evaluations") or {}
    for dataset_name, payload in evaluations.items():
        metrics = payload.get("metrics") or {}
        for metric_name in ["f1", "precision", "recall", "parse_rate", "total"]:
            if metric_name in metrics:
                row[f"eval.{dataset_name}.{metric_name}"] = metrics[metric_name]

    return row


def main() -> None:
    args = parse_args()

    grid_path = resolve_path(args.grid)
    grid_config = load_yaml(grid_path)
    base_config_path = resolve_path(grid_config.get("base_config", "config.yaml"))
    base_config = load_yaml(base_config_path)

    constants = grid_config.get("constants", {})
    matrix = grid_config.get("matrix", {})
    combinations = expand_matrix(matrix)
    if args.max_runs is not None:
        combinations = combinations[:args.max_runs]

    run_root = resolve_path(args.run_root or grid_config.get("run_root", "runs"))
    run_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    grid_runs: list[dict[str, Any]] = []
    failures = 0

    for index, combination in enumerate(combinations, start=1):
        overrides = dict(constants)
        overrides.update(combination)
        resolved_config = apply_overrides(base_config, overrides)

        if grid_config.get("name_template"):
            run_name = render_name_template(grid_config["name_template"], resolved_config)
        else:
            run_name = build_run_name(resolved_config, overrides)

        command = build_run_command(
            grid_config,
            python_executable=args.python,
            run_root=run_root,
            run_name=run_name,
            overrides=overrides,
            cli_args=args,
        )

        print(f"[{index}/{len(combinations)}] {run_name}", flush=True)
        print(" ".join(command), flush=True)

        if not args.dry_run:
            try:
                subprocess.run(command, cwd=REPO_ROOT, check=True)
            except subprocess.CalledProcessError:
                failures += 1
                if not args.continue_on_error:
                    raise

        summary = load_run_summary(run_root, run_name)
        if summary:
            rows.append(summary_to_row(summary, overrides))
            grid_runs.append(
                {
                    "run_name": run_name,
                    "status": summary.get("status"),
                    "overrides": overrides,
                    "summary_path": str(run_root / run_name / "experiment_summary.json"),
                }
            )
        else:
            rows.append(
                {
                    "run_name": run_name,
                    "status": "missing_summary" if not args.dry_run else "dry_run",
                    **{f"override.{key}": value for key, value in overrides.items()},
                }
            )
            grid_runs.append(
                {
                    "run_name": run_name,
                    "status": "missing_summary" if not args.dry_run else "dry_run",
                    "overrides": overrides,
                }
            )

    grid_name = grid_path.stem
    summary_json_path = run_root / f"{grid_name}_summary.json"
    summary_csv_path = run_root / f"{grid_name}_summary.csv"

    save_json(
        summary_json_path,
        {
            "created_at": now_iso(),
            "grid_path": str(grid_path),
            "base_config": str(base_config_path),
            "run_root": str(run_root),
            "total_runs": len(combinations),
            "failures": failures,
            "runs": grid_runs,
            "rows": rows,
        },
    )
    write_csv(summary_csv_path, rows)


if __name__ == "__main__":
    main()
