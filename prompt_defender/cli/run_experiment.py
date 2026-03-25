from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from prompt_defender.pipeline.experiment_utils import (
    apply_overrides,
    build_run_name,
    load_json,
    load_yaml,
    now_iso,
    parse_named_paths,
    parse_override_strings,
    save_json,
    save_yaml,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one experiment end-to-end: train -> export -> eval."
    )
    parser.add_argument("--config", default="config.yaml", help="Base YAML config")
    parser.add_argument("--run-root", default="runs", help="Directory for experiment runs")
    parser.add_argument("--run-name", default=None, help="Optional explicit run name")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Config override, repeatable",
    )
    parser.add_argument(
        "--eval-data",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Evaluation dataset, repeatable; defaults to val dataset from config",
    )
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--strategy", default="auto")
    parser.add_argument("--logger", default="tensorboard", choices=["tensorboard", "wandb"])
    parser.add_argument("--resume", default=None, help="Optional Lightning checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to export instead of train summary best checkpoint")
    parser.add_argument("--model-path", default=None, help="Exported HF model path for eval-only runs")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if completed summary already exists")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use")
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-max-new-tokens", type=int, default=256)
    parser.add_argument("--eval-max-input-length", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--eval-device", default=None)
    parser.add_argument(
        "--eval-dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
    )
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def stringify_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_command(command: list[str], *, cwd: Path, dry_run: bool) -> None:
    print(stringify_command(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, check=True)


def load_summary_if_exists(path: Path) -> dict[str, Any] | None:
    if path.exists():
        return load_json(path)
    return None


def resolve_eval_datasets(
    args: argparse.Namespace,
    resolved_config: dict[str, Any],
) -> list[tuple[str, Path]]:
    if args.eval_data:
        return [
            (name, resolve_path(path))
            for name, path in parse_named_paths(args.eval_data)
        ]

    val_path = resolved_config.get("data", {}).get("val_path")
    if not val_path:
        return []
    return [("val", resolve_path(val_path))]


def resolve_checkpoint_path(
    args: argparse.Namespace,
    checkpoints_dir: Path,
    train_summary: dict[str, Any] | None,
) -> Path | None:
    if args.checkpoint:
        return resolve_path(args.checkpoint)

    if train_summary and train_summary.get("best_checkpoint"):
        return Path(train_summary["best_checkpoint"])

    candidates = sorted(checkpoints_dir.glob("guard-best-*.ckpt"))
    if candidates:
        return candidates[-1]

    final_checkpoint = checkpoints_dir / "final.ckpt"
    if final_checkpoint.exists():
        return final_checkpoint

    return None


def resolve_model_path(
    args: argparse.Namespace,
    run_dir: Path,
    train_summary: dict[str, Any] | None,
) -> Path | None:
    if args.model_path:
        return resolve_path(args.model_path)

    default_export_dir = run_dir / "export" / "hf_model"
    if default_export_dir.exists():
        return default_export_dir

    if train_summary and train_summary.get("hf_export_dir"):
        export_dir = Path(train_summary["hf_export_dir"])
        if export_dir.exists():
            return export_dir

    return None


def dry_run_checkpoint_placeholder(args: argparse.Namespace, checkpoints_dir: Path) -> Path:
    if args.checkpoint:
        return resolve_path(args.checkpoint)
    return checkpoints_dir / "guard-best-placeholder.ckpt"


def dry_run_model_placeholder(args: argparse.Namespace, run_dir: Path) -> Path:
    if args.model_path:
        return resolve_path(args.model_path)
    return run_dir / "export" / "hf_model"


def collect_eval_summaries(eval_dir: Path, eval_datasets: list[tuple[str, Path]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for name, _ in eval_datasets:
        metrics_path = eval_dir / f"{name}_metrics.json"
        if metrics_path.exists():
            summaries[name] = load_json(metrics_path)
    return summaries


def main() -> None:
    args = parse_args()

    config_path = resolve_path(args.config)
    base_config = load_yaml(config_path)
    overrides = parse_override_strings(args.overrides)
    resolved_config = apply_overrides(base_config, overrides)
    if args.seed is not None:
        resolved_config.setdefault("training", {})["seed"] = args.seed

    run_root = resolve_path(args.run_root)
    run_name = build_run_name(resolved_config, overrides, args.run_name)
    run_dir = run_root / run_name
    checkpoints_dir = run_dir / "checkpoints"
    export_dir = run_dir / "export" / "hf_model"
    eval_dir = run_dir / "eval"
    resolved_config.setdefault("output", {})["dir"] = str(checkpoints_dir)

    resolved_config_path = run_dir / "resolved_config.yaml"
    summary_path = run_dir / "experiment_summary.json"

    existing_summary = load_summary_if_exists(summary_path)
    if (
        args.skip_existing
        and existing_summary
        and existing_summary.get("status") == "completed"
    ):
        print(f"Skipping completed run: {run_name}")
        return

    eval_datasets = resolve_eval_datasets(args, resolved_config)
    commands: list[str] = []
    summary: dict[str, Any] = {
        "status": "running",
        "created_at": now_iso(),
        "run_name": run_name,
        "run_dir": str(run_dir),
        "config_path": str(config_path),
        "resolved_config_path": str(resolved_config_path),
        "overrides": overrides,
        "devices": args.devices,
        "num_nodes": args.num_nodes,
        "strategy": args.strategy,
        "logger": args.logger,
        "seed": args.seed,
        "commands": commands,
        "dry_run": args.dry_run,
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(resolved_config_path, resolved_config)
    save_json(summary_path, summary)

    try:
        if not args.skip_train:
            train_command = [
                args.python,
                "train.py",
                "--config",
                str(resolved_config_path),
                "--devices",
                str(args.devices),
                "--num_nodes",
                str(args.num_nodes),
                "--strategy",
                args.strategy,
                "--logger",
                args.logger,
                "--skip-auto-export",
            ]
            if args.resume:
                train_command.extend(["--resume", args.resume])
            if args.seed is not None:
                train_command.extend(["--seed", str(args.seed)])
            commands.append(stringify_command(train_command))
            run_command(train_command, cwd=REPO_ROOT, dry_run=args.dry_run)

        train_summary = load_summary_if_exists(checkpoints_dir / "train_summary.json")
        checkpoint_path = resolve_checkpoint_path(args, checkpoints_dir, train_summary)
        if checkpoint_path is None and args.dry_run:
            checkpoint_path = dry_run_checkpoint_placeholder(args, checkpoints_dir)

        if not args.skip_export:
            if checkpoint_path is None:
                raise FileNotFoundError("Could not resolve a checkpoint to export")
            export_command = [
                args.python,
                "export_checkpoint.py",
                "--config",
                str(resolved_config_path),
                "--checkpoint",
                str(checkpoint_path),
                "--output-dir",
                str(export_dir),
                "--summary-file",
                str(run_dir / "export" / "export_summary.json"),
            ]
            commands.append(stringify_command(export_command))
            run_command(export_command, cwd=REPO_ROOT, dry_run=args.dry_run)

        model_path = resolve_model_path(args, run_dir, train_summary)
        if model_path is None and args.dry_run:
            model_path = dry_run_model_placeholder(args, run_dir)
        if not args.skip_eval:
            if model_path is None:
                raise FileNotFoundError("Could not resolve an exported HF model for evaluation")
            if not eval_datasets:
                raise ValueError("No evaluation datasets configured; pass --eval-data or set data.val_path")

            eval_dir.mkdir(parents=True, exist_ok=True)
            for dataset_name, dataset_path in eval_datasets:
                eval_command = [
                    args.python,
                    "eval.py",
                    "--model-path",
                    str(model_path),
                    "--data-path",
                    str(dataset_path),
                    "--batch-size",
                    str(args.eval_batch_size),
                    "--max-new-tokens",
                    str(args.eval_max_new_tokens),
                    "--dtype",
                    args.eval_dtype,
                    "--metrics-file",
                    str(eval_dir / f"{dataset_name}_metrics.json"),
                ]
                if args.eval_max_input_length is not None:
                    eval_command.extend(
                        ["--max-input-length", str(args.eval_max_input_length)]
                    )
                if args.eval_limit is not None:
                    eval_command.extend(["--limit", str(args.eval_limit)])
                if args.eval_device:
                    eval_command.extend(["--device", args.eval_device])
                if args.disable_thinking:
                    eval_command.append("--disable-thinking")
                if args.save_predictions:
                    eval_command.extend(
                        [
                            "--output-file",
                            str(eval_dir / f"{dataset_name}_predictions.jsonl"),
                        ]
                    )
                commands.append(stringify_command(eval_command))
                run_command(eval_command, cwd=REPO_ROOT, dry_run=args.dry_run)

        export_summary = load_summary_if_exists(run_dir / "export" / "export_summary.json")
        eval_summaries = collect_eval_summaries(eval_dir, eval_datasets)
        summary.update(
            {
                "status": "completed" if not args.dry_run else "dry_run",
                "completed_at": now_iso(),
                "train": train_summary,
                "export": export_summary,
                "evaluations": eval_summaries,
            }
        )
        save_json(summary_path, summary)
    except Exception as exc:
        summary.update(
            {
                "status": "failed",
                "completed_at": now_iso(),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        save_json(summary_path, summary)
        raise


if __name__ == "__main__":
    main()
