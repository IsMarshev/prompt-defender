from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import yaml


_NON_ALNUM_RE = re.compile(r"[^a-zA-Z0-9]+")
_EXPERIMENT_NAME_RE = re.compile(r"^(?P<model>.+)-experiment-(?P<number>\d+)$")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=False, sort_keys=False)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_override_strings(items: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got: {item!r}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Override key is empty: {item!r}")
        try:
            parsed_value = json.loads(raw_value)
        except json.JSONDecodeError:
            parsed_value = yaml.safe_load(raw_value)
        overrides[key] = parsed_value
    return overrides


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    resolved = deepcopy(config)
    for key, value in overrides.items():
        cursor = resolved
        parts = key.split(".")
        for part in parts[:-1]:
            next_value = cursor.get(part)
            if not isinstance(next_value, dict):
                next_value = {}
                cursor[part] = next_value
            cursor = next_value
        cursor[parts[-1]] = value
    return resolved


def flatten_dict(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flattened.update(flatten_dict(value, path))
        else:
            flattened[path] = value
    return flattened


def slugify(value: Any) -> str:
    text = str(value).strip().lower().replace("/", "-")
    text = _NON_ALNUM_RE.sub("-", text)
    text = text.strip("-")
    return text or "value"


def backbone_slug(backbone: str) -> str:
    return slugify(backbone.split("/")[-1] if "/" in backbone else backbone)


def backbone_display_name(backbone: str) -> str:
    text = str(backbone).strip()
    if "/" in text:
        text = text.split("/")[-1]
    text = text.replace("\\", "-").replace("/", "-").strip()
    return text or "model"


def infer_experiment_root(checkpoints_dir: str | Path) -> Path:
    ckpt_dir = Path(checkpoints_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = (Path.cwd() / ckpt_dir).resolve()
    else:
        ckpt_dir = ckpt_dir.resolve()

    run_dir = ckpt_dir.parent
    if ckpt_dir.name == "checkpoints" and (
        (run_dir / "resolved_config.yaml").exists()
        or (run_dir / "experiment_summary.json").exists()
    ):
        return ckpt_dir.parent.parent
    return ckpt_dir.parent


def next_experiment_name(model_name: str, experiment_root: str | Path) -> str:
    display_name = backbone_display_name(model_name)
    root = Path(experiment_root)
    logs_root = root / "logs"
    highest_number = 0
    registry_path = logs_root / "experiment_registry.json"

    if logs_root.exists():
        for candidate in logs_root.iterdir():
            match = _EXPERIMENT_NAME_RE.match(candidate.name)
            if not match:
                continue
            if match.group("model") != display_name:
                continue
            highest_number = max(highest_number, int(match.group("number")))

    if registry_path.exists():
        registry = load_json(registry_path)
        highest_number = max(
            highest_number,
            int(registry.get("models", {}).get(display_name, 0)),
        )

    return f"{display_name}-experiment-{highest_number + 1}"


def reserve_experiment_name(model_name: str, experiment_root: str | Path) -> tuple[str, Path]:
    root = Path(experiment_root)
    logs_root = root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    registry_path = logs_root / "experiment_registry.json"

    experiment_name = next_experiment_name(model_name, root)
    match = _EXPERIMENT_NAME_RE.match(experiment_name)
    if not match:
        raise ValueError(f"Generated invalid experiment name: {experiment_name}")

    registry: dict[str, Any]
    if registry_path.exists():
        registry = load_json(registry_path)
    else:
        registry = {"models": {}}

    models = registry.setdefault("models", {})
    models[match.group("model")] = int(match.group("number"))
    save_json(registry_path, registry)

    return experiment_name, registry_path


def build_run_name(
    resolved_config: dict[str, Any],
    overrides: dict[str, Any],
    explicit_name: str | None = None,
) -> str:
    if explicit_name:
        return explicit_name

    pieces = [
        timestamp_tag(),
        backbone_slug(resolved_config.get("model", {}).get("backbone", "model")),
    ]
    for key in [
        "training.learning_rate",
        "training.batch_size",
        "training.gradient_accumulation_steps",
    ]:
        if key in overrides:
            pieces.append(f"{key.split('.')[-1]}-{slugify(overrides[key])}")

    for key, value in sorted(overrides.items()):
        short_key = key.split(".")[-1]
        token = f"{short_key}-{slugify(value)}"
        if token not in pieces:
            pieces.append(token)
        if len(pieces) >= 6:
            break

    return "-".join(pieces)


def parse_named_paths(items: list[str]) -> list[tuple[str, str]]:
    named_paths: list[tuple[str, str]] = []
    for item in items:
        if "=" in item:
            name, path = item.split("=", 1)
        else:
            path = item
            name = Path(path).stem
        named_paths.append((slugify(name), path))
    return named_paths


def expand_matrix(matrix: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not matrix:
        return [{}]

    keys = list(matrix.keys())
    values: list[list[Any]] = []
    for key in keys:
        options = matrix[key]
        if not isinstance(options, list) or not options:
            raise ValueError(f"Matrix key {key!r} must contain a non-empty list")
        values.append(options)

    return [
        dict(zip(keys, combination))
        for combination in product(*values)
    ]


def render_name_template(template: str, payload: dict[str, Any]) -> str:
    context = {
        key.replace(".", "_"): slugify(value)
        for key, value in flatten_dict(payload).items()
    }
    return template.format(**context)
