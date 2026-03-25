import argparse
import json
from pathlib import Path

import torch

from experiment_utils import load_yaml, now_iso, save_json
from model import PromptGuardGenModel


def load_config(path: str) -> dict:
    return load_yaml(path)


def extract_model_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    extracted = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            extracted[key[len("model."):]] = value
    if not extracted:
        raise ValueError("Could not find model.* keys in checkpoint state_dict")
    return extracted


def main():
    parser = argparse.ArgumentParser(
        description="Export a Lightning checkpoint to Hugging Face save_pretrained format."
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for exported model")
    parser.add_argument(
        "--summary-file",
        type=str,
        default=None,
        help="Optional path to export summary JSON",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    bundle = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = bundle.get("state_dict", bundle)
    model_state = extract_model_state_dict(state_dict)

    wrapper = PromptGuardGenModel(
        model_name=cfg["model"]["backbone"],
        freeze_backbone=False,
    )
    missing, unexpected = wrapper.load_state_dict(model_state, strict=False)
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    wrapper.model.save_pretrained(output_dir)
    wrapper.tokenizer.save_pretrained(output_dir)

    export_meta = {
        "source_checkpoint": str(checkpoint_path),
        "base_model": cfg["model"]["backbone"],
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }
    with (output_dir / "export_meta.json").open("w", encoding="utf-8") as f:
        json.dump(export_meta, f, ensure_ascii=False, indent=2)

    export_summary = {
        "status": "completed",
        "created_at": now_iso(),
        "config_path": args.config,
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "base_model": cfg["model"]["backbone"],
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "export_meta_file": str(output_dir / "export_meta.json"),
    }
    summary_path = Path(args.summary_file) if args.summary_file else output_dir / "export_summary.json"
    save_json(summary_path, export_summary)

    print(f"Exported model to {output_dir}")


if __name__ == "__main__":
    main()
