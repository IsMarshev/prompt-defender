import argparse
import os
import random

import numpy as np
import torch
import yaml
from transformers import set_seed

from prompt_guard.data.dataset import PromptSafetyDataset, PackedBatchSampler, PackedDataCollator
from prompt_guard.evaluation.eval import run_evaluation
from prompt_guard.model.model import ModelConfig, load_model_and_tokenizer
from prompt_guard.training.sft import (
    PromptSafetyModule,
    build_lightning_trainer,
    verify_loss_masking,
    verify_packed_batches,
)


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def _lazy_split(
    file_path: str,
    tokenizer,
    max_length: int,
    ratio: float,
    seed: int,
    eval_n: int,
) -> tuple[PromptSafetyDataset, PromptSafetyDataset]:
    """
    Scan the JSONL file once, split by index, and return:
      - train: lazy file-backed dataset
      - val:   eval_n random samples fully in memory
    No sample text is kept in RAM for the train split.
    """
    # Single scan — builds offset map and lengths for all lines
    base = PromptSafetyDataset(file_path, tokenizer, max_length, log_distribution=False)
    n_total = len(base)

    rng = random.Random(seed)
    all_idx = list(range(n_total))
    rng.shuffle(all_idx)

    n_val = max(1, int(n_total * ratio))
    val_file_idx = all_idx[:n_val]
    train_file_idx = all_idx[n_val:]

    # Val: load eval_n random samples into memory
    chosen_val = rng.sample(val_file_idx, min(eval_n, len(val_file_idx)))
    val_samples = [base._read_file_sample(i) for i in chosen_val]
    val_dataset = PromptSafetyDataset.from_samples(
        val_samples, tokenizer, max_length, log_distribution=True
    )

    # Train: lazy subset — reuses the already-scanned offset map, no re-scan
    train_dataset = base.with_indices(train_file_idx)

    return train_dataset, val_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PromptDefender safety classifier")
    parser.add_argument("--config", default="prompt_guard/configs/train_config.yaml")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Verify pipeline on 3 packed batches, then exit.",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    _set_all_seeds(seed)

    # -- Model & tokenizer --
    model_cfg = ModelConfig(
        model_name_or_path=cfg["model_name_or_path"],
        bf16=cfg.get("bf16", True),
        use_lora=cfg.get("use_lora", False),
    )
    print(f"Loading model: {cfg['model_name_or_path']}")
    model, tokenizer, use_fa2 = load_model_and_tokenizer(model_cfg)
    model.train()
    max_length = cfg.get("max_length", 2048)
    eval_n = cfg.get("eval_subset_size", 128)

    # -- Datasets --
    if cfg.get("val_file"):
        print(f"\nLoading train data (lazy): {cfg['train_file']}")
        train_dataset = PromptSafetyDataset(
            cfg["train_file"], tokenizer, max_length=max_length, log_distribution=True
        )
        print(f"Loading val subset ({eval_n} samples): {cfg['val_file']}")
        val_dataset = PromptSafetyDataset.random_eval_subset(
            cfg["val_file"],
            tokenizer,
            n=eval_n,
            seed=seed,
            max_length=max_length,
        )
    else:
        ratio = cfg.get("val_split_ratio", 0.02)
        print(f"\nLoading train data (lazy, {ratio*100:.1f}% split): {cfg['train_file']}")
        train_dataset, val_dataset = _lazy_split(
            cfg["train_file"], tokenizer, max_length, ratio, seed, eval_n
        )

    verify_loss_masking(train_dataset, n=2)

    # -- Dry-run --
    if args.dry_run:
        print("\n[DRY RUN] Verifying first 3 packed batches, then exiting.\n")
        sampler = PackedBatchSampler(
            dataset=train_dataset, max_length=max_length, shuffle=True, seed=seed
        )
        collator = PackedDataCollator(
            tokenizer=tokenizer, max_length=max_length, use_fa2=use_fa2
        )
        verify_packed_batches(train_dataset, sampler, collator, n_batches=3)
        print("[DRY RUN] Done. Exiting before training.")
        return

    # -- Train --
    module = PromptSafetyModule(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        cfg=cfg,
        use_fa2=use_fa2,
    )
    trainer = build_lightning_trainer(cfg, use_wandb=args.use_wandb, has_val=True)

    print("\nStarting training...")
    trainer.fit(module)

    # -- Save final model --
    final_dir = os.path.join(cfg["output_dir"], "final")
    os.makedirs(final_dir, exist_ok=True)
    print(f"\nSaving final model to {final_dir}")
    module.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # -- Post-training eval on the held-out val subset --
    print("\nRunning final evaluation on val subset...")
    run_evaluation(
        model_path=final_dir,
        benchmarks=[("val", _write_eval_jsonl(val_dataset, final_dir))],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


def _write_eval_jsonl(dataset: PromptSafetyDataset, out_dir: str) -> str:
    """Dump val samples to a temp JSONL for run_evaluation."""
    import json
    path = os.path.join(out_dir, "_eval_tmp.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for s in dataset._samples or []:
            label_bin = "Unsafe" if s["label"] in ("Unsafe", "Controversial") else "Safe"
            f.write(json.dumps({"text": s["user_message"], "label": label_bin}) + "\n")
    return path


if __name__ == "__main__":
    main()
