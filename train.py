import argparse
import json
import os
import random
import tempfile

import numpy as np
import torch
import yaml
from transformers import set_seed

from prompt_guard.data.dataset import (
    PackedBatchSampler,
    PackedDataCollator,
    PromptSafetyDataset,
)
from prompt_guard.evaluation.eval import run_evaluation
from prompt_guard.model.model import ModelConfig, load_model_and_tokenizer
from prompt_guard.training.sft import (
    run_sft,
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


def _split_train_val(samples: list, ratio: float, seed: int) -> tuple[list, list]:
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    n_val = max(1, int(len(samples) * ratio))
    val_idx = set(indices[:n_val])
    train = [s for i, s in enumerate(samples) if i not in val_idx]
    val = [s for i, s in enumerate(samples) if i in val_idx]
    return train, val



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

    max_length = cfg.get("max_length", 2048)

    # -- Datasets --
    print(f"\nLoading train data: {cfg['train_file']}")
    train_dataset = PromptSafetyDataset(
        cfg["train_file"], tokenizer, max_length=max_length, log_distribution=True
    )

    val_dataset = None
    if cfg.get("val_file"):
        print(f"Loading val data: {cfg['val_file']}")
        val_dataset = PromptSafetyDataset(
            cfg["val_file"], tokenizer, max_length=max_length, log_distribution=True
        )
    else:
        ratio = cfg.get("val_split_ratio", 0.02)
        print(f"\nSplitting {ratio * 100:.1f}% of train data as hold-out val set...")
        train_samples, val_samples = _split_train_val(
            train_dataset.samples, ratio=ratio, seed=seed
        )
        train_dataset.set_samples(train_samples)
        val_dataset = PromptSafetyDataset.from_samples(
            val_samples, tokenizer, max_length=max_length
        )

    # -- Verify individual sample loss masking --
    verify_loss_masking(train_dataset, n=2)

    # -- Dry-run: verify 3 packed batches then exit --
    if args.dry_run:
        print("\n[DRY RUN] Verifying first 3 packed batches, then exiting.\n")
        sampler = PackedBatchSampler(
            dataset=train_dataset,
            max_length=max_length,
            shuffle=True,
            seed=seed,
        )
        collator = PackedDataCollator(
            tokenizer=tokenizer,
            max_length=max_length,
            use_fa2=use_fa2,
        )
        verify_packed_batches(train_dataset, sampler, collator, n_batches=3)
        print("[DRY RUN] Done. Exiting before training.")
        return

    # -- Train --
    trainer = run_sft(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        cfg=cfg,
        use_fa2=use_fa2,
        use_wandb=args.use_wandb,
    )

    print("\nStarting training...")
    trainer.train()

    # -- Save final model --
    final_dir = os.path.join(cfg["output_dir"], "final")
    print(f"\nSaving final model to {final_dir}")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    # -- Evaluate on val set --
    if val_dataset is not None:
        print("\nRunning final evaluation on val set...")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as tmp:
            for s in val_dataset.samples:
                label_bin = "Unsafe" if s["label"] in ("Unsafe", "Controversial") else "Safe"
                tmp.write(json.dumps({"text": s["user_message"], "label": label_bin}) + "\n")
            tmp_path = tmp.name

        run_evaluation(
            model_path=final_dir,
            benchmarks=[("val", tmp_path)],
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        os.unlink(tmp_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
