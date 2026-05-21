from typing import Optional

import numpy as np
import wandb
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import (
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import EvalPrediction

from prompt_guard.data.dataset import (
    DataCollatorForSafety,
    PackedBatchSampler,
    PackedDataCollator,
    PromptSafetyDataset,
)
from prompt_guard.inference.classifier import _parse_output

LABEL2ID = {"Safe": 0, "Unsafe": 1, "Controversial": 2}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _decode_predictions(
    predictions: np.ndarray,
    tokenizer: PreTrainedTokenizerBase,
) -> list[str]:
    token_ids = predictions.argmax(-1) if predictions.ndim == 3 else predictions
    results = []
    for row in token_ids:
        valid = row[row != -100]
        results.append(tokenizer.decode(valid, skip_special_tokens=True))
    return results


def build_compute_metrics(tokenizer: PreTrainedTokenizerBase):
    def compute_metrics(eval_pred: EvalPrediction) -> dict:
        predictions, labels = eval_pred
        label_texts = _decode_predictions(labels, tokenizer)
        pred_texts = _decode_predictions(predictions, tokenizer)

        y_true, y_pred = [], []
        for lt, pt in zip(label_texts, pred_texts):
            true_label, _ = _parse_output(lt)
            pred_label, _ = _parse_output(pt)
            y_true.append(LABEL2ID.get(true_label, 1))
            y_pred.append(LABEL2ID.get(pred_label, 1))

        return {
            "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        }

    return compute_metrics


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def verify_loss_masking(dataset: PromptSafetyDataset, n: int = 2) -> None:
    print("\n--- Loss masking verification ---")
    for i in range(min(n, len(dataset))):
        sample = dataset[i]
        labels = sample["labels"]
        input_ids = sample["input_ids"]
        n_masked = (labels == -100).sum().item()
        n_total = labels.size(0)
        n_target = n_total - n_masked

        first_nonneg = (labels != -100).nonzero(as_tuple=True)[0]
        split_idx = first_nonneg[0].item() if len(first_nonneg) > 0 else -1
        target_preview = dataset.tokenizer.decode(
            input_ids[split_idx:].tolist(), skip_special_tokens=True
        )
        print(f"  Sample {i}: total={n_total}, masked={n_masked}, target_tokens={n_target}")
        print(f"  Target text preview: {repr(target_preview[:80])}")
    print("---------------------------------\n")


def verify_packed_batches(
    dataset: PromptSafetyDataset,
    sampler: PackedBatchSampler,
    collator: PackedDataCollator,
    n_batches: int = 3,
) -> None:
    print("\n--- Packed batch verification ---")
    sampler_iter = iter(sampler)

    for batch_idx in range(n_batches):
        try:
            bin_indices = next(sampler_iter)
        except StopIteration:
            break

        samples = [dataset[i] for i in bin_indices]
        batch = collator(samples)

        labels = batch["labels"][0]          # (max_length,)
        position_ids = batch["position_ids"][0]  # (max_length,)
        n_real = sum(dataset.lengths[i] for i in bin_indices)
        n_label_tokens = (labels != -100).sum().item()

        print(f"\n  Batch {batch_idx + 1}:")
        print(f"    Packed samples:       {len(bin_indices)}")
        print(
            f"    Real tokens:          {n_real}/{collator.max_length} "
            f"({100 * n_real / collator.max_length:.1f}% efficiency)"
        )
        print(f"    Non-(-100) labels:    {n_label_tokens}")

        # Verify position_ids restart from 0 at each sample boundary.
        cumlen = 0
        ok = True
        for k, idx in enumerate(bin_indices):
            start_pos = position_ids[cumlen].item()
            if start_pos != 0:
                ok = False
            cumlen += dataset.lengths[idx]

        print(f"    Position IDs restart: {'OK' if ok else 'FAIL'}")
        if not ok:
            cumlen = 0
            for k, idx in enumerate(bin_indices):
                print(
                    f"      sample[{k}] offset={cumlen} "
                    f"pos[offset]={position_ids[cumlen].item()}"
                )
                cumlen += dataset.lengths[idx]

        if "attention_mask" in batch:
            mask = batch["attention_mask"][0, 0]  # (T, T)
            size = min(8, mask.shape[0])
            print(f"    Attention mask [{size}x{size} top-left slice]:")
            slice_str = mask[:size, :size].cpu()
            for row in slice_str:
                print("      " + "  ".join(
                    "  0" if v == 0.0 else "-inf" for v in row.tolist()
                ))

    print("---------------------------------\n")


# ---------------------------------------------------------------------------
# Training arguments
# ---------------------------------------------------------------------------

def build_training_args(cfg: dict) -> TrainingArguments:
    report_to = "wandb" if cfg.get("use_wandb", False) else "none"

    return TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg.get("num_train_epochs", 3),
        # With packed batches each DataLoader item is one full packed sequence;
        # batching is controlled by PackedBatchSampler, not Trainer.
        per_device_train_batch_size=1,
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 16),
        learning_rate=float(cfg.get("learning_rate", 2e-5)),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.05),
        weight_decay=cfg.get("weight_decay", 0.01),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        bf16=cfg.get("bf16", True),
        logging_steps=cfg.get("logging_steps", 10),
        eval_strategy=cfg.get("eval_strategy", "steps"),
        eval_steps=cfg.get("eval_steps", 500),
        save_strategy=cfg.get("save_strategy", "steps"),
        save_steps=cfg.get("save_steps", 500),
        load_best_model_at_end=cfg.get("load_best_model_at_end", True),
        metric_for_best_model=cfg.get("metric_for_best_model", "f1"),
        greater_is_better=True,
        seed=cfg.get("seed", 42),
        report_to=report_to,
        save_total_limit=3,
        dataloader_num_workers=cfg.get("dataloader_num_workers", 0),
        remove_unused_columns=False,
        predict_with_generate=False,
        include_inputs_for_metrics=False,
    )


# ---------------------------------------------------------------------------
# Trainer subclass that injects the packed train DataLoader
# ---------------------------------------------------------------------------

class PackedTrainer(Trainer):
    def __init__(
        self,
        *args,
        packed_sampler: PackedBatchSampler,
        packed_collator: PackedDataCollator,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._packed_sampler = packed_sampler
        self._packed_collator = packed_collator

    def get_train_dataloader(self) -> DataLoader:
        # Trainer's internal _inner_training_loop calls
        # batch_sampler.set_epoch(epoch) automatically when the attribute exists.
        return DataLoader(
            self.train_dataset,
            batch_sampler=self._packed_sampler,
            collate_fn=self._packed_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def run_sft(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: PromptSafetyDataset,
    val_dataset: Optional[PromptSafetyDataset],
    cfg: dict,
    use_fa2: bool = False,
    use_wandb: bool = False,
) -> PackedTrainer:
    if use_wandb:
        wandb.init(
            project=cfg.get("wandb_project", "prompt-defender"),
            name=cfg.get("wandb_run_name", None),
            config=cfg,
        )

    cfg["use_wandb"] = use_wandb
    max_length = cfg.get("max_length", 2048)

    packed_sampler = PackedBatchSampler(
        dataset=train_dataset,
        max_length=max_length,
        shuffle=True,
        seed=cfg.get("seed", 42),
    )
    packed_collator = PackedDataCollator(
        tokenizer=tokenizer,
        max_length=max_length,
        use_fa2=use_fa2,
    )
    # Standard padding collator for eval (Trainer's default get_eval_dataloader uses this).
    eval_collator = DataCollatorForSafety(tokenizer=tokenizer, pad_to_multiple_of=8)

    training_args = build_training_args(cfg)

    callbacks = []
    if cfg.get("load_best_model_at_end", True) and val_dataset is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

    trainer = PackedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=eval_collator,
        packed_sampler=packed_sampler,
        packed_collator=packed_collator,
        compute_metrics=build_compute_metrics(tokenizer) if val_dataset else None,
        callbacks=callbacks,
    )

    return trainer
