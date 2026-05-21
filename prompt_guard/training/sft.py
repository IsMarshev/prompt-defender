import os
from typing import Optional

import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_cosine_schedule_with_warmup,
)
from typing import NamedTuple

class EvalPrediction(NamedTuple):
    predictions: "np.ndarray"
    label_ids: "np.ndarray"

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

        labels = batch["labels"][0]
        position_ids = batch["position_ids"][0]
        n_real = sum(dataset.lengths[i] for i in bin_indices)
        n_label_tokens = (labels != -100).sum().item()

        print(f"\n  Batch {batch_idx + 1}:")
        print(f"    Packed samples:       {len(bin_indices)}")
        print(
            f"    Real tokens:          {n_real}/{collator.max_length} "
            f"({100 * n_real / collator.max_length:.1f}% efficiency)"
        )
        print(f"    Non-(-100) labels:    {n_label_tokens}")

        cumlen = 0
        ok = True
        for idx in bin_indices:
            if position_ids[cumlen].item() != 0:
                ok = False
            cumlen += dataset.lengths[idx]

        print(f"    Position IDs restart: {'OK' if ok else 'FAIL'}")

        if "attention_mask" in batch:
            mask = batch["attention_mask"][0, 0]
            size = min(8, mask.shape[0])
            print(f"    Attention mask [{size}x{size} top-left slice]:")
            for row in mask[:size, :size].cpu():
                print("      " + "  ".join(
                    "  0" if v == 0.0 else "-inf" for v in row.tolist()
                ))

    print("---------------------------------\n")


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class PromptSafetyModule(L.LightningModule):
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: PromptSafetyDataset,
        val_dataset: Optional[PromptSafetyDataset],
        cfg: dict,
        use_fa2: bool = False,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.use_fa2 = use_fa2
        self._val_outputs: list[dict] = []
        self._packed_sampler: Optional[PackedBatchSampler] = None

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self.model(**batch).loss
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        with torch.no_grad():
            outputs = self.model(**batch)
        self._val_outputs.append({
            "loss": outputs.loss.detach().cpu(),
            "logits": outputs.logits.detach().cpu(),
            "labels": batch["labels"].detach().cpu(),
        })

    def on_validation_epoch_end(self) -> None:
        if not self._val_outputs:
            return

        avg_loss = torch.stack([o["loss"] for o in self._val_outputs]).mean()
        self.log("val/loss", avg_loss, prog_bar=True, sync_dist=True)

        # Decode per sample to avoid seq_len mismatch when batches are padded
        # to different lengths by DataCollatorForSafety.
        y_true: list[int] = []
        y_pred: list[int] = []
        for o in self._val_outputs:
            pred_ids = o["logits"].argmax(-1)  # (B, T)
            label_ids = o["labels"]            # (B, T)
            for pred_row, label_row in zip(pred_ids, label_ids):
                mask = label_row != -100
                pred_text = self.tokenizer.decode(
                    pred_row[mask].tolist(), skip_special_tokens=True
                )
                label_text = self.tokenizer.decode(
                    label_row[mask].tolist(), skip_special_tokens=True
                )
                pred_label, _ = _parse_output(pred_text)
                true_label, _ = _parse_output(label_text)
                y_pred.append(LABEL2ID.get(pred_label, 1))
                y_true.append(LABEL2ID.get(true_label, 1))

        metrics = {
            "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        }
        self.log_dict(
            {f"val/{k}": v for k, v in metrics.items()},
            prog_bar=True,
            sync_dist=True,
        )
        self._val_outputs.clear()

    def on_train_epoch_start(self) -> None:
        if self._packed_sampler is not None:
            self._packed_sampler.set_epoch(self.current_epoch)

    def train_dataloader(self) -> DataLoader:
        max_length = self.cfg.get("max_length", 2048)
        self._packed_sampler = PackedBatchSampler(
            self.train_dataset,
            max_length=max_length,
            shuffle=True,
            seed=self.cfg.get("seed", 42),
        )
        packed_collator = PackedDataCollator(
            tokenizer=self.tokenizer,
            max_length=max_length,
            use_fa2=self.use_fa2,
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=self._packed_sampler,
            collate_fn=packed_collator,
            num_workers=self.cfg.get("dataloader_num_workers", 0),
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        eval_collator = DataCollatorForSafety(self.tokenizer, pad_to_multiple_of=8)
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.get("per_device_eval_batch_size", 4),
            collate_fn=eval_collator,
            shuffle=False,
            num_workers=self.cfg.get("dataloader_num_workers", 0),
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.cfg.get("learning_rate", 2e-5)),
            weight_decay=float(self.cfg.get("weight_decay", 0.01)),
            eps=1e-8,
        )

        max_length = self.cfg.get("max_length", 2048)
        grad_accum = self.cfg.get("gradient_accumulation_steps", 16)
        num_epochs = self.cfg.get("num_train_epochs", 3)

        # Estimate number of optimizer steps for the LR scheduler
        est_sampler = PackedBatchSampler(
            self.train_dataset, max_length=max_length, shuffle=False
        )
        steps_per_epoch = max(1, len(est_sampler) // grad_accum)
        total_steps = steps_per_epoch * num_epochs
        warmup_steps = max(1, int(total_steps * self.cfg.get("warmup_ratio", 0.05)))

        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


# ---------------------------------------------------------------------------
# Trainer factory
# ---------------------------------------------------------------------------

def build_lightning_trainer(
    cfg: dict,
    use_wandb: bool = False,
    has_val: bool = True,
) -> L.Trainer:
    callbacks: list = [LearningRateMonitor(logging_interval="step")]

    if has_val:
        callbacks += [
            ModelCheckpoint(
                dirpath=os.path.join(cfg["output_dir"], "checkpoints"),
                monitor="val/f1",
                mode="max",
                save_top_k=cfg.get("save_total_limit", 3),
                filename="epoch={epoch}-step={step}-f1={val/f1:.4f}",
                save_last=True,
            ),
            EarlyStopping(
                monitor="val/f1",
                patience=cfg.get("early_stopping_patience", 3),
                mode="max",
            ),
        ]
    else:
        callbacks.append(
            ModelCheckpoint(
                dirpath=os.path.join(cfg["output_dir"], "checkpoints"),
                save_last=True,
                every_n_train_steps=cfg.get("save_steps", 500),
            )
        )

    if use_wandb:
        logger = WandbLogger(
            project=cfg.get("wandb_project", "prompt-defender"),
            name=cfg.get("wandb_run_name", None),
            config=cfg,
        )
    else:
        logger = True  # default TensorBoard logger

    return L.Trainer(
        max_epochs=cfg.get("num_train_epochs", 3),
        precision="bf16-mixed" if cfg.get("bf16", True) else "32-true",
        accumulate_grad_batches=cfg.get("gradient_accumulation_steps", 16),
        gradient_clip_val=cfg.get("max_grad_norm", 1.0),
        log_every_n_steps=cfg.get("logging_steps", 10),
        val_check_interval=cfg.get("eval_steps", 500) if has_val else 1.0,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        default_root_dir=cfg["output_dir"],
    )
