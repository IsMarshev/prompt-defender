"""
Generative Qwen3Guard training with PyTorch Lightning.

Single GPU:
    python train.py --config config.yaml

Multi-GPU (DDP):
    python train.py --config config.yaml --devices 4 --strategy ddp

Multi-node:
    python train.py --config config.yaml --devices 4 --strategy ddp --num_nodes 2
"""

import argparse
import re
from pathlib import Path

import yaml
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

from model import PromptGuardGenModel
from dataset import SAFETY_LABEL_TO_ID, build_dataloader


UNKNOWN_SAFETY_LABEL_ID = len(SAFETY_LABEL_TO_ID)
SAFETY_PATTERN = re.compile(
    r'(?i)(?:^|[\{\n,])\s*"?safety"?\s*[:=]\s*"?'
    r"(safe|unsafe|controversial)\b"
)


def parse_safety_label(text: str) -> int:
    match = SAFETY_PATTERN.search(text)
    if not match:
        return UNKNOWN_SAFETY_LABEL_ID

    normalized = match.group(1).strip().lower().capitalize()
    return SAFETY_LABEL_TO_ID.get(normalized, UNKNOWN_SAFETY_LABEL_ID)


def summarize_confusion(confusion: torch.Tensor) -> dict[str, torch.Tensor]:
    confusion = confusion.to(dtype=torch.float32)
    class_count = len(SAFETY_LABEL_TO_ID)

    f1_scores = []
    recalls = []
    for class_idx in range(class_count):
        tp = confusion[class_idx, class_idx]
        fp = confusion[:, class_idx].sum() - tp
        fn = confusion[class_idx, :].sum() - tp

        recalls.append(tp / (tp + fn).clamp_min(1.0))
        f1_scores.append((2.0 * tp) / (2.0 * tp + fp + fn).clamp_min(1.0))

    total = confusion.sum().clamp_min(1.0)
    parsed = confusion[:, :class_count].sum()
    correct = confusion[:, :class_count].diag().sum()

    return {
        "macro_f1": torch.stack(f1_scores).mean(),
        "macro_recall": torch.stack(recalls).mean(),
        "accuracy": correct / total,
        "parse_rate": parsed / total,
    }


def compute_metrics(pred_texts: list[str], true_label_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    class_count = len(SAFETY_LABEL_TO_ID)
    confusion = torch.zeros(
        (class_count, class_count + 1),
        dtype=torch.long,
        device=true_label_ids.device,
    )

    for pred_text, true_label_id in zip(pred_texts, true_label_ids.tolist()):
        pred_label_id = parse_safety_label(pred_text)
        confusion[int(true_label_id), pred_label_id] += 1

    metrics = summarize_confusion(confusion)
    metrics["confusion"] = confusion
    return metrics


class GenerativeGuardModule(L.LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        self.model = PromptGuardGenModel(
            model_name=cfg["model"]["backbone"],
            freeze_backbone=cfg["model"]["freeze_backbone"],
        )
        class_count = len(SAFETY_LABEL_TO_ID)
        self.register_buffer(
            "val_confusion",
            torch.zeros(class_count, class_count + 1, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "val_generative_examples",
            torch.tensor(0, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "val_generative_batches",
            torch.tensor(0, dtype=torch.long),
            persistent=False,
        )

    def forward(self, batch):
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

    def _generative_eval_batch_limit(self) -> int:
        return int(self.cfg["logging"].get("generative_eval_batches", 32))

    def _should_run_generative_eval(self, batch_idx: int) -> bool:
        if self.trainer.sanity_checking:
            return False

        batch_limit = self._generative_eval_batch_limit()
        if batch_limit == 0:
            return False
        if batch_limit < 0:
            return True
        return batch_idx < batch_limit

    def _generate_safety_predictions(self, batch) -> list[str]:
        max_new_tokens = self.cfg["data"].get("eval_max_new_tokens", 32)
        use_synced_gpus = self.trainer.world_size > 1
        prompt_lengths = batch["prompt_lengths"]
        batch_size = prompt_lengths.size(0)

        if batch_size == 0:
            return []

        max_prompt_len = int(prompt_lengths.max().item())
        if max_prompt_len <= 0:
            return [""] * batch_size

        prompt_input_ids = batch["input_ids"][:, :max_prompt_len].clone()
        prompt_attention_mask = torch.zeros_like(prompt_input_ids)

        for row_idx, prompt_length in enumerate(prompt_lengths.tolist()):
            if prompt_length <= 0:
                prompt_input_ids[row_idx].fill_(self.model.pad_token_id)
                continue

            prompt_attention_mask[row_idx, :prompt_length] = 1
            if prompt_length < max_prompt_len:
                prompt_input_ids[row_idx, prompt_length:max_prompt_len] = self.model.pad_token_id

        generated = self.model.model.generate(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.model.pad_token_id,
            do_sample=False,
            synced_gpus=use_synced_gpus,
        )
        generated_ids = generated[:, max_prompt_len:]

        pred_texts = []
        for row_idx, prompt_length in enumerate(prompt_lengths.tolist()):
            if prompt_length <= 0:
                pred_texts.append("")
                continue
            pred_texts.append(
                self.model.tokenizer.decode(
                    generated_ids[row_idx],
                    skip_special_tokens=True,
                )
            )

        return pred_texts

    def _shared_step(self, batch):
        output = self(batch)
        loss = output.loss

        # Exact-match accuracy over the supervised safety label tokens.
        with torch.no_grad():
            preds = output.logits[:, :-1].argmax(dim=-1)
            target = batch["labels"][:, 1:]
            mask = target != -100
            valid_rows = mask.any(dim=1)
            if valid_rows.any():
                token_match = (~mask) | (preds == target)
                sample_match = token_match.all(dim=1) & valid_rows
                acc = sample_match[valid_rows].float().mean()
            else:
                acc = torch.tensor(0.0, device=loss.device)

        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)

        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_loss", loss, prog_bar=False, sync_dist=True)
        self.log("train/token_acc", acc, prog_bar=True, sync_dist=True)
        self.log("train_token_acc", acc, prog_bar=False, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)

        if self._should_run_generative_eval(batch_idx):
            pred_texts = self._generate_safety_predictions(batch)
            batch_metrics = compute_metrics(pred_texts, batch["safety_label_ids"])
            self.val_confusion += batch_metrics["confusion"]
            self.val_generative_examples += batch["safety_label_ids"].new_tensor(
                batch["safety_label_ids"].numel()
            )
            self.val_generative_batches += self.val_generative_batches.new_tensor(1)

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_loss", loss, prog_bar=False, sync_dist=True)
        self.log("val/token_acc", acc, prog_bar=True, sync_dist=True)
        self.log("val_token_acc", acc, prog_bar=False, sync_dist=True)

    def on_validation_epoch_start(self):
        self.val_confusion.zero_()
        self.val_generative_examples.zero_()
        self.val_generative_batches.zero_()

    def on_validation_epoch_end(self):
        confusion = self.val_confusion
        generative_examples = self.val_generative_examples
        generative_batches = self.val_generative_batches
        if self.trainer.world_size > 1:
            confusion = self.all_gather(confusion).sum(dim=0)
            generative_examples = self.all_gather(generative_examples).sum()
            generative_batches = self.all_gather(generative_batches).sum()

        metrics = summarize_confusion(confusion)
        for name, value in metrics.items():
            self.log(
                f"val/{name}",
                value,
                prog_bar=name in {"macro_f1", "macro_recall", "accuracy"},
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )
            self.log(
                f"val_{name}",
                value,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )
        self.log(
            "val/generative_examples",
            generative_examples.to(dtype=torch.float32),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )
        self.log(
            "val_generative_examples",
            generative_examples.to(dtype=torch.float32),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )
        self.log(
            "val/generative_batches",
            generative_batches.to(dtype=torch.float32),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )
        self.log(
            "val_generative_batches",
            generative_batches.to(dtype=torch.float32),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=False,
        )

    def configure_optimizers(self):
        cfg_opt = self.cfg["optimizer"]
        cfg_train = self.cfg["training"]
        cfg_sched = self.cfg["scheduler"]

        OptClass = AdamW if cfg_opt["name"] == "adamw" else Adam
        parameters = [p for p in self.parameters() if p.requires_grad]
        if not parameters:
            raise ValueError(
                "No trainable parameters found. "
                "Set model.freeze_backbone=false or add trainable adapters."
            )
        optimizer = OptClass(
            parameters,
            lr=cfg_train["learning_rate"],
            weight_decay=cfg_train["weight_decay"],
            betas=tuple(cfg_opt["betas"]),
            eps=cfg_opt["eps"],
        )

        total_steps = max(int(self.trainer.estimated_stepping_batches), 1)
        warmup_steps = int(total_steps * cfg_train["warmup_ratio"])
        warmup_steps = min(warmup_steps, max(total_steps - 1, 0))
        decay_steps = max(total_steps - warmup_steps, 1)

        if cfg_sched["name"] == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=decay_steps)
        else:
            scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=decay_steps,
            )

        if warmup_steps > 0:
            warmup = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            combined = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, scheduler],
                milestones=[warmup_steps],
            )
        else:
            combined = scheduler

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": combined,
                "interval": "step",
            },
        }


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint")
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"], help="Logger to use")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # --- dataloaders ---
    train_dl = build_dataloader(
        data_path=cfg["data"]["train_path"],
        tokenizer_name=cfg["model"]["backbone"],
        batch_size=cfg["training"]["batch_size"],
        max_length=cfg["data"]["max_length"],
        shuffle=True,
        num_workers=cfg["output"]["num_workers"],
        include_response_tasks=cfg["data"].get("include_response_tasks", True),
    )
    val_dl = build_dataloader(
        data_path=cfg["data"]["val_path"],
        tokenizer_name=cfg["model"]["backbone"],
        batch_size=cfg["training"]["batch_size"],
        max_length=cfg["data"]["max_length"],
        shuffle=False,
        num_workers=cfg["output"]["num_workers"],
        include_response_tasks=cfg["data"].get("include_response_tasks", True),
    )

    # --- model ---
    module = GenerativeGuardModule(cfg)

    # --- callbacks ---
    ckpt_dir = Path(cfg["output"]["dir"])
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="guard-best-{epoch}-{step}-{val_loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            auto_insert_metric_name=False,
        ),
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="guard-step-{epoch}-{step}",
            save_top_k=-1,
            every_n_train_steps=cfg["logging"]["save_every"],
            save_on_train_epoch_end=False,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # --- strategy ---
    strategy = args.strategy
    if strategy == "ddp":
        strategy = DDPStrategy(find_unused_parameters=False)
    
    if strategy == "fsdp":
        strategy = L.strategies.FSDPStrategy(
            auto_wrap_policy=L.strategies.fsdp.auto_wrap.AutoWrapPolicy(
                {PromptGuardGenModel}
            ),
            param_init_fn=L.strategies.fsdp.default_param_init_fn,
        )

    # --- precision ---
    if cfg["training"]["bf16"]:
        precision = "bf16-mixed"
    elif cfg["training"]["fp16"]:
        precision = "16-mixed"
    else:
        precision = "32-true"
            
    # --- logger ---
    if args.logger == "wandb":
        experiment_logger = WandbLogger(project="generative_guard", save_dir=ckpt_dir / "logs")
    else:
        experiment_logger = TensorBoardLogger(save_dir=ckpt_dir / "logs", name="generative_guard")

    # --- trainer ---
    trainer = L.Trainer(
        max_epochs=cfg["training"]["epochs"],
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=strategy,
        precision=precision,
        accumulate_grad_batches=cfg["training"]["gradient_accumulation_steps"],
        gradient_clip_val=cfg["training"]["max_grad_norm"],
        log_every_n_steps=cfg["logging"]["log_every"],
        val_check_interval=cfg["logging"]["eval_every"],
        limit_val_batches=cfg["logging"].get("limit_val_batches", 128),
        callbacks=callbacks,
        logger=experiment_logger,
        enable_progress_bar=True,
    )

    trainer.fit(module, train_dl, val_dl, ckpt_path=args.resume)
    trainer.save_checkpoint(ckpt_dir / "final.ckpt")


if __name__ == "__main__":
    main()
