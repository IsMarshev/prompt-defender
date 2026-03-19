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
from pathlib import Path

import yaml
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

from model import PromptGuardGenModel
from dataset import build_dataloader


class GenerativeGuardModule(L.LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        self.model = PromptGuardGenModel(
            model_name=cfg["model"]["backbone"],
            freeze_backbone=cfg["model"]["freeze_backbone"],
        )

    def forward(self, batch):
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

    def _shared_step(self, batch):
        output = self(batch)
        loss = output.loss

        # Token-level accuracy on target tokens (where labels != -100)
        with torch.no_grad():
            preds = output.logits[:, :-1].argmax(dim=-1)
            target = batch["labels"][:, 1:]
            mask = target != -100
            if mask.any():
                acc = (preds[mask] == target[mask]).float().mean()
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

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_loss", loss, prog_bar=False, sync_dist=True)
        self.log("val/token_acc", acc, prog_bar=True, sync_dist=True)
        self.log("val_token_acc", acc, prog_bar=False, sync_dist=True)

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
            filename="guard-{epoch}-{step}-{val_loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            every_n_train_steps=cfg["logging"]["save_every"],
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # --- strategy ---
    strategy = args.strategy
    if strategy == "ddp":
        strategy = DDPStrategy(find_unused_parameters=False)

    # --- precision ---
    if cfg["training"]["bf16"]:
        precision = "bf16-mixed"
    elif cfg["training"]["fp16"]:
        precision = "16-mixed"
    else:
        precision = "32-true"

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
        callbacks=callbacks,
        logger=TensorBoardLogger(save_dir=ckpt_dir / "logs", name="generative_guard"),
        enable_progress_bar=True,
    )

    trainer.fit(module, train_dl, val_dl, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
