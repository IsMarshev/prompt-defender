"""
Stream Qwen3Guard training with PyTorch Lightning.

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

from model import StreamGuard
from loss import QueryLoss, ResponseLoss
from dataset import build_dataloader


class StreamGuardModule(L.LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        self.model = StreamGuard(
            backbone_name=cfg["model"]["backbone"],
            freeze_backbone=cfg["model"]["freeze_backbone"],
        )
        self.query_loss_fn = QueryLoss()
        self.response_loss_fn = ResponseLoss()

    def forward(self, batch):
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            query_end_idx=batch["query_end_idx"],
        )

    def _shared_step(self, batch):
        q_risk, q_cat, r_risk, r_cat = self(batch)

        loss_q = self.query_loss_fn(
            y_risk=batch["q_risk"],
            y_cat=batch["q_cat"],
            logits_risk=q_risk,
            logits_cat=q_cat,
        )
        loss_r = self.response_loss_fn(
            y_risk=batch["r_risk"],
            y_cat=batch["r_cat"],
            logits_risk=r_risk,
            logits_cat=r_cat,
        )
        loss = loss_q + loss_r

        # --- metrics ---
        with torch.no_grad():
            q_risk_acc = (q_risk.argmax(-1) == batch["q_risk"]).float().mean()
            r_risk_pred = r_risk.argmax(-1)
            mask = batch["attention_mask"].bool()
            r_risk_acc = (r_risk_pred[mask] == batch["r_risk"][mask]).float().mean()

        return loss, loss_q, loss_r, q_risk_acc, r_risk_acc

    def training_step(self, batch, batch_idx):
        loss, loss_q, loss_r, q_acc, r_acc = self._shared_step(batch)

        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        self.log("train/loss_query", loss_q, sync_dist=True)
        self.log("train/loss_response", loss_r, sync_dist=True)
        self.log("train/q_risk_acc", q_acc, prog_bar=True, sync_dist=True)
        self.log("train/r_risk_acc", r_acc, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_q, loss_r, q_acc, r_acc = self._shared_step(batch)

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/loss_query", loss_q, sync_dist=True)
        self.log("val/loss_response", loss_r, sync_dist=True)
        self.log("val/q_risk_acc", q_acc, prog_bar=True, sync_dist=True)
        self.log("val/r_risk_acc", r_acc, sync_dist=True)

    def configure_optimizers(self):
        cfg_opt = self.cfg["optimizer"]
        cfg_train = self.cfg["training"]
        cfg_sched = self.cfg["scheduler"]

        OptClass = AdamW if cfg_opt["name"] == "adamw" else Adam
        optimizer = OptClass(
            self.parameters(),
            lr=cfg_train["learning_rate"],
            weight_decay=cfg_train["weight_decay"],
            betas=tuple(cfg_opt["betas"]),
            eps=cfg_opt["eps"],
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * cfg_train["warmup_ratio"])

        if cfg_sched["name"] == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
        else:
            scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps - warmup_steps)

        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        combined = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, scheduler],
            milestones=[warmup_steps],
        )

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
    )
    val_dl = build_dataloader(
        data_path=cfg["data"]["val_path"],
        tokenizer_name=cfg["model"]["backbone"],
        batch_size=cfg["training"]["batch_size"],
        max_length=cfg["data"]["max_length"],
        shuffle=False,
        num_workers=cfg["output"]["num_workers"],
    )

    # --- model ---
    module = StreamGuardModule(cfg)

    # --- callbacks ---
    ckpt_dir = Path(cfg["output"]["dir"])
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="guard-{epoch}-{step}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            every_n_train_steps=cfg["logging"]["save_every"],
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
        logger=TensorBoardLogger(save_dir=ckpt_dir / "logs", name="stream_guard"),
        enable_progress_bar=True,
    )

    trainer.fit(module, train_dl, val_dl, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
