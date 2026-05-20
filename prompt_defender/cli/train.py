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
from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

from prompt_defender.core.dataset import SAFETY_LABEL_TO_ID, SAFETY_PREFIX, build_dataloader
from prompt_defender.core.evaluation import (
    normalize_generated_safety_text,
    parse_safety_label as parse_safety_label_text,
)
from prompt_defender.core.model import PromptGuardGenModel
from prompt_defender.pipeline.experiment_utils import (
    infer_experiment_root,
    load_json,
    now_iso,
    reserve_experiment_name,
    save_json,
)


UNKNOWN_SAFETY_LABEL_ID = len(SAFETY_LABEL_TO_ID)


def parse_safety_label(text: str) -> int:
    return SAFETY_LABEL_TO_ID.get(
        parse_safety_label_text(text),
        UNKNOWN_SAFETY_LABEL_ID,
    )


def summarize_confusion(confusion: torch.Tensor) -> dict[str, torch.Tensor]:
    confusion = confusion.to(dtype=torch.float32)
    class_count = len(SAFETY_LABEL_TO_ID)

    f1_scores = []
    precisions = []
    recalls = []
    for class_idx in range(class_count):
        tp = confusion[class_idx, class_idx]
        fp = confusion[:, class_idx].sum() - tp
        fn = confusion[class_idx, :].sum() - tp

        precisions.append(tp / (tp + fp).clamp_min(1.0))
        recalls.append(tp / (tp + fn).clamp_min(1.0))
        f1_scores.append((2.0 * tp) / (2.0 * tp + fp + fn).clamp_min(1.0))

    total = confusion.sum().clamp_min(1.0)
    parsed = confusion[:, :class_count].sum()
    correct = confusion[:, :class_count].diag().sum()

    return {
        "macro_f1": torch.stack(f1_scores).mean(),
        "macro_precision": torch.stack(precisions).mean(),
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


def compute_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    grads = [parameter.grad.detach() for parameter in parameters if parameter.grad is not None]
    if not grads:
        return torch.tensor(0.0)

    if norm_type == float("inf"):
        return torch.stack([grad.abs().max() for grad in grads]).max()

    per_parameter_norms = torch.stack([torch.norm(grad, norm_type) for grad in grads])
    return torch.norm(per_parameter_norms, norm_type)


class GenerativeGuardModule(L.LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        self.model = PromptGuardGenModel(
            model_name=cfg["model"]["backbone"],
            freeze_backbone=cfg["model"]["freeze_backbone"],
            attn_implementation=cfg["model"].get("attn_implementation"),
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

    def on_fit_start(self) -> None:
        if self.trainer.is_global_zero:
            print(f"[guard] attention backend: {self.model.attn_implementation}")
            print(f"[guard] model dtype: {next(self.model.parameters()).dtype}")

    def forward(self, batch):
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=batch["labels"],
            position_ids=batch.get("position_ids"),
        )

    def _generative_eval_batch_limit(self) -> int:
        return int(self.cfg["logging"].get("generative_eval_batches", 32))

    def _should_run_generative_eval(self, batch: dict, batch_idx: int) -> bool:
        if self.trainer.sanity_checking:
            return False
        if batch.get("packed"):
            return False  # generative eval requires non-packed val batches

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

        prefix_ids = self.model.tokenizer(
            SAFETY_PREFIX,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        prefix_len = len(prefix_ids)
        max_prompt_len = int(prompt_lengths.max().item())
        total_input_len = max_prompt_len + prefix_len

        prompt_input_ids = batch["input_ids"].new_full(
            (batch_size, total_input_len),
            self.model.pad_token_id,
        )
        prompt_attention_mask = batch["input_ids"].new_zeros((batch_size, total_input_len))
        prefix_tensor = batch["input_ids"].new_tensor(prefix_ids)

        for row_idx, prompt_length in enumerate(prompt_lengths.tolist()):
            sequence_len = prompt_length + prefix_len
            start_idx = total_input_len - sequence_len
            if prompt_length > 0:
                prompt_input_ids[row_idx, start_idx:start_idx + prompt_length] = batch["input_ids"][
                    row_idx, :prompt_length
                ]
            prompt_input_ids[
                row_idx,
                start_idx + prompt_length:start_idx + sequence_len,
            ] = prefix_tensor
            prompt_attention_mask[row_idx, start_idx:start_idx + sequence_len] = 1

        generated = self.model.model.generate(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.model.pad_token_id,
            do_sample=False,
            synced_gpus=use_synced_gpus,
        )
        generated_ids = generated[:, total_input_len:]

        pred_texts = []
        for row_idx in range(batch_size):
            pred_texts.append(
                normalize_generated_safety_text(
                    self.model.tokenizer.decode(
                        generated_ids[row_idx],
                        skip_special_tokens=True,
                    )
                )
            )

        return pred_texts

    def _shared_step(self, batch):
        output = self(batch)
        loss = output.loss

        with torch.no_grad():
            preds = output.logits[:, :-1].argmax(dim=-1)
            target = batch["labels"][:, 1:]
            mask = target != -100
            if mask.any():
                acc = (preds == target)[mask].float().mean()
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

    def on_before_optimizer_step(self, optimizer):
        grad_norm = compute_grad_norm(self.parameters()).to(self.device)
        self.log(
            "train/grad_norm",
            grad_norm,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=False,
        )
        self.log(
            "train_grad_norm",
            grad_norm,
            prog_bar=False,
            on_step=True,
            on_epoch=False,
            sync_dist=False,
        )

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)

        if self._should_run_generative_eval(batch, batch_idx):
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
                prog_bar=name in {"macro_f1", "macro_precision", "macro_recall", "accuracy"},
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


def maybe_to_float(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    return float(value)


def is_distributed_run(args: argparse.Namespace) -> bool:
    return int(args.devices) * int(args.num_nodes) > 1


def resolve_experiment_name(model_name: str, ckpt_dir: Path) -> tuple[str, Path, Path]:
    metadata_path = ckpt_dir / "experiment_metadata.json"
    if metadata_path.exists():
        metadata = load_json(metadata_path)
        existing_name = metadata.get("experiment_name")
        existing_model_name = metadata.get("model_name")
        if (
            isinstance(existing_name, str)
            and existing_name.strip()
            and existing_model_name == model_name
        ):
            registry_path = infer_experiment_root(ckpt_dir) / "logs" / "experiment_registry.json"
            return existing_name.strip(), metadata_path, registry_path

    experiment_root = infer_experiment_root(ckpt_dir)
    experiment_name, registry_path = reserve_experiment_name(model_name, experiment_root)
    save_json(
        metadata_path,
        {
            "created_at": now_iso(),
            "model_name": model_name,
            "experiment_name": experiment_name,
            "experiment_root": str(experiment_root),
            "experiment_registry_path": str(registry_path),
            "checkpoints_dir": str(ckpt_dir),
        },
    )
    return experiment_name, metadata_path, registry_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint")
    parser.add_argument("--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skip-auto-export", action="store_true")
    args = parser.parse_args()

    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.set_float32_matmul_precision("high")

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.get("training", {}).get("seed")
    distributed_run = is_distributed_run(args)
    train_drop_last = bool(cfg["data"].get("train_drop_last", distributed_run))
    val_drop_last = bool(cfg["data"].get("val_drop_last", False))
    packing = bool(cfg["data"].get("packing", False))
    template_tokenizer = cfg["model"].get("template_tokenizer")

    if seed is not None:
        L.seed_everything(seed, workers=True)

    _dl_common = dict(
        tokenizer_name=cfg["model"]["backbone"],
        max_length=cfg["data"]["max_length"],
        num_workers=cfg["output"]["num_workers"],
        include_response_tasks=cfg["data"].get("include_response_tasks", True),
        template_tokenizer_name=template_tokenizer,
    )

    train_dl = build_dataloader(
        data_path=cfg["data"]["train_path"],
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        drop_last=train_drop_last,
        packing=packing,
        **_dl_common,
    )
    # Val always non-packed so generative eval can extract per-example sequences
    val_dl = build_dataloader(
        data_path=cfg["data"]["val_path"],
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        drop_last=val_drop_last,
        packing=False,
        **_dl_common,
    )

    module = GenerativeGuardModule(cfg)

    ckpt_dir = Path(cfg["output"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    experiment_name, experiment_metadata_path, experiment_registry_path = resolve_experiment_name(
        cfg["model"]["backbone"],
        ckpt_dir,
    )
    logger_logs_root = ckpt_dir / "logs"
    logger_logs_root.mkdir(parents=True, exist_ok=True)

    best_ckpt_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="guard-best-{epoch}-{step}-{val_loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )
    step_ckpt_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="guard-step-{epoch}-{step}",
        save_top_k=-1,
        every_n_train_steps=cfg["logging"]["save_every"],
        save_on_train_epoch_end=False,
        auto_insert_metric_name=False,
    )
    callbacks = [
        best_ckpt_callback,
        step_ckpt_callback,
        LearningRateMonitor(logging_interval="step"),
    ]

    strategy = args.strategy
    if strategy == "ddp":
        strategy = DDPStrategy(find_unused_parameters=False)
    elif strategy == "fsdp":
        strategy = FSDPStrategy(auto_wrap_policy={PromptGuardGenModel})

    if cfg["training"]["bf16"]:
        precision = "bf16-mixed"
    elif cfg["training"]["fp16"]:
        precision = "16-mixed"
    else:
        precision = "32-true"

    if args.logger == "wandb":
        try:
            from lightning.pytorch.loggers import WandbLogger
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "wandb logger requested, but wandb is not installed."
            ) from exc
        experiment_logger = WandbLogger(
            project="generative_guard",
            name=experiment_name,
            save_dir=logger_logs_root,
        )
    else:
        experiment_logger = TensorBoardLogger(
            save_dir=logger_logs_root,
            name=experiment_name,
            version="",
        )

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
        use_distributed_sampler=True,
    )

    trainer.fit(module, train_dl, val_dl, ckpt_path=args.resume)
    final_ckpt_path = ckpt_dir / "final.ckpt"
    trainer.save_checkpoint(final_ckpt_path)

    train_summary_path = ckpt_dir / "train_summary.json"
    train_summary = {
        "status": "completed",
        "created_at": now_iso(),
        "config_path": args.config,
        "output_dir": str(ckpt_dir),
        "experiment_name": experiment_name,
        "experiment_metadata_path": str(experiment_metadata_path),
        "experiment_registry_path": str(experiment_registry_path),
        "logs_root": str(logger_logs_root),
        "logger": args.logger,
        "devices": args.devices,
        "num_nodes": args.num_nodes,
        "strategy": args.strategy,
        "distributed_run": distributed_run,
        "train_drop_last": train_drop_last,
        "val_drop_last": val_drop_last,
        "packing": packing,
        "attn_implementation": cfg["model"].get("attn_implementation"),
        "resume_checkpoint": args.resume,
        "seed": seed,
        "best_checkpoint": best_ckpt_callback.best_model_path or None,
        "best_score": maybe_to_float(best_ckpt_callback.best_model_score),
        "last_checkpoint": best_ckpt_callback.last_model_path or None,
        "final_checkpoint": str(final_ckpt_path),
        "step_checkpoint_count": len(step_ckpt_callback.best_k_models),
        "hf_export_dir": None,
    }
    if trainer.is_global_zero:
        save_json(train_summary_path, train_summary)

    if trainer.is_global_zero and not args.skip_auto_export:
        best_ckpt = best_ckpt_callback.best_model_path
        if best_ckpt:
            print(f"Экспорт лучшего чекпоинта: {best_ckpt}")
            best_module = GenerativeGuardModule.load_from_checkpoint(best_ckpt, cfg=cfg, map_location="cpu")
            hf_export_dir = ckpt_dir / "hf_model"
            hf_export_dir.mkdir(parents=True, exist_ok=True)
            best_module.model.model.save_pretrained(hf_export_dir)
            # Save the template tokenizer (with the guard chat template) if configured,
            # otherwise fall back to the base tokenizer. This ensures the exported model
            # is always evaluated with the same prompt format it was trained on.
            if template_tokenizer:
                from transformers import AutoTokenizer as _AT
                _tmpl_tok = _AT.from_pretrained(template_tokenizer, trust_remote_code=True)
                _tmpl_tok.save_pretrained(hf_export_dir)
            else:
                best_module.model.tokenizer.save_pretrained(hf_export_dir)
            print(f"HuggingFace модель успешно сохранена в {hf_export_dir}")
            train_summary["hf_export_dir"] = str(hf_export_dir)
            save_json(train_summary_path, train_summary)


if __name__ == "__main__":
    main()
