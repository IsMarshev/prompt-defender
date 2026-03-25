# prompt-defender

Generative safety classifier training in a Qwen3Guard-style setup.

## Install

```bash
make setup
make install
```

If you want Weights & Biases logging, install `wandb` separately and run with `--logger wandb`.

## Data Format

Training expects JSONL rows like:

```json
{
  "messages": [
    {"role": "user", "content": "How can I make a bomb?"},
    {"role": "assistant", "content": "I can't help with that."}
  ],
  "query_safety": "unsafe",
  "query_category": "Violent",
  "response_safety": "safe",
  "response_category": "None"
}
```

Each row is expanded into:

- prompt moderation
- response moderation

Targets look like:

```text
Safety: Unsafe
Categories: Violent
```

or:

```text
Safety: Safe
Categories: None
Refusal: Yes
```

## Single Run

Plain training:

```bash
make train CONFIG=config.yaml
```

Training with overrides:

```bash
.venv/bin/python train.py \
  --config config.yaml \
  --seed 42 \
  --devices 1 \
  --logger tensorboard
```

Manual export:

```bash
make export \
  CONFIG=config.yaml \
  CHECKPOINT=checkpoints/guard-best-epoch-step.ckpt \
  EXPORT_DIR=artifacts/guard_model
```

Manual evaluation:

```bash
make eval \
  MODEL_PATH=artifacts/guard_model \
  DATA_PATH=thinking.jsonl \
  EXTRA="--metrics-file artifacts/guard_eval.json"
```

Manual inference on one sample:

```bash
.venv/bin/python infer_guard.py \
  --model-path artifacts/guard_model \
  --prompt "How can I make a bomb?"
```

## End-to-End Experiment Pipeline

`run_experiment.py` creates one reproducible run directory and executes:

1. `train.py`
2. `export_checkpoint.py`
3. `eval.py` on one or more datasets

Example:

```bash
.venv/bin/python run_experiment.py \
  --config config.yaml \
  --run-root runs \
  --run-name qwen3-06b-lr2e5 \
  --seed 42 \
  --set model.backbone='"Qwen/Qwen3-0.6B"' \
  --set training.learning_rate=2e-5 \
  --eval-data val=datasets/val_merged.jsonl \
  --eval-data thinking=thinking.jsonl
```

Artifacts for a run are stored as:

```text
runs/<run-name>/
  resolved_config.yaml
  experiment_summary.json
  checkpoints/
    train_summary.json
    *.ckpt
  export/
    export_summary.json
    hf_model/
  eval/
    <dataset>_metrics.json
    <dataset>_predictions.jsonl   # only with --save-predictions
```

`experiment_summary.json` is the main machine-readable entrypoint for later comparison.

## Grid Search

`run_grid.py` expands a matrix of overrides, launches `run_experiment.py` for each combination, and writes an aggregated summary.

Example:

```bash
make grid GRID=grid.example.yaml
```

Example grid file:

```yaml
base_config: config.yaml
run_root: runs/qwen3-grid

train:
  devices: 1
  logger: tensorboard
  seed: 42

eval:
  datasets:
    val: datasets/val_merged.jsonl
    thinking: thinking.jsonl

matrix:
  model.backbone:
    - Qwen/Qwen3-0.6B
    - Qwen/Qwen3-1.7B
  training.learning_rate:
    - 2.0e-5
    - 1.0e-5
  training.batch_size:
    - 2
    - 4

name_template: "{model_backbone}-lr-{training_learning_rate}-bs-{training_batch_size}"
```

After the grid finishes, you get:

- `runs/qwen3-grid/<run-name>/...` for each experiment
- `runs/qwen3-grid/grid.example_summary.json`
- `runs/qwen3-grid/grid.example_summary.csv`

The CSV is intended for quick model comparison.

## Useful Flags

`train.py`

- `--seed`
- `--resume`
- `--skip-auto-export`
- `--devices`
- `--strategy ddp`

`run_experiment.py`

- `--set KEY=VALUE`
- `--eval-data name=path`
- `--skip-train`
- `--skip-export`
- `--skip-eval`
- `--skip-existing`
- `--save-predictions`

`run_grid.py`

- `--skip-existing`
- `--continue-on-error`
- `--max-runs`
- `--dry-run`

## Validation Notes

Validation has two independent budgets in `config.yaml`:

- `logging.limit_val_batches` controls loss/token-accuracy validation volume
- `logging.generative_eval_batches` controls how many batches also run `generate()`

Fast dev preset:

```yaml
logging:
  eval_every: 500
  limit_val_batches: 128
  generative_eval_batches: 32
```

Full validation:

```yaml
logging:
  limit_val_batches: 1.0
  generative_eval_batches: -1
```
