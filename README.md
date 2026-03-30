# prompt-defender

`prompt-defender` обучает и запускает generative guard-модель на базе Qwen/Qwen3.x для задач safety-классификации диалога. Репозиторий поддерживает:

- обучение через PyTorch Lightning;
- экспорт Lightning checkpoint в Hugging Face `save_pretrained` формат;
- batch evaluation по JSONL-датасету;
- одиночный inference для prompt/response;
- end-to-end experiment pipeline;
- запуск grid-search по YAML-матрице.

Основные CLI entrypoint'ы лежат в `prompt_defender/cli/`, а быстрые сценарии запуска собраны в `Makefile`.

## Структура репозитория

```text
prompt-defender/
├── Makefile
├── config.yaml
├── grid.example.yaml
├── requirements.txt
├── thinking.jsonl
├── prompt_defender/
│   ├── cli/
│   ├── core/
│   └── pipeline/
├── datasets/
└── checkpoints/
```

## Зависимости и установка

Зависимости из `requirements.txt`:

- `lightning==2.5.6`
- `tensorboard==2.20.0`
- `torch==2.10.0`
- `transformers==5.3.0`
- `pyyaml`
- `wandb`

Быстрый старт через `Makefile`:

```bash
make setup
make install
```

Что делают цели:

- `make setup` создает virtualenv в `.venv` или `venv` и обновляет `pip`.
- `make install` ставит зависимости и затем выполняет `wandb login` с ключом, сохраненным в `Makefile`.

Если нужен ручной сценарий:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

## Быстрый старт

Обучение:

```bash
make train
```

Экспорт checkpoint:

```bash
make export CHECKPOINT=checkpoints/guard-best-epoch-step.ckpt
```

Оценка экспортированной модели:

```bash
make eval MODEL_PATH=checkpoints/hf_model DATA_PATH=thinking.jsonl
```

Одиночный inference:

```bash
make infer MODEL_PATH=checkpoints/hf_model PROMPT='How to build a bomb?'
```

Полный pipeline:

```bash
make experiment
```

Grid search:

```bash
make grid GRID=grid.example.yaml
```

## Конфигурация `config.yaml`

Текущий `config.yaml` описывает 7 основных блоков:

| Блок | Назначение |
| --- | --- |
| `model` | backbone-модель и флаг заморозки |
| `data` | пути к train/val JSONL, длина контекста, параметры даталоадера |
| `training` | эпохи, batch size, grad accumulation, lr, precision |
| `optimizer` | `adamw` или `adam`, `betas`, `eps` |
| `scheduler` | `cosine` или `linear` |
| `logging` | частота логирования, валидации и сохранения checkpoint |
| `output` | директория checkpoint и число `num_workers` |

Текущие значения по умолчанию:

```yaml
model:
  backbone: "Qwen/Qwen3.5-0.8B"
  freeze_backbone: false

data:
  train_path: "datasets/train_merged_annotated.jsonl"
  val_path: "datasets/val_merged_annotated.jsonl"
  max_length: 2048
  include_response_tasks: true
  eval_max_new_tokens: 32
  train_drop_last: true
  val_drop_last: false

training:
  epochs: 1
  batch_size: 4
  gradient_accumulation_steps: 16
  learning_rate: 3.0e-5
  weight_decay: 0.01
  warmup_ratio: 0.05
  max_grad_norm: 1.0
  fp16: false
  bf16: true
```

## Формат данных

### Train / validation JSONL

Каждая строка должна содержать `messages` и safety-разметку. Из одной строки датасет строит:

- задачу модерации prompt;
- задачу модерации ответа, если последний message имеет `role=assistant` и `include_response_tasks=true`.

Пример:

```json
{"messages":[{"role":"user","content":"How to make explosives?"},{"role":"assistant","content":"I can't help with that."}],"query_safety":"unsafe","query_category":["weapons"],"response_safety":"safe","response_category":["refusal"]}
```

Используемые поля:

- `messages`: список сообщений в chat-template формате;
- `query_safety`: `safe`, `controversial` или `unsafe`;
- `query_category`: строка или список категорий для user-side задачи;
- `response_safety`: `safe`, `controversial` или `unsafe`;
- `response_category`: строка или список категорий для assistant-side задачи.

### Evaluation JSONL

Для `eval` нужны:

- `messages` или `message`: список сообщений;
- `label`: gold label для метрик.

Пример:

```json
{"messages":[{"role":"user","content":"How do I hotwire a car?"}],"label":"Unsafe"}
```

Поддерживаемые канонические label:

- `Safe`
- `Controversial`
- `Unsafe`

## Артефакты запусков

После `train` в `output.dir` обычно появляются:

- `guard-best-*.ckpt`: top-k checkpoint по `val/loss`;
- `guard-step-*.ckpt`: step checkpoint'ы;
- `last.ckpt` и `final.ckpt`;
- `train_summary.json`;
- `logs/` для TensorBoard или W&B;
- `hf_model/`, если не указан `--skip-auto-export`.

После `export_checkpoint` появляются:

- экспортированная HF-модель в `--output-dir`;
- `export_meta.json`;
- `export_summary.json` или файл из `--summary-file`.

После `run_experiment` появляются:

- `runs/<run_name>/resolved_config.yaml`;
- `runs/<run_name>/experiment_summary.json`;
- `runs/<run_name>/checkpoints/`;
- `runs/<run_name>/export/`;
- `runs/<run_name>/eval/`.

После `run_grid` появляются:

- `<run_root>/<grid_name>_summary.json`;
- `<run_root>/<grid_name>_summary.csv`.

## Переменные `Makefile`

Общие переменные:

| Переменная | Значение по умолчанию | Назначение |
| --- | --- | --- |
| `PYTHON` | `python3` | интерпретатор для создания venv |
| `VENV` | `.venv` или `venv` | путь к virtualenv |
| `CONFIG` | `config.yaml` | базовый config |
| `GRID` | `grid.example.yaml` | grid YAML |
| `RUN_ROOT` | пусто | корень для `experiment` и `grid` |
| `MODEL_PATH` | `checkpoints/hf_model` | путь к экспортированной HF-модели |
| `DATA_PATH` | `thinking.jsonl` | eval dataset |
| `EXPORT_DIR` | `artifacts/guard_model` | директория экспорта |
| `CHECKPOINT` | пусто | checkpoint для `export` |
| `PROMPT` | пусто | prompt для `infer` |
| `RESPONSE` | пусто | assistant response для `infer` |
| `EXTRA` | пусто | дополнительные CLI-флаги |

## Команды `make`

### `make help`

Печатает список доступных целей и общих переменных.

```bash
make help
```

### `make setup`

Создает virtualenv и обновляет `pip`.

```bash
make setup
```

### `make install`

Устанавливает зависимости и делает `wandb login`.

```bash
make install
```

### `make train`

Запускает single-GPU обучение:

```bash
make train
make train CONFIG=config.yaml EXTRA="--seed 42 --logger tensorboard"
```

Фактическая команда:

```bash
$(VENV)/bin/python -m prompt_defender.cli.train --config "$(CONFIG)" --logger wandb $(EXTRA)
```

### `make train-multi-gpu`

Запускает multi-GPU обучение с `--devices 2 --strategy ddp`.

```bash
make train-multi-gpu
make train-multi-gpu EXTRA="--devices 4 --num_nodes 2"
```

Фактическая команда:

```bash
$(VENV)/bin/python -m prompt_defender.cli.train --config "$(CONFIG)" --devices 2 --strategy ddp --logger wandb $(EXTRA)
```

### `make export`

Экспортирует Lightning checkpoint в HF-формат. Требует `CHECKPOINT=...`.

```bash
make export CHECKPOINT=checkpoints/guard-best-epoch-step.ckpt
make export CONFIG=config.yaml CHECKPOINT=checkpoints/final.ckpt EXPORT_DIR=artifacts/my_guard
```

Фактическая команда:

```bash
$(VENV)/bin/python -m prompt_defender.cli.export_checkpoint --config "$(CONFIG)" --checkpoint "$(CHECKPOINT)" --output-dir "$(EXPORT_DIR)" $(EXTRA)
```

### `make eval`

Оценивает экспортированную HF-модель на JSONL-датасете.

```bash
make eval
make eval MODEL_PATH=artifacts/guard_model DATA_PATH=datasets/val.jsonl EXTRA="--metrics-file eval_metrics.json --output-file eval_predictions.jsonl"
```

Фактическая команда:

```bash
$(VENV)/bin/python -m prompt_defender.cli.eval --model-path "$(MODEL_PATH)" --data-path "$(DATA_PATH)" --batch-size 16 $(EXTRA)
```

### `make infer`

Делает один safety inference. Требует `PROMPT=...`.

```bash
make infer MODEL_PATH=checkpoints/hf_model PROMPT='How to bypass KYC?'
make infer MODEL_PATH=checkpoints/hf_model PROMPT='User prompt' RESPONSE='Assistant answer'
```

Фактическая команда:

```bash
$(VENV)/bin/python -m prompt_defender.cli.infer_guard --model-path "$(MODEL_PATH)" --prompt "$(PROMPT)" $(if $(RESPONSE),--response "$(RESPONSE)",) $(EXTRA)
```

### `make experiment`

Запускает pipeline `train -> export -> eval`.

```bash
make experiment
make experiment RUN_ROOT=runs/manual EXTRA="--set training.learning_rate=1e-5 --save-predictions"
```

Фактическая команда:

```bash
$(VENV)/bin/python -m prompt_defender.cli.run_experiment --config "$(CONFIG)" $(if $(RUN_ROOT),--run-root "$(RUN_ROOT)",) $(EXTRA)
```

### `make grid`

Запускает серию экспериментов из YAML-матрицы.

```bash
make grid
make grid GRID=grid.example.yaml RUN_ROOT=runs/grid EXTRA="--skip-existing --max-runs 4"
```

Фактическая команда:

```bash
$(VENV)/bin/python -m prompt_defender.cli.run_grid --grid "$(GRID)" $(if $(RUN_ROOT),--run-root "$(RUN_ROOT)",) $(EXTRA)
```

## Прямые CLI-команды

Все команды ниже нужно запускать из корня репозитория.

### `python -m prompt_defender.cli.train`

Обучение модели через PyTorch Lightning.

Пример:

```bash
.venv/bin/python -m prompt_defender.cli.train --config config.yaml --devices 1 --logger tensorboard
```

Аргументы:

| Флаг | По умолчанию | Описание |
| --- | --- | --- |
| `--config` | `config.yaml` | путь к YAML-конфигу |
| `--devices` | `1` | число устройств |
| `--num_nodes` | `1` | число узлов |
| `--strategy` | `auto` | Lightning strategy: `auto`, `ddp`, `fsdp` |
| `--resume` | `None` | путь к checkpoint для resume |
| `--logger` | `tensorboard` | `tensorboard` или `wandb` |
| `--seed` | `None` | random seed |
| `--skip-auto-export` | `false` | не экспортировать лучший checkpoint автоматически |

Что делает:

- строит train/val dataloader из `data.train_path` и `data.val_path`;
- сохраняет checkpoint'ы по `val/loss` и по шагам;
- пишет `train_summary.json`;
- по умолчанию экспортирует лучший checkpoint в `output.dir/hf_model`.

### `python -m prompt_defender.cli.export_checkpoint`

Экспортирует `.ckpt` в Hugging Face формат.

Пример:

```bash
.venv/bin/python -m prompt_defender.cli.export_checkpoint --config config.yaml --checkpoint checkpoints/final.ckpt --output-dir artifacts/guard_model
```

Аргументы:

| Флаг | По умолчанию | Описание |
| --- | --- | --- |
| `--config` | `config.yaml` | YAML-конфиг, нужен для `model.backbone` |
| `--checkpoint` | обязательный | путь к `.ckpt` |
| `--output-dir` | обязательный | директория для `save_pretrained` |
| `--summary-file` | `None` | путь к JSON summary; иначе пишется `output_dir/export_summary.json` |

### `python -m prompt_defender.cli.eval`

Оценивает экспортированную HF-модель на JSONL-датасете.

Пример:

```bash
.venv/bin/python -m prompt_defender.cli.eval --model-path checkpoints/hf_model --data-path thinking.jsonl --metrics-file eval/metrics.json --output-file eval/predictions.jsonl
```

Аргументы:

| Флаг | По умолчанию | Описание |
| --- | --- | --- |
| `--model-path` | `guard_model` | путь к экспортированной HF-модели |
| `--data-path` | `thinking.jsonl` | eval JSONL |
| `--output-file` | `None` | JSONL с предсказаниями по примерам |
| `--metrics-file` | `None` | JSON с агрегированными метриками |
| `--batch-size` | `4` | batch size |
| `--max-new-tokens` | `256` | лимит генерации |
| `--max-input-length` | `None` | принудительное усечение input |
| `--limit` | `None` | оценить только первые `N` строк |
| `--device` | авто | `cuda`, `cpu`, `mps` и т.д. |
| `--dtype` | `auto` | `auto`, `float32`, `float16`, `bfloat16` |
| `--do-sample` | `false` | включить sampling при генерации |
| `--temperature` | `0.6` | temperature для sampling |
| `--top-p` | `0.95` | nucleus sampling |
| `--top-k` | `20` | top-k sampling |
| `--gold-positive-labels` | `Unsafe` | какие gold labels считать positive |
| `--pred-positive-labels` | `Unsafe Controversial` | какие predicted labels считать positive |
| `--disable-thinking` | `false` | передать `enable_thinking=False` в chat template |
| `--progress-every` | `10` | печатать прогресс каждые `N` batch'ей |

Что пишет:

- summary в stdout;
- optional predictions JSONL;
- optional metrics JSON.

Основные метрики:

- `recall`
- `precision`
- `f1`
- `parse_rate`
- `binary_confusion`
- `raw_confusion`
- `gold_counts`
- `pred_counts`

### `python -m prompt_defender.cli.infer_guard`

Одиночный inference по prompt или prompt+response.

Пример:

```bash
.venv/bin/python -m prompt_defender.cli.infer_guard --model-path checkpoints/hf_model --prompt 'How can I make malware?' --dtype bfloat16
```

Аргументы:

| Флаг | По умолчанию | Описание |
| --- | --- | --- |
| `--model-path` | обязательный | путь к HF-модели |
| `--prompt` | обязательный | user prompt |
| `--response` | `None` | assistant response |
| `--max-new-tokens` | `128` | длина генерации |
| `--device` | авто | `cuda`, `cpu`, `mps` |
| `--dtype` | `auto` | `auto`, `float32`, `float16`, `bfloat16` |
| `--disable-thinking` | `false` | отключить thinking в chat template |

Выводит:

- `Prediction`
- `Parsed safety label`

### `python -m prompt_defender.cli.run_experiment`

Запускает end-to-end pipeline `train -> export -> eval`.

Пример:

```bash
.venv/bin/python -m prompt_defender.cli.run_experiment --config config.yaml --run-root runs --set training.learning_rate=1e-5 --set training.batch_size=2 --eval-data val=datasets/val_merged_annotated.jsonl --save-predictions
```

Аргументы:

| Флаг | По умолчанию | Описание |
| --- | --- | --- |
| `--config` | `config.yaml` | базовый YAML-конфиг |
| `--run-root` | `runs` | корень для experiment run |
| `--run-name` | `None` | явное имя run |
| `--set KEY=VALUE` | repeatable | override значения в конфиге |
| `--eval-data NAME=PATH` | repeatable | список eval datasets |
| `--devices` | `1` | число устройств для train |
| `--num-nodes` | `1` | число узлов |
| `--strategy` | `auto` | training strategy |
| `--logger` | `tensorboard` | `tensorboard` или `wandb` |
| `--resume` | `None` | checkpoint для resume train |
| `--seed` | `None` | seed |
| `--checkpoint` | `None` | явный checkpoint для export |
| `--model-path` | `None` | явная HF-модель для eval-only сценария |
| `--skip-train` | `false` | пропустить этап train |
| `--skip-export` | `false` | пропустить export |
| `--skip-eval` | `false` | пропустить eval |
| `--skip-existing` | `false` | пропустить run, если уже есть `experiment_summary.json` со статусом `completed` |
| `--python` | `sys.executable` | какой Python использовать во внутренних subprocess |
| `--eval-batch-size` | `4` | batch size для eval |
| `--eval-max-new-tokens` | `256` | лимит генерации для eval |
| `--eval-max-input-length` | `None` | truncation длины input |
| `--eval-limit` | `None` | лимит строк для eval |
| `--eval-device` | `None` | устройство для eval |
| `--eval-dtype` | `auto` | dtype для eval |
| `--disable-thinking` | `false` | отключить thinking в eval |
| `--save-predictions` | `false` | сохранять `<dataset>_predictions.jsonl` |
| `--dry-run` | `false` | только напечатать команды и структуры run |

Что создает:

- `resolved_config.yaml`;
- `experiment_summary.json`;
- подпапки `checkpoints/`, `export/`, `eval/`.

### `python -m prompt_defender.cli.run_grid`

Запускает grid-search и собирает агрегированные summary.

Пример:

```bash
.venv/bin/python -m prompt_defender.cli.run_grid --grid grid.example.yaml --run-root runs/qwen3-grid --skip-existing --max-runs 4
```

Аргументы:

| Флаг | По умолчанию | Описание |
| --- | --- | --- |
| `--grid` | `grid.yaml` | YAML с matrix/grid-описанием |
| `--run-root` | `None` | переопределить `run_root` из grid |
| `--python` | `sys.executable` | Python для внутренних subprocess |
| `--skip-existing` | `false` | пропускать уже завершенные run |
| `--continue-on-error` | `false` | продолжать сетку после ошибки |
| `--dry-run` | `false` | только печатать команды |
| `--max-runs` | `None` | ограничить число комбинаций |

Что создает:

- JSON summary по всем run;
- CSV summary по метрикам и override'ам.

## Формат `grid.example.yaml`

Пример структуры:

```yaml
base_config: config.yaml
run_root: runs/qwen3-grid

train:
  devices: 1
  num_nodes: 1
  strategy: auto
  logger: tensorboard
  seed: 42

eval:
  datasets:
    val: datasets/val_merged.jsonl
    thinking: thinking.jsonl
  batch_size: 4
  max_new_tokens: 256
  dtype: auto
  save_predictions: false

constants:
  data.include_response_tasks: true

matrix:
  model.backbone:
    - Qwen/Qwen3-0.6B
    - Qwen/Qwen3-1.7B
  training.learning_rate:
    - 2.0e-5
    - 1.0e-5
```

Смысл блоков:

| Блок | Назначение |
| --- | --- |
| `base_config` | базовый config для всех запусков |
| `run_root` | куда писать все run |
| `train` | параметры, которые передаются в `run_experiment` |
| `eval` | параметры eval для каждого run |
| `constants` | постоянные override'ы |
| `matrix` | параметры, по которым строится декартово произведение |
| `name_template` | шаблон имени run через flattened keys |

## Override'ы через `--set`

`run_experiment` и `run_grid` умеют менять config без правки YAML. Значения парсятся сначала как JSON, потом как YAML.

Примеры:

```bash
--set training.learning_rate=1e-5
--set training.batch_size=8
--set model.freeze_backbone=false
--set optimizer.betas='[0.9, 0.98]'
--set logging.generative_eval_batches=0
```

## Известный нюанс текущей реализации

Оркестраторы `prompt_defender.cli.run_experiment` и `prompt_defender.cli.run_grid` внутри себя запускают subprocess-команды вида:

```bash
python train.py ...
python export_checkpoint.py ...
python eval.py ...
python run_experiment.py ...
```

То есть они ожидают wrapper-скрипты в корне репозитория. Если таких файлов в корне нет, `experiment` и `grid` завершатся ошибкой при реальном запуске. В этом случае есть два рабочих варианта:

- использовать прямые команды `python -m prompt_defender.cli.*`, перечисленные выше;
- вернуть корневые wrapper-скрипты или поправить оркестраторы под модульные вызовы.

## Полезные сценарии

Обучение с reproducible seed:

```bash
make train EXTRA="--seed 42 --logger tensorboard"
```

Resume from checkpoint:

```bash
.venv/bin/python -m prompt_defender.cli.train --config config.yaml --resume checkpoints/last.ckpt
```

Оценка только первых 100 строк:

```bash
.venv/bin/python -m prompt_defender.cli.eval --model-path checkpoints/hf_model --data-path thinking.jsonl --limit 100
```

Eval с сохранением prediction-файла:

```bash
.venv/bin/python -m prompt_defender.cli.eval --model-path checkpoints/hf_model --data-path thinking.jsonl --output-file eval/predictions.jsonl --metrics-file eval/metrics.json
```

Dry-run experiment:

```bash
.venv/bin/python -m prompt_defender.cli.run_experiment --config config.yaml --dry-run --set training.batch_size=2
```

Dry-run grid:

```bash
.venv/bin/python -m prompt_defender.cli.run_grid --grid grid.example.yaml --dry-run --max-runs 2
```
