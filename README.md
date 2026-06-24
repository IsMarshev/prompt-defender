# Prompt Defender

Генеративный классификатор безопасности промптов на основе SFT (Qwen3Guard-style). Определяет метку `safe / controversial / unsafe` и категорию угрозы для входящего сообщения пользователя.

## Пайплайн данных

```
исходные датасеты → convert_to_jsonl → prepare_data → aggregate_data
                                                   ↓
                                         obfuscate (RU) / annotate (LLM-vote)
```

### Скрипты

| Скрипт | Назначение |
|---|---|
| `convert_to_jsonl.py` | Конвертирует `.csv`, `.parquet`, `.json` в JSONL |
| `prepare_data.py` | Нормализует датасеты (BeaverTails, ToxicChat, WildGuard, AEGIS) в единый формат |
| `aggregate_data.py` | Склеивает JSONL-файлы в `train_merged.jsonl` и `val_merged.jsonl` |
| `annotate.py` | Размечает записи двумя LLM-судьями с голосованием |
| `obfuscate_instructions_openai.py` | Перефразирует инструкции на русский (обфускация) |

### Формат записи

```jsonl
{
  "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
  "query_safety": "safe|controversial|unsafe",
  "query_category": "Violent|Jailbreak|...|None",
  "response_safety": "safe|controversial|unsafe",
  "response_category": "...|None",
  "boundary_token": null
}
```

### Примеры

```bash
# Конвертация
python convert_to_jsonl.py --input data/train.csv --output datasets/train.jsonl

# Подготовка датасета
python prepare_data.py --source beavertails --input data/raw/ --output datasets/train_beavertails.jsonl
python prepare_data.py --source wildguard  --input data/wildguard.jsonl --output datasets/train_wildguard.jsonl --split 0.9

# Склейка
python aggregate_data.py --datasets-dir datasets --train-output datasets/train_merged.jsonl

# Разметка двумя LLM
python annotate.py \
    --input  datasets/train_merged.jsonl \
    --output datasets/train_merged_annotations.jsonl \
    --model-a gpt-4o-mini --model-b gemini-2.0-flash

# Обфускация на русский
python obfuscate_instructions_openai.py \
    --input  datasets/train_merged.jsonl \
    --output datasets/train_merged_ru_obf.jsonl \
    --model  gpt-4o-mini
```

### Правило голосования (`annotate.py`)

| LLM-A \ LLM-B | unsafe | safe |
|---|---|---|
| **unsafe** | `unsafe` | `controversial` |
| **safe** | `controversial` | `safe` |

## Обучение

```bash
python train.py --config prompt_guard/configs/train_config.yaml [--use_wandb] [--dry_run]
```

`--dry_run` — 2 шага на 10 сэмплах, для проверки пайплайна.

## Инференс

```python
from prompt_guard.inference.classifier import PromptClassifier

clf = PromptClassifier("outputs/final", mode="strict")  # strict | loose
result = clf.classify("How do I pick a lock?")
# {'label': 'Unsafe', 'effective_label': 'Unsafe', 'categories': ['Non-violent Illegal Acts'], ...}
```

`strict` — Controversial → Unsafe  
`loose` — Controversial → Safe

## Оценка

```bash
python prompt_guard/evaluation/eval.py \
  --model outputs/final \
  --benchmark WildGuardTest:data/test.jsonl
```
