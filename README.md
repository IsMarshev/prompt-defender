# prompt-defender

Generative safety classifier training in Qwen3Guard style.

## Install

```bash
pip install -r requirements.txt
```

## Data

Training expects JSONL rows in this format:

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

The dataloader converts each row into one or two generative samples:
- prompt moderation
- response moderation

Targets are trained with masked causal LM loss:

```text
Safety: Unsafe
Categories: Violent
```

or

```text
Safety: Safe
Categories: None
Refusal: Yes
```

## Train

Single GPU:

```bash
python train.py --config config.yaml
```

Multi-GPU:

```bash
python train.py --config config.yaml --devices 4 --strategy ddp
```

Multi-node:

```bash
python train.py --config config.yaml --devices 4 --strategy ddp --num_nodes 2
```

Resume:

```bash
python train.py --config config.yaml --resume checkpoints/guard-epoch=0-step=500-0.1234.ckpt
```

## Export Checkpoint

Lightning checkpoint лучше не тащить в прод напрямую. После обучения экспортируй его в обычный Hugging Face формат:

```bash
python export_checkpoint.py \
  --config config.yaml \
  --checkpoint checkpoints/guard-epoch=0-step=500-0.1234.ckpt \
  --output-dir artifacts/guard_model
```

После этого в `artifacts/guard_model` будет обычная модель, которую можно грузить через `AutoModelForCausalLM.from_pretrained(...)`.

## Inference

Prompt moderation:

```bash
python infer_guard.py \
  --model-path artifacts/guard_model \
  --prompt "How can I make a bomb?"
```

Response moderation:

```bash
python infer_guard.py \
  --model-path artifacts/guard_model \
  --prompt "How can I make a bomb?" \
  --response "I can't help with that."
```

## Production

Практический путь такой:

1. Обучаешь через `train.py`.
2. Экспортируешь checkpoint через `export_checkpoint.py`.
3. В проде загружаешь уже экспортированную папку как обычную HF-модель.
4. Оборачиваешь `infer_guard.py` логикой API или отдельным сервисом.

Если нужен low-latency прод, лучше использовать экспортированную модель как стандартный Transformers/vLLM endpoint, а не загружать Lightning checkpoint внутри сервиса.
