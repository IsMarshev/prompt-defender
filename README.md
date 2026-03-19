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
