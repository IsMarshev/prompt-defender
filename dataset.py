"""
Dataset for Qwen3Guard-Gen style generative training.

Each JSONL row is converted into one or two instruction-following samples:
1. Prompt moderation: classify the user-side conversation state.
2. Response moderation: classify the assistant response, if present.
"""

import json
from functools import partial

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


RISK_DISPLAY = {
    "safe": "Safe",
    "controversial": "Controversial",
    "unsafe": "Unsafe",
}
SAFETY_PREFIX = "Safety: "
SAFETY_LABEL_TO_ID = {
    "Safe": 0,
    "Controversial": 1,
    "Unsafe": 2,
}

REFUSAL_PREFIXES = (
    "i'm sorry",
    "i cannot",
    "i can't",
    "i wont",
    "i won't",
    "sorry",
    "as a responsible",
    "i must refuse",
    "i cannot help",
    "i can't help",
)


def format_safety(label: str | None) -> str:
    return RISK_DISPLAY.get((label or "safe").lower(), "Safe")


def format_categories(value) -> str:
    if value is None:
        return "None"
    if isinstance(value, list):
        categories = [str(item).strip() for item in value if str(item).strip()]
    else:
        raw = str(value).strip()
        if not raw or raw.lower() == "none":
            return "None"
        categories = [part.strip() for part in raw.split(",") if part.strip()]
    return ", ".join(categories) if categories else "None"


def derive_refusal_label(sample: dict, response_text: str | None) -> str:
    query_safety = (sample.get("query_safety") or "safe").lower()
    response_safety = (sample.get("response_safety") or "safe").lower()

    if response_text:
        normalized = response_text.strip().lower()
        if normalized.startswith(REFUSAL_PREFIXES):
            return "Yes"

    if query_safety in {"unsafe", "controversial"} and response_safety == "safe":
        return "Yes"
    return "No"


def build_prompt_target(sample: dict) -> str:
    return (
        f"Safety: {format_safety(sample.get('query_safety'))}\n"
        f"Categories: {format_categories(sample.get('query_category'))}"
    )


def build_response_target(sample: dict, response_text: str | None) -> str:
    return (
        f"Safety: {format_safety(sample.get('response_safety'))}\n"
        f"Categories: {format_categories(sample.get('response_category'))}\n"
        f"Refusal: {derive_refusal_label(sample, response_text)}"
    )


def trim_messages_for_prompt_moderation(messages: list[dict]) -> list[dict]:
    if messages and messages[-1].get("role") == "assistant":
        return messages[:-1]
    return messages


def _find_subsequence(sequence: list[int], subsequence: list[int]) -> int:
    if not subsequence or len(subsequence) > len(sequence):
        return -1

    last_start = len(sequence) - len(subsequence)
    for start in range(last_start + 1):
        if sequence[start:start + len(subsequence)] == subsequence:
            return start
    return -1


def build_target_ids_and_loss_mask(
    tokenizer,
    target_text: str,
) -> tuple[list[int], list[bool], str]:
    safety_line = target_text.splitlines()[0]
    if not safety_line.startswith(SAFETY_PREFIX):
        raise ValueError(f"Unsupported target format: {target_text!r}")

    safety_label = safety_line[len(SAFETY_PREFIX):].strip()
    if safety_label not in SAFETY_LABEL_TO_ID:
        raise ValueError(f"Unsupported safety label: {safety_label!r}")

    target_with_eos = target_text + tokenizer.eos_token
    use_offsets = getattr(tokenizer, "is_fast", False)
    tokenized = tokenizer(
        target_with_eos,
        add_special_tokens=False,
        return_attention_mask=False,
        return_offsets_mapping=use_offsets,
    )

    target_ids = tokenized["input_ids"]
    label_start = len(SAFETY_PREFIX)
    label_end = label_start + len(safety_label)

    if use_offsets:
        target_loss_mask = [
            offset_end > label_start and offset_start < label_end
            for offset_start, offset_end in tokenized["offset_mapping"]
        ]
    else:
        target_loss_mask = [False] * len(target_ids)
        label_variants = [
            f" {safety_label}",
            safety_label,
        ]
        for variant in label_variants:
            variant_ids = tokenizer(
                variant,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
            start = _find_subsequence(target_ids, variant_ids)
            if start == -1:
                continue
            for idx in range(start, start + len(variant_ids)):
                target_loss_mask[idx] = True
            break

        if not any(target_loss_mask):
            raise ValueError("Could not locate safety label tokens in target text.")

    return target_ids, target_loss_mask, safety_label


def truncate_pair(
    prompt_ids: list[int],
    target_ids: list[int],
    target_loss_mask: list[bool],
    max_length: int,
) -> tuple[list[int], list[int], list[bool]]:
    if len(target_ids) >= max_length:
        return [], target_ids[:max_length], target_loss_mask[:max_length]

    max_prompt_len = max_length - len(target_ids)
    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]

    return prompt_ids, target_ids, target_loss_mask


class GuardDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer_name: str,
        max_length: int = 2048,
        include_response_tasks: bool = True,
    ):
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    sample = json.loads(line)
                    self.samples.extend(
                        self._expand_sample(sample, include_response_tasks=include_response_tasks)
                    )

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

    def _expand_sample(self, sample: dict, include_response_tasks: bool) -> list[dict]:
        messages = sample.get("messages") or []
        if not messages:
            return []

        examples = []

        prompt_messages = trim_messages_for_prompt_moderation(messages)
        if prompt_messages:
            examples.append(
                {
                    "messages": prompt_messages,
                    "target_text": build_prompt_target(sample),
                }
            )

        if include_response_tasks and messages[-1].get("role") == "assistant":
            response_text = messages[-1].get("content")
            examples.append(
                {
                    "messages": messages,
                    "target_text": build_response_target(sample, response_text),
                }
            )

        return examples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        prompt_text = self.tokenizer.apply_chat_template(
            sample["messages"],
            tokenize=False,
        )
        target_text = sample["target_text"]

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        target_ids, target_loss_mask, safety_label = build_target_ids_and_loss_mask(
            self.tokenizer,
            target_text,
        )

        prompt_ids, target_ids, target_loss_mask = truncate_pair(
            prompt_ids,
            target_ids,
            target_loss_mask,
            self.max_length,
        )

        input_ids = prompt_ids + target_ids
        attention_mask = [1] * len(input_ids)
        loss_mask = [False] * len(prompt_ids) + target_loss_mask

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.bool),
            "prompt_length": torch.tensor(len(prompt_ids), dtype=torch.long),
            "safety_label_id": torch.tensor(
                SAFETY_LABEL_TO_ID[safety_label],
                dtype=torch.long,
            ),
        }


def collate_fn(batch: list[dict], pad_token_id: int) -> dict:
    max_len = max(b["input_ids"].size(0) for b in batch)

    input_ids = []
    attention_mask = []
    loss_masks = []
    prompt_lengths = []
    safety_label_ids = []

    for item in batch:
        pad_len = max_len - item["input_ids"].size(0)
        input_ids.append(
            torch.nn.functional.pad(item["input_ids"], (0, pad_len), value=pad_token_id)
        )
        attention_mask.append(
            torch.nn.functional.pad(item["attention_mask"], (0, pad_len), value=0)
        )
        loss_masks.append(
            torch.nn.functional.pad(item["loss_mask"], (0, pad_len), value=False)
        )
        prompt_lengths.append(item["prompt_length"])
        safety_label_ids.append(item["safety_label_id"])

    input_ids = torch.stack(input_ids)
    loss_masks = torch.stack(loss_masks)
    labels = input_ids.masked_fill(~loss_masks, -100)

    return {
        "input_ids": input_ids,
        "attention_mask": torch.stack(attention_mask),
        "labels": labels,
        "prompt_lengths": torch.stack(prompt_lengths),
        "safety_label_ids": torch.stack(safety_label_ids),
    }


def build_dataloader(
    data_path: str,
    tokenizer_name: str,
    batch_size: int = 4,
    max_length: int = 2048,
    shuffle: bool = True,
    num_workers: int = 0,
    include_response_tasks: bool = True,
) -> DataLoader:
    dataset = GuardDataset(
        data_path=data_path,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        include_response_tasks=include_response_tasks,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_fn, pad_token_id=dataset.tokenizer.pad_token_id),
        num_workers=num_workers,
        pin_memory=True,
    )
