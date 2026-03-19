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


def truncate_pair(prompt_ids: list[int], target_ids: list[int], max_length: int) -> tuple[list[int], list[int]]:
    if len(target_ids) >= max_length:
        return [], target_ids[:max_length]

    max_prompt_len = max_length - len(target_ids)
    if len(prompt_ids) > max_prompt_len:
        prompt_ids = prompt_ids[-max_prompt_len:]

    return prompt_ids, target_ids


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
        target_text = sample["target_text"] + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        target_ids = self.tokenizer(
            target_text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]

        prompt_ids, target_ids = truncate_pair(prompt_ids, target_ids, self.max_length)

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch: list[dict], pad_token_id: int) -> dict:
    max_len = max(b["input_ids"].size(0) for b in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for item in batch:
        pad_len = max_len - item["input_ids"].size(0)
        input_ids.append(
            torch.nn.functional.pad(item["input_ids"], (0, pad_len), value=pad_token_id)
        )
        attention_mask.append(
            torch.nn.functional.pad(item["attention_mask"], (0, pad_len), value=0)
        )
        labels.append(
            torch.nn.functional.pad(item["labels"], (0, pad_len), value=-100)
        )

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
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
