"""
Dataset for Qwen3Guard-Gen style generative training.

Each JSONL row is converted into one or two instruction-following samples:
1. Prompt moderation: classify the user-side conversation state.
2. Response moderation: classify the assistant response, if present.

Packing mode (packing=True):
  Examples are sorted by length and grouped into same-length buckets so each
  micro-batch is filled to near-max_length with minimal padding.  A 4-D
  block-diagonal causal attention mask is built per batch item, preventing
  any cross-example attention.  Works with attn_implementation="sdpa" or
  "eager"; Flash Attention 2 uses its own varlen path via _upad_input when
  attention_mask contains padding zeros.
"""

from __future__ import annotations

import json
import random
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, BatchSampler
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


def build_target_ids(tokenizer, target_text: str) -> tuple[list[int], str]:
    """Tokenise target_text + EOS, return (token_ids, safety_label)."""
    safety_line = target_text.splitlines()[0]
    if not safety_line.startswith(SAFETY_PREFIX):
        raise ValueError(f"Unsupported target format: {target_text!r}")

    safety_label = safety_line[len(SAFETY_PREFIX):].strip()
    if safety_label not in SAFETY_LABEL_TO_ID:
        raise ValueError(f"Unsupported safety label: {safety_label!r}")

    target_with_eos = target_text + tokenizer.eos_token
    target_ids = tokenizer(
        target_with_eos,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    return target_ids, safety_label


def truncate_to_max(
    prompt_ids: list[int],
    target_ids: list[int],
    max_length: int,
) -> tuple[list[int], list[int]]:
    if len(target_ids) >= max_length:
        return [], target_ids[:max_length]
    max_prompt_len = max_length - len(target_ids)
    return prompt_ids[-max_prompt_len:], target_ids


class GuardDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer_name: str,
        max_length: int = 2048,
        include_response_tasks: bool = True,
        template_tokenizer_name: str | None = None,
    ):
        self.samples: list[dict] = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    sample = json.loads(line)
                    self.samples.extend(
                        self._expand_sample(sample, include_response_tasks=include_response_tasks)
                    )

        # Tokenizer used for actual encoding
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            padding_side="left",
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Separate tokenizer for apply_chat_template (guard model has the right template)
        if template_tokenizer_name and template_tokenizer_name != tokenizer_name:
            self.template_tokenizer = AutoTokenizer.from_pretrained(
                template_tokenizer_name,
                trust_remote_code=True,
            )
        else:
            self.template_tokenizer = self.tokenizer

        self.max_length = max_length

        # Pre-tokenize all samples once and cache.  This eliminates double
        # tokenisation during training and gives exact lengths for packing.
        self._cache: list[dict] = []
        self._lengths: list[int] = []
        for sample in self.samples:
            item = self._tokenize(sample)
            self._cache.append(item)
            self._lengths.append(item["input_ids"].size(0))

    @staticmethod
    def _normalize_sample(sample: dict) -> dict:
        """Convert flat instruction-format records to the messages-based format."""
        if sample.get("messages"):
            return sample
        instruction = sample.get("instruction")
        if not instruction:
            return sample
        return {
            **sample,
            "messages": [{"role": "user", "content": instruction}],
            "query_safety": sample.get("label", "safe"),
            "query_category": sample.get("category", "None"),
        }

    def _expand_sample(self, sample: dict, include_response_tasks: bool) -> list[dict]:
        sample = self._normalize_sample(sample)
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

    def _tokenize(self, sample: dict) -> dict:
        prompt_text = self.template_tokenizer.apply_chat_template(
            sample["messages"],
            tokenize=False,
            add_generation_prompt=False,
            chat_template_kwargs={"enable_thinking": False},
        )
        prompt_ids: list[int] = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]

        target_ids, safety_label = build_target_ids(self.tokenizer, sample["target_text"])
        prompt_ids, target_ids = truncate_to_max(prompt_ids, target_ids, self.max_length)

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + list(target_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "prompt_length": torch.tensor(len(prompt_ids), dtype=torch.long),
            "safety_label_id": torch.tensor(SAFETY_LABEL_TO_ID[safety_label], dtype=torch.long),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self._cache[idx]


# ---------------------------------------------------------------------------
# Collators
# ---------------------------------------------------------------------------

def collate_fn(batch: list[dict], pad_token_id: int) -> dict:
    """Standard padded collator (no packing)."""
    max_len = max(b["input_ids"].size(0) for b in batch)

    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    prompt_lengths = []
    safety_label_ids = []

    for item in batch:
        pad_len = max_len - item["input_ids"].size(0)
        input_ids_list.append(
            F.pad(item["input_ids"], (0, pad_len), value=pad_token_id)
        )
        attention_mask_list.append(
            F.pad(torch.ones(item["input_ids"].size(0), dtype=torch.long), (0, pad_len))
        )
        labels_list.append(
            F.pad(item["labels"], (0, pad_len), value=-100)
        )
        prompt_lengths.append(item["prompt_length"])
        safety_label_ids.append(item["safety_label_id"])

    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "labels": torch.stack(labels_list),
        "prompt_lengths": torch.stack(prompt_lengths),
        "safety_label_ids": torch.stack(safety_label_ids),
    }


def _build_block_diagonal_causal_mask(
    seqlens: list[int],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Build an additive causal block-diagonal attention mask.

    Within each sub-sequence: lower-triangular (0.0 = attend, -inf = blocked).
    Across sub-sequences: fully blocked (-inf).

    Returns shape (1, 1, total_len, total_len) ready for model forward.
    """
    total = sum(seqlens)
    # Start fully blocked
    mask = torch.full((total, total), torch.finfo(dtype).min, dtype=dtype, device=device)
    pos = 0
    for l in seqlens:
        # triu(fill(-inf), diagonal=1) gives 0 on+below diagonal, -inf above diagonal
        causal_block = torch.triu(
            torch.full((l, l), torch.finfo(dtype).min, dtype=dtype, device=device),
            diagonal=1,
        )
        mask[pos : pos + l, pos : pos + l] = causal_block
        pos += l
    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)


def packed_collate_fn(batch: list[dict], pad_token_id: int) -> dict:
    """
    Packing collator: concatenates examples in each batch item into a single
    sequence with a 4-D block-diagonal causal attention mask to prevent any
    cross-example token interaction.

    Each element of `batch` is already one packed item produced by
    PackingCollator.__call__, i.e. a dict with keys:
      input_ids, labels, position_ids, seqlens (list[int]), safety_label_ids, prompt_lengths
    """
    # Each batch element is already packed; we need to stack them.
    max_len = max(b["input_ids"].size(0) for b in batch)

    input_ids_list = []
    labels_list = []
    position_ids_list = []
    all_seqlens = []
    safety_label_ids = []
    prompt_lengths = []

    for item in batch:
        pad_len = max_len - item["input_ids"].size(0)
        input_ids_list.append(F.pad(item["input_ids"], (0, pad_len), value=pad_token_id))
        labels_list.append(F.pad(item["labels"], (0, pad_len), value=-100))
        position_ids_list.append(F.pad(item["position_ids"], (0, pad_len), value=0))
        all_seqlens.append(item["seqlens"])
        safety_label_ids.extend(item["safety_label_ids"])
        prompt_lengths.extend(item["prompt_lengths"])

    input_ids = torch.stack(input_ids_list)
    labels = torch.stack(labels_list)
    position_ids = torch.stack(position_ids_list)

    # 4-D block-diagonal mask — one per batch item
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    masks = []
    for seqlens in all_seqlens:
        total = sum(seqlens)
        m = _build_block_diagonal_causal_mask(seqlens, dtype=dtype, device=input_ids.device)
        # Pad mask spatial dims to max_len
        if total < max_len:
            pad = max_len - total
            m = F.pad(m, (0, pad, 0, pad), value=torch.finfo(dtype).min)
        masks.append(m)  # (1, 1, max_len, max_len)

    attention_mask_4d = torch.cat(masks, dim=0)  # (B, 1, max_len, max_len)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask_4d,
        "labels": labels,
        "position_ids": position_ids,
        "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
        "safety_label_ids": torch.tensor(safety_label_ids, dtype=torch.long),
        "packed": True,
    }


class PackingCollator:
    """
    Wraps the dataset's raw items.  For each call (list of raw examples),
    greedily packs them into one sequence and returns a single packed item.
    Used together with LengthGroupedBatchSampler so the sampler hands over
    examples that already fit within max_length when concatenated.
    """

    def __init__(self, dataset: GuardDataset, max_length: int):
        self.dataset = dataset
        self.max_length = max_length

    def __call__(self, items: list[dict]) -> dict:
        input_ids_parts = []
        labels_parts = []
        position_ids_parts = []
        seqlens = []
        safety_label_ids = []
        prompt_lengths = []

        for item in items:
            n = item["input_ids"].size(0)
            input_ids_parts.append(item["input_ids"])
            labels_parts.append(item["labels"])
            position_ids_parts.append(torch.arange(n, dtype=torch.long))
            seqlens.append(n)
            safety_label_ids.append(item["safety_label_id"].item())
            prompt_lengths.append(item["prompt_length"].item())

        return {
            "input_ids": torch.cat(input_ids_parts),
            "labels": torch.cat(labels_parts),
            "position_ids": torch.cat(position_ids_parts),
            "seqlens": seqlens,
            "safety_label_ids": safety_label_ids,
            "prompt_lengths": prompt_lengths,
        }


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

class LengthGroupedBatchSampler(BatchSampler):
    """
    Groups examples by token length so each batch has similar-length sequences,
    minimising padding.  For packing mode, each "batch" is a group of examples
    whose total length fits within max_packed_length.
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 42,
        max_packed_length: int | None = None,
    ):
        self.drop_last = drop_last
        self._batches: list[list[int]] = []

        sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])

        if max_packed_length is not None:
            # Greedy bin-packing: fill each bin up to max_packed_length
            current_bin: list[int] = []
            current_total = 0
            for idx in sorted_indices:
                l = lengths[idx]
                if l > max_packed_length:
                    continue  # single example too long; skip
                if current_total + l > max_packed_length and current_bin:
                    self._batches.append(current_bin)
                    current_bin = []
                    current_total = 0
                current_bin.append(idx)
                current_total += l
            if current_bin and (not drop_last or len(current_bin) > 0):
                self._batches.append(current_bin)
        else:
            for i in range(0, len(sorted_indices), batch_size):
                batch = sorted_indices[i : i + batch_size]
                if drop_last and len(batch) < batch_size:
                    continue
                self._batches.append(batch)

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(self._batches)

    def __iter__(self):
        yield from self._batches

    def __len__(self) -> int:
        return len(self._batches)


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def build_dataloader(
    data_path: str,
    tokenizer_name: str,
    batch_size: int = 4,
    max_length: int = 2048,
    shuffle: bool = True,
    num_workers: int = 0,
    include_response_tasks: bool = True,
    drop_last: bool = False,
    packing: bool = False,
    template_tokenizer_name: str | None = None,
) -> DataLoader:
    dataset = GuardDataset(
        data_path=data_path,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        include_response_tasks=include_response_tasks,
        template_tokenizer_name=template_tokenizer_name,
    )
    pad_token_id = dataset.tokenizer.pad_token_id

    if packing:
        packing_collator = PackingCollator(dataset, max_length)

        sampler = LengthGroupedBatchSampler(
            lengths=dataset._lengths,
            batch_size=batch_size,  # used only as fallback; packing uses max_packed_length
            drop_last=drop_last,
            shuffle=shuffle,
            max_packed_length=max_length,
        )

        return DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=lambda items: packed_collate_fn(
                [packing_collator(items)], pad_token_id
            ),
            num_workers=num_workers,
            pin_memory=True,
        )

    sampler = LengthGroupedBatchSampler(
        lengths=dataset._lengths,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=partial(collate_fn, pad_token_id=pad_token_id),
        num_workers=num_workers,
        pin_memory=True,
    )
