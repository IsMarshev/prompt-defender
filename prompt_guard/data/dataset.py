import json
import random
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset, Sampler
from transformers import PreTrainedTokenizerBase

INSTRUCTION_TEMPLATE = """\
# Task:
1) Evaluate whether THE LAST USER's query belongs to safe or unsafe content based on our safety policy.
2) Identify the category of unsafe content found.

<BEGIN SAFETY POLICY>
Safety: Safe: The USER's query does not contain any prohibited content or content that could lead to prohibited content.
Safety: Unsafe: The USER's query contains prohibited content.
Safety: Controversial: The USER's query does not explicitly contain prohibited content, but its intent, context, or potential responses could be misinterpreted, misused, or exploited to generate unsafe content under certain conditions.
<END SAFETY POLICY>

<BEGIN UNSAFE CONTENT CATEGORIES>
Violent.
Non-violent Illegal Acts.
Sexual Content or Sexual Acts.
PII.
Suicide & Self-Harm.
Unethical Acts.
Politically Sensitive Topics.
Copyright Violation.
Jailbreak.
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>
USER: {user_message}
<END CONVERSATION>

Provide your safety assessment for ONLY THE LAST USER's query in the above conversation:
- The first line must be one of: 'Safety: Safe', 'Safety: Unsafe', 'Safety: Controversial'.
- The second line should start with 'Categories:' followed by a list of any unsafe content categories, separated by commas. If the content is safe, use 'Categories: None'.\
"""

LABEL_MAP = {
    "safe": "Safe",
    "unsafe": "Unsafe",
    "controversial": "Controversial",
}


def _normalize_label(raw: str) -> str:
    return LABEL_MAP.get(raw.lower().strip(), raw.strip().capitalize())


def _normalize_categories(raw: str, label: str) -> str:
    if label == "Safe":
        return "None"
    if not raw or raw.strip().lower() in ("none", ""):
        return "None"
    parts = [c.strip() for c in raw.split(",") if c.strip()]
    return ", ".join(parts) if parts else "None"


def _build_messages(user_message: str) -> list[dict]:
    return [{"role": "user", "content": INSTRUCTION_TEMPLATE.format(user_message=user_message)}]


def _build_target(label: str, categories: str) -> str:
    return f"Safety: {label}\nCategories: {categories}"


def _prompt_text(tokenizer: PreTrainedTokenizerBase, user_message: str) -> str:
    messages = _build_messages(user_message)
    has_template = (
        hasattr(tokenizer, "apply_chat_template")
        and tokenizer.chat_template is not None
    )
    if has_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    return f"<|user|>\n{messages[0]['content']}\n<|assistant|>\n"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PromptSafetyDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 2048,
        log_distribution: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load(file_path)

        if log_distribution:
            self._log_distribution()

        # Precompute token lengths for PackedBatchSampler (one tokenise pass per sample).
        self.lengths = [self._token_length(s) for s in self.samples]

    def _load(self, path: str) -> list[dict]:
        raw = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw.append(json.loads(line))

        samples = []
        for item in raw:
            label = _normalize_label(item["label"])
            categories = _normalize_categories(item.get("category", "None"), label)
            samples.append(
                {
                    "user_message": item["instruction"],
                    "label": label,
                    "categories": categories,
                }
            )
        return samples

    def _log_distribution(self):
        counts = Counter(s["label"] for s in self.samples)
        total = len(self.samples)
        print(f"Dataset distribution (total={total}):")
        for label in ("Safe", "Unsafe", "Controversial"):
            n = counts.get(label, 0)
            print(f"  {label}: {n} ({100 * n / total:.1f}%)")

    def _token_length(self, sample: dict) -> int:
        prompt = _prompt_text(self.tokenizer, sample["user_message"])
        target = _build_target(sample["label"], sample["categories"])
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = self.tokenizer.encode(target, add_special_tokens=False)
        total = len(prompt_ids) + len(target_ids)
        if self.tokenizer.eos_token_id is not None:
            total += 1
        return min(total, self.max_length)

    @classmethod
    def from_samples(
        cls,
        samples: list[dict],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 2048,
        log_distribution: bool = True,
    ) -> "PromptSafetyDataset":
        """Construct a dataset from an already-loaded sample list (no JSONL file needed)."""
        obj = cls.__new__(cls)
        obj.tokenizer = tokenizer
        obj.max_length = max_length
        obj.samples = samples
        obj.lengths = [obj._token_length(s) for s in samples]
        if log_distribution:
            obj._log_distribution()
        return obj

    def set_samples(self, samples: list[dict]) -> None:
        """Replace samples in-place and recompute precomputed lengths."""
        self.samples = samples
        self.lengths = [self._token_length(s) for s in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        prompt = _prompt_text(self.tokenizer, sample["user_message"])
        target = _build_target(sample["label"], sample["categories"])

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = self.tokenizer.encode(target, add_special_tokens=False)

        input_ids = prompt_ids + target_ids
        if self.tokenizer.eos_token_id is not None:
            input_ids = input_ids + [self.tokenizer.eos_token_id]

        if len(input_ids) > self.max_length:
            warnings.warn(
                f"Sample {idx} exceeds max_length "
                f"({len(input_ids)} > {self.max_length}), truncating."
            )
            input_ids = input_ids[: self.max_length]

        n_prompt = min(len(prompt_ids), len(input_ids))
        labels = [-100] * n_prompt + input_ids[n_prompt:]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Eval collator (standard padding, used by Trainer's eval dataloader)
# ---------------------------------------------------------------------------

@dataclass
class DataCollatorForSafety:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]

        max_len = max(x.size(0) for x in input_ids)
        if self.pad_to_multiple_of:
            max_len = (
                (max_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        pad_id = self.tokenizer.pad_token_id or 0

        padded_ids, attn_masks, padded_labels = [], [], []
        for ids, labs in zip(input_ids, labels):
            L = ids.size(0)
            n = max_len - L
            if n > 0:
                pad = torch.full((n,), pad_id, dtype=torch.long)
                padded_ids.append(torch.cat([ids, pad]))
                attn_masks.append(
                    torch.cat([torch.ones(L, dtype=torch.long), torch.zeros(n, dtype=torch.long)])
                )
                padded_labels.append(
                    torch.cat([labs, torch.full((n,), -100, dtype=torch.long)])
                )
            else:
                padded_ids.append(ids)
                attn_masks.append(torch.ones(L, dtype=torch.long))
                padded_labels.append(labs)

        return {
            "input_ids": torch.stack(padded_ids),
            "attention_mask": torch.stack(attn_masks),
            "labels": torch.stack(padded_labels),
        }


# ---------------------------------------------------------------------------
# PackedBatchSampler — greedy bin-packing
# ---------------------------------------------------------------------------

class PackedBatchSampler(Sampler):
    def __init__(
        self,
        dataset: PromptSafetyDataset,
        max_length: int,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.lengths = dataset.lengths
        self.max_length = max_length
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self._cached_bins: Optional[list[list[int]]] = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._cached_bins = None

    def _pack(self) -> list[list[int]]:
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(indices)

        bins: list[list[int]] = []
        current: list[int] = []
        current_len = 0

        for idx in indices:
            sample_len = self.lengths[idx]
            if sample_len > self.max_length:
                # Already truncated by dataset, but be defensive.
                sample_len = self.max_length

            if current_len + sample_len <= self.max_length:
                current.append(idx)
                current_len += sample_len
            else:
                if current:
                    bins.append(current)
                current = [idx]
                current_len = sample_len

        if current:
            bins.append(current)

        return bins

    @property
    def _bins(self) -> list[list[int]]:
        if self._cached_bins is None:
            self._cached_bins = self._pack()
        return self._cached_bins

    def __len__(self) -> int:
        return len(self._bins)

    def __iter__(self):
        bins = self._bins
        total_real = sum(self.lengths[i] for b in bins for i in b)
        total_cap = len(bins) * self.max_length

        for b in bins:
            yield b

        if total_cap > 0:
            print(
                f"[PackedBatchSampler] epoch={self.epoch}  "
                f"packing efficiency = {total_real / total_cap:.1%}  "
                f"({total_real:,} / {total_cap:,} tokens)"
            )

        # Invalidate so the next epoch's set_epoch call starts clean.
        self._cached_bins = None


# ---------------------------------------------------------------------------
# PackedDataCollator — concatenates a bin into one packed sequence
# ---------------------------------------------------------------------------

@dataclass
class PackedDataCollator:
    tokenizer: PreTrainedTokenizerBase
    max_length: int
    use_fa2: bool

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        all_input_ids: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        all_position_ids: list[torch.Tensor] = []
        # cu_seqlens[k] = start token index of sample k in the packed sequence
        cu_seqlens: list[int] = [0]

        for feat in features:
            ids = feat["input_ids"]   # (L,)
            labs = feat["labels"]     # (L,)
            L = ids.size(0)

            all_input_ids.append(ids)
            all_labels.append(labs)
            # Position ids restart from 0 for every sample in the pack.
            all_position_ids.append(torch.arange(L, dtype=torch.long))
            cu_seqlens.append(cu_seqlens[-1] + L)

        input_ids = torch.cat(all_input_ids)
        labels = torch.cat(all_labels)
        position_ids = torch.cat(all_position_ids)
        total_len = input_ids.size(0)

        # Pad to max_length
        pad_len = self.max_length - total_len
        if pad_len > 0:
            pad_id = self.tokenizer.pad_token_id or 0
            input_ids = torch.cat(
                [input_ids, torch.full((pad_len,), pad_id, dtype=torch.long)]
            )
            labels = torch.cat(
                [labels, torch.full((pad_len,), -100, dtype=torch.long)]
            )
            # Position ids for padding: continue monotonically (values don't matter,
            # padding outputs are ignored, but must stay in-range for the embedding table).
            position_ids = torch.cat(
                [position_ids, torch.arange(pad_len, dtype=torch.long)]
            )

        # Add batch dimension (batch_size = 1 per packed sequence).
        input_ids = input_ids.unsqueeze(0)       # (1, max_length)
        labels = labels.unsqueeze(0)             # (1, max_length)
        position_ids = position_ids.unsqueeze(0) # (1, max_length)

        result: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
        }

        if not self.use_fa2:
            result["attention_mask"] = self._build_block_diagonal_mask(
                cu_seqlens, total_len
            )

        return result

    def _build_block_diagonal_mask(
        self, cu_seqlens: list[int], total_len: int
    ) -> torch.Tensor:
        T = self.max_length
        # Additive bias: 0.0 = attend, -inf = block.
        # Initialise everything to blocked; we'll open up intra-sample causal windows.
        mask = torch.full((T, T), float("-inf"))

        for s in range(len(cu_seqlens) - 1):
            start = cu_seqlens[s]
            end = cu_seqlens[s + 1]
            block_size = end - start
            # Lower-triangular (including diagonal) → 0, upper → -inf.
            upper = torch.triu(
                torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1
            )
            block = torch.zeros(block_size, block_size)
            block[upper] = float("-inf")
            mask[start:end, start:end] = block

        # Padding positions must not produce -inf in every column (NaN in softmax).
        # Allow them to attend to the first real token; their outputs are masked by labels=-100.
        if total_len < T:
            mask[total_len:, 0] = 0.0

        # Shape: (1, 1, T, T) — broadcast over batch and heads.
        return mask.unsqueeze(0).unsqueeze(0)
