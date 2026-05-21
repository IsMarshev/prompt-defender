import json
import os
import random
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np
from tqdm import tqdm

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
    """
    File-backed dataset with lazy per-sample loading.

    Stores only a byte-offset map in memory; full text is read from disk on
    __getitem__. Lengths are computed in a single scan pass so PackedBatchSampler
    never waits for a second tokenisation pass.

    For small in-memory datasets (eval subsets) use PromptSafetyDataset.from_samples().
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 2048,
        log_distribution: bool = True,
        indices: Optional[list[int]] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._file_path = file_path
        self._samples: Optional[list[dict]] = None  # None → lazy file mode

        # Single pass: build offset map and compute all lengths
        self._offset_map, self._all_lengths = self._scan(file_path)

        # Active file-line indices (default: all)
        self._active = indices if indices is not None else list(range(len(self._offset_map)))

        # Lengths for the active subset — no extra disk reads needed
        self.lengths = [self._all_lengths[i] for i in self._active]

        if log_distribution:
            self._log_distribution()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scan(self, path: str) -> tuple[list[tuple[int, str, str]], list[int]]:
        """
        Two-phase scan:
          1. Read JSONL line by line (fast, tqdm by bytes) → build offset map, collect raw texts.
          2. Batch-tokenise to get lengths (fast tokenizer batch call) → cache to .npy on disk.
        On subsequent runs phase 2 is replaced by a single numpy load.
        """
        # ── Phase 1: read lines ──────────────────────────────────────────────
        offset_map: list[tuple[int, str, str]] = []
        instructions: list[str] = []
        labels_meta: list[str] = []
        cats_meta: list[str] = []

        file_size = os.path.getsize(path)
        byte_offset = 0
        with tqdm(
            total=file_size, unit="B", unit_scale=True,
            desc=f"Reading  {os.path.basename(path)}", leave=True,
        ) as bar, open(path, "rb") as f:
            for raw in f:
                off = byte_offset
                byte_offset += len(raw)
                bar.update(len(raw))
                s = raw.decode("utf-8").strip()
                if not s:
                    continue
                item = json.loads(s)
                label = _normalize_label(item["label"])
                cats = _normalize_categories(item.get("category", "None"), label)
                offset_map.append((off, label, cats))
                instructions.append(item["instruction"])
                labels_meta.append(label)
                cats_meta.append(cats)

        # ── Phase 2: lengths (cached) ────────────────────────────────────────
        cache = path + ".lengths.npy"
        if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(path):
            all_lengths = np.load(cache).tolist()
            print(f"Lengths loaded from cache ({len(all_lengths)} samples).")
        else:
            all_lengths = self._batch_tokenize_lengths(instructions, labels_meta, cats_meta)
            try:
                np.save(cache, np.array(all_lengths, dtype=np.int32))
            except OSError:
                pass  # read-only filesystem — skip cache write

        return offset_map, all_lengths

    def _batch_tokenize_lengths(
        self,
        instructions: list[str],
        labels: list[str],
        cats: list[str],
        batch_size: int = 512,
    ) -> list[int]:
        """Batch-encode prompts+targets; much faster than N individual encode() calls."""
        has_eos = self.tokenizer.eos_token_id is not None
        lengths: list[int] = []

        for i in tqdm(
            range(0, len(instructions), batch_size),
            desc="Tokenising",
            unit="batch",
            leave=True,
        ):
            sl = slice(i, i + batch_size)
            prompts = [_prompt_text(self.tokenizer, instr) for instr in instructions[sl]]
            targets = [_build_target(lbl, cat) for lbl, cat in zip(labels[sl], cats[sl])]

            p_ids = self.tokenizer(prompts, add_special_tokens=False)["input_ids"]
            t_ids = self.tokenizer(targets, add_special_tokens=False)["input_ids"]

            for pi, ti in zip(p_ids, t_ids):
                n = len(pi) + len(ti) + (1 if has_eos else 0)
                lengths.append(min(n, self.max_length))

        return lengths

    def _token_length(self, sample: dict) -> int:
        prompt = _prompt_text(self.tokenizer, sample["user_message"])
        target = _build_target(sample["label"], sample["categories"])
        n = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        n += len(self.tokenizer.encode(target, add_special_tokens=False))
        if self.tokenizer.eos_token_id is not None:
            n += 1
        return min(n, self.max_length)

    def _read_file_sample(self, file_idx: int) -> dict:
        """Read and return sample dict by absolute file-line index."""
        off, label, cats = self._offset_map[file_idx]
        with open(self._file_path, "rb") as f:
            f.seek(off)
            item = json.loads(f.readline().decode("utf-8"))
        return {"user_message": item["instruction"], "label": label, "categories": cats}

    def _get_sample(self, local_idx: int) -> dict:
        if self._samples is not None:
            return self._samples[local_idx]
        return self._read_file_sample(self._active[local_idx])

    def _log_distribution(self) -> None:
        if self._samples is not None:
            labels = [s["label"] for s in self._samples]
        else:
            labels = [self._offset_map[i][1] for i in self._active]
        counts = Counter(labels)
        total = len(labels)
        print(f"Dataset distribution (total={total}):")
        for lbl in ("Safe", "Unsafe", "Controversial"):
            n = counts.get(lbl, 0)
            print(f"  {lbl}: {n} ({100 * n / total:.1f}%)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def with_indices(self, indices: list[int]) -> "PromptSafetyDataset":
        """
        Return a new lazy dataset restricted to the given file-line indices,
        reusing the already-scanned offset map (no re-scan of the file).
        """
        obj = PromptSafetyDataset.__new__(PromptSafetyDataset)
        obj.tokenizer = self.tokenizer
        obj.max_length = self.max_length
        obj._file_path = self._file_path
        obj._samples = None
        obj._offset_map = self._offset_map      # shared — no copy
        obj._all_lengths = self._all_lengths    # shared — no copy
        obj._active = indices
        obj.lengths = [self._all_lengths[i] for i in indices]
        obj._log_distribution()
        return obj

    @classmethod
    def from_samples(
        cls,
        samples: list[dict],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 2048,
        log_distribution: bool = True,
    ) -> "PromptSafetyDataset":
        """Construct a fully in-memory dataset from an already-loaded sample list."""
        obj = cls.__new__(cls)
        obj.tokenizer = tokenizer
        obj.max_length = max_length
        obj._file_path = None
        obj._offset_map = []
        obj._all_lengths = []
        obj._active = []
        obj._samples = samples
        obj.lengths = [obj._token_length(s) for s in samples]
        if log_distribution:
            obj._log_distribution()
        return obj

    @classmethod
    def random_eval_subset(
        cls,
        file_path: str,
        tokenizer: PreTrainedTokenizerBase,
        n: int = 128,
        seed: int = 0,
        max_length: int = 2048,
    ) -> "PromptSafetyDataset":
        """
        Load n random samples from a JSONL file fully into memory.
        Intended for eval: small, fixed subset that fits in RAM.
        """
        all_samples: list[dict] = []
        file_size = os.path.getsize(file_path)
        with tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            desc=f"Reading {os.path.basename(file_path)}",
            leave=True,
        ) as bar:
            with open(file_path, "rb") as f:
                for raw in f:
                    bar.update(len(raw))
                    line = raw.decode("utf-8").strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    label = _normalize_label(item["label"])
                    cats = _normalize_categories(item.get("category", "None"), label)
                    all_samples.append({
                        "user_message": item["instruction"],
                        "label": label,
                        "categories": cats,
                    })
        chosen = random.Random(seed).sample(all_samples, min(n, len(all_samples)))
        return cls.from_samples(chosen, tokenizer, max_length, log_distribution=True)

    def set_samples(self, samples: list[dict]) -> None:
        """Replace in-memory samples and recompute lengths (in-memory datasets only)."""
        self._samples = samples
        self.lengths = [self._token_length(s) for s in samples]

    def __len__(self) -> int:
        return len(self._samples) if self._samples is not None else len(self._active)

    def __getitem__(self, idx: int) -> dict:
        sample = self._get_sample(idx)
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
# Eval collator (standard padding, used for eval dataloader)
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
            sample_len = min(self.lengths[idx], self.max_length)
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
        cu_seqlens: list[int] = [0]

        for feat in features:
            ids = feat["input_ids"]
            labs = feat["labels"]
            L = ids.size(0)

            all_input_ids.append(ids)
            all_labels.append(labs)
            all_position_ids.append(torch.arange(L, dtype=torch.long))
            cu_seqlens.append(cu_seqlens[-1] + L)

        input_ids = torch.cat(all_input_ids)
        labels = torch.cat(all_labels)
        position_ids = torch.cat(all_position_ids)
        total_len = input_ids.size(0)

        pad_len = self.max_length - total_len
        if pad_len > 0:
            pad_id = self.tokenizer.pad_token_id or 0
            input_ids = torch.cat(
                [input_ids, torch.full((pad_len,), pad_id, dtype=torch.long)]
            )
            labels = torch.cat(
                [labels, torch.full((pad_len,), -100, dtype=torch.long)]
            )
            position_ids = torch.cat(
                [position_ids, torch.arange(pad_len, dtype=torch.long)]
            )

        input_ids = input_ids.unsqueeze(0)
        labels = labels.unsqueeze(0)
        position_ids = position_ids.unsqueeze(0)

        result: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
        }

        if not self.use_fa2:
            result["attention_mask"] = self._build_block_diagonal_mask(cu_seqlens, total_len)

        return result

    def _build_block_diagonal_mask(
        self, cu_seqlens: list[int], total_len: int
    ) -> torch.Tensor:
        T = self.max_length
        mask = torch.full((T, T), float("-inf"))

        for s in range(len(cu_seqlens) - 1):
            start = cu_seqlens[s]
            end = cu_seqlens[s + 1]
            block_size = end - start
            upper = torch.triu(
                torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1
            )
            block = torch.zeros(block_size, block_size)
            block[upper] = float("-inf")
            mask[start:end, start:end] = block

        if total_len < T:
            mask[total_len:, 0] = 0.0

        return mask.unsqueeze(0).unsqueeze(0)
