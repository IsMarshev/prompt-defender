"""
Dataset and DataLoader for Stream Qwen3Guard training.

Expected JSONL format (one JSON object per line):
{
    "messages": [
        {"role": "user", "content": "How to make a bomb?"},
        {"role": "assistant", "content": "I'm sorry, I can't help with that."}
    ],
    "query_safety": "unsafe",          // "safe" | "controversial" | "unsafe"
    "query_category": "Violent",       // category name or "None"
    "response_safety": "safe",         // "safe" | "controversial" | "unsafe"
    "response_category": "None",       // category name or "None"
    "boundary_token": null             // int | null -- first unsafe token index in response
}

Token-level label logic (Section 4.2):
  - All tokens before boundary_token -> safe (risk=0, cat=0)
  - Tokens from boundary_token onward -> sample-level label
  - If boundary_token is null and response is unsafe -> entire response is unsafe
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from model import NUM_QUERY_CAT, NUM_RESPONSE_CAT

RISK_MAP = {"safe": 0, "controversial": 1, "unsafe": 2}

QUERY_CAT_MAP = {
    "None": 0,
    "Violent": 1,
    "Non-violent Illegal Acts": 2,
    "Sexual Content or Sexual Acts": 3,
    "PII": 4,
    "Suicide & Self-Harm": 5,
    "Unethical Acts": 6,
    "Politically Sensitive Topics": 7,
    "Copyright Violation": 8,
    "Jailbreak": 9,
}

RESPONSE_CAT_MAP = {
    "None": 0,
    "Violent": 1,
    "Non-violent Illegal Acts": 2,
    "Sexual Content or Sexual Acts": 3,
    "PII": 4,
    "Suicide & Self-Harm": 5,
    "Unethical Acts": 6,
    "Politically Sensitive Topics": 7,
    "Copyright Violation": 8,
}


class GuardDataset(Dataset):
    def __init__(self, data_path: str, tokenizer_name: str, max_length: int = 2048):
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        messages = sample["messages"]

        # --- Tokenize the full conversation via chat template ---
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding.input_ids.squeeze(0)       # (L,)
        attention_mask = encoding.attention_mask.squeeze(0)  # (L,)

        # --- Find query end position (first <|im_end|> token) ---
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        end_positions = (input_ids == im_end_id).nonzero(as_tuple=True)[0]
        query_end_idx = end_positions[0].item() if len(end_positions) > 0 else 0

        # --- Determine where the response tokens start ---
        # Response tokens begin after the second <|im_start|> block
        if len(end_positions) >= 1:
            response_start = query_end_idx + 1
        else:
            response_start = len(input_ids)

        # --- Query-level labels ---
        q_risk = RISK_MAP[sample["query_safety"]]
        q_cat = QUERY_CAT_MAP.get(sample.get("query_category", "None"), 0)

        # --- Response token-level labels ---
        r_safety = RISK_MAP.get(sample.get("response_safety", "safe"), 0)
        r_cat_id = RESPONSE_CAT_MAP.get(sample.get("response_category", "None"), 0)
        boundary = sample.get("boundary_token")

        seq_len = len(input_ids)
        r_risk_labels = torch.zeros(seq_len, dtype=torch.long)
        r_cat_labels = torch.zeros(seq_len, dtype=torch.long)

        if r_safety != 0:  # not safe
            if boundary is not None:
                start = min(response_start + boundary, seq_len)
            else:
                start = response_start
            r_risk_labels[start:] = r_safety
            r_cat_labels[start:] = r_cat_id

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "query_end_idx": query_end_idx,
            "q_risk": torch.tensor(q_risk, dtype=torch.long),
            "q_cat": torch.tensor(q_cat, dtype=torch.long),
            "r_risk": r_risk_labels,
            "r_cat": r_cat_labels,
        }


def collate_fn(batch: list[dict]) -> dict:
    """Pad sequences to the longest in the batch."""
    max_len = max(b["input_ids"].size(0) for b in batch)

    input_ids = []
    attention_mask = []
    query_end_idx = []
    q_risk = []
    q_cat = []
    r_risk = []
    r_cat = []

    for b in batch:
        pad_len = max_len - b["input_ids"].size(0)

        input_ids.append(
            torch.nn.functional.pad(b["input_ids"], (0, pad_len), value=0)
        )
        attention_mask.append(
            torch.nn.functional.pad(b["attention_mask"], (0, pad_len), value=0)
        )
        r_risk.append(
            torch.nn.functional.pad(b["r_risk"], (0, pad_len), value=0)
        )
        r_cat.append(
            torch.nn.functional.pad(b["r_cat"], (0, pad_len), value=0)
        )
        query_end_idx.append(b["query_end_idx"])
        q_risk.append(b["q_risk"])
        q_cat.append(b["q_cat"])

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "query_end_idx": torch.tensor(query_end_idx, dtype=torch.long),
        "q_risk": torch.stack(q_risk),
        "q_cat": torch.stack(q_cat),
        "r_risk": torch.stack(r_risk),
        "r_cat": torch.stack(r_cat),
    }


def build_dataloader(
    data_path: str,
    tokenizer_name: str,
    batch_size: int = 4,
    max_length: int = 2048,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    ds = GuardDataset(data_path, tokenizer_name, max_length)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
