import re
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt_guard.data.dataset import INSTRUCTION_TEMPLATE

_SAFETY_RE = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)", re.IGNORECASE)
_CATEGORIES_RE = re.compile(r"Categories:\s*(.+)", re.IGNORECASE)

VALID_CATEGORIES = {
    "Violent",
    "Non-violent Illegal Acts",
    "Sexual Content or Sexual Acts",
    "PII",
    "Suicide & Self-Harm",
    "Unethical Acts",
    "Politically Sensitive Topics",
    "Copyright Violation",
    "Jailbreak",
}


def _parse_output(text: str) -> tuple[str, list[str]]:
    label_match = _SAFETY_RE.search(text)
    if not label_match:
        return "Unsafe", []

    label = label_match.group(1).capitalize()

    cat_match = _CATEGORIES_RE.search(text)
    if not cat_match:
        return label, []

    raw_cats = cat_match.group(1).strip()
    if raw_cats.lower() == "none":
        return label, []

    categories = [c.strip() for c in raw_cats.split(",") if c.strip()]
    return label, categories


class PromptClassifier:
    def __init__(
        self,
        model_path: str,
        mode: str = "strict",
        device: str = "cuda",
    ):
        assert mode in ("strict", "loose"), "mode must be 'strict' or 'loose'"
        self.mode = mode
        self.device = device if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _build_prompt(self, user_message: str) -> str:
        messages = [
            {"role": "user", "content": INSTRUCTION_TEMPLATE.format(user_message=user_message)}
        ]
        has_template = (
            hasattr(self.tokenizer, "apply_chat_template")
            and self.tokenizer.chat_template is not None
        )
        if has_template:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        return f"<|user|>\n{messages[0]['content']}\n<|assistant|>\n"

    def _effective_label(self, label: str) -> str:
        if label == "Controversial":
            return "Unsafe" if self.mode == "strict" else "Safe"
        return label

    def classify(self, user_message: str) -> dict:
        prompt = self._build_prompt(user_message)
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=1024,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = out[0][enc["input_ids"].shape[1]:]
        raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        label, categories = _parse_output(raw_output)
        return {
            "label": label,
            "effective_label": self._effective_label(label),
            "categories": categories,
            "raw_output": raw_output,
        }

    def classify_batch(self, messages: list[str]) -> list[dict]:
        prompts = [self._build_prompt(m) for m in messages]
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            add_special_tokens=False,
        ).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=50,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        results = []
        prompt_len = enc["input_ids"].shape[1]
        for i, row in enumerate(out):
            new_tokens = row[prompt_len:]
            raw_output = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            label, categories = _parse_output(raw_output)
            results.append(
                {
                    "label": label,
                    "effective_label": self._effective_label(label),
                    "categories": categories,
                    "raw_output": raw_output,
                }
            )
        return results
