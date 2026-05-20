from __future__ import annotations

import torch
import torch.nn as nn
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# transformers 5.0 renamed torch_dtype → dtype
_DTYPE_KWARG = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"


def _load_model_with_best_attn(model_name: str, attn_implementation: str | None):
    """Try flash_attention_2 → sdpa → eager, return (model, actual_impl)."""
    dtype = torch.bfloat16

    candidates = []
    if attn_implementation:
        candidates.append(attn_implementation)
    if "flash_attention_2" not in candidates:
        candidates.append("flash_attention_2")
    if "sdpa" not in candidates:
        candidates.append("sdpa")
    candidates.append(None)  # eager fallback

    last_err = None
    for impl in candidates:
        kwargs = {"trust_remote_code": True, _DTYPE_KWARG: dtype}
        if impl:
            kwargs["attn_implementation"] = impl
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            return model, impl or "eager"
        except (ValueError, ImportError, NotImplementedError) as exc:
            last_err = exc
            continue

    raise RuntimeError(f"Could not load {model_name} with any attention backend") from last_err


class PromptGuardGenModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        freeze_backbone: bool = False,
        attn_implementation: str | None = None,
    ):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model, self.attn_implementation = _load_model_with_best_attn(
            model_name, attn_implementation
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    def forward(self, input_ids, attention_mask=None, labels=None, position_ids=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
        )
