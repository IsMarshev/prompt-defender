import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class PromptGuardGenModel(nn.Module):
    def __init__(self, model_name: str, freeze_backbone: bool = False):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
