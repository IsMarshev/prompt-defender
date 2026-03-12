"""
Stream Qwen3Guard model.

Architecture (Section 4.1 of the paper):
  - Qwen3 backbone produces last hidden state h
  - Two parallel classification heads (query & response), each with:
      x = LayerNorm(W_pre @ h)
      y_risk = Softmax(W_risk @ x)   -- 3 classes: safe, controversial, unsafe
      y_cat  = Softmax(W_cat  @ x)   -- category prediction
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


NUM_RISK_CLASSES = 3   # safe=0, controversial=1, unsafe=2
NUM_QUERY_CAT = 10     # 9 categories + none
NUM_RESPONSE_CAT = 9   # 8 categories + none (no Jailbreak for responses)


class ClassificationHead(nn.Module):
    """One classification head: pre-projection + LayerNorm -> risk & category logits."""

    def __init__(self, hidden_size: int, num_cat: int):
        super().__init__()
        self.pre = nn.Linear(hidden_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.risk_head = nn.Linear(hidden_size, NUM_RISK_CLASSES)
        self.cat_head = nn.Linear(hidden_size, num_cat)

    def forward(self, h: torch.Tensor):
        x = self.ln(self.pre(h))
        return self.risk_head(x), self.cat_head(x)


class StreamGuard(nn.Module):
    """Stream Qwen3Guard with two classification heads on top of a Qwen3 backbone."""

    def __init__(self, backbone_name: str, freeze_backbone: bool = False):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            backbone_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        hidden_size = self.backbone.config.hidden_size

        self.query_head = ClassificationHead(hidden_size, NUM_QUERY_CAT)
        self.response_head = ClassificationHead(hidden_size, NUM_RESPONSE_CAT)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        query_end_idx: torch.Tensor,
    ):
        """
        Args:
            input_ids:      (B, L)
            attention_mask: (B, L)
            query_end_idx:  (B,) index of the last query token (e.g. <|im_end|>)
                            per sample, used to extract the query representation.

        Returns:
            q_risk:  (B, NUM_RISK_CLASSES)       query risk logits
            q_cat:   (B, NUM_QUERY_CAT)          query category logits
            r_risk:  (B, L, NUM_RISK_CLASSES)     per-token response risk logits
            r_cat:   (B, L, NUM_RESPONSE_CAT)     per-token response category logits
        """
        h = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state  # (B, L, H)

        # --- Query head: classify at the query end token only ---
        batch_idx = torch.arange(h.size(0), device=h.device)
        h_query = h[batch_idx, query_end_idx]  # (B, H)
        q_risk, q_cat = self.query_head(h_query)

        # --- Response head: classify every token ---
        r_risk, r_cat = self.response_head(h)

        return q_risk, q_cat, r_risk, r_cat
