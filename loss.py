import torch

import torch.nn as nn
import torch.nn.functional as F


class QueryLoss(nn.Module):
    def __init__(self, safe_label=0):
        super().__init__()
        self.safe_label = safe_label

    def forward(self, y_risk, y_cat, logits_risk, logits_cat):

        loss_risk = F.cross_entropy(logits_risk, y_risk)

        mask = y_risk != self.safe_label

        if mask.any():
            loss_cat = F.cross_entropy(
                logits_cat[mask],
                y_cat[mask]
            )
        else:
            loss_cat = torch.tensor(
                0.0,
                device=logits_risk.device
            )

        return loss_risk + loss_cat
    
class ResponseLoss(nn.Module):
    def __init__(self, safe_label=0):
        super().__init__()
        self.safe_label = safe_label

    def forward(self, y_risk, y_cat, logits_risk, logits_cat):

        y_risk = y_risk.flatten()
        y_cat = y_cat.flatten()

        logits_risk = logits_risk.view(-1, logits_risk.size(-1))
        logits_cat = logits_cat.view(-1, logits_cat.size(-1))

        loss_risk = F.cross_entropy(logits_risk, y_risk, reduction="mean")

        y_cat = y_cat.masked_fill(y_risk == self.safe_label, -100)

        if (y_cat != -100).any():
            loss_cat = F.cross_entropy(
                logits_cat,
                y_cat,
                ignore_index=-100,
                reduction="mean"
            )
        else:
            loss_cat = torch.tensor(0.0, device=logits_risk.device)

        return loss_risk + loss_cat