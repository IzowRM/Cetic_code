import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossMultiLabel(nn.Module):
    """
    Focal Loss pour classification multilabel.
    - logits: [B, C] (sortie brute du modèle)
    - targets: [B, C] en {0,1} float
    """
    def __init__(self, gamma=2.0, alpha=None, reduction="mean", eps=1e-9):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # None, float, ou tensor [C]
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits, targets):
        targets = targets.float()

        # probas sigmoid
        probs = torch.sigmoid(logits).clamp(self.eps, 1. - self.eps)

        # p_t = p si y=1 sinon (1-p)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # BCE par élément (sans réduction)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # facteur focal
        focal_factor = (1 - p_t) ** self.gamma

        loss = focal_factor * bce

        # alpha weighting (optionnel)
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            else:
                # alpha tensor [C] -> broadcast sur batch
                alpha = self.alpha.to(logits.device)
                alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
