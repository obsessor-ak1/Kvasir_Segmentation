import torch
from torch import nn


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred, y):
        probs = torch.sigmoid(y_pred)
        if y.ndim == 3:
            y = y.unsqueeze(dim=1)

        y = y.float()
        probs = probs.reshape(probs.size(0), -1)
        y = y.reshape(y.size(0), -1)
        intersection = (probs * y).sum(dim=1)
        denominator = probs.sum(dim=1) + y.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, y_pred, y):
        return self.alpha * self.bce(y_pred, y) + (1 - self.alpha) * self.dice(y_pred, y)