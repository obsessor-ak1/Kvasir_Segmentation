import torch
from torch import nn

class DiceLoss(nn.Module):
    """Dice Loss Implementation for Binary Segmentation."""
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred, y):
        y_pred = torch.sigmoid(y_pred)
        if y.ndim != 4:
            y = y.unsqueeze(dim=1)
        intersection = (y_pred * y)
        union = y_pred + y
        dice = 2 * (intersection + 1e-6) / (union + 1e-6)
        dice = dice.mean()
        return 1 - dice


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, y_pred, y):
        return self.alpha * self.bce(y_pred, y) + (1 - self.alpha) * self.dice(y_pred, y)