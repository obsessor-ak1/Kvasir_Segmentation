from ignite import utils
import torch


def train_output_transform(output):
    """Transform for training metrics. output is (loss, y_pred, y)."""
    y_pred, y = output[1:]
    y_pred = y_pred.squeeze(dim=1).sigmoid().round().long()
    y_pred = utils.to_onehot(y_pred, num_classes=2)
    y = y.squeeze(dim=1).long()
    return y_pred, y


def val_output_transform(output):
    """Transform for validation/evaluation metrics. output is (y_pred, y)."""
    y_pred, y = output
    y_pred = y_pred.squeeze(dim=1).sigmoid().round().long()
    y_pred = utils.to_onehot(y_pred, num_classes=2)
    y = y.squeeze(dim=1).long()
    return y_pred, y
