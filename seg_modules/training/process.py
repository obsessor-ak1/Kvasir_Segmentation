from abc import ABC, abstractmethod

import torch
from torch.amp import autocast


class ProcessFunction(ABC):
    """A Process function for torch ignite engine."""
    @abstractmethod
    def __call__(self, X):
        pass


class SegmentationTrainerProcess(ProcessFunction):
    """Trainer process for segmentation models."""
    def __init__(
        self, model, optimizer, loss_fn, grad_scaler, device="cuda", use_amp=True
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.grad_scaler = grad_scaler
        self.device = device
        self.use_amp = use_amp

    def __call__(self, engine, batch):
        self.model.train()
        self.optimizer.zero_grad()

        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y = (y > 127)
        device_type = self.device.type
        with autocast(device_type=device_type, enabled=self.use_amp and device_type=="cuda"):
            y_pred = self.model(x)
            loss = self.loss_fn(y_pred.view(y.shape), y.float())

        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        return loss.item(), y_pred.detach(), y.detach()


class SegmentationEvaluatorProcess(ProcessFunction):
    """Evaluator process for segmentation models."""
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device

    def __call__(self, engine, batch):
        self.model.eval()

        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y = (y > 127)
        with torch.inference_mode():
            y_pred = self.model(x)

        return y_pred, y


class UNetPlusPlusTrainerProcess(ProcessFunction):
    """Trainer process for UNetPlusPlus model supporting deep supervision."""
    def __init__(
        self, model, optimizer, loss_fn, grad_scaler, device="cuda", use_amp=True
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.grad_scaler = grad_scaler
        self.device = device
        self.use_amp = use_amp

    def __call__(self, engine, batch):
        self.model.train()
        self.optimizer.zero_grad()

        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        y = (y > 127)
        device_type = self.device.type
        with autocast(device_type=device_type, enabled=self.use_amp and device_type=="cuda"):
            y_preds = self.model(x)
            # UNetPlusPlus returns a list of outputs during training.
            # We compute deep supervision loss as the average loss across all outputs.
            loss = sum(self.loss_fn(pred.view(y.shape), y.float()) for pred in y_preds) / len(y_preds)

        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        # Detach the final branch's predictions for metric calculations
        return loss.item(), y_preds[-1].detach(), y.detach()


