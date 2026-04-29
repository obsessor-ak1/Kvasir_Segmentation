"""A script to train UNet model for medical image segmentation."""

from argparse import ArgumentParser
import sys

import torch
from torch.amp import autocast, GradScaler
from torch import nn
from torchvision.transforms import v2

from ignite import utils
import ignite.distributed as idist
from ignite.engine import Engine, Events
from ignite.handlers import (
    Checkpoint,
    DiskSaver,
    LRScheduler,
    global_step_from_engine,
    EarlyStopping,
)
from ignite.handlers.tqdm_logger import ProgressBar
from ignite.handlers.wandb_logger import WandBLogger, OutputHandler
from ignite.metrics import (
    Accuracy,
    ConfusionMatrix,
    mIoU,
    DiceCoefficient,
    Loss,
    RunningAverage,
)

from seg_modules.data import Kvasir1Dataset
from seg_modules.unet import UNet

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 16
NUM_WORKERS = 2

train_transform = v2.Compose(
    [
        v2.Resize(IMAGE_SIZE, antialias=True),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        v2.ToDtype(torch.float32, scale=True),
    ]
)

val_transform = v2.Compose(
    [v2.Resize(IMAGE_SIZE, antialias=True), v2.ToDtype(torch.float32, scale=True)]
)

train_bar = ProgressBar(persist=False, desc="Training")
val_bar = ProgressBar(persist=False, desc="Validation")

def get_distributed_config():
    if not torch.cuda.is_available():
        print("Detected NO GPU. Running in CPU-only mode. 🐢")
        return None, 1
    num_gpus = torch.cuda.device_count()    
    if num_gpus == 1:
        print(f"Detected 1 GPU: {torch.cuda.get_device_name(0)}. Running in single-process mode. 🚀")
        return None, None
        
    if sys.platform == "win32":
        print(f"Detected Windows with {num_gpus} GPUs. Forcing 'gloo' backend. 🚀🚀")
        return "gloo", num_gpus
        
    print(f"Detected Linux with {num_gpus} GPUs. Using optimal 'nccl' backend. 🚀🚀")
    return "nccl", num_gpus



def get_dataloaders(path, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    with idist.one_rank_first(local=True):
        train_dataset = Kvasir1Dataset(
            root_path=path, transform=train_transform, mode="train"
        )
        val_dataset = Kvasir1Dataset(
            root_path=path, transform=val_transform, mode="val"
        )

    train_loader = idist.auto_dataloader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    val_loader = idist.auto_dataloader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader


class TrainerProcess:
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

    @staticmethod
    def binary_output_transform(output):
        y_pred, y = output[1:]
        y_pred = y_pred.squeeze(dim=1).sigmoid().round().long()
        y_pred = utils.to_onehot(y_pred, num_classes=2)
        y = y.squeeze(dim=1).long()
        return y_pred, y


class EvaluatorProcess:
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

    @staticmethod
    def binary_output_transform(output):
        y_pred, y = output
        y_pred = y_pred.squeeze(dim=1).sigmoid().round().long()
        y_pred = utils.to_onehot(y_pred, num_classes=2)
        y = y.squeeze(dim=1).long()
        return y_pred, y


def log_trainer_metrics(trainer_engine):
    train_bar.log_message(f"Train Epoch: {trainer_engine.state.epoch}")
    for name, val in trainer_engine.state.metrics.items():
        if name != "confusion_matrix":
            train_bar.log_message(f"{name}: {val}")


def log_evaluator_metrics(validator_engine):
    val_bar.log_message("Validation Metrics:")
    for name, val in validator_engine.state.metrics.items():
        if name != "confusion_matrix":
            val_bar.log_message(f"{name}: {val}")


def attach_wandb_logger(trainer, evaluator, config):
    """Attach Weights & Biases logger to the trainer and evaluator."""
    step_source = global_step_from_engine(trainer, Events.EPOCH_COMPLETED)
    logger = WandBLogger(
        project="Kvasir_Segmentation",
        config=config,
    )
    logger.attach(
        trainer,
        log_handler=OutputHandler(
            tag="training",
            metric_names="all",
            global_step_transform=lambda engine, _: engine.state.epoch,
        ),
        event_name=Events.EPOCH_COMPLETED,
    )
    logger.attach(
        evaluator,
        log_handler=OutputHandler(
            tag="validation",
            metric_names="all",
            global_step_transform=step_source,
        ),
        event_name=Events.EPOCH_COMPLETED,
    )
    print("Weights & Biases logger attached...✅")


def start_training(local_rank, config):
    device = idist.device()
    model = UNet(in_channels=3, num_classes=1)
    model = idist.auto_model(model)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    optimizer = idist.auto_optim(optimizer)
    grad_scaler = GradScaler(enabled=device.type == "cuda" and config["use_amp"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    train_loader, val_loader = get_dataloaders(
        config["data_path"], config["batch_size"], config["num_workers"]
    )

    # Creating the metrics
    cm_train = ConfusionMatrix(
        num_classes=2, output_transform=TrainerProcess.binary_output_transform
    )
    cm_val = ConfusionMatrix(
        num_classes=2, output_transform=EvaluatorProcess.binary_output_transform
    )
    train_metrics = {
        "loss": RunningAverage(output_transform=lambda output: output[0]),
        "accuracy": Accuracy(output_transform=TrainerProcess.binary_output_transform),
        "confusion_matrix": cm_train,
        "mIoU": mIoU(cm_train),
        "dice": DiceCoefficient(cm_train),
    }
    val_metrics = {
        "loss": Loss(loss_fn, output_transform=lambda output: (output[0].view(output[1].shape), output[1].float())),
        "accuracy": Accuracy(output_transform=EvaluatorProcess.binary_output_transform),
        "confusion_matrix": cm_val,
        "mIoU": mIoU(cm_val),
        "dice": DiceCoefficient(cm_val),
    }
    # Preparing the engine
    train_process = TrainerProcess(
        model, optimizer, loss_fn, grad_scaler, device, config["use_amp"]
    )
    val_process = EvaluatorProcess(model, device)
    trainer = Engine(train_process)
    evaluator = Engine(val_process)
    for name, metric in train_metrics.items():
        metric.attach(trainer, name)
    for name, metric in val_metrics.items():
        metric.attach(evaluator, name)
    schedule_handler = LRScheduler(scheduler)
    trainer.add_event_handler(Events.EPOCH_STARTED, schedule_handler)
    # Creating and loading checkpoints
    to_track = {
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": scheduler,
        "grad_scaler": grad_scaler,
        "trainer": trainer,
    }
    if config["checkpoint_path"]:
        Checkpoint.load_objects(to_load=to_track, filename=config["checkpoint_path"])

    checkpoint_handler = Checkpoint(
        to_save=to_track, 
        save_handler=DiskSaver("./checkpoints", create_dir=True, require_empty=False),
        score_function=lambda engine: engine.state.metrics["dice"][1].item(),
        score_name="dice",
        n_saved=3,
        global_step_transform=global_step_from_engine(trainer, Events.EPOCH_COMPLETED),
    )
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler)
    # Adding early stopping criteria
    stopping_handler = EarlyStopping(
        mode="max",
        patience=5,
        score_function=lambda engine: engine.state.metrics["dice"][1].item(),
        trainer=trainer,
    )
    evaluator.add_event_handler(Events.EPOCH_COMPLETED, stopping_handler)
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader)
    )
    if idist.get_rank() == 0:
        train_bar.attach(trainer, metric_names=["loss"])
        val_bar.attach(evaluator)
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED, log_trainer_metrics
        )
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, log_evaluator_metrics)
        attach_wandb_logger(trainer, evaluator, config)
    trainer.run(train_loader, max_epochs=config["num_epochs"])


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("data_path", type=str, help="Dataset path")
    parser.add_argument("num_epochs", type=int, help="Number of epochs to train")
    parser.add_argument(
        "batch_size", type=int, help="Batch size for training and validation"
    )
    parser.add_argument(
        "learning_rate", type=float, help="Learning rate for the optimizer"
    )

    parser.add_argument(
        "--use_amp", action="store_true", help="Enable Automatic Mixed Precision"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=NUM_WORKERS,
        help="Number of workers for dataloaders",
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None, help="Path to a checkpoint to load"
    )

    args = parser.parse_args()
    config = vars(args)

    backend, nproc = get_distributed_config()
    with idist.Parallel(backend=backend, nproc_per_node=nproc) as parallel:
        parallel.run(start_training, config)


if __name__ == "__main__":
    main()
