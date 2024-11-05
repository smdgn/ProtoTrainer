import torch
import torch.optim.swa_utils as swa

from pathlib import Path
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.tracking import TensorBoardTracker


# move class to extra file to prevent circular dependencies
class TrainerInterface:
    """
    Attributes:
        tracker: The tracker accelerator is initialized with. Defaults to a *tensorboard* tracker
        device: The device the trainer is placed at. Defaults to GPU based training
        accelerator: The initialized accelerator instance
        model: The initialized model instance
        optimizer: The initialized optimizer instance
        scheduler: The initialized scheduler instance if given
        ema_model: The weight-averaged model instance if used
        train_dataloader: The dataloader used to train the model
        validation_dataloader: The dataloader used to validate the model, if given
        project_dir: The active working directory of the trainer instance.
        logging_dir: The directory used to store tensorboard event files
        epochs: The number of epochs to train the model
        total_steps: The total number of steps to train the model
        total_steps_per_epoch: The number of steps per epoch
        accelerator_initialized: If accelerator is fully initialized
        compiled: If the trainer instance is compiled
    """
    compiled: bool = False
    validation_dataloader:  DataLoader | None = None
    train_dataloader: DataLoader | None = None
    accelerator_initialized: bool = False
    accelerator: Accelerator | None = None
    tracker: TensorBoardTracker | None = None
    device: str | None = None
    project_dir: Path | None = None
    logging_dir: Path | None

    # dispatched variables
    model: torch.nn.Module | None = None
    ema_model: swa.AveragedModel | None = None
    optimizer: torch.optim.Optimizer | None = None
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

    # misc
    epochs: int = 0
    total_steps: int = 0
    total_steps_per_epoch: int = 0

    start_epoch: int = 1
    start_step: int = 1