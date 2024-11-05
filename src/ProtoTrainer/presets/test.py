from ProtoTrainer.presets.mixins import SaveAndLoadMixin
from ProtoTrainer.core.trainer import BaseTrainer
from utils.gradient_logger import LogGradientsMixin
import torch

class TrainerA(BaseTrainer, SaveAndLoadMixin, LogGradientsMixin):
    def preprocess(self, batch: dict) -> torch.Tensor:
        pass

    def compute_loss(self, processed_batch) -> torch.Tensor | dict[torch.Tensor]:
        pass

    def __init__(self):
        super().__init__()


t = TrainerA()

