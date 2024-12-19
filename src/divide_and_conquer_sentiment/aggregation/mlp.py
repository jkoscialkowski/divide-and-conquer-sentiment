from dataclasses import dataclass

import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn, optim
from torch.nn import functional as F

from .base import AggregatorBase


@dataclass()
class MLPAggregator(AggregatorBase):
    model: L.LightningModule
    trainer: L.Trainer

    def aggregate(self, data):
        # TODO: Check if the model has been trained
        return self.model.predict(data)

    def feature_func(self, pred_array: torch.Tensor):
        return torch.hstack(
            [
                pred_array.mean(0),
                pred_array.min(0),
                pred_array.max(0),
                pred_array.std(0),
                torch.ptp(pred_array, 0),
                torch.bincount(torch.argmax(pred_array, 1), minlength=4),
                pred_array.shape[0],
            ]
        )

    def train(self, train_loader, val_loader, epochs=10):
        self.trainer.fit(self.model, train_loader, val_loader)


class MLP(L.LightningModule):
    def __init__(self, input_size, output_size, hidden_layer_sizes=(128, 64)):
        super().__init__()
        sizes = [input_size] + list(hidden_layer_sizes) + [output_size]
        self.layers = nn.ModuleList([nn.Linear(inp, out) for inp, out in zip(sizes[:-1], sizes[1:])])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = optim.Adam(self.parameters())
        return optimizer
