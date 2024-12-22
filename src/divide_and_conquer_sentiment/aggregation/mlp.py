from dataclasses import dataclass

import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn, optim
from torch.nn import functional as F

from .base import AggregatorBase


@dataclass()
class MLPAggregator(AggregatorBase):
    model: "MLP"
    trainer: L.Trainer

    def __init__(self, mlp: "MLP"):
        self.model = mlp
        self.trainer = L.Trainer(max_epochs=10)

    def aggregate(self, subpredictions: list[torch.Tensor]) -> list[torch.Tensor]:
        # TODO: Check if the model has been trained
        return self.model.predict(subpredictions)

    def train(self, train_loader, val_loader, epochs=10):
        self.trainer.fit(self.model, train_loader, val_loader)


class MLP(L.LightningModule):
    def __init__(
        self, input_size: int, output_size: int, hidden_layer_sizes: tuple[int, ...] = (128, 64), lr: float = 0.001
    ):
        super().__init__()
        sizes = [input_size] + list(hidden_layer_sizes) + [output_size]
        self.layers = nn.ModuleList([nn.Linear(inp, out) for inp, out in zip(sizes[:-1], sizes[1:])])
        self.lr = lr

    def predict(self, subpredictions: list[torch.Tensor]) -> list[torch.Tensor]:
        return [F.sigmoid(self(subpred)) for subpred in subpredictions]

    def forward(self, subpredictions: list[torch.Tensor]) -> torch.Tensor:
        x = torch.vstack(list(map(self._feature_func, subpredictions)))
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def _feature_func(self, pred_array: torch.Tensor):
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
