from dataclasses import dataclass

import lightning as L
import torch
from datasets import Dataset
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .base import AggregatorBase


@dataclass()
class MLPAggregator(AggregatorBase):
    model: "MLP"
    trainer: L.Trainer

    def __init__(self, mlp: "MLP"):
        self.model = mlp
        self.trainer = L.Trainer(callbacks=[EarlyStopping(monitor="val_loss", patience=3, mode="min")])

    def aggregate(self, subpredictions: list[torch.Tensor], defaults=None) -> torch.Tensor:
        # TODO: Check if the model has been trained
        return self.model.predict(subpredictions)

    def train(self, train_dataset: Dataset, val_dataset: Dataset) -> None:
        """
        Both datasets should have a column "subpredictions" containing the subpredictions with a size that complies with
        the input size of the model.
        :param train_dataset: Training dataset
        :param val_dataset: Validation dataset
        """
        train_loader = DataLoader(
            train_dataset.with_format("torch"), batch_size=256, shuffle=True, collate_fn=self.collate_fn
        )
        val_loader = DataLoader(val_dataset.with_format("torch"), batch_size=256, collate_fn=self.collate_fn)
        self.trainer.fit(model=self.model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    @staticmethod
    def collate_fn(x):
        return ([item["subpreds"] for item in x], torch.tensor([item["label"] for item in x]))


class MLP(L.LightningModule):
    def __init__(
        self, input_size: int, output_size: int, hidden_layer_sizes: tuple[int, ...] = (128, 64), lr: float = 0.001
    ):
        super().__init__()
        sizes = [5 * input_size + 1] + list(hidden_layer_sizes) + [output_size]
        self.input_size = input_size
        self.layers = nn.ModuleList([nn.Linear(inp, out) for inp, out in zip(sizes[:-1], sizes[1:])])
        self.lr = lr

    def predict(self, subpredictions: list[torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            return F.softmax(self(subpredictions), dim=1)

    def forward(self, subpredictions: list[torch.Tensor]) -> torch.Tensor:
        x = torch.vstack(list(map(self._feature_func, subpredictions)))
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)

    def training_step(self, batch: tuple[list[torch.Tensor], torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, batch_size=len(x))
        return loss

    def validation_step(self, batch: tuple[list[torch.Tensor], torch.Tensor], batch_idx: int):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        val_accuracy = (y == torch.argmax(y_hat, dim=1)).float().mean()
        self.log("val_loss", val_loss, prog_bar=True, batch_size=len(x))
        self.log("val_accuracy", val_accuracy, prog_bar=True, batch_size=len(x))

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def _feature_func(self, pred_array: torch.Tensor):
        return torch.hstack(
            [
                pred_array.mean(dim=0),
                pred_array.min(dim=0).values,
                pred_array.max(dim=0).values,
                pred_array.std(dim=0, correction=0),
                torch.bincount(torch.argmax(pred_array, dim=1), minlength=self.input_size),
                torch.tensor(pred_array.shape[0], device=pred_array.device),
            ]
        )
