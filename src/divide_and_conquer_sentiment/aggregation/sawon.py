from dataclasses import dataclass

import torch

from .base import AggregatorBase


@dataclass()
class SawonAggregator(AggregatorBase):
    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold

    def aggregate(self, subpredictions: list[torch.Tensor], defaults: list[torch.Tensor] | None = None) -> torch.Tensor:
        if defaults is None:
            raise ValueError("Defaults should be provided")

        aggregated = []
        for subpred, default in zip(subpredictions, defaults):
            aggregated.append(self.awon(subpred, default))
        return torch.vstack(aggregated)

    def awon(self, scores_array: torch.Tensor, default) -> torch.Tensor:
        if scores_array.shape[0] == 1:
            return default
        mask = scores_array[:, 1] <= self.threshold

        if not max(mask):
            return default
        scores_masked = scores_array[mask]
        return torch.mean(scores_masked, dim=0)
