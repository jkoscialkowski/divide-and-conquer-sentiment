from abc import ABC, abstractmethod

import torch


class AggregatorBase(ABC):
    @abstractmethod
    def aggregate(self, subpredictions: list[torch.Tensor], defaults: list[torch.Tensor] | None = None) -> torch.Tensor:
        pass

    def classify(self, subpredictions: list[torch.Tensor], defaults: list[torch.Tensor] | None = None) -> torch.Tensor:
        return torch.argmax(self.aggregate(subpredictions, defaults), dim=1)
