from abc import ABC, abstractmethod

import torch


class AggregatorBase(ABC):
    @abstractmethod
    def aggregate(self, subpredictions: list[torch.Tensor]) -> list[torch.Tensor]:
        pass
