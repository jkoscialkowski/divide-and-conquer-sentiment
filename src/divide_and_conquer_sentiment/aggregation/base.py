from abc import ABC, abstractmethod

import torch


class AggregatorBase(ABC):
    @abstractmethod
    def aggregate(self, subpredictions: list[torch.Tensor], **kwargs) -> list[torch.Tensor]:
        pass

    def classify(self, predictions: list[torch.Tensor]) -> list[int]:
        res = []
        for pred in predictions:
            res.append(torch.argmax(pred.squeeze()).item())
        return res
