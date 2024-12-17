from abc import ABC, abstractmethod

import torch


class SubpredictorBase(ABC):
    @abstractmethod
    def predict(self, inputs: list[str]) -> list[torch.Tensor]:
        pass
