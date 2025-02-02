from abc import ABC, abstractmethod

import torch
from setfit.span.modeling import PolarityModel

class SentimentModel(ABC):
    @abstractmethod
    def predict(self, subsentences: list[str]) -> list[torch.Tensor]:
        pass

class PolaritySentimentModel(SentimentModel):

    def __init__(self, sentiment_model: PolarityModel):
        self.model = sentiment_model

    def predict(self, subsentences: list[str]) -> list[torch.Tensor]:
        try:
            torch_matrix = self.model.predict_proba(subsentences, show_progress_bar=False)
        except ValueError:
            print(f'Prediction failed for subsentences {subsentences}, len = {len(subsentences)}')
            raise ValueError
        tensor_list = [torch_matrix[i][1:4].unsqueeze(0) for i in range(torch_matrix.size(0))]
        return tensor_list