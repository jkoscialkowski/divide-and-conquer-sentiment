from dataclasses import dataclass
from .base import AggregatorBase
import torch
from divide_and_conquer_sentiment import SentimentModel

@dataclass()
class SawonAggregator(AggregatorBase):

    def __init__(self, sentiment_model: SentimentModel, treshold: float = 0.9):

        self.sentiment_model = sentiment_model
        self.treshold = treshold

    def aggregate(self, subpredictions: list[torch.Tensor] , **kwargs) -> list[torch.Tensor]:
        passages = kwargs['passages']
        defaults =  self.sentiment_model.predict(passages)
        result = []
        for i in range(len(subpredictions)):
            scores_array = subpredictions[i]
            default = defaults[i]
            result.append(self.awon(scores_array, default))
        return result

    def awon(self, scores_array: torch.Tensor, default) -> torch.Tensor:
        if scores_array.shape[0] == 1:
            return default
        mask = scores_array[:, 1] <= self.treshold
        if max(mask) == False:
            return default
        scores_masked = scores_array[mask]
        return torch.mean(scores_masked, dim=0)