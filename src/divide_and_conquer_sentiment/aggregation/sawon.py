from .base import AggregatorBase
import torch
from transformers import TextClassificationPipeline

class SawonAggregator(AggregatorBase):

    def __init__(self, passages: list[str], sentiment_model: TextClassificationPipeline, treshold: float = 0.9):
        self.passages = passages
        self.sentiment_model = sentiment_model
        self.treshold = treshold

    def aggregate(self, subpredictions: list[torch.Tensor]) -> list[torch.Tensor]:
        defaults = self.full_passage_prediction()
        result = []
        for i in range(len(subpredictions)):
            scores_array = subpredictions[i]
            default = defaults[i]
            result.append(self.awon(scores_array, default))
        return result

    def full_passage_prediction(self) -> list[torch.Tensor]:
        return self.sentiment_model.predict(self.passages)

    def awon(self, scores_array: torch.Tensor, default) -> torch.Tensor:
        if scores_array.dim() == 1:
            return default
        mask = scores_array[:, 1] <= self.treshold
        if max(mask) == False:
            return default
        scores_masked = scores_array[mask]
        return torch.mean(scores_masked, dim=0)