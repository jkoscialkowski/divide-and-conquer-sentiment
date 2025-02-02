from dataclasses import dataclass

import torch

from divide_and_conquer_sentiment.subprediction.absa import SubpredictorBase

from .aggregation.base import AggregatorBase


@dataclass
class DACSModel:
    subpredictor: SubpredictorBase
    aggregator: AggregatorBase

    def __init__(
        self,
        subpredictor: SubpredictorBase,
        aggregator: AggregatorBase,
    ):
        self.subpredictor = subpredictor
        self.aggregator = aggregator

    def predict(self, inputs: list[str], **kwargs) -> torch.Tensor:
        return self.aggregator.aggregate(self.subpredictor.predict(inputs), **kwargs)

    def classify(self, inputs: list[str], **kwargs) -> torch.Tensor:
        return self.aggregator.classify(self.subpredictor.predict(inputs), **kwargs)
