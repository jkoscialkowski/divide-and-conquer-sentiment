from dataclasses import dataclass

from .aggregation.base import AggregatorBase
from .constituents import SubpredictorBase


@dataclass
class DACModel:
    subpredictor: SubpredictorBase
    aggregator: AggregatorBase

    def __init__(self, subpredictor: SubpredictorBase, aggregator: AggregatorBase):
        self.subpredictor = subpredictor
        self.aggregator = aggregator

    def predict(self, inputs: list[str]):
        return self.aggregator.aggregate(self.subpredictor.predict(inputs))
