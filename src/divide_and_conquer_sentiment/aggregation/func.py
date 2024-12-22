from typing import Callable

import torch

from .base import AggregatorBase


class FuncAggregator(AggregatorBase):
    def __init__(self, agg_func: Callable[[torch.Tensor], torch.Tensor]):
        """

        :param agg_func: A function that takes a 2-dim tensor, performs some aggregation and returns a 1-dim tensor
        """
        self.agg_func = agg_func

    def aggregate(self, subpredictions: list[torch.Tensor]) -> list[torch.Tensor]:
        return list(map(self.agg_func, subpredictions))
