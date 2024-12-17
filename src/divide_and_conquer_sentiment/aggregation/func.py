from base import AggregatorBase


class FuncAggregator(AggregatorBase):
    def aggregate(self, data):
        raise NotImplementedError
