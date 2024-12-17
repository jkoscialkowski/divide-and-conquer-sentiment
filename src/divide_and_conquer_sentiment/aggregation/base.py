from abc import ABC, abstractmethod


class AggregatorBase(ABC):
    @abstractmethod
    def aggregate(self, data):
        pass
