import pytest
import torch

from divide_and_conquer_sentiment.aggregation.base import AggregatorBase


class TestAggregatorBase(AggregatorBase):
    def aggregate(self, subpredictions: list[torch.Tensor], defaults: list[torch.Tensor] | None = None) -> torch.Tensor:
        return subpredictions[0]


@pytest.mark.parametrize(
    "inputs, expected",
    [
        (torch.tensor([[0.8, 0.1, 0.1]]), torch.Tensor([0])),
        (torch.tensor([[0.1, 0.8, 0.1]]), torch.Tensor([1])),
        (torch.tensor([[0.1, 0.1, 0.8]]), torch.Tensor([2])),
        (torch.tensor([[0.1, 0.1, 0.1]]), torch.Tensor([0])),
        (torch.tensor([[0.1, 0.1, 0.8], [0.1, 0.1, 0.1]]), torch.Tensor([2, 0])),
    ],
)
def test_classify(inputs, expected, mocker):
    aggregator = TestAggregatorBase()
    mocker.patch.object(aggregator, "aggregate", return_value=inputs)
    output = aggregator.classify([torch.Tensor([0])])

    assert torch.equal(output, expected)
