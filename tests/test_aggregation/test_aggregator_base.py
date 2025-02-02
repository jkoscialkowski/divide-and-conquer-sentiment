import pytest
import torch
from divide_and_conquer_sentiment.aggregation import AggregatorBase


class TestAggregatorBase(AggregatorBase):
    def __init__(self):
        self.nothing = None
    def aggregate(self, x):
        return list[torch.Tensor.new_empty()]

@pytest.mark.parametrize(
    "inputs, expected",
    [
        ([torch.tensor([0.8,0.1,0.1])],[0]),
        ([torch.tensor([0.1,0.8,0.1])],[1]),
        ([torch.tensor([0.1,0.1,0.8])],[2]),
        ([torch.tensor([0.1,0.1,0.1])],[0]),
        ([torch.tensor([0.1,0.1,0.8]),torch.tensor([0.1,0.1,0.1])],[2,0]),
    ]
)
def test_classify(inputs, expected):
    base_model = TestAggregatorBase()
    output = base_model.classify(inputs)

    assert output == expected