import pytest
import torch

from divide_and_conquer_sentiment.aggregation.func import FuncAggregator


@pytest.mark.parametrize(
    "input, expected",
    [(torch.ones((5, 3)), 5 * torch.ones(3)), (torch.range(0, 11).reshape(4, -1), torch.tensor([18, 22, 26]))],
)
def test_sum(input, expected):
    def agg_func_sum(x: torch.Tensor):
        return x.sum(0)

    fa = FuncAggregator(agg_func_sum)

    assert torch.equal(fa.aggregate([input])[0], expected)


@pytest.mark.parametrize(
    "input, expected",
    [(torch.ones((5, 3)), torch.ones(3)), (torch.range(0, 11).reshape(4, -1), torch.tensor([4.5, 5.5, 6.5]))],
)
def test_mean(input, expected):
    def agg_func_mean(x: torch.Tensor):
        return x.mean(0)

    fa = FuncAggregator(agg_func_mean)

    assert torch.equal(fa.aggregate([input])[0], expected)
