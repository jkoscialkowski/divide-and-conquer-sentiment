import pytest
import torch

from divide_and_conquer_sentiment.aggregation.sawon import SawonAggregator
from tests.test_utils import assert_tensor_lists_equal

#
# @pytest.fixture
# def mock_sawon(mocker):
#     mock_model = mocker.Mock(spec=SentimentModel)
#     mock_model.predict.return_value = [torch.zeros(3), torch.zeros(3), torch.zeros(3)]
#     sawon = SawonAggregator(mock_model)
#
#     return sawon


@pytest.mark.parametrize(
    "input, expected",
    [
        (torch.ones((5, 3)), torch.zeros(3)),
        (torch.ones(1, 3) / 2, torch.zeros(3)),
        (torch.ones((5, 3)) / 2, torch.ones(3) / 2),
        (torch.tensor([[0.10, 0.91, 0.00], [0.9000, 0.1000, 0.0000]]), torch.tensor([0.9000, 0.1000, 0.0000])),
    ],
)
def test_awon(input, expected):
    sawon = SawonAggregator()

    output = sawon.awon(input, default=torch.zeros(3))

    assert torch.equal(output, expected)


@pytest.mark.parametrize(
    "subpredictions, defaults, expected",
    [
        (
            [torch.ones((5, 3)), torch.ones(1, 3), torch.ones((5, 3)) / 2],
            [torch.zeros(3)] * 3,
            [torch.zeros(3), torch.zeros(3), torch.ones(3) / 2],
        )
    ],
)
def test_aggregate(subpredictions, defaults, expected):
    sawon = SawonAggregator()
    output = sawon.aggregate(subpredictions, defaults=defaults)
    assert_tensor_lists_equal(output, expected)
