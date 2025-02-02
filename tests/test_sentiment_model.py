import pytest
import torch
from setfit.span.modeling import PolarityModel

import tests.test_utils as utils
from divide_and_conquer_sentiment import PolaritySentimentModel


@pytest.fixture
def mocks(mocker):
    mock_model = mocker.Mock(spec=PolarityModel)

    return mock_model


@pytest.mark.parametrize(
    "inputs, expected",
    [
        (torch.tensor([[0.0069, 0.0068, 0.0235, 0.9628]]), [torch.tensor([[0.0068, 0.0235, 0.9628]])]),
        (
            torch.tensor([[0.0069, 0.0068, 0.0235, 0.9628], [0.0651, 0.6133, 0.2772, 0.0444]]),
            [torch.tensor([[0.0068, 0.0235, 0.9628]]), torch.tensor([[0.6133, 0.2772, 0.0444]])],
        ),
    ],
)
def test_predict(mocks, inputs, expected):
    mock_model = mocks
    mock_model.predict_proba.return_value = inputs
    sentiment_model = PolaritySentimentModel(mock_model)
    output = sentiment_model.predict(["not important"])

    utils.assert_tensor_lists_equal(output, expected)

    # assert torch.equal(output, expected)
