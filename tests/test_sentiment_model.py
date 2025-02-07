import pytest
import torch

from setfit.span.modeling import PolarityModel

from divide_and_conquer_sentiment import PolaritySentimentModel
from divide_and_conquer_sentiment import RobertaSentimentModel

import tests.test_utils as utils

@pytest.fixture
def mocks(mocker):
    mock_polarity = mocker.Mock(spec=PolarityModel)
    mock_roberta = mocker.Mock(spec=RobertaSentimentModel)
    return mock_polarity, mock_roberta

@pytest.mark.parametrize(
    "inputs, expected",
    [(torch.tensor([[0.0069, 0.0068, 0.0235, 0.9628]]), [torch.tensor([[.0068, 0.0235, 0.9628]])])
     ,(torch.tensor([[0.0069, 0.0068, 0.0235, 0.9628],
        [0.0651, 0.6133, 0.2772, 0.0444]]),[torch.tensor([[0.0068, 0.0235, 0.9628]]), torch.tensor([[0.6133, 0.2772, 0.0444]])] )
       ]
)
def test_predict_polarity(mocks, inputs, expected):
    mock_polarity, mock_roberta = mocks
    mock_polarity.predict_proba.return_value = inputs
    sentiment_model = PolaritySentimentModel(mock_polarity)
    output =sentiment_model.predict(["not important"])

    utils.assert_tensor_lists_equal(output, expected)

@pytest.mark.parametrize(
    "inputs, expected",
[([[{'label': 'negative', 'score': 0.0005},
  {'label': 'neutral', 'score': 0.0068},
  {'label': 'positive', 'score': 0.9926}],
 [{'label': 'negative', 'score': 0.9991},
  {'label': 'neutral', 'score': 0.0005},
  {'label': 'positive', 'score': 0.0002}],
 [{'label': 'negative', 'score': 0.0004},
  {'label': 'neutral', 'score': 0.9984},
  {'label': 'positive', 'score': 0.0011}]], [torch.tensor([[ 0.0005, 0.0068, 0.9926]]),torch.tensor([[0.9991, 0.0005, 0.0002]]),torch.tensor([[0.0004, 0.9984, 0.0011]])] )
 ])

def test_predict_roberta(mocks, inputs, expected):
    mock_polarity, mock_roberta = mocks
    mock_roberta.return_value = inputs
    sentiment_model = RobertaSentimentModel(mock_roberta)
    output =sentiment_model.predict(["not important"])

    utils.assert_tensor_lists_equal(output, expected)