from unittest import mock

import pytest
import torch
from transformers import pipeline
from src.divide_and_conquer_sentiment.aggregation.sawon import SawonAggregator

# Helper function to assert lists of tensors
def assert_tensor_lists_equal(list1, list2):
    assert len(list1) == len(list2), f"Lengths do not match: {len(list1)} != {len(list2)}"
    for tensor1, tensor2 in zip(list1, list2):
        assert torch.equal(tensor1, tensor2), f"Tensors do not match: {tensor1} != {tensor2}"

@pytest.fixture
def mocks(mocker):
    mock_pipeline = mocker.Mock(spec=pipeline)
    mock_pipeline._postprocess_params = {"top_k": None}
    passages = [""]
    sawon = SawonAggregator(passages, mock_pipeline)

    mock_tensor = [torch.zeros(3), torch.zeros(3), torch.zeros(3)]
    # Mock `my_function` to return `mock_tensor`
    mocker.patch.object(sawon,'full_passage_prediction', return_value=mock_tensor)
    return sawon

def test_full_passage_prediction(mocks):
    expected_tensor = [torch.zeros(3), torch.zeros(3), torch.zeros(3)]
    result = mocks.full_passage_prediction()

    assert_tensor_lists_equal(result, expected_tensor), "The tensors do not match!"


@pytest.mark.parametrize(
    "input, expected",
    [(torch.ones((5, 3)), torch.zeros(3))
        , (torch.ones(3), torch.zeros(3))
        , (torch.ones((5, 3))/2, torch.ones(3)/2)
        ,( torch.tensor([[0.10, 0.91, 0.00],[0.9000, 0.1000, 0.0000]]), torch.tensor([0.9000, 0.1000, 0.0000]) )],
)

def test_awon(input, expected, mocks):
    sawon =mocks

    output = sawon.awon(input,default=torch.zeros(3))

    assert torch.equal(output, expected)

@pytest.mark.parametrize(
    "subpredictions, expected",
    [ ([torch.ones((5, 3)), torch.ones(3), torch.ones((5, 3))/2] , [torch.zeros(3),torch.zeros(3),torch.ones(3)/2])
     ]
)
def test_aggregate(subpredictions, expected, mocks):
    sawon =mocks
    output = sawon.aggregate(subpredictions)
    assert_tensor_lists_equal(output, expected)