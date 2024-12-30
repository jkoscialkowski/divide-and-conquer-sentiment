from unittest import mock

import pytest
import torch
from transformers import pipeline
from divide_and_conquer_sentiment.aggregation.sawon import SawonAggregator

# Helper function to assert lists of tensors
def assert_tensor_lists_equal(list1, list2):
    assert len(list1) == len(list2), f"Lengths do not match: {len(list1)} != {len(list2)}"
    for tensor1, tensor2 in zip(list1, list2):
        assert torch.equal(tensor1, tensor2), f"Tensors do not match: {tensor1} != {tensor2}"

@pytest.fixture
def mocks(mocker):
    mock_pipeline = mocker.Mock(spec=pipeline)
    mock_pipeline.return_value = [torch.zeros(3), torch.zeros(3), torch.zeros(3)]
    sawon = SawonAggregator(mock_pipeline)

    return sawon

@pytest.mark.parametrize(
    "input, expected",
    [(torch.ones((5, 3)), torch.zeros(3))
        , (torch.ones(1, 3)/2, torch.zeros(3))
        , (torch.ones((5, 3))/2, torch.ones(3)/2)
        ,( torch.tensor([[0.10, 0.91, 0.00],[0.9000, 0.1000, 0.0000]]), torch.tensor([0.9000, 0.1000, 0.0000]) )],
)

def test_awon(input, expected, mocks):
    sawon =mocks

    output = sawon.awon(input,default=torch.zeros(3))

    assert torch.equal(output, expected)

@pytest.mark.parametrize(
    "subpredictions, expected",
    [ ([torch.ones((5, 3)), torch.ones(1,3), torch.ones((5, 3))/2] , [torch.zeros(3),torch.zeros(3),torch.ones(3)/2])
     ]
)
def test_aggregate(subpredictions, expected, mocks):
    sawon =mocks
    output = sawon.aggregate(subpredictions, passages = [])
    assert_tensor_lists_equal(output, expected)