import pytest
import torch

from divide_and_conquer_sentiment import SentimentModel
from divide_and_conquer_sentiment.subprediction.sentence import Chunker, ChunkSubpredictor


@pytest.fixture
def mocks(mocker):
    mock_chunker = mocker.Mock(spec=Chunker)
    mock_model = mocker.Mock(spec=SentimentModel)
    # mock_pipeline._postprocess_params = {"top_k": None}
    return mock_chunker, mock_model


# def test_error_on_init_if_sentiment_model_returns_one_score(mocks):
#     mocks[1]._postprocess_params = {"top_k": 1}
#     with pytest.raises(ValueError):
#         ChunkSubpredictor(*mocks)


@pytest.mark.parametrize("inputs", [[], ["text11.text12", "text2", "text31.text32.text33"]])
def test_predict(inputs, mocks, mocker):
    # Given
    chunker, sentiment_model = mocks
    chunk_subpredictor = ChunkSubpredictor(chunker, sentiment_model)
    chunk_list_rv = [inp.split() for inp in inputs]
    chunker.chunk_list.return_value = chunk_list_rv
    mocker.patch.object(chunk_subpredictor, "_chunk_to_tensor")

    # When
    chunk_subpredictor.predict(inputs)

    # Then
    chunker.chunk_list.assert_called_once_with(inputs)
    assert chunk_subpredictor._chunk_to_tensor.call_args_list == list(map(mocker.call, chunk_list_rv))


@pytest.mark.parametrize(
    "inputs, expected",
    [
        (
            [torch.tensor([[0.1, 0.8, 0.1]]), torch.tensor([[0.1, 0.8, 0.1]])],
            torch.tensor([[0.1, 0.8, 0.1], [0.1, 0.8, 0.1]]),
        ),
        ([torch.tensor([[0.1, 0.8, 0.1]])], torch.tensor([[0.1, 0.8, 0.1]])),
    ],
)
def test__chunk_to_tensor(mocks, inputs, expected):
    chunker, sentiment_model = mocks
    sentiment_model.predict.return_value = inputs
    chunk_subpredictor = ChunkSubpredictor(chunker, sentiment_model)
    output = chunk_subpredictor._chunk_to_tensor(["it doesnt matter"])
    assert torch.equal(output, expected)
