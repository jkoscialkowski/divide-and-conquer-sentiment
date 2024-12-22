import pytest
import torch
from transformers import pipeline

from divide_and_conquer_sentiment.subprediction.sentence import Chunker, ChunkSubpredictor


@pytest.fixture
def mocks(mocker):
    mock_chunker = mocker.Mock(spec=Chunker)
    mock_pipeline = mocker.Mock(spec=pipeline)
    mock_pipeline._postprocess_params = {"top_k": None}
    return mock_chunker, mock_pipeline


def test_error_on_init_if_sentiment_model_returns_one_score(mocks):
    mocks[1]._postprocess_params = {"top_k": 1}
    with pytest.raises(ValueError):
        ChunkSubpredictor(*mocks)


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


@pytest.mark.parametrize("inputs,preds", [([], [[]]), (["text11", "text12"], [[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])])
def test_chunk_to_tensor(inputs, preds, mocks):
    # Given
    labels = ["negative", "neutral", "positive"]
    chunker, sentiment_model = mocks
    chunk_subpredictor = ChunkSubpredictor(chunker, sentiment_model)
    sentiment_model.return_value = [
        [{"label": label, "score": score} for label, score in zip(labels, scores)] for scores in preds
    ]

    # When
    result = chunk_subpredictor._chunk_to_tensor(inputs)

    # Then
    sentiment_model.assert_called_once_with(inputs)
    assert torch.equal(result, torch.tensor(preds))
