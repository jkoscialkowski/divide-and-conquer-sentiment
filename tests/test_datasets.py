import pytest

from datasets import Dataset

from divide_and_conquer_sentiment.dataloaders import _train_val_test_split


@pytest.fixture
def dataset():
    return Dataset.from_list([{"number": x} for x in range(100)])


@pytest.mark.parametrize("val_test_perc,expected", [
    ((0.1, 0.1), (80, 10, 10)),
    ((0.1, 0.2), (70, 10, 20)),
    ((0.25, 0.25), (50, 25, 25))
])
def test_train_val_test_split(val_test_perc, expected, dataset):
    dt_dict = _train_val_test_split(dataset, val_test_perc, 123)
    assert list(dt_dict.shape.values()) == [(e, 1) for e in expected]