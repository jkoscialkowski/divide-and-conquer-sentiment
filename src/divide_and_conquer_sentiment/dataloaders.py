import os
import tempfile
from datasets import Dataset, DatasetDict, load_dataset
from kaggle import api


def load_kaggle_dataset(
    kaggle_dataset_path: str,
    val_test_perc: tuple[float, float] | None,
    seed: int | None
) -> Dataset | DatasetDict:
    dataset = _download_from_kaggle(kaggle_dataset_path)

    if val_test_perc and seed:
        return _train_val_test_split(dataset, val_test_perc, seed)
    elif val_test_perc is None and seed is None:
        return dataset


def _download_from_kaggle(kaggle_dataset_path: str) -> Dataset:
    with tempfile.TemporaryDirectory() as tmp_dirname:
        api.dataset_download_files(kaggle_dataset_path, path=tmp_dirname, unzip=True)
        csv_paths = [path for path in os.listdir(tmp_dirname) if path.endswith(".csv")]
        return load_dataset("csv", data_files=csv_paths)


def _sanitize_dataset(dataset: Dataset):
    raise NotImplementedError


def _train_val_test_split(dataset: Dataset, val_test_perc: tuple[float, float] | None, seed: int | None) -> DatasetDict:
    val_perc, test_perc = val_test_perc
    split_dataset_trainval_test = dataset.train_test_split(test_size=test_perc, seed=seed)
    split_dataset_train_val = split_dataset_trainval_test["train"].train_test_split(
        test_size=val_perc / (1 - test_perc), seed=seed
    )

    return DatasetDict({
        "train": split_dataset_train_val["train"],
        "val": split_dataset_train_val["test"],
        "test": split_dataset_trainval_test["test"]
    })
