import os
import tempfile

from datasets import ClassLabel, Dataset, DatasetDict, load_dataset
from kaggle import api


def load_kaggle_dataset(
    kaggle_dataset_path: str,
    colname_mapping: dict[str, str],
    val_test_perc: tuple[float, float] | None = None,
    seed: int | None = None,
) -> Dataset | DatasetDict:
    _check_colname_mapping(colname_mapping)

    dataset = _download_from_kaggle(kaggle_dataset_path)
    sanitized_dataset = _sanitize_dataset(dataset, colname_mapping)

    if val_test_perc:
        return _train_val_test_split(sanitized_dataset, val_test_perc, seed)
    else:
        return sanitized_dataset


def _check_colname_mapping(
    colname_mapping: dict[str, str],
):
    if len(colname_mapping) != 2:
        raise ValueError("Wrong size: colname_mapping should only contain 2 values: 'text' and 'label'.")
    if set(colname_mapping.values()) != {"text", "label"}:
        raise ValueError("Wrong values: the only allowed values are 'text' and 'label'.")


def _download_from_kaggle(kaggle_dataset_path: str) -> Dataset:
    with tempfile.TemporaryDirectory() as tmp_dirname:
        api.dataset_download_files(kaggle_dataset_path, path=tmp_dirname, unzip=True)
        csv_paths = [os.path.join(tmp_dirname, path) for path in os.listdir(tmp_dirname) if path.endswith(".csv")]
        return load_dataset("csv", data_files=csv_paths, split="all")


def _sanitize_dataset(dataset: Dataset, colname_mapping: dict[str, str]) -> Dataset:
    ds = dataset.rename_columns(colname_mapping).select_columns(list(colname_mapping.values()))

    unique_label_values = set(ds.unique("label"))

    if unique_label_values == {"negative", "neutral", "positive"}:
        label_mapping = {"negative": 0, "neutral": 1, "positive": 2}
    elif unique_label_values == {1, 2, 3, 4, 5}:
        label_mapping = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}  # type: ignore

    result = (
        ds.map(lambda x: {"text": x["text"] or ""})
        .map(lambda x: {"label": label_mapping[x["label"]]})
        .cast_column("label", ClassLabel(names=["negative", "neutral", "positive"]))
    )

    return result


def _train_val_test_split(dataset: Dataset, val_test_perc: tuple[float, float], seed: int | None) -> DatasetDict:
    val_perc, test_perc = val_test_perc
    split_dataset_trainval_test = dataset.train_test_split(test_size=test_perc, seed=seed)
    split_dataset_train_val = split_dataset_trainval_test["train"].train_test_split(
        test_size=val_perc / (1 - test_perc), seed=seed
    )

    return DatasetDict(
        {
            "train": split_dataset_train_val["train"],
            "val": split_dataset_train_val["test"],
            "test": split_dataset_trainval_test["test"],
        }
    )
