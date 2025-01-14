from collections import defaultdict
from typing import Literal

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from numpy.typing import ArrayLike
from sklearn.metrics import classification_report

ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}


def simple_classification_report(y_true: ArrayLike, y_pred: ArrayLike, output_dict: bool = False):
    if isinstance(y_true, list):
        y_true = _coerce_preds_list(y_true)

    if isinstance(y_pred, list):
        y_pred = _coerce_preds_list(y_pred)
    elif isinstance(y_pred, np.ndarray):
        y_pred = _coerce_preds_list(y_pred.tolist())
    elif isinstance(y_pred, torch.Tensor):
        y_pred = _coerce_preds_list(y_pred.tolist())
    else:
        raise NotImplementedError("Only lists, numpy arrays and torch tensors are supported")

    return classification_report(y_true, y_pred, output_dict=output_dict)


def _coerce_preds_list(preds: ArrayLike):
    if isinstance(preds, list):
        if isinstance(preds[0], int):
            return [ID2LABEL[y] for y in preds]
        else:
            if set(preds) != {"negative", "neutral", "positive"}:
                raise ValueError("Unknown label values")
            else:
                return preds
    else:
        raise NotImplementedError("Only array-likes of strings or integers are supported")


Metric = Literal["accuracy", "macro f1", "weighted f1"]


def compare_multiple(labels_dict: dict[str, ArrayLike], preds_dict: dict[str, dict[str, ArrayLike]], metric: Metric):
    match metric:
        case "accuracy":
            accessor = lambda x: x["accuracy"]
        case "macro f1":
            accessor = lambda x: x["macro avg"]["f1-score"]
        case "weighted f1":
            accessor = lambda x: x["weighted avg"]["f1-score"]
        case _:
            raise ValueError("Invalid metric")

    metrics_dict: dict[str, dict[str, float]] = defaultdict(dict)
    for dataset_name, model_dict in preds_dict.items():
        for model_name, preds in model_dict.items():
            metrics_dict[dataset_name][model_name] = accessor(
                simple_classification_report(labels_dict[dataset_name], preds, output_dict=True)
            )

    return pd.DataFrame.from_dict(metrics_dict, orient="columns")


def plot_metrics_per_token_count_bins(
    dataset: Dataset,
):
    pass
