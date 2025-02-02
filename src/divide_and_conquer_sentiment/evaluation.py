from collections import defaultdict
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import Dataset
from numpy.typing import ArrayLike
from sklearn.metrics import classification_report

ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}


def simple_classification_report(y_true: ArrayLike, y_pred: ArrayLike, **kwargs):
    return classification_report(_coerce_list(y_true), _coerce_list(y_pred), **kwargs)


def _coerce_list(values: ArrayLike):
    if isinstance(values, list):
        pass
    elif isinstance(values, np.ndarray):
        values = values.tolist()
    elif isinstance(values, pd.Series):
        values = values.to_list()
    elif isinstance(values, torch.Tensor):
        values = values.tolist()
    else:
        raise NotImplementedError("Only array-likes of strings or integers are supported")

    if len(values) == 0:
        return []

    if isinstance(values[0], int):
        return [ID2LABEL[y] for y in values]
    else:
        if set(values).issubset({"negative", "neutral", "positive"}):
            return values
        else:
            raise ValueError("Unknown label values")


Metric = Literal["accuracy", "macro f1", "weighted f1"]


def model_dataset_comparison(
    dataset_dict: dict[str, Dataset], preds_dict: dict[str, dict[str, ArrayLike]], metric: Metric
):
    accessor = _get_metric_accessor(metric)

    metrics_dict: dict[str, dict[str, float]] = defaultdict(dict)
    for dataset_name, model_dict in preds_dict.items():
        for model_name, preds in model_dict.items():
            metrics_dict[dataset_name][model_name] = accessor(
                simple_classification_report(dataset_dict[dataset_name]["label"], preds, output_dict=True)
            )

    return pd.DataFrame.from_dict(metrics_dict, orient="columns")


def plot_metrics_per_token_count_bins(
    dataset_dict: dict[str, Dataset], preds_dict: dict[str, dict[str, ArrayLike]], metric: Metric
):
    pd_dataset_dict = {}
    for dataset_name, dataset in dataset_dict.items():
        df = dataset.to_pandas()
        df["bin10"] = pd.cut(df["text"].str.split().map(len), bins=range(0, 101, 10), right=False).astype(str)
        pd_dataset_dict[dataset_name] = df

    plotting_data_dict = {}
    for dataset_name, model_dict in preds_dict.items():
        df_list = []
        for model_name, preds in model_dict.items():
            df_list.append(_aggregate(pd_dataset_dict[dataset_name], preds, metric, model_name))
        plotting_data_dict[dataset_name] = pd.concat(df_list).reset_index()

    _create_plot(plotting_data_dict, metric)


def _get_metric_accessor(metric: Metric):
    match metric:
        case "accuracy":
            return lambda x: x["accuracy"]
        case "macro f1":
            return lambda x: x["macro avg"]["f1-score"]
        case "weighted f1":
            return lambda x: x["weighted avg"]["f1-score"]
        case _:
            raise ValueError("Invalid metric")


def _aggregate(df: pd.DataFrame, preds: ArrayLike, metric: Metric, model: str):
    return (
        df.assign(preds=preds)
        .groupby("bin10")
        .apply(
            lambda x: simple_classification_report(x.label, x.preds, output_dict=True, zero_division=np.nan),
            include_groups=False,
        )
        .reset_index()
        .assign(
            support=lambda x: x.iloc[:, 1].map(lambda y: y["macro avg"]["support"]),
            metric=lambda x: x.iloc[:, 1].map(_get_metric_accessor(metric)),
            model=model,
        )
        .sort_index()
    )


def _create_plot(plotting_data_dict: dict[str, pd.DataFrame], metric: Metric):
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    matplotlib.rcParams.update({"font.size": 16})

    for i, (name, plotting_data) in enumerate(plotting_data_dict.items()):
        ax1 = axs[i]
        ax2 = ax1.twinx()
        ax1.label_outer()
        sns.barplot(data=plotting_data, x="bin10", y="support", ax=ax1, color="skyblue")
        ax1.set_ylabel("NOBS")
        sns.lineplot(data=plotting_data, x="bin10", y="metric", ax=ax2, hue="model", marker="o")
        ax2.set_ylabel(metric)
        ax1.set_xlabel("Tokens")
        ax1.set_title(name)
        ax2.get_legend().remove()
        ax1.tick_params("x", labelrotation=45)

    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center")

    plt.show()
