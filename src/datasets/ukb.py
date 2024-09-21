# pylint: disable=too-many-function-args, invalid-name
""" UKB ICA dataset loading script"""
import numpy as np
import pandas as pd

from omegaconf import DictConfig


def load_data(
    cfg: DictConfig,
    dataset_path: str = "/data/users2/ppopov1/datasets/ukb/UKB_",
    indices_path: str = "/data/users2/ppopov1/datasets/ukb/correct_indices_GSP.csv",
):
    """
    Return UKB data

    Input:
    dataset_path: str = "/data/users2/ppopov1/datasets/ukb/UKB_"
    - path to the dataset (incomplete)
    indices_path: str = "/data/users2/ppopov1/datasets/correct_indices_GSP.csv"
    - path to correct indices/components

    Output:
    features, labels
    """

    if "tuning_holdout" in cfg.dataset and cfg.dataset.tuning_holdout:
        if cfg.mode.name == "tune":
            with np.load(dataset_path+"tune.npz") as npzfile:
                data = npzfile["data"]
                labels = npzfile["sex"]
        else:
            with np.load(dataset_path+"exp.npz") as npzfile:
                data = npzfile["data"]
                labels = npzfile["sex"]
    else:
        with np.load(dataset_path+"all.npz") as npzfile:
            data = npzfile["data"]
            labels = npzfile["sex"]


    if cfg.dataset.filter_indices:
        # get correct indices/components
        indices = pd.read_csv(indices_path, header=None)
        idx = indices[0].values - 1
        data = data[:, idx, :]

    data = np.swapaxes(data, 1, 2)
    # data.shape = [n_samples, time_length, feature_size]

    return data, labels
