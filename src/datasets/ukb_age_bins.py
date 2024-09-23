# pylint: disable=too-many-function-args, invalid-name
""" UKB ICA (with age bins (X) sex) labels dataset loading script"""
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
    dataset_path: str = "/data/users2/ppopov1/datasets/ukb/UKB_sex_data.npz"
    - path to the dataset with lablels
    indices_path: str = "/data/users2/ppopov1/datasets/ukb/correct_indices_GSP.csv"
    - path to correct indices/components

    Output:
    features, labels
    """

    if "tuning_holdout" in cfg.dataset and cfg.dataset.tuning_holdout:
        if cfg.mode.name == "tune":
            with np.load(dataset_path+"tune.npz") as npzfile:
                data = npzfile["data"]
                sexes = npzfile["sex"]
                ages = npzfile["age"]
        else:
            with np.load(dataset_path+"exp.npz") as npzfile:
                data = npzfile["data"]
                sexes = npzfile["sex"]
                ages = npzfile["age"]
    else:
        with np.load(dataset_path+"all.npz") as npzfile:
            data = npzfile["data"]
            sexes = npzfile["sex"]
            ages = npzfile["age"]

    bins = np.histogram_bin_edges(ages)
    ages = np.digitize(ages, bins)

    ages[ages == bins.shape[0]] = bins.shape[0] - 1
    ages = ages - 1

    labels = ages + sexes * np.unique(ages).shape[0]

    if cfg.dataset.filter_indices:
        # get correct indices/components
        indices = pd.read_csv(indices_path, header=None)
        idx = indices[0].values - 1
        data = data[:, idx, :]

    data = np.swapaxes(data, 1, 2)
    
    return data, labels