# pylint: disable=too-many-function-args, invalid-name
""" UKB ICA (with age bins (X) sex) labels dataset loading script"""
import numpy as np

from omegaconf import DictConfig

from src.datasets.ukb import load_data as load_sex_data


def load_data(
    cfg: DictConfig,
    dataset_path: str = "/anonpath/datasets/ukb/UKB_age_data.npz",
):
    """
    Return UKB data

    Input:
    dataset_path: str = "/anonpath/datasets/ukb/UKB_sex_data.npz"
    - path to the dataset with lablels
    indices_path: str = "/anonpath/datasets/ukb/correct_indices_GSP.csv"
    - path to correct indices/components

    Output:
    features, labels
    """

    data, sexes = load_sex_data(cfg)

    with np.load(dataset_path) as npzfile:
        ages = npzfile["labels"]

    bins = np.histogram_bin_edges(ages)
    ages = np.digitize(ages, bins)

    ages[ages == bins.shape[0]] = bins.shape[0] - 1
    ages = ages - 1

    labels = ages + sexes * np.unique(ages).shape[0]

    return data, labels
