# pylint: disable=too-many-function-args, invalid-name
""" ABIDE ICA dataset loading script"""
import h5py
import numpy as np
import pandas as pd

from pathlib import Path
DATA_ROOT = Path().joinpath("../assets/data").resolve()


def load_data(
    dataset_path: str = DATA_ROOT.joinpath("abide/ABIDE1_AllData.h5"),
    indices_path: str = DATA_ROOT.joinpath("ICA_correct_order.csv"),
    labels_path: str = DATA_ROOT.joinpath("abide/labels_ABIDE1.csv"),
):
    """
    Return ABIDE1 data

    Input:
    dataset_path: str = DATA_ROOT.joinpath("abide/ABIDE1_AllData.h5")
    - path to the dataset
    indices_path: str = DATA_ROOT.joinpath("ICA_correct_order.csv")
    - path to correct indices/components
    labels_path: str = DATA_ROOT.joinpath("abide/labels_ABIDE1.csv")
    - path to labels

    Output:
    data, labels
    """

    # get data and reshape to correct format
    hf = h5py.File(dataset_path, "r")
    data = hf.get("ABIDE1_dataset")
    data = np.array(data)

    num_subjects = data.shape[0]
    num_components = 100
    data = data.reshape(num_subjects, num_components, -1)

    # get correct indices/components
    indices = pd.read_csv(indices_path, header=None)
    idx = indices[0].values - 1
    data = data[:, idx, :]

    # get labels
    labels = pd.read_csv(labels_path, header=None)
    labels = labels.values.flatten().astype("int") - 1

    data = np.swapaxes(data, 1, 2)
    # data.shape = [n_samples, time_length, feature_size]
    # 569 - sessions - data.shape[0]
    # 140 - time points - data.shape[1]
    # 53 - components - data.shape[2]

    return data, labels
