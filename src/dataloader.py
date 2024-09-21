# pylint: disable=no-member, invalid-name, too-many-locals, too-many-arguments, consider-using-dict-items
""" Scripts for creating dataloaders """
from importlib import import_module
from copy import deepcopy

from numpy.random import default_rng

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import torch
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import open_dict

import math

def dataloader_factory(cfg, data, k, trial=None):
    """Return dataloader according to the used model"""
    if "custom_dataloader" not in cfg.model or not cfg.model.custom_dataloader:
        dataloader = common_dataloader(cfg, data, k, trial)
    else:
        try:
            model_module = import_module(f"src.models.{cfg.model.name}")
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"No module named '{cfg.model.name}' \
                                    found in 'src.models'. Check if model name \
                                    in config file and its module name are the same"
            ) from e

        try:
            get_dataloader = model_module.get_dataloader
        except AttributeError as e:
            raise AttributeError(
                f"'src.models.{cfg.model.name}' has no function\
                                'get_dataloader'. Is the function misnamed/not defined?"
            ) from e

        dataloader = get_dataloader(cfg, data, k, trial)

    return dataloader


def common_dataloader(cfg, original_data, k, trial=None):
    """
    Return common dataloaders
    dataloaders are a dictionary of
    {
        "train": dataloader,
        "valid": dataloader,
        "test": dataloader,
        "any additional test": dataloader,
    }
    Expects the data produced by src.data.data_factory() with common_processor:
        Input data is a dict of
        {
            "main":
                {
                "TS": data, (if model is TS or TS-FNC)
                "FNC": data, (if model is FNC, tri-FNC, or TS-FNC)
                "labels": data,
                },
            "additional test datasets with similar shape"
        }

    Output dataloaders return tuples with ("TS", "FNC", "labels"), ("TS", "labels"), or ("FNC", "labels") data order
    """
    data = deepcopy(original_data)
    split_data = {"train": {}, "valid": {}, "test": {}}

    # train/test split
    split_data["train"], split_data["test"] = cross_validation_split(
        data["main"], cfg.mode.n_splits, k
    )

    # train/val split
    splitter = StratifiedShuffleSplit(
        n_splits=cfg.mode.n_trials,
        test_size=split_data["train"]["labels"].shape[0] // cfg.mode.n_splits,
        random_state=42,
    )
    tr_val_splits = list(
        splitter.split(split_data["train"]["labels"], split_data["train"]["labels"])
    )
    train_index, val_index = (
        tr_val_splits[0] if cfg.mode.name == "tune" else tr_val_splits[trial]
    )
    for key in split_data["train"]:
        split_data["train"][key], split_data["valid"][key] = (
            split_data["train"][key][train_index],
            split_data["train"][key][val_index],
        )

    # shuffle training data along time axis
    if "permute" in cfg and cfg.permute == "Single":
        rng = default_rng(seed=42)
        for i in range(split_data["train"]["TS"].shape[0]):
            # shuffle time points of each subject independently
            # axis=0 - time axis of a subject
            rng.shuffle(split_data["train"]["TS"][i], axis=0)

    # add additional test datasets to split_data
    for key in data:
        if key != "main":
            split_data[key] = data[key]

    # create dataloaders
    dataloaders = {}
    key_order = ["TS", "FNC", "labels"]
    for key in split_data:
        for data_key in split_data[key]:
            if data_key == "labels":
                split_data[key][data_key] = torch.tensor(
                    split_data[key][data_key], dtype=torch.int64
                )
            else:
                split_data[key][data_key] = torch.tensor(
                    split_data[key][data_key], dtype=torch.float32
                )
        # order-wise unpacking: 'key_order' order should be followed
        unpacked_tensors = [
            split_data[key].get(data_key)
            for data_key in key_order
            if data_key in split_data[key]
        ]
        assert len(unpacked_tensors) == len(split_data[key])

        dataloaders[key] = DataLoader(
            TensorDataset(*unpacked_tensors),
            batch_size=cfg.mode.batch_size,
            num_workers=0,
            shuffle=key == "train",
        )

    # write info on dataloaders to the config
    dataloader_info = {}
    for key, dl in dataloaders.items():
        dataloader_info[key] = {
            "n_samples": len(dl.dataset),
            "batch_size": dl.batch_size,
            "n_batches": math.ceil(len(dl.dataset)/dl.batch_size)
        }
    with open_dict(cfg):
        cfg.dataloader = dataloader_info
    return dataloaders


def cross_validation_split(data, n_splits, k):
    """
    Split data into train and test data using StratifiedKFold.
    Input data should be dict
    {
        "single_or_multiple_keys": data,
        "labels": data (required),
    }
    Output train and test data inherit the same keys
    """
    train_data = {}
    test_data = {}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    CV_folds = list(skf.split(data["labels"], data["labels"]))
    train_index, test_index = CV_folds[k]
    for key in data:
        train_data[key], test_data[key] = (
            data[key][train_index],
            data[key][test_index],
        )

    return train_data, test_data
