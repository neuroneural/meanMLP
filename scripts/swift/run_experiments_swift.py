# pylint: disable=too-many-statements, too-many-locals, invalid-name, unbalanced-tuple-unpacking, no-value-for-parameter
"""Script for running experiments: tuning and testing hypertuned models"""
import os
from copy import deepcopy

from omegaconf import OmegaConf, DictConfig
import hydra

import pandas as pd
import numpy as np
import math

from src.utils import set_project_name, set_run_name, validate_config, get_resume_params
from src.data import data_factory, data_postfactory
from src.dataloader import dataloader_factory, cross_validation_split
from src.model import model_config_factory, model_factory
from src.model_utils import criterion_factory, optimizer_factory, scheduler_factory
from src.logger import logger_factory
from src.trainer import trainer_factory

from torch.utils.data import DataLoader, TensorDataset
import torch
from omegaconf import open_dict

@hydra.main(version_base=None, config_path="../src/conf", config_name="exp_config")
def start(cfg: DictConfig):
    """Main script for starting experiments"""

    # check if config is correct
    validate_config(cfg)

    # set wandb environment
    os.environ["WANDB_SILENT"] = "true" if cfg.wandb_silent else "false"
    os.environ["WANDB_MODE"] = "offline" if cfg.wandb_offline else "online"

    # set project name and directory
    set_project_name(cfg)
    # load interrupted config
    if "resume" in cfg and cfg.resume:
        cfg = get_resume_params(cfg)
    else:
        os.makedirs(cfg.project_dir, exist_ok=True)
    with open(f"{cfg.project_dir}/general_config.yaml", "w", encoding="utf8") as f:
        OmegaConf.save(cfg, f)

    print("General config:")
    print(OmegaConf.to_yaml(cfg))

    # load dataset, compute FNCs if model requires them.
    original_data = data_factory(cfg)
    # resume flags check
    is_interupted = "resume" in cfg and cfg.resume

    experiment(cfg=cfg, original_data=original_data, is_interupted=is_interupted)



def experiment(cfg, original_data, is_interupted=False):
    """Given config and data, run cross-validated rounds with optimal HPs"""
    if cfg.dataset.name == "hcp":
        with open_dict(cfg):
            cfg.dataset.data_info.main.data_shape[1] = 1200
            cfg.dataset.data_info.main.data_shape[2] = 400
            cfg.dataloader = {"train": {"n_batches": math.ceil(555 // cfg.mode.batch_size)}}
        dataloaders = {}
        model_cfg = model_config_factory(cfg, 0)
        for key in ["train", "valid", "test"]:
            raw_data = np.load(f"/data/users2/ppopov1/volume_fmri/hcp/mlp/schaefer_{key}_{cfg.mode.split_idx}.npz")

            features = torch.tensor(raw_data["data"], dtype=torch.float32)
            labels = torch.tensor(raw_data["labels"], dtype=torch.int64)
            print(torch.any(torch.isnan(features)))
            print(torch.any(torch.isnan(labels)))

            dataloaders[key] = DataLoader(
                    TensorDataset(features, labels),
                    batch_size=cfg.mode.batch_size,
                    num_workers=0,
                    shuffle=key == "train",
                )
            
    elif cfg.dataset.name == "fbirn":
        with open_dict(cfg):
            cfg.dataset.data_info.main.data_shape[1] = 140
            cfg.dataset.data_info.main.data_shape[2] = 400
            cfg.dataloader = {"train": {"n_batches": math.ceil(555 // cfg.mode.batch_size)}}
        dataloaders = {}
        model_cfg = model_config_factory(cfg, 0)
        for key in ["train", "valid", "test"]:
            raw_data = np.load(f"/data/users2/ppopov1/volume_fmri/fbirn/mlp/schaefer_{key}_{cfg.mode.split_idx}.npz")

            features = torch.tensor(raw_data["data"], dtype=torch.float32)
            labels = torch.tensor(raw_data["labels"], dtype=torch.int64)
            print(torch.any(torch.isnan(features)))
            print(torch.any(torch.isnan(labels)))

            dataloaders[key] = DataLoader(
                    TensorDataset(features, labels),
                    batch_size=cfg.mode.batch_size,
                    num_workers=0,
                    shuffle=key == "train",
                )
        
    set_run_name(cfg, outer_k=0, trial=0)
    os.makedirs(cfg.run_dir, exist_ok=True)


    
    results = run_trial(cfg, model_cfg, dataloaders)

    # save run's results in the folds directory
    df = pd.DataFrame(results, index=[0])
    with open(f"{cfg.k_dir}/fold_runs.csv", "a", encoding="utf8") as f:
        df.to_csv(f, header=f.tell() == 0, index=False)

    # save outer_k's model config
    with open(f"{cfg.k_dir}/model_config.yaml", "w", encoding="utf8") as f:
        OmegaConf.save(model_cfg, f)

    # load and save the fold's results in the project directory
    df = pd.read_csv(f"{cfg.k_dir}/fold_runs.csv")
    with open(f"{cfg.project_dir}/runs.csv", "a", encoding="utf8") as f:
        df.to_csv(f, header=f.tell() == 0, index=False)


def run_trial(cfg, model_cfg, dataloaders):
    """Given config and prepared dataloaders, build and train the model and return test results"""
    model = model_factory(cfg, model_cfg)
    criterion = criterion_factory(cfg, model_cfg)
    optimizer = optimizer_factory(cfg, model_cfg, model)
    scheduler = scheduler_factory(cfg, model_cfg, optimizer)
    logger = logger_factory(cfg, model_cfg)

    trainer = trainer_factory(
        cfg,
        model_cfg,
        dataloaders,
        model,
        criterion,
        optimizer,
        scheduler,
        logger,
    )

    results = trainer.run()
    logger.finish()

    return results


if __name__ == "__main__":
    start()
