# pylint: disable=too-many-statements, too-many-locals, invalid-name, unbalanced-tuple-unpacking, no-value-for-parameter
"""Script for running experiments: tuning and testing hypertuned models"""
import os
from copy import deepcopy

from omegaconf import OmegaConf, DictConfig
import hydra

import pandas as pd
import numpy as np

from src.utils import set_project_name, set_run_name, verify_config
from src.data import data_factory, data_postfactory
from src.dataloader import dataloader_factory, cross_validation_split
from src.model import model_config_factory, model_factory
from src.model_utils import criterion_factory, optimizer_factory, scheduler_factory
from src.logger import logger_factory
from src.trainer import trainer_factory


@hydra.main(version_base=None, config_path="../src/conf", config_name="exp_config")
def start(cfg: DictConfig):
    """Main script for starting experiments"""

    # check if config is correct
    verify_config(cfg)

    # set wandb environment
    os.environ["WANDB_SILENT"] = "true" if cfg.wandb_silent else "false"
    os.environ["WANDB_MODE"] = "offline" if cfg.wandb_offline else "online"

    # set project name and directory
    set_project_name(cfg)
    os.makedirs(cfg.project_dir, exist_ok=True)
    with open(f"{cfg.project_dir}/general_config.yaml", "w", encoding="utf8") as f:
        OmegaConf.save(cfg, f)

    # load dataset, compute FNCs if model requires them.
    original_data = data_factory(cfg)

    print(OmegaConf.to_yaml(cfg))

    if cfg.mode.name == "tune":
        if ("single_HPs" in cfg and cfg.single_HPs) or (
            "tuning_holdout" in cfg.dataset and cfg.dataset.tuning_holdout
        ):
            # use the whole original_data for tuning.
            # the obtained HPs are better to be used on other datasets
            # (unless cfg.dataset.tuning_holdout is True),
            # otherwise there is a testing-on-train-data danger
            tune(cfg=cfg, original_data=original_data)

        else:
            # tune each original_data CV fold independently
            for k in range(0, cfg.mode.n_splits):
                print(f"Outer k: {k:02d}")
                tune_fold_data = deepcopy(original_data)
                tune_fold_data["main"], _ = cross_validation_split(
                    tune_fold_data["main"], cfg.mode.n_splits, k
                )
                tune(cfg=cfg, original_data=tune_fold_data, outer_k=k)

    elif cfg.mode.name == "exp":
        experiment(cfg=cfg, original_data=original_data)


def tune(cfg, original_data, outer_k=None):
    """Given config and data, run several cross-validated rounds of optimal HP search"""

    # for each trial get new set of HPs, test them using CV
    for trial in range(0, cfg.mode.n_trials):
        print(f"Trial: {trial:04d}")
        # get random model config
        model_cfg = model_config_factory(cfg)
        # reshape data according to model config (if needed)
        data = data_postfactory(
            cfg,
            model_cfg,
            original_data,
        )
        # run nested CV
        for inner_k in range(0, cfg.mode.n_splits):
            print(f"Inner k: {inner_k:02d}")
            set_run_name(cfg, outer_k=outer_k, trial=trial, inner_k=inner_k)
            os.makedirs(cfg.run_dir, exist_ok=True)
            dataloaders = dataloader_factory(cfg, data, k=inner_k)
            results = run_trial(cfg, model_cfg, dataloaders)

            # save results of nested CV in the trial directory
            df = pd.DataFrame(results, index=[0])
            with open(f"{cfg.trial_dir}/CV_runs.csv", "a", encoding="utf8") as f:
                df.to_csv(f, header=f.tell() == 0, index=False)

        # save model config
        with open(f"{cfg.trial_dir}/model_config.yaml", "w", encoding="utf8") as f:
            OmegaConf.save(model_cfg, f)

        # summarize the trial's CV results and save them
        df = pd.read_csv(f"{cfg.trial_dir}/CV_runs.csv")
        score = np.mean(df["test_score"].to_numpy())
        loss = np.mean(df["test_average_loss"].to_numpy())
        time = np.mean(df["training_time"].to_numpy())
        df = pd.DataFrame(
            {
                "trial": trial,
                "score": score,
                "loss": loss,
                "time": time,
                "path_to_config": f"{cfg.trial_dir}/model_config.yaml",
            },
            index=[0],
        )
        with open(f"{cfg.k_dir}/trial_runs.csv", "a", encoding="utf8") as f:
            df.to_csv(f, header=f.tell() == 0, index=False)

    # get optimal config and save it
    df = pd.read_csv(f"{cfg.k_dir}/trial_runs.csv")
    best_idx = df["score"].idxmax()
    best_config_path = df.loc[best_idx]["path_to_config"]
    with open(best_config_path, "r", encoding="utf8") as f:
        best_config = OmegaConf.load(f)
    with open(f"{cfg.k_dir}/best_config.yaml", "w", encoding="utf8") as f:
        OmegaConf.save(best_config, f)


def experiment(cfg, original_data):
    """Given config and data, run cross-validated rounds with optimal HPs"""

    for outer_k in range(0, cfg.mode.n_splits):
        print(f"k: {outer_k:02d}")
        # for each fold get optimal set of HPs,
        # unless single_HP is True,
        # or model HPs are hardcoded
        model_cfg = model_config_factory(cfg, outer_k)
        # reshape data according to model config (if needed)
        data = data_postfactory(
            cfg,
            model_cfg,
            original_data,
        )
        # for outer_k test fold, train model n_trials times,
        # using different train/valid split each time
        for trial in range(0, cfg.mode.n_trials):
            print(f"Trial: {trial:04d}")
            set_run_name(cfg, outer_k=outer_k, trial=trial)
            os.makedirs(cfg.run_dir, exist_ok=True)
            dataloaders = dataloader_factory(cfg, data, k=outer_k, trial=trial)
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
