# pylint: disable=W0201,W0223,C0103,C0115,C0116,R0902,E1101,R0914
"""Main training script for simple models that output logits"""
import os
import sys
import argparse
import json
import shutil

import pandas as pd
from apto.utils.misc import boolean_flag
from apto.utils.report import get_classification_report
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from animus import EarlyStoppingCallback, IExperiment
from animus.torch.callbacks import TorchCheckpointerCallback
import wandb

from src.settings import LOGS_ROOT, UTCNOW
from src.ts_data import load_dataset

# deferred import
# from src.ts_model import (
#     get_config,
#     get_model,
#     get_criterion,
#     get_optimizer,
# )


class Experiment(IExperiment):
    """
    Animus-based training script. For more info see animus documentation
    (it's quite simple)
    """

    def __init__(
        self,
        mode: str,
        path: str,
        model: str,
        dataset: str,
        multiclass: bool,
        scaled: bool,
        test_datasets: list,
        prefix: str,
        n_splits: int,
        n_trials: int,
        max_epochs: int,
        batch_size: int,
    ) -> None:
        super().__init__()
        self.config = {}

        self.utcnow = self.config["default_prefix"] = UTCNOW
        # starting fold/trial; used in resumed experiments
        self.start_k = 0
        self.start_trial = 0

        if mode == "resume":
            # get params of the resumed experiment, reset starting fold/trial
            (
                mode,
                model,
                dataset,
                multiclass,
                scaled,
                test_datasets,
                prefix,
                n_splits,
                n_trials,
                max_epochs,
            ) = self.acquire_params(path)

        self.mode = self.config["mode"] = mode  # tune or experiment mode
        self.model = self.config["model"] = model  # model name
        # main dataset name (used for training)
        self.dataset_name = self.config["dataset"] = dataset
        # if dataset should be scaled by sklearn's StandardScaler
        self.multiclass = self.config["multiclass"] = multiclass
        self._scaled = self.config["scaled"] = scaled

        if test_datasets is None:  # additional test datasets
            self.test_datasets = []
        else:
            if self.mode == "tune":
                print("'Tune' mode overrides additional test datasets")
                self.test_datasets = []
            else:
                self.test_datasets = test_datasets

        if self.dataset_name in self.test_datasets:
            # Fraction of the main dataset is always used as a test dataset;
            # no need for it in the list of test datasets
            print(
                f"Received main dataset {self.dataset_name} among test datasets {self.test_datasets}; removed"
            )
            self.test_datasets.remove(self.dataset_name)
        self.config["test_datasets"] = self.test_datasets

        # num of splits for StratifiedKFold
        self.n_splits = self.config["n_splits"] = n_splits
        # num of trials for each fold
        self.n_trials = self.config["n_trials"] = n_trials
        self.max_epochs = self.config["max_epochs"] = max_epochs
        self.batch_size = self.config["batch_size"] = batch_size

        # set project name prefix
        if len(prefix) == 0:
            self.project_prefix = f"{self.utcnow}"
        else:
            # '-'s are reserved for name parsing
            self.project_prefix = prefix.replace("-", "_")
        self.config["prefix"] = self.project_prefix

        proj_dataset_name = self.dataset_name
        if self.multiclass:
            proj_dataset_name = f"multiclass_{proj_dataset_name}"
        if self._scaled:
            proj_dataset_name = f"scaled_{proj_dataset_name}"
        self.project_name = f"{self.mode}-{self.model}-{proj_dataset_name}"
        if len(self.test_datasets) != 0:
            project_ending = "-tests-" + "_".join(self.test_datasets)
            self.project_name += project_ending
        self.config["project_name"] = self.project_name

        self.logdir = f"{LOGS_ROOT}/{self.project_prefix}-{self.project_name}/"
        self.config["logdir"] = self.logdir

        # create experiment directory
        os.makedirs(self.logdir, exist_ok=True)
        # save initial config
        logfile = f"{self.logdir}/config.json"
        with open(logfile, "w") as fp:
            json.dump(self.config, fp, indent=2)

        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"

        print(f"Used device: {dev}")
        self.device = torch.device(dev)

    def acquire_params(self, path):
        """
        Used for extracting experiment set-up from the
        given path for resuming an interrupted experiment
        """
        # load experiment config
        with open(f"{path}/config.json", "r") as fp:
            config = json.load(fp)

        # find when the experiment got interrupted
        with open(path + "/runs.csv", "r") as fp:
            lines = len(fp.readlines()) - 1
            self.start_k = lines // config["n_trials"]
            self.start_trial = lines - self.start_k * config["n_trials"]

        # delete failed run
        faildir = path + f"/k_{self.start_k}/{self.start_trial:04d}"
        print("Deleting interrupted run logs in " + faildir)
        try:
            shutil.rmtree(faildir)
        except FileNotFoundError:
            print("Could not delete interrupted run logs - FileNotFoundError")

        # reset the default prefix
        self.utcnow = self.config["default_prefix"] = config["default_prefix"]

        return (
            config["mode"],
            config["model"],
            config["dataset"],
            config["multiclass"],
            config["scaled"],
            config["test_datasets"],
            config["prefix"],
            config["n_splits"],
            config["n_trials"],
            config["max_epochs"],
        )

    def initialize_data(self, dataset, for_test=False):
        # load core dataset (or additional datasets if for_test==True)
        # your dataset should have shape [n_features; n_channels; time_len]

        features, labels = load_dataset(dataset, self.multiclass)
        features = np.swapaxes(features, 1, 2)  # [n_features; time_len; n_channels;]

        if self._scaled:
            # Time scaling
            # TODO: check if it is correct
            features_shape = features.shape  # [n_features; time_len; n_channels;]
            features = features.reshape(-1, features_shape[1])
            features = features.swapaxes(0, 1)

            scaler = StandardScaler()
            features = scaler.fit_transform(features)  # first dimension is scaled

            features = features.swapaxes(0, 1)
            features = features.reshape(features_shape)

        if for_test:
            # if dataset is loaded for tests, it should not be
            # split into train/val/test.
            # it is called in `on_experiment_end` when testing time comes

            return TensorDataset(
                torch.tensor(features, dtype=torch.float32),
                torch.tensor(labels, dtype=torch.int64),
            )

        self.data_shape = features.shape  # [n_features; time_len; n_channels;]
        self.n_classes = np.unique(labels).shape[0]

        print("data shape: ", self.data_shape)
        # train-val/test split
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        skf.get_n_splits(features, labels)

        train_index, test_index = list(skf.split(features, labels))[self.k]

        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=self.data_shape[0] // self.n_splits,
            random_state=42 + self.trial,
            stratify=y_train,
        )

        self._train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.int64),
        )
        self._valid_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.int64),
        )
        self._test_ds = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.int64),
        )

    def initialize_model(self):
        # deferred import - there is a cyclic import
        from src.ts_model import (
            get_config,
            get_model,
            get_criterion,
            get_optimizer,
        )

        _, self.model_config = get_config(self)
        self.config["batch_size"] = self.batch_size

        self._model = get_model(self.model, self.model_config)
        self._model = self._model.to(self.device)

        self.criterion = get_criterion(self.model)

        lr, self.optimizer = get_optimizer(self, self.model_config)
        self.model_config["lr"] = lr

    def on_experiment_start(self, exp: "IExperiment"):
        super().on_experiment_start(exp)

        # create run's path directory
        self.config["runpath"] = f"{self.logdir}k_{self.k}/{self.trial:04d}"
        self.config["run_config_path"] = f"{self.config['runpath']}/config.json"
        os.makedirs(self.config["runpath"], exist_ok=True)

        # init wandb logger
        self.wandb_logger: wandb.run = wandb.init(
            project=f"{self.project_prefix}-{self.project_name}",
            name=f"k_{self.k}-trial_{self.trial}",
            save_code=True,
        )

        # init data
        self.initialize_data(self.dataset_name)

        # init model
        self.initialize_model()

        # setup data loaders
        self.datasets = {
            "train": DataLoader(
                self._train_ds,
                batch_size=self.batch_size,
                num_workers=0,
                shuffle=True,
            ),
            "valid": DataLoader(
                self._valid_ds,
                batch_size=self.batch_size,
                num_workers=0,
                shuffle=False,
            ),
        }

        # setup callbacks
        self.callbacks = {
            "early-stop": EarlyStoppingCallback(
                minimize=True,
                patience=30,
                dataset_key="valid",
                metric_key="loss",
                min_delta=0.001,
            ),
            "checkpointer": TorchCheckpointerCallback(
                exp_attr="_model",
                logdir=self.config["runpath"],
                dataset_key="valid",
                metric_key="loss",
                minimize=True,
            ),
        }

        self.num_epochs = self.max_epochs

        self.wandb_logger.config.update(self.config)
        self.wandb_logger.config.update(self.model_config)

        # update saved config
        with open(f"{self.logdir}/config.json", "w") as fp:
            json.dump(self.config, fp)
        # save model params in the run's directory
        with open(self.config["run_config_path"], "w") as fp:
            json.dump(self.model_config, fp)

    def run_dataset(self, thr_tune: bool = False):
        all_scores, all_targets = [], []
        total_loss = 0.0
        self._model.train(self.is_train_dataset)

        with torch.set_grad_enabled(self.is_train_dataset):
            for self.dataset_batch_step, (data, target) in enumerate(tqdm(self.dataset)):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                logits = self._model(data)
                loss = self.criterion(logits, target)
                score = torch.softmax(logits, dim=-1)

                all_scores.append(score.cpu().detach().numpy())
                all_targets.append(target.cpu().detach().numpy())
                total_loss += loss.sum().item()
                if self.is_train_dataset:
                    loss.backward()
                    self.optimizer.step()

        total_loss /= self.dataset_batch_step + 1

        y_test = np.hstack(all_targets)
        y_score = np.vstack(all_scores)
        if thr_tune is True:
            return y_test, y_score

        if self.dataset_key == "test":
            y_score_tuned = y_score - self.best_threshold
            y_pred = np.argmax(y_score_tuned, axis=-1).astype(np.int32)
        else:
            y_pred = np.argmax(y_score, axis=-1).astype(np.int32)

        report = get_classification_report(
            y_true=y_test, y_pred=y_pred, y_score=y_score, beta=0.5
        )

        self.dataset_metrics = {
            "accuracy": report["precision"].loc["accuracy"],
            "score": report["auc"].loc["weighted"],
            "loss": total_loss,
        }

    def on_epoch_end(self, exp: "IExperiment") -> None:
        super().on_epoch_end(self)
        self.wandb_logger.log(
            {
                "train_accuracy": self.epoch_metrics["train"]["accuracy"],
                "train_score": self.epoch_metrics["train"]["score"],
                "train_loss": self.epoch_metrics["train"]["loss"],
                "valid_accuracy": self.epoch_metrics["valid"]["accuracy"],
                "valid_score": self.epoch_metrics["valid"]["score"],
                "valid_loss": self.epoch_metrics["valid"]["loss"],
            },
        )

    def thr_gen(self, depth, summ, constr_threshold):
        if depth == 1:
            yield constr_threshold + [summ]
            return
        for thr in np.arange(0.0, summ + 0.0001, 0.001):
            rest_sum = summ - thr
            if rest_sum >= 0.0:
                new_constr_threshold = constr_threshold + [thr]
                yield from self.thr_gen(depth - 1, rest_sum, new_constr_threshold)

    def on_experiment_end(self, exp: "IExperiment") -> None:
        super().on_experiment_end(exp)

        print("Run threshold tuning")
        # load best weights
        logpath = f"{self.config['runpath']}/_model.best.pth"
        checkpoint = torch.load(logpath, map_location=lambda storage, loc: storage)
        self._model.load_state_dict(checkpoint)
        self._model = self._model.to(self.device)

        self.dataset_key = "valid"
        self.dataset = self.datasets["valid"]

        val_y_test, val_y_score = self.run_dataset(thr_tune=True)

        best_acc = 0.0
        self.best_threshold = [np.array(([0] * self.n_classes))]
        for thr in self.thr_gen(self.n_classes, 1.0, []):
            thr = np.array(thr)
            new_y_score = val_y_score - thr

            new_y_pred = np.argmax(new_y_score, axis=-1).astype(np.int32)
            acc = accuracy_score(val_y_test, new_y_pred)
            if acc > best_acc:
                best_acc = acc
                self.best_threshold = [thr]
            elif acc == best_acc:
                self.best_threshold += [thr]

        self.best_threshold = np.mean(np.stack(self.best_threshold), axis=0)

        print("Run test dataset")

        self.dataset_key = "test"
        self.dataset = DataLoader(
            self._test_ds,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
        )
        self.run_dataset()

        print("Test results:")
        print("Accuracy ", self.dataset_metrics["accuracy"])
        print("AUC ", self.dataset_metrics["score"])
        print("Loss ", self.dataset_metrics["loss"])
        results = {
            "test_accuracy": self.dataset_metrics["accuracy"],
            "test_score": self.dataset_metrics["score"],
            "test_loss": self.dataset_metrics["loss"],
            "config_path": self.config["run_config_path"],
        }

        print("Run additional test datasets")
        for dataset_name in self.test_datasets:
            test_ds = self.initialize_data(dataset_name, for_test=True)
            self.dataset = DataLoader(
                test_ds,
                batch_size=self.batch_size,
                num_workers=0,
                shuffle=False,
            )
            self.run_dataset()

            print(f"On {dataset_name}:")
            print("Accuracy ", self.dataset_metrics["accuracy"])
            print("AUC ", self.dataset_metrics["score"])
            print("Loss ", self.dataset_metrics["loss"])

            results[f"{dataset_name}_test_accuracy"] = self.dataset_metrics["accuracy"]
            results[f"{dataset_name}_test_score"] = self.dataset_metrics["score"]
            results[f"{dataset_name}_test_loss"] = self.dataset_metrics["loss"]

        df = pd.DataFrame(results, index=[0])
        with open(f"{self.logdir}/runs.csv", "a") as f:
            df.to_csv(f, header=f.tell() == 0, index=False)
        results.pop("config_path")
        self.wandb_logger.log(results)
        self.wandb_logger.finish()

    def start(self):
        # get experiment folds-trial pairs
        # if mode is 'resume', then it won't have already completed folds
        # else it is just all folds-trial pairs from (0, 0) to (n_splits, n_trials)
        folds_of_interest = []

        if self.start_k < self.n_splits:
            # interrupted fold
            for trial in range(self.start_trial, self.n_trials):
                folds_of_interest += [(self.start_k, trial)]

            # the rest of folds
            for k in range(self.start_k + 1, self.n_splits):
                for trial in range(self.n_trials):
                    folds_of_interest += [(k, trial)]
        else:
            raise IndexError

        # run through folds
        for k, trial in folds_of_interest:
            self.k = k  # k'th test fold
            self.trial = trial  # trial'th trial on the k'th fold
            self.run()


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    for_continue = "resume" in sys.argv

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "tune",
            "experiment",
            "resume",
        ],
        required=True,
        help="'tune' for model hyperparams tuning; \
            'experiment' for experiments with tuned model; \
                'resume' for resuming interrupted experiment",
    )
    parser.add_argument(
        "--path",
        type=str,
        required=for_continue,
        help="Path to the interrupted experiment \
            (e.g., /Users/user/mlp_project/assets/logs/prefix-mode-model-ds), \
                used in 'resume' mode",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "mlp",
            "wide_mlp",
            "deep_mlp",
            "attention_mlp",
            "new_attention_mlp",
            "meta_mlp",
            "pe_mlp",
            "pe_att_mlp",
            "mlp_tf",
            "lstm",
            "noah_lstm",
            "transformer",
            "mean_transformer",
            "first_transformer",
            "pe_transformer",
        ],
        required=not for_continue,
        help="Name of the model to run",
    )
    parser.add_argument(
        "--ds",
        type=str,
        choices=[
            "oasis",
            "adni",
            "abide",
            "fbirn",
            "cobre",
            "abide_869",
            "ukb",
            "ukb_age_bins",
            "bsnip",
            "time_fbirn",
            "fbirn_100",
            "fbirn_200",
            "fbirn_400",
            "fbirn_1000",
            "hcp",
            "hcp_roi",
            "abide_roi",
        ],
        required=not for_continue,
        help="Name of the dataset to use for training",
    )

    parser.add_argument(
        "--test-ds",
        nargs="*",
        type=str,
        choices=[
            "oasis",
            "adni",
            "abide",
            "fbirn",
            "cobre",
            "abide_869",
            "ukb",
            "ukb_age_bins",
            "bsnip",
            "time_fbirn",
            "fbirn_100",
            "fbirn_200",
            "fbirn_400",
            "fbirn_1000",
            "hcp",
            "hcp_roi",
            "abide_roi",
        ],
        help="Additional datasets for testing",
    )

    # oasis, bsnip, adni and ukb_age_bins have multiple classes;
    # ukb_age_bins is ukb age dataset with ages splitted into bins
    boolean_flag(parser, "multiclass", default=False)

    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix for the project name (body of the project name \
            is '$mode-$model-$dataset'): default: UTC time",
    )

    # whehter dataset should be scaled by sklearn's StandardScaler
    boolean_flag(parser, "scaled", default=False)

    parser.add_argument(
        "--max-epochs",
        type=int,
        default=30,
        help="Max number of epochs (min 30)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=10,
        help="Number of trials to run on each test fold",
    )
    parser.add_argument(
        "--num-splits",
        type=int,
        default=5,
        help="Number of splits for StratifiedKFold (affects the number of test folds)",
    )
    args = parser.parse_args()

    Experiment(
        mode=args.mode,
        path=args.path,
        model=args.model,
        dataset=args.ds,
        multiclass=args.multiclass,
        scaled=args.scaled,
        test_datasets=args.test_ds,
        prefix=args.prefix,
        n_splits=args.num_splits,
        n_trials=args.num_trials,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
    ).start()
