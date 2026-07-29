"""
Collection of helper functions, common model routines, and a training script.
"""

import numpy as np
import pandas as pd

import torch
from torch.nn.functional import cross_entropy, softmax
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, confusion_matrix,
)

import time
from copy import deepcopy


def basic_ce_loss(logits, targets):
    """
    Default cross entropy loss function used by many models.
    
    Returns
    -------
    loss : Tensor
        CE loss for backprop.
    logs : dict
        Loss dictionary for logs.
    """
    loss = cross_entropy(logits, targets)

    return loss, {"CE_loss": float(loss.detach().cpu().item())}


def basic_Adam_optimizer(model, lr):
    """ Basic Adam optimizer """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return optimizer


def compute_metrics(y_prob, y_pred, y_true, compute_auc=True, compute_confusion=True):
    """
    Compute a bundle of classification metrics.

    Parameters
    ----------
    y_prob : array (N, C)
        Predicted class probabilities.
    y_pred : array (N,)
        Predicted class labels.
    y_true : array (N,)
        True class labels.
    compute_auc : bool, optional
        Whether to compute ROC AUC.
    compute_confusion : bool, optional
        Whether to add confusion matrix counts to the log.

    Returns
    -------
    log : dict
        Metric name -> value. The caller is expected to prefix keys with
        "train_"/"val_"/"test_".
    """
    log = {}
    log["accuracy"] = accuracy_score(y_true, y_pred)
    log["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    if compute_auc:
        if y_prob.shape[1] == 2: # binary classification
            log["auc"] = roc_auc_score(y_true, y_prob[:, 1])
        else: # multiclass classification
            log["auc"] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    log["f1_macro"] = f1_score(y_true, y_pred, average="macro")

    if compute_confusion:
        log.update(confusion_counts(y_pred, y_true, n_classes=y_prob.shape[1]))

    return log


def confusion_counts(y_pred, y_true, n_classes):
    """
    Confusion matrix as a flat dict of counts, suitable for a dataframe row.

    For a binary problem with the usual "class 1 is positive" reading, the four
    keys map onto the conventional names as:

        cm_true0_pred0 = TN        cm_true0_pred1 = FP
        cm_true1_pred0 = FN        cm_true1_pred1 = TP

    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    return {
        f"cm_true{i}_pred{j}": int(cm[i, j])
        for i in range(n_classes)
        for j in range(n_classes)
    }


def basic_handle_batch(model, batch):
    """
    Standard batch handling routine for models. 
    Works for the majority of models that expect single input (time series or FNC).
    Only exception is FBNetGen that expects both time series and FNC.

    Returns
    -------
    loss : Tensor
        Loss for backprop (if you need it)
    batch_log : dict
        Dictionary of classification metrics and losses for logs.
    """
    # load batch into model
    
    data, labels = batch[:-1], batch[-1] # most of the time batch contains (data, labels), but sometimes there's more
    logits, loss_load = model(*data)
    loss, loss_log = model.compute_loss(loss_load, labels)

    # stash predictions for later scoring
    y_prob = softmax(logits, dim=1).detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()

    batch_log = dict(loss_log)
    batch_log["y_prob"] = y_prob
    batch_log["y_true"] = y_true

    return loss, batch_log


def basic_dataloader(data,
                     labels,
                     type,
                     batch_size: int = 64,
                     shuffle: bool = True,
                     zscore=True):
    """
    Dataloader factory for time series and FNC data.
    Creates a DataLoader from time series data and labels, derives FNC as PCC if needed.
    Most of the models hook their `prepare_dataloader` methods to this function.

    Args
    ----
    data : array-like, shape (B, T, D)
        Time series data of shape Batch x Time x (D)Features 
    labels : array-like, shape (B,)
        Class labels for the data.
    type : str
        Determines the type of data stored in the dataloader:
        "TS" for time series, "FNC" for connectivity matrices, "ALL" for both.
    batch_size : int, optional
        Batch size for the DataLoader.
    shuffle : bool, optional
        Whether to shuffle batching in the DataLoader.
    zscore : bool, optional
        Whether to z-score time series.

    Returns
    -------
    DataLoader
        A PyTorch DataLoader generating the batches of data (as TS or FNC, or both) and labels.
    """

    if type == "TS":
        if zscore:
            data = zscore_np(data, axis=1) # (B, T, D)
        data = torch.tensor(data, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        dataset = TensorDataset(data, labels)
    elif type == "FNC":
        fnc = corrcoef_batch(data)  # (B, D, D)
        fnc = torch.tensor(fnc, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        dataset = TensorDataset(fnc, labels)
    elif type == "ALL":
        if zscore:
            data = zscore_np(data, axis=1) # (B, T, D)
        fnc = corrcoef_batch(data)  # (B, D, D)
        data = torch.tensor(data, dtype=torch.float32)
        fnc = torch.tensor(fnc, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        dataset = TensorDataset(data, fnc, labels)
    else:
        raise ValueError(f"Unknown data type: {type}. Supported types are 'TS', 'FNC', and 'ALL'.")
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def zscore_np(x, axis=0, eps=1e-8):
    """
    numpy-only z-score

    Parameters
    ----------
    x : array_like
        data to z-score
    axis : int or tuple of int
        Axis/axes along which to compute mean/std. (e.g., axis=1 for (B,T,D) time axis)

    Returns
    -------
    xz : ndarray
        Z-scored array (same shape as x).
    """
    x = np.asarray(x)
    mu = np.mean(x, axis=axis, keepdims=True)
    sd = np.std(x, axis=axis, keepdims=True)
    return (x - mu) / (sd + eps)


def corrcoef_batch(x, eps=1e-8):
    """
    Batched Pearson correlation over features for time-series data.

    Parameters
    ----------
    X : array_like, shape (B, T, D)
        Data to correlate, arranged as Batch x Time x (D)Features.

    Returns
    -------
    C : ndarray, shape (B, D, D)
        Pearson correlation matrices per batch item.
    """
    x = np.asarray(x)
    B, T, D = x.shape

    # z-score along time
    Z = zscore_np(x, axis=1, eps=eps)   # (B, T, D)

    # Correlation = (Z^T @ Z) / T for each batch element
    C = np.einsum('btd,bte->bde', Z, Z) / T             # (B, D, D)

    # Symmetrize & fix diagonals (nice-to-have for numeric drift)
    C = 0.5 * (C + C.transpose(0, 2, 1))
    batch_indices = np.arange(B)[:, None]
    diag_indices = np.arange(D)
    C[batch_indices, diag_indices, diag_indices] = 1.0

    return C


class BasicTrainer:
    """
    Basic training loop using model-provided helpers. 
    Trains the model for a given number of epochs but stops if validation loss doesn't improve for `patience` epochs.

    Args
    ----
    model : torch.nn.Module
        One of the models from `ml4fmri.models` package.
    train_loader, val_loader, test_loader : DataLoader
        Dataloaders that yield batches with data appropriate for the model. `model.prepare_dataloader` can be used to create them.
    epochs : int, optional
        Max epochs (default: 200).
    lr : float, optional
        Optimizer LR (default: use model's LR).
    device : str, optional
        "cuda", "mps", or "cpu". Default: auto-detect (cuda -> mps -> cpu).
    patience : int, optional
        Early stopping patience on validation loss (default: 30).

    Returns on `run()`
    ------------------
    (train_log, test_log, predictions_log) : tuple of pandas.DataFrame
        Each with "model" already stamped; cvbench adds "fold" (and, for
        predictions_log, "sample_id").
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int = 200,
        lr: float = None,
        device: str = None,
        patience: int = 30,
    ) -> None:
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.epochs = epochs
        if lr is None:
            lr = model.lr
        self.lr = lr
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda") if torch.cuda.is_available() \
                else torch.device("mps") if torch.backends.mps.is_available() \
                    else torch.device("cpu")
        self.patience = patience


        # Optimizer
        self.optimizer = model.get_optimizer(lr)

        # bookkeeping
        self.n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.best_state = None
        self.best_val_loss = None

        # move model to device
        self.model.to(self.device)


    def _epoch(self, loader: DataLoader, train: bool):
        with torch.set_grad_enabled(train):
            self.model.train(train)
            total_loss = 0.0
            n_batches = len(loader)
            agg_log = {}
            probs, trues = [], []   # pooled across the whole dataloader

            for batch in loader:
                batch = tuple(t.to(self.device) for t in batch) # cast batch to device
                loss, batch_log = self.model.handle_batch(batch)

                total_loss += float(loss.detach().cpu().item())

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # accumulate per-batch scalar loss terms
                for k, v in batch_log.items():
                    if isinstance(v, (int, float)):
                        agg_log[k] = agg_log.get(k, 0.0) + float(v)
                probs.append(batch_log["y_prob"])
                trues.append(batch_log["y_true"])

            # losses: per-batch mean; metrics: computed once over the pooled epoch predictions
            agg_log = {k: v / n_batches for k, v in agg_log.items()}
            agg_log["loss"] = total_loss / n_batches
            y_prob = np.vstack(probs)
            y_true = np.hstack(trues)
            agg_log.update(compute_metrics(y_prob, y_prob.argmax(axis=1), y_true))
            # y_prob/y_true are pooled in loader order; for a shuffle=False loader
            # that is the order of the underlying dataset, which is what lets
            # cvbench map test predictions back to sample indices. MAY BREAK IN THE FUTURE.
            return agg_log, y_prob, y_true

    def run(self):
        train_logs = []
        stopper = EarlyStopping(patience=self.patience) # monitors training, stops if validation loss does not improve

        ### Training loop
        start = time.time()
        for epoch in range(self.epochs):
            train_log, _, _ = self._epoch(self.train_loader, train=True)
            val_log, _, _ = self._epoch(self.val_loader, train=False)

            # handle logs
            epoch_log = {"model": self.model.__class__.__name__, "epoch": epoch, "lr": self.lr}
            epoch_log.update({f"train_{k}": v for k, v in train_log.items()})
            epoch_log.update({f"val_{k}": v for k, v in val_log.items()})
            train_logs.append(epoch_log)

            # track training via validation loss
            improved = stopper.step(val_log["loss"])
            if improved:
                self.best_state = deepcopy(self.model.state_dict())


            if stopper.stop:
                break
        training_time = time.time() - start
        
        # build train history DataFrame
        train_logs = pd.DataFrame(train_logs)

        # restore best weights
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)


        ### Test loop
        model_name = self.model.__class__.__name__
        test_metrics = {"model": model_name}
        test_log, test_y_prob, test_y_true = self._epoch(self.test_loader, train=False)
        test_metrics.update({f"test_{k}": v for k, v in test_log.items()})
        test_metrics.update({
            "train_time": training_time,
            "n_params": self.n_params,
        })
        test_logs = pd.DataFrame([test_metrics])

        # raw test-fold predictions, in test-loader order (the order the test set
        # was passed in); cvbench attaches "sample_id" and "fold"
        predictions_log = pd.DataFrame({
            "model": model_name,
            "y_true": test_y_true.astype(int),
            "y_pred": test_y_prob.argmax(axis=1).astype(int),
        })
        for c in range(test_y_prob.shape[1]):
            predictions_log[f"p_{c}"] = test_y_prob[:, c]

        return train_logs, test_logs, predictions_log

class EarlyStopping:
    """Early stopping mechanism, watches if metric is minimized."""
    def __init__(self, patience: int = 30):
        self.patience = patience
        self.best_score = None
        self.counter = 0
        self.stop = False

    def step(self, value: float) -> bool:
        """
        Returns a boolean indicating if the metric improved.
        Toggles stop flag if patience is exceeded.
        """
        if self.best_score is None or value < self.best_score - 1e-12:
            self.best_score = value
            self.counter = 0
            return True  # improved
        self.counter += 1
        if self.counter >= self.patience:
            self.stop = True
        return False