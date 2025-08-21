# pylint: disable=invalid-name, missing-function-docstring
""" 
LR model module
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import time

from .helper_functions import corrcoef_batch, compute_metrics

class LR():
    """
    FNC MODEL
    Basic logistic regression model for fMRI data.
    Based on the sklearn's logistic regression, basically a wrapper that adds cvbench-required API.
    Expected input shape: [batch_size, input_feature_size, input_feature_size].
    Output: [batch_size, n_classes], loss_load = {"logits": logits}
    """

    def __init__(
            self,
            input_size: int,
            output_size: int,
    ):
        """
        Initialize LR model.
        """
        self.model = LogisticRegression()
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        """ For compatibility """
        return 0, {"logits": None} # all models return (1) logits and (2) a dictionary of data for computing loss

    def compute_loss(self, loss_load, targets):
        """ For compatibility """

        loss, loss_log = 0, {"dummy_loss": 0}
        return loss, loss_log

    def handle_batch(self, batch):
        """ For compatibility """

        loss, batch_log = 0, {"dummy_loss": 0, "dummy_score": 1}
        return loss, batch_log
    
    def get_optimizer(self, lr=None):
        """ For compatibility """
        return None

    @staticmethod
    def prepare_dataloader(data,
                           labels,
                           batch_size: int = 64,
                           shuffle: bool = True):
        """
        Returns FNC and label numpy arrays that can be used for training of sklearn's LR.
        Args:
            data (array-like): Time series data of shape (B, T, D).
            labels (array-like): Class labels for the data.
            batch_size (int, optional): Batch size for the DataLoader. Defaults to 64.
            shuffle (bool, optional): Whether to shuffle batching in the DataLoader. Defaults to True.
        
        Returns:
            (fnc, labels): FNC and labels as numpy arrays.
        """
        fnc = corrcoef_batch(data)
        n_rois = fnc.shape[1]
        tril_idx = np.tril_indices(n_rois, k=-1)
        fnc = fnc[:, tril_idx[0], tril_idx[1]]

        return (fnc, labels)
    
    def train_model(self,
              train_loader,
              val_loader,
              test_loader,
              epochs= None,
              lr= None,
              device= None,
              patience= None,
        ):
        """
        Standard model training routine, rewired for LR.
        Args:
            train_loader (fnc, labels): Tuple of numpy arrays for LR training returned by prepare_dataloader
            val_loader (fnc, labels): Merged with train data
            test_loader (fnc, labels): Test data
            epochs (optional): Kept for compatibility
            lr (optional): Kept for compatibility
            device (optional): Kept for compatibility
            patience (optional): Kept for compatibility
        """
        train_data, train_labels = train_loader
        val_data, val_labels = val_loader
        test_data, test_labels = test_loader

        # merge training and validation sets
        train_data = np.concatenate([train_data, val_data], axis=0)
        train_labels = np.concatenate([train_labels, val_labels], axis=0)

        # train
        start_time = time.time()
        self.model.fit(
            X=train_data,
            y=train_labels,
        )
        training_time = time.time() - start_time

        # train log
        train_logs = {"model": "LR", "epoch": 0, "lr": 0}
        y_score = self.model.predict_proba(train_data)
        y_pred = np.argmax(y_score, axis=-1).astype(np.int32)
        train_log = compute_metrics(y_true=train_labels, y_pred=y_pred, y_prob=y_score)
        train_logs.update({f"train_{k}": v for k, v in train_log.items()})
        train_logs.update({f"val_{k}": v for k, v in train_log.items()})
        train_logs = pd.DataFrame([train_logs])
        
        # test
        y_score = self.model.predict_proba(test_data)
        y_pred = np.argmax(y_score, axis=-1).astype(np.int32)
        test_logs = {"model": "LR"}
        test_log = compute_metrics(y_true=test_labels, y_pred=y_pred, y_prob=y_score)
        test_logs.update({f"test_{k}": v for k, v in test_log.items()})
        test_logs.update({
            "train_time": training_time,
            "n_params": self.model.coef_.size,
        })
        test_logs = pd.DataFrame([test_logs])

        return train_logs, test_logs
