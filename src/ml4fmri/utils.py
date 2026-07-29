"""
Shared types and helpers.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class FoldResults:
    """
    A dataclass of everything a single (model, fold) run in cvbench produces; the return type of `train_model`.

    Attributes
    ----------
    train_log : pandas.DataFrame
        One row per epoch, with "train_"/"val_"-prefixed metrics. Non-iterative
        models (e.g. LR) return a single row.
    test_metrics : dict
        A flat dict of final test-set scores with "test_"-prefixed keys, plus bookkeeping such as
        "train_time" and "n_params".
    predictions : dict, optional
        {"y_prob": (N, C) array, "y_true": (N,) array} for the test fold, in the
        order the test set was passed in.
    """

    train_log: pd.DataFrame
    test_metrics: dict
    predictions: Optional[dict] = None

    def __post_init__(self):
        # Validate here rather than in cvbench, so that someone writing a custom
        # model gets the complaint from the object they just built.
        if not isinstance(self.train_log, pd.DataFrame):
            raise TypeError(
                f"FoldResults.train_log must be a pandas DataFrame with one row "
                f"per epoch; got {type(self.train_log).__name__}"
            )
        if not isinstance(self.test_metrics, dict):
            raise TypeError(
                f"FoldResults.test_metrics must be a flat dict of test scores; "
                f"got {type(self.test_metrics).__name__}"
            )
        if self.predictions is not None:
            missing = {"y_prob", "y_true"} - set(self.predictions)
            if missing:
                raise ValueError(
                    f"FoldResults.predictions is missing {sorted(missing)}; "
                    'expected {"y_prob": (N, C) array, "y_true": (N,) array}'
                )
