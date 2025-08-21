"""
Cross-validation benchmarking runner and report object for ml4fmri models.

Use example:
```python
from ml4fmri.report import cvbench
report = cvbench(data, labels, models=[meanMLP, LSTM], n_folds=8)
report.plot_scores()
report.plot_training_curves()
df_test = report.get_test_dataframe()
df_train = report.get_train_dataframe()
```

Notes
-----
- Assumes time-series data shaped (B, T, D) and integer labels shaped (B,).
- Infers `input_size=D` and `output_size=n_classes`.
"""

import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import torch

import warnings
import logging
import inspect
import os
logging.basicConfig(
    format="%(name)s %(levelname)s: %(message)s",
    level=logging.INFO,
)

# -----------------------------
# CV runner (cvbench)
# -----------------------------

MODELS = [
    'LR',
    'meanMLP',
    'MILC',
    'Transformer',
    'meanTransformer',
    'BNT',
    'BrainNetCNN',
    'LSTM',
    'meanLSTM',
    'FBNetGen',
    'DICE',
    'Glacier',
    'BolT'
]

LITE_MODELS = [
    'LR',
    'meanMLP',
    'MILC',
    'meanTransformer',
    'BNT',
    'BrainNetCNN',
    'meanLSTM',
]

TS_MODELS = [
    'meanMLP',
    'MILC',
    'Transformer',
    'meanTransformer',
    'LSTM',
    'meanLSTM',
    'FBNetGen',
    'DICE',
    'Glacier',
    'BolT'
]

FNC_MODELS = [
    'LR',
    'BNT',
    'BrainNetCNN',
]

def _discover_models():
    """
    Find model classes under ml4fmri.models.
    Finds classes that have `prepare_dataloader` and `train_model` methods.
    """
    import ml4fmri.models as mdl

    found = {}
    for name, obj in inspect.getmembers(mdl, inspect.isclass):
        if hasattr(obj, "prepare_dataloader") and hasattr(obj, "train_model"):
            found[name] = obj
    return found

def cvbench(
    data,
    labels,
    models: str | list[str] = "lite",
    n_folds: int = 10,
    val_ratio: float = 0.2,
    random_state: int = 42,
    epochs: int = 200,
    lr: float = None,
    device: str = None,
    patience: int = 30,
):
    """
    Run cross-validation across multiple models and return a report (2 dataframes with a couple of viz functions).

    Parameters
    ----------
    data : array (B, T, D)
    labels : array (B,)
    models : "lite" (default) | "all" | "ts" | "fnc" | "model_name" (e.g., "meanMLP") | list of model names (e.g., ["meanMLP", "meanLSTM"]).
    n_folds : number of CV folds.
    val_ratio : fraction of the training fold to reserve for validation.
    random_state : seed for the CV splits.
    epochs : maximum number of epochs to train each model.
    lr : learning rate for the optimizer (if None, uses model's default).
    device : device to run the training on (if None, uses cuda -> apple mps -> cpu).
    patience : early stopping patience for training.
    """

    LOGGER = logging.getLogger("cvbench")

    # check data
    data = np.asarray(data)
    labels = np.asarray(labels)
    assert data.shape[0] == labels.shape[0], f"data and labels batch dimensions mismatch (data {data.shape}[0] != labels {labels.shape[0]})"
    assert data.ndim == 3, f"Expected data with 3 dimensions (Batch, Time, (D)Features); got {data.shape}"
    # check for NaNs
    if np.isnan(data).any():
        warnings.warn("Found NaN values in 'data' array", UserWarning)
    if np.isnan(labels).any():
        warnings.warn("Found NaN values in 'labels' array", UserWarning)

    B, T, D = data.shape
    C = np.unique(labels).shape[0]
    assert C >= 2, f"Expected at least 2 classes in labels; got {C}"

    # automated model discovery and selection routine
    available_model_dict = _discover_models() # scan ml4fmri.models for model classes
    if models == 'all':
        chosen = MODELS
    elif models == 'lite':
        chosen = LITE_MODELS
    elif models == 'ts':
        chosen = TS_MODELS
    elif models == 'fnc':
        chosen = FNC_MODELS
    elif isinstance(models, str):
        chosen = [models]
        assert models in available_model_dict, f"Model '{models}' not found among available models: {list(available_model_dict.keys())}"
    elif isinstance(models, list):
        chosen = models
        missing = [m for m in models if m not in available_model_dict]
        assert not missing, f"Models {missing} not found among available models: {list(available_model_dict.keys())}"
          
    chosen_model_dict = {m: available_model_dict[m] for m in chosen}
    LOGGER.info(f"Running models: {chosen}")

    # Run CV for each chosen model
    skf = StratifiedKFold(n_splits=int(n_folds), shuffle=True, random_state=int(random_state))

    if device is not None:
        LOGGER.info(f"Using device: {device}")
    else:
        device = "cuda" if torch.cuda.is_available() \
            else "mps" if torch.backends.mps.is_available() \
                else "cpu"

        LOGGER.info(f"Using device: {device}")
    
    train_logs = []
    test_logs = []
    train_splits, val_splits, test_splits = [], [], []
    
    for model_name, model_class in chosen_model_dict.items(): # Model loop
        LOGGER.info(f"Training model: {model_name}")
            
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(data, labels)): # CV loop

            start = time.time()

            # Split data into train and test sets
            X_train_full, y_train_full = data[train_idx], labels[train_idx]
            X_test, y_test = data[test_idx], labels[test_idx]

            # Inner split for validation from the training fold
            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
            tr_idx, val_idx = next(sss.split(X_train_full, y_train_full))
            X_train, y_train = X_train_full[tr_idx], y_train_full[tr_idx]
            X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]

            train_splits.append(train_idx[tr_idx].tolist())
            val_splits.append(train_idx[val_idx].tolist())
            test_splits.append(test_idx.tolist())

            # Prepare dataloaders via model class helper, handle data transforms inside if needed 
            # (e.g., FNC derivation from time series, or z-scoring)
            train_loader = model_class.prepare_dataloader(X_train, y_train, shuffle=True)
            val_loader   = model_class.prepare_dataloader(X_val, y_val, shuffle=False)
            test_loader  = model_class.prepare_dataloader(X_test, y_test, shuffle=False)

            # Instantiate model
            model = model_class(input_size=D, output_size=C)

            # Train model
            train_df, test_df = model.train_model(train_loader, val_loader, test_loader,
                                                  epochs=epochs, lr=lr, device=device, patience=patience)

            time_elapsed = time.time() - start
            
            # Annotate and save logs
            train_df["fold"] = fold_idx
            test_df["fold"] = fold_idx

            train_logs.append(train_df)
            test_logs.append(test_df)

            fold_logger = LOGGER.getChild(f"{model_name}")
            # locate the epoch with minimum validation loss and get its train AUC
            best_idx = train_df['val_loss'].idxmin()
            train_score = train_df.loc[best_idx, 'train_auc']
            val_score = train_df.loc[best_idx, 'val_auc']
            test_score = test_df['test_auc'].iloc[-1]
            fold_logger.info(f"Fold {(fold_idx+1):02d}/{n_folds:02d}: Train/Val/Test AUC {train_score:.3f}/{val_score:.3f}/{test_score:.3f}: Time elapsed {time_elapsed:.2f} s")

    train_df_all = pd.concat(train_logs, ignore_index=True)
    test_df_all = pd.concat(test_logs, ignore_index=True)

    meta = {
        "models": [m for m in chosen],
        "n_folds": int(n_folds),
        "val_ratio": float(val_ratio),
        "random_state": int(random_state),
        "input_size": int(D),
        "n_classes": int(C),
        "train_indices": train_splits[:n_folds],
        "val_indices":   val_splits[:n_folds],
        "test_indices":  test_splits[:n_folds],
    }
    return Report(train_df_all, test_df_all, meta)


# -----------------------------
# Public Report object
# -----------------------------

class Report(object):
    def __init__(self, train_df, test_df, meta):
        self.train_df = pd.DataFrame(train_df).copy()
        self.test_df = pd.DataFrame(test_df).copy()
        self.meta = dict(meta)

    def get_train_dataframe(self):
        return self.train_df.copy()

    def get_test_dataframe(self):
        return self.test_df.copy()

    def get_meta(self):
        return self.meta.copy()

    def save(self, path=None):
        """Save train and test dataframes at path directory"""
        if path is not None:
            assert os.path.isdir(path), f"Path '{path}' is not a valid directory"

        train_path = os.path.join(path, "cvbench_train.csv")
        test_path = os.path.join(path, "cvbench_test.csv")
        meta_path = os.path.join(path, "cvbench_meta.json")

        train_df = self.train_df.copy()
        test_df = self.test_df.copy()
        meta = self.meta.copy()
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        with open(meta_path, "w") as f:
            json.dump(meta, f)

    def plot_scores(self, metric="test_auc", show=True, show_outliers=False):
        """Boxplots of a test metric per model across folds (default: 'test_auc')."""
        if metric not in self.test_df.columns:
            raise KeyError("Metric '%s' not found in test_df columns: %s" % (metric, list(self.test_df.columns)))
        order = (
            self.test_df.groupby("model")[metric]
            .median()
            .sort_values(ascending=False)
            .index.tolist()
        )
        data = [self.test_df[self.test_df["model"] == m][metric].dropna().values for m in order]
        fig, ax = plt.subplots(figsize=(7.5, 4.5))

        # draw blue dashed line at y = 0.5 if plotting an AUC metric
        if 'auc' in metric:
            ax.axhline(y=0.5, color='lightblue', linestyle='dashed', linewidth=1)

        ax.boxplot(data, tick_labels=order, showfliers=show_outliers)
        ax.set_ylabel("Test AUC" if metric == "test_auc" else metric)
        ax.grid(True, axis="y", linestyle=":", linewidth=0.5)

        # stagger x tick labels to avoid overlap
        for i, lbl in enumerate(ax.get_xticklabels()):
            offset = 0.05
            x, y = lbl.get_position()
            lbl.set_position((x, y if i % 2 == 0 else y - offset))

        fig.tight_layout()

        if show:
            plt.show()
            return None
        return fig
    
    def plot_scores_h(self, metric="test_auc", show=True, show_outliers=False):
        """
        Horizontal boxplots of a test metric per model across folds.
        - Best (highest median) model appears at the TOP.
        - Y labels are drawn inside the axes, slightly above each box, left-aligned.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        if metric not in self.test_df.columns:
            raise KeyError("Metric '%s' not found in test_df columns: %s"
                        % (metric, list(self.test_df.columns)))

        # Order models by median score (desc) → best first.
        order = (
            self.test_df.groupby("model")[metric]
            .median()
            .sort_values(ascending=False)
            .index.tolist()
        )
        # Data arrays per model, in that order.
        data = [self.test_df.loc[self.test_df["model"] == m, metric].dropna().values
                for m in order]

        n = len(order)
        # Positions reversed so the first (best) ends up at the TOP.
        positions = np.arange(n, 0, -1)

        # Figure height scales with number of models.
        fig_h = max(4.0, 0.5 * n + 1.5)
        fig, ax = plt.subplots(figsize=(7.5, fig_h))

        ax.boxplot(
            data,
            vert=False,
            positions=positions,
            showfliers=show_outliers,
            manage_ticks=False,   # we'll manage ticks/labels ourselves
        )
        # draw blue dashed line at x = 0.5 if plotting an AUC metric
        if 'auc' in metric:
            ax.axvline(x=0.5, color='lightblue', linestyle='dashed', linewidth=1)

        ax.set_xlabel("Test AUC" if metric == "test_auc" else metric)
        ax.set_yticks([])  # hide default tick labels
        ax.grid(True, axis="x", linestyle=":", linewidth=0.5)
        ax.set_ylim(0.5, n + 0.5)

        # Add labels inside the plot, left-aligned, slightly above each box.
        # Use a blended transform: x in axes-coords (0..1), y in data-coords.
        trans = ax.get_yaxis_transform()
        y_offset = 0.15  # how much above the box center
        x_inset = 0.01   # 1% from the left edge inside the axes

        for y, label in zip(positions, order):
            ax.text(x_inset, y + y_offset, label,
                    transform=trans, ha="left", va="bottom")

        fig.tight_layout()

        if show:
            plt.show()
            return None
        return fig
        

    def plot_training_curves(self, show=True, per_model=True):
        """
        Plot train/val curves across epochs for all models & folds.
        Creates TWO figures: (1) loss, (2) AUC (if available).
        - Per-model subplots do NOT share x-limits; each shows its own epoch ticks.
        - Legend shows only two example lines (black solid = train, black dashed = val).
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
        from matplotlib.lines import Line2D

        df = self.train_df.copy()
        if "model" not in df or "fold" not in df or "epoch" not in df:
            raise KeyError("train_df must contain 'model', 'fold', and 'epoch' columns.")

        models = list(df["model"].unique())

        plots = [
            ("loss",  "train_loss", "val_loss"),
            ("auc",   "train_auc",  "val_auc"),
        ]

        figs = []

        for name, train_key, val_key in plots:
            # skip figure if columns are absent entirely
            if train_key not in df.columns or val_key not in df.columns:
                continue

            if per_model:
                n = len(models)
                fig, axs = plt.subplots(n, 1, figsize=(7.5, max(3.0 * n, 3.5)), sharex=False)
                if n == 1:
                    axs = [axs]
                for ax, model in zip(axs, models):
                    d = df[df["model"] == model]
                    # plot each fold without adding labels (we’ll use custom legend)
                    for _, g in d.groupby("fold"):
                        if train_key in g.columns and val_key in g.columns:
                            ax.plot(g["epoch"], g[train_key], color="C0", alpha=0.6, linewidth=1.2)
                            ax.plot(g["epoch"], g[val_key], color="C0", alpha=0.9, linestyle="--", linewidth=1.2)
                    ax.set_title(f"{model} – {name}")
                    ax.set_ylabel(name)
                    ax.grid(True, linestyle=":", linewidth=0.5)
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                    ax.set_xlabel("epoch")
                # custom legend: only two example lines
                handles = [
                    Line2D([0], [0], color="black", linestyle="-", linewidth=1.5, label="train"),
                    Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, label="val"),
                ]
                fig.legend(handles=handles, loc="upper right")
                fig.tight_layout(rect=[0, 0, 0.9, 1])
            else:
                fig, ax = plt.subplots(figsize=(7.5, 4.5))
                for _, d in df.groupby("model"):
                    for _, g in d.groupby("fold"):
                        if train_key in g.columns and val_key in g.columns:
                            ax.plot(g["epoch"], g[train_key], color="C0", alpha=0.4, linewidth=1.2)
                            ax.plot(g["epoch"], g[val_key], color="C0", alpha=0.9, linestyle="--", linewidth=1.2)
                ax.set_title(f"Training/Validation {name}")
                ax.set_xlabel("epoch")
                ax.set_ylabel(name)
                ax.grid(True, linestyle=":", linewidth=0.5)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                # custom legend
                handles = [
                    Line2D([0], [0], color="black", linestyle="-", linewidth=1.5, label="train"),
                    Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, label="val"),
                ]
                fig.legend(handles=handles, loc="upper right")
                fig.tight_layout()

            figs.append(fig)

        if show:
            for f in figs:
                plt.show(f)
            return (None, None) if len(figs) == 2 else (None,) * len(figs)

        # return (loss_fig, auc_fig) when both exist; otherwise whatever is available
        return tuple(figs)
