"""
Cross-validation runner and report object for ml4fmri models.

Design goals
------------
- Keep models as plain nn.Module with your standardized helpers (`prepare_dataloader`, `train_model`).
- No disk I/O; keep best checkpoints in memory inside each model's own training.
- Provide a simple, polyssifier-like entrypoint:

    from ml4fmri.report import cvbench
    report = cvbench(data, labels, models=[meanMLP, LSTM], n_folds=8)
    report.plot_scores()
    report.plot_training_curves()
    df_test = report.get_test_dataframe()
    df_train = report.get_train_dataframe()

Notes
-----
- Assumes time-series data shaped (B, T, D) and integer labels shaped (B,).
- Infers `input_size=D` and `output_size=n_classes` per model unless overridden via `model_kwargs`.
- Uses StratifiedKFold for outer CV and a random stratified split of the training fold to create a validation set.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import logging
logging.basicConfig(
    format="%(name)s %(levelname)s: %(message)s",
    level=logging.INFO,
)

# -----------------------------
# CV runner (cvbench)
# -----------------------------


def _discover_models():
    """
    Find model classes under ml4fmri.models.
    Finds classes that have `prepare_dataloader` and `train_model` methods.
    """
    import ml4fmri.models as mdl
    found = {}
    for name in dir(mdl):
        obj = getattr(mdl, name, None)
        nested_obj = getattr(obj, name, None)
        if hasattr(obj, "prepare_dataloader") and hasattr(obj, "train_model"):
            found[name] = obj
        if (nested_obj and hasattr(nested_obj, "prepare_dataloader") and hasattr(nested_obj, "train_model")):
            found[name] = nested_obj
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
    models : "lite" (default) | "all" | "model_name" (e.g., "meanMLP") | list of model names (e.g., ["meanMLP", "meanLSTM"]).
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
    assert data.ndim == 3, f"Expected data with shape (Batch, Time, (D)Features); got {data.shape}"
    B, T, D = data.shape
    C = np.unique(labels).shape[0]
    assert C >= 2, f"Expected at least 2 classes in labels; got {C}"

    # automated model discovery and selection routine
    available_model_dict = _discover_models() # scan ml4fmri.models for model classes
    if models == 'all':
        chosen = list(available_model_dict.keys())
    elif models == 'lite':
        chosen = ["meanMLP"]
    elif isinstance(models, str):
        chosen = [models]
        assert models in available_model_dict, f"Model '{models}' not found among available models: {list(available_model_dict.keys())}"
    elif isinstance(models, list):
        chosen = models
        missing = [m for m in models if m not in available_model_dict]
        assert not missing, f"Models {missing} not found among available models: {list(available_model_dict.keys())}"
            
    chosen_model_dict = {m: available_model_dict[m] for m in chosen}


    # Run CV for each chosen model
    skf = StratifiedKFold(n_splits=int(n_folds), shuffle=True, random_state=int(random_state))

    train_logs = []
    test_logs = []

    for model_name, model_class in chosen_model_dict.items(): # Model loop
        LOGGER.info(f"Training model: {model_name}")

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(data, labels)): # CV loop

            # Split data into train and test sets
            X_train_full, y_train_full = data[train_idx], labels[train_idx]
            X_test, y_test = data[test_idx], labels[test_idx]

            # Inner split for validation from the training fold
            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
            train_idx, val_idx = next(sss.split(X_train_full, y_train_full))
            X_train, y_train = X_train_full[train_idx], y_train_full[train_idx]
            X_val, y_val = X_train_full[val_idx], y_train_full[val_idx]

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

            # Annotate and save logs
            train_df["fold"] = fold_idx
            test_df["fold"] = fold_idx

            train_logs.append(train_df)
            test_logs.append(test_df)

            fold_logger = LOGGER.getChild(f"{model_name}")
            fold_logger.info(f"Fold {fold_idx+1}/{n_folds}: Test AUC {test_df['test_auc'].iloc[-1]:.3f}")

    train_df_all = pd.concat(train_logs, ignore_index=True)
    test_df_all = pd.concat(test_logs, ignore_index=True)

    meta = {
        "n_folds": int(n_folds),
        "val_ratio": float(val_ratio),
        "models": [m for m in chosen],
        "random_state": int(random_state),
        "input_size": int(D),
        "n_classes": int(C),
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

    def plot_scores(self, metric="test_auc", show=True):
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
        ax.boxplot(data, labels=order, showmeans=True)
        ax.set_ylabel(metric)
        ax.set_title(f"Test AUC across models in {self.meta['n_folds']}-fold CV")
        ax.grid(True, axis="y", linestyle=":", linewidth=0.5)
        fig.tight_layout()

        if show:
            plt.show()
            return None
        return fig
        

    def plot_training_curves(self, metric="auc", loss_key="loss", per_model=True, show=True):
        """
        Plot train/val curves across epochs for all models & folds.
        Defaults assume columns like 'train_auc', 'val_auc', 'train_loss', 'val_loss'.
        """
        train_key, val_key = "train_%s" % metric, "val_%s" % metric
        tloss_key, vloss_key = "train_%s" % loss_key, "val_%s" % loss_key
        df = self.train_df.copy()
        models = df["model"].unique().tolist()

        # Metric figure
        if per_model:
            n = len(models)
            fig_m, axs_m = plt.subplots(n, 1, figsize=(7.5, 3.0 * n), sharex=True)
            if n == 1:
                axs_m = [axs_m]
            for ax, model in zip(axs_m, models):
                d = df[df["model"] == model]
                for fold, g in d.groupby("fold"):
                    if train_key in g and val_key in g:
                        ax.plot(g["epoch"], g[train_key], alpha=0.6, label="fold %s train" % fold)
                        ax.plot(g["epoch"], g[val_key], alpha=0.9, linestyle="--", label="fold %s val" % fold)
                ax.set_title("%s – %s" % (model, metric))
                ax.set_ylabel(metric)
                ax.grid(True, linestyle=":", linewidth=0.5)
            axs_m[-1].set_xlabel("epoch")
            handles, labels = axs_m[0].get_legend_handles_labels()
            if handles:
                fig_m.legend(handles, labels, loc="upper right")
            fig_m.tight_layout(rect=[0, 0, 0.9, 1])
        else:
            fig_m, ax = plt.subplots(figsize=(7.5, 4.5))
            for model, d in df.groupby("model"):
                for fold, g in d.groupby("fold"):
                    if train_key in g and val_key in g:
                        ax.plot(g["epoch"], g[train_key], alpha=0.4, label="%s f%s train" % (model, fold))
                        ax.plot(g["epoch"], g[val_key], alpha=0.8, linestyle="--", label="%s f%s val" % (model, fold))
            ax.set_title("Training/Validation %s" % metric)
            ax.set_xlabel("epoch")
            ax.set_ylabel(metric)
            ax.grid(True, linestyle=":", linewidth=0.5)
            fig_m.tight_layout()

        # Loss figure
        if per_model:
            n = len(models)
            fig_l, axs_l = plt.subplots(n, 1, figsize=(7.5, 3.0 * n), sharex=True)
            if n == 1:
                axs_l = [axs_l]
            for ax, model in zip(axs_l, models):
                d = df[df["model"] == model]
                for fold, g in d.groupby("fold"):
                    if tloss_key in g and vloss_key in g:
                        ax.plot(g["epoch"], g[tloss_key], alpha=0.6, label="fold %s train" % fold)
                        ax.plot(g["epoch"], g[vloss_key], alpha=0.9, linestyle="--", label="fold %s val" % fold)
                ax.set_title("%s – %s" % (model, loss_key))
                ax.set_ylabel(loss_key)
                ax.grid(True, linestyle=":", linewidth=0.5)
            axs_l[-1].set_xlabel("epoch")
            handles, labels = axs_l[0].get_legend_handles_labels()
            if handles:
                fig_l.legend(handles, labels, loc="upper right")
            fig_l.tight_layout(rect=[0, 0, 0.9, 1])
        else:
            fig_l, ax = plt.subplots(figsize=(7.5, 4.5))
            for model, d in df.groupby("model"):
                for fold, g in d.groupby("fold"):
                    if tloss_key in g and vloss_key in g:
                        ax.plot(g["epoch"], g[tloss_key], alpha=0.4, label="%s f%s train" % (model, fold))
                        ax.plot(g["epoch"], g[vloss_key], alpha=0.8, linestyle="--", label="%s f%s val" % (model, fold))
            ax.set_title("Training/Validation %s" % loss_key)
            ax.set_xlabel("epoch")
            ax.set_ylabel(loss_key)
            ax.grid(True, linestyle=":", linewidth=0.5)
            fig_l.tight_layout()

        if show:
            plt.show(fig_m)
            plt.show(fig_l)
            return None, None
        return fig_m, fig_l