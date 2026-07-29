"""
Cross-validation benchmarking runner and report object for ml4fmri models.

Use example:
```python
from ml4fmri import cvbench
report = cvbench(data, labels, models=["meanMLP", "LSTM"], n_folds=8,
                 sample_ids=subject_ids, save_dir="results/mdd_vs_hc")
report.plot_scores()
report.plot_training_curves()
report.plot_confusion()
df_test = report.get_test_dataframe()
df_train = report.get_train_dataframe()
df_pred = report.get_predictions_dataframe()   # raw test-fold probabilities
```

Results are written to disk as the run proceeds, so an interrupted run still leaves
usable partial results behind. Layout of a results directory:

```
<save_dir>/
├── cvbench_meta.json            # version, seeds, run configuration, status
├── cvbench_train.csv            # model,fold,epoch,... per-epoch training log
├── cvbench_test.csv             # model,fold,... one row per (model, fold)
├── cvbench_predictions.csv      # model,fold,sample_id,y_true,y_pred,p_0,...,p_{C-1}
└── fold_records/
    ├── sample_order.csv         # pos,sample_id -- decodes indices.json; only with sample_ids=
    ├── fold_00/
    │   ├── indices.json         # positional train/val/test indices (model-independent)
    │   └── checkpoints/         # best-validation weights per model
    │       ├── meanMLP.pt
    │       └── LR.joblib
    └── fold_01/ ...
```

Notes
-----
- Assumes time-series data shaped (B, T, D) and integer labels shaped (B,).
- Infers `input_size=D` and `output_size=n_classes`.
- Test-fold predictions are logged in the order of the test split, which relies on
  `prepare_dataloader(..., shuffle=False)` preserving sample order. cvbench asserts
  this holds; custom models that reorder or drop samples may fail loudly.
"""

import json
import time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from .utils import _seed_everything, _save_checkpoint

import torch

from collections import OrderedDict
from datetime import datetime
from importlib.metadata import version
import builtins
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
    'BNT',
]

TS_MODELS = [
    'meanMLP',
    'MILC',
    'Transformer',
    'meanTransformer',
    'LSTM',
    'meanLSTM',
    'DICE',
    'Glacier',
    'BolT'
]

FNC_MODELS = [
    'LR',
    'BNT',
    'BrainNetCNN',
    'FBNetGen',
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


# -----------------------------
# Saving helpers
# -----------------------------

def _order_columns(df, first=("model", "fold")):
    """Move identifier columns to the front, preserving the order of the rest."""
    lead = [c for c in first if c in df.columns]
    rest = [c for c in df.columns if c not in lead]
    return df[lead + rest]


def cvbench(
    data,
    labels,
    models: str | list[str] = None,
    custom_models: type | list[type] = None,
    n_folds: int = 10,
    val_ratio: float = 0.2,
    cv_seed: int = 42,
    val_seed: int = 42,
    init_seed: int = 42,
    epochs: int = 200,
    lr: float = None,
    device: str = None,
    patience: int = 30,
    sample_ids=None,
    save_dir=None,
    save_checkpoints: bool = True,
):
    """
    Run cross-validation across multiple models and return a Report.

    Parameters
    ----------
    data : array (B, T, D)
    labels : array (B,)
    models : "lite" | "all" | "ts" | "fnc" | "model_name" (e.g., "meanMLP") | list of model names (e.g., ["meanMLP", "meanLSTM"]). Default: "lite" if using CPU, "all" if GPU or Apple MPS is available.
    custom_models: model class (or a list of them) with API compatible with ml4fmri, will be used along the vanilla models, or override them (if named the same). Check Colab tutorial for examples.
    n_folds : number of CV folds.
    val_ratio : fraction of the training fold to reserve for validation.
    cv_seed : seed for the outer (train+val)/test fold assignment.
    val_seed : seed for the inner train/val split within each fold.
    init_seed : seed for model initialisation; dropout and batch order follow from it.
        Each fold uses `init_seed + fold_idx`, so folds are independent draws.
    epochs : maximum number of epochs to train each model.
    lr : learning rate for the optimizer (if None, uses model's default).
    device : device to run the training on (if None, uses cuda -> apple mps -> cpu).
    patience : early stopping patience for training.
    sample_ids : optional unique identifiers (ints or strings), length B, logged with
        the predictions. Defaults to `np.arange(B)`. Real subject IDs keep results
        joinable across runs.
    save_dir : results directory. Defaults to a timestamped `./cvbench_<timestamp>/`;
        pass `False` to keep results in memory only.
    save_checkpoints : store best-validation weights under `fold_records/fold_XX/checkpoints/`.
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

    ## sample identifiers: default to positional, but let the caller supply real IDs
    sample_ids_provided = sample_ids is not None
    if sample_ids is None:
        sample_ids = np.arange(B)
    else:
        sample_ids = np.asarray(sample_ids)
        assert sample_ids.shape[0] == B, \
            f"sample_ids has {sample_ids.shape[0]} entries but data has {B} samples"
        assert len(np.unique(sample_ids)) == B, \
            "sample_ids must be unique; duplicates would make logged predictions ambiguous"

    LOGGER.info(f"Got DATA in shape {data.shape} and LABELS in shape {labels.shape}")
    LOGGER.info(f"Unique labels: {np.unique(labels)}, Counts: {np.bincount(labels)}")
    LOGGER.info(f"Assuming that (#Samples = {B}, Time = {T}, #Features = {D})")

    # detect devices: if anything other than CPU is available, test all models; otherwise use only lite models.
    if device is not None:
        LOGGER.info(f"Using device: {device}")
    else:
        device = "cuda" if torch.cuda.is_available() \
            else "mps" if torch.backends.mps.is_available() \
                else "cpu"

        LOGGER.info(f"Using device: {device}")

    if models is None:
        if device == "cpu":
            models = "lite"
        else:
            models = "all"


    # model discovery and selection routine
    chosen_model_dict = {}

    available_model_dict = _discover_models() # scan ml4fmri.models for model classes
    if models == 'all' or models == builtins.all: # second option for users who forget to put quotes
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
    else:
        raise ValueError(f"{models} (type {type(models)}) is not a valid model specification")
    chosen_model_dict = {m: available_model_dict[m] for m in chosen}

    # handle custom models passed by the user
    custom_models_list = [] if custom_models is None else custom_models if isinstance(custom_models, list) else [custom_models]
    # rough check; it doesn't guarantee that the models will work yet, but it's a start for debugging
    for model_class in custom_models_list:
        assert isinstance(model_class, type), f"Custom model '{model_class}' is not a class"
        assert hasattr(model_class, "prepare_dataloader") and hasattr(model_class, "train_model"), \
            f"Custom model '{model_class}' must have `prepare_dataloader` and `train_model` methods defined;\
                 see ml4fmri.models.LR and ml4fmri.models.meanMLP for examples"
        if model_class.__name__ in chosen_model_dict:
            LOGGER.warning(
                f"Custom model '{model_class.__name__}' has the same name as one of the bundled models; it will replace it"
            )
        chosen_model_dict[model_class.__name__] = model_class
    custom_model_dict = {model_class.__name__: model_class for model_class in custom_models_list}

    final_model_dict = OrderedDict()
    final_model_dict.update(custom_model_dict)
    for k, v in chosen_model_dict.items():
        if k not in final_model_dict:
            final_model_dict[k] = v

    LOGGER.info(
        f"Running models: {list(final_model_dict.keys())}"
    )

    if any(fnc_model in final_model_dict for fnc_model in FNC_MODELS):
        if D*D > 10000:
            LOGGER.warning(f"HIGH DIM FNC: Given input time series with {D} features, the FNC matrices derived for FNC models will have {D*D} elements")

    # Build CV train/val/test data splits
    skf = StratifiedKFold(n_splits=int(n_folds), shuffle=True, random_state=int(cv_seed))
    splits = []
    for fold_idx, (train_full_idx, test_idx) in enumerate(skf.split(data, labels)):
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=val_ratio, random_state=int(val_seed)
        )
        tr_pos, val_pos = next(sss.split(data[train_full_idx], labels[train_full_idx]))
        splits.append({
            "train": train_full_idx[tr_pos],
            "val": train_full_idx[val_pos],
            "test": test_idx,
        })

    # Prepare the results directory
    started = datetime.now()
    out_dir = None
    if save_dir is not False:
        # handle the main directory
        if save_dir is None:
            save_dir = os.path.join(
                os.getcwd(), f"cvbench_{started.strftime('%Y%m%d_%H%M%S')}"
            )
        out_dir = os.fspath(save_dir)
        os.makedirs(out_dir, exist_ok=True)
        LOGGER.info(f"Saving results to: {out_dir}")

        # handle the directories for folds and save records
        records_dir = os.path.join(out_dir, "fold_records")
        os.makedirs(records_dir, exist_ok=True)

        for fold_idx, split in enumerate(splits):
            fold_dir = os.path.join(records_dir, f"fold_{fold_idx:02d}")
            os.makedirs(fold_dir, exist_ok=True)
            with open(os.path.join(fold_dir, "indices.json"), "w") as f:
                json.dump(
                    {k: split[k].tolist() for k in ("train", "val", "test")},
                    f, indent=2,
                )

        if sample_ids_provided:
            pd.DataFrame({"pos": np.arange(B), "sample_id": sample_ids}).to_csv(
                os.path.join(records_dir, "sample_order.csv"), index=False)

        # create CSV for predictions
        predictions_path = os.path.join(out_dir, "cvbench_predictions.csv")
        pred_columns = ["model", "fold", "sample_id", "y_true", "y_pred"] + [f"p_{c}" for c in range(C)]
        pd.DataFrame(columns=pred_columns).to_csv(predictions_path, index=False)

    # Initialize and save meta information about the run
    meta = {
        "status": "running",
        "ml4fmri_version": version("ml4fmri"),
        "created": started.isoformat(timespec="seconds"),
        "finished": None,
        "elapsed_seconds": None,
        "models": list(final_model_dict.keys()),
        "custom_models": list(custom_model_dict.keys()),
        "data": {
            "n_samples": int(B),
            "n_timepoints": int(T),
            "input_size": int(D),
            "n_classes": int(C),
            "class_counts": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
            "sample_ids_provided": bool(sample_ids_provided),
        },
        "cv": {
            "n_folds": int(n_folds),
            "val_ratio": float(val_ratio),
        },
        "seeds": {
            "cv_seed": int(cv_seed),
            "val_seed": int(val_seed),
            "init_seed": int(init_seed),
        },
        "training": {
            "epochs": int(epochs),
            "lr": lr,
            "patience": int(patience),
            "device": str(device),
        },
        "save_checkpoints": bool(save_checkpoints),
    }
    if out_dir is not None:
        with open(os.path.join(out_dir, "cvbench_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    train_logs = []
    test_logs = []
    prediction_logs = []

    for model_name, model_class in final_model_dict.items(): # Model loop
        LOGGER.info(f"Training model: {model_name}")

        for fold_idx, split in enumerate(splits): # CV loop

            start = time.time()
            train_idx, val_idx, test_idx = split["train"], split["val"], split["test"]

            X_train, y_train = data[train_idx], labels[train_idx]
            X_val, y_val = data[val_idx], labels[val_idx]
            X_test, y_test = data[test_idx], labels[test_idx]

            # random seed for dataloader shuffling, model init and things like that
            _seed_everything(int(init_seed) + fold_idx)

            # Prepare dataloaders via model class helper, handle data transforms inside if needed
            # (e.g., FNC derivation from time series, or z-scoring)
            train_loader = model_class.prepare_dataloader(X_train, y_train, shuffle=True)
            val_loader   = model_class.prepare_dataloader(X_val, y_val, shuffle=False)
            test_loader  = model_class.prepare_dataloader(X_test, y_test, shuffle=False)

            # Instantiate model
            model = model_class(input_size=D, output_size=C)

            # Train model and get results: (train_log, test_log, predictions_log),
            # each a DataFrame with "model" already stamped by the model itself.
            train_df, test_df, pred_df = model.train_model(
                train_loader, val_loader, test_loader,
                epochs=epochs, lr=lr, device=device, patience=patience,
            )

            time_elapsed = time.time() - start
            train_df["fold"] = fold_idx
            test_df["fold"] = fold_idx

            train_logs.append(train_df)
            test_logs.append(test_df)

            # handle raw predictions log: check that the order of test samples matches the split, and add fold/sample_id columns
            assert len(pred_df) == len(test_idx), (
                f"{model_name}: got {len(pred_df)} test predictions for {len(test_idx)} "
                "test samples; prepare_dataloader must not add or drop samples"
            )
            assert np.array_equal(pred_df["y_true"].to_numpy(), y_test), (
                f"{model_name}: test labels came back in a different order than the "
                "test split; prepare_dataloader must not shuffle when shuffle=False"
            )
            pred_df["fold"] = fold_idx
            pred_df["sample_id"] = np.asarray(sample_ids[test_idx])
            p_cols = [c for c in pred_df.columns if c.startswith("p_")]
            pred_df = pred_df[["model", "fold", "sample_id", "y_true", "y_pred"] + p_cols]
            prediction_logs.append(pred_df)

            # Save results to disk
            if out_dir is not None:
                if save_checkpoints:
                    ckpt_dir = os.path.join(
                        out_dir, "fold_records", f"fold_{fold_idx:02d}", "checkpoints"
                    )
                    os.makedirs(ckpt_dir, exist_ok=True)
                    _save_checkpoint(model, os.path.join(ckpt_dir, model_name))

                _order_columns(pd.concat(train_logs, ignore_index=True)).to_csv(
                    os.path.join(out_dir, "cvbench_train.csv"), index=False)
                _order_columns(pd.concat(test_logs, ignore_index=True)).to_csv(
                    os.path.join(out_dir, "cvbench_test.csv"), index=False)
                pred_df.to_csv(
                    predictions_path, mode="a", header=False, index=False)

            fold_logger = LOGGER.getChild(f"{model_name}")
            # locate the epoch with minimum validation loss and get its train AUC
            best_idx = train_df['val_loss'].idxmin()
            train_score = train_df.loc[best_idx, 'train_auc']
            val_score = train_df.loc[best_idx, 'val_auc']
            test_score = test_df['test_auc'].iloc[-1]
            fold_logger.info(f"Fold {(fold_idx+1):02d}/{n_folds:02d}: Train/Val/Test AUC {train_score:.3f}/{val_score:.3f}/{test_score:.3f}: Time elapsed {time_elapsed:.2f} s")

    train_df_all = _order_columns(pd.concat(train_logs, ignore_index=True))
    test_df_all = _order_columns(pd.concat(test_logs, ignore_index=True))
    predictions_df_all = pd.concat(prediction_logs, ignore_index=True)

    finished = datetime.now()
    meta["status"] = "completed"
    meta["finished"] = finished.isoformat(timespec="seconds")
    meta["elapsed_seconds"] = round((finished - started).total_seconds(), 1)
    if out_dir is not None:
        with open(os.path.join(out_dir, "cvbench_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    return Report(
        train_df_all, test_df_all, predictions_df_all, meta,
    )


# -----------------------------
# Public Report object
# -----------------------------

class Report(object):
    """
    Results of a `cvbench` run: the logs, the raw test-fold predictions, and the
    plotting helpers.
    """

    def __init__(self, train_df, test_df, predictions_df, meta):
        self.train_df = pd.DataFrame(train_df).copy()
        self.test_df = pd.DataFrame(test_df).copy()
        self.predictions_df = pd.DataFrame(predictions_df).copy()
        self.meta = dict(meta)

    def get_train_dataframe(self):
        return self.train_df.copy()

    def get_test_dataframe(self):
        return self.test_df.copy()

    def get_predictions_dataframe(self):
        return self.predictions_df.copy()

    def get_meta(self):
        return self.meta.copy()
    
    def plot_confusion(self, normalize="true", show=True, max_cols=3):
        """
        Confusion matrix heatmap per model, pooled across folds.

        Parameters
        ----------
        normalize : "true" (row-normalised, recall on the diagonal), 
                    "pred" (column-normalised, precision), 
                    or None for raw counts. 
                    Cells are always annotated with raw counts.
        show : call plt.show() instead of returning the figure.
        max_cols : maximum panels per row.
        """
        # sum the per-fold cm_true{i}_pred{j} counts already in the test log. Summing
        # is exact, not an average: every sample is tested exactly once across folds.
        cells = {c: tuple(int(v) for v in c.removeprefix("test_cm_true").split("_pred"))
                 for c in self.test_df.columns if c.startswith("test_cm_true")}
        if not cells:
            raise KeyError("No test_cm_true*_pred* columns in this report's test log")
        n_classes = 1 + max(i for i, _ in cells.values())

        matrices = {}
        for model, g in self.test_df.groupby("model"):
            cm = np.zeros((n_classes, n_classes), dtype=int)
            for col, (i, j) in cells.items():
                cm[i, j] = g[col].sum()
            matrices[model] = cm

        # best first, matching plot_scores' ordering where a metric is available
        if "test_auc" in self.test_df.columns:
            order = (self.test_df.groupby("model")["test_auc"].median()
                     .sort_values(ascending=False).index)
            matrices = {m: matrices[m] for m in order if m in matrices}

        n = len(matrices)
        n_cols = min(max_cols, n)
        n_rows = int(np.ceil(n / n_cols))
        fig, axs = plt.subplots(n_rows, n_cols,
                                figsize=(3.6 * n_cols, 3.9 * n_rows), squeeze=False)

        for ax, (model, cm) in zip(axs.ravel(), matrices.items()):
            counts = cm
            if normalize == "true":
                denom = cm.sum(axis=1, keepdims=True)
            elif normalize == "pred":
                denom = cm.sum(axis=0, keepdims=True)
            elif normalize is None:
                denom = None
            else:
                raise ValueError(f"normalize must be 'true', 'pred' or None; got {normalize!r}")
            shown = counts / np.maximum(denom, 1) if denom is not None else counts

            im = ax.imshow(shown, cmap="Blues",
                           vmin=0, vmax=1 if denom is not None else shown.max())
            n_classes = cm.shape[0]
            ax.set_xticks(range(n_classes)); ax.set_yticks(range(n_classes))
            ax.set_xlabel("predicted"); ax.set_ylabel("true")
            acc = np.trace(cm) / max(cm.sum(), 1)
            ax.set_title(f"{model} (acc {acc:.2f})")

            # annotate with raw counts, flipping text colour on dark cells
            hi = shown.max() if shown.size else 1
            for i in range(n_classes):
                for j in range(n_classes):
                    ax.text(j, i, f"{counts[i, j]:d}", ha="center", va="center",
                            fontsize=9,
                            color="white" if shown[i, j] > 0.6 * hi else "black")
            fig.colorbar(im, ax=ax, fraction=0.046)

        for ax in axs.ravel()[n:]:
            ax.axis("off")

        fig.tight_layout()
        if show:
            plt.show()
            return None
        return fig

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
            all_vals = np.concatenate(data)
            if np.nanmin(all_vals) <= 0.5 <= np.nanmax(all_vals):
                ax.axhline(y=0.5, color='lightblue', linestyle='dashed', linewidth=1)

        bp = ax.boxplot(
            data, 
            tick_labels=order, 
            showfliers=show_outliers, 
            patch_artist=True,
            boxprops=dict(facecolor='white', edgecolor='black')
        )
        for patch, model in zip(bp['boxes'], order):
            if model in FNC_MODELS:
                patch.set_hatch('//')

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
        """

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

        bp = ax.boxplot(
            data,
            vert=False,
            positions=positions,
            showfliers=show_outliers,
            manage_ticks=False,   # we'll manage ticks/labels ourselves
            patch_artist=True,
            boxprops=dict(facecolor='white', edgecolor='black'),
        )
        for patch, model in zip(bp['boxes'], order):
            if model in FNC_MODELS:
                patch.set_hatch('//')

        # draw blue dashed line at x = 0.5 if plotting an AUC metric
        if 'auc' in metric:
            all_vals = np.concatenate(data)
            if np.nanmin(all_vals) <= 0.5 <= np.nanmax(all_vals):
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
        """

        train_color, train_style = "blue", "-"
        val_color,   val_style   = "orange", "--"

        df = self.train_df.copy()
        if "model" not in df or "fold" not in df or "epoch" not in df:
            raise KeyError("train_df must contain 'model', 'fold', and 'epoch' columns.")

        models = list(df["model"].unique())
        models = [m for m in models if m != 'LR'] # LR training logs aren't really logs

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
                            ax.plot(g["epoch"], g[train_key],
                                color=train_color, linestyle=train_style,
                                alpha=0.4, linewidth=1.2)
                            ax.plot(g["epoch"], g[val_key],
                                color=val_color,   linestyle=val_style,
                                alpha=0.4, linewidth=1.2)
        
                    ax.set_title(f"{model} – {name}")
                    ax.set_ylabel(name)
                    ax.grid(True, linestyle=":", linewidth=0.5)
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                    ax.set_xlabel("epoch")

                # then update the legend handles accordingly:
                handles = [
                    Line2D([0], [0],
                    color=train_color, linestyle=train_style,
                    linewidth=1.5, label="train"),
                    Line2D([0], [0],
                    color=val_color, linestyle=val_style,
                    linewidth=1.5, label="val"),
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
