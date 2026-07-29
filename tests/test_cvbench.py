import json
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from ml4fmri.cvreport import cvbench, Report
from ml4fmri.models import LR


class MyLR(LR):
    """Module-level so that it is importable, and therefore picklable."""


def test_cvbench_smoke(toy_data, tmp_path):
    X, y = toy_data
    # Use your discovery inside cvbench; 'lite' defaults to a small set (e.g., meanMLP)
    rep = cvbench(X, y, models="lite", n_folds=2, val_ratio=0.2,
                  cv_seed=0, val_seed=0, init_seed=0, save_dir=tmp_path)
    tdf = rep.get_test_dataframe()
    trn = rep.get_train_dataframe()
    assert isinstance(tdf, pd.DataFrame) and not tdf.empty
    assert isinstance(trn, pd.DataFrame) and "epoch" in trn.columns

    # Plotting shouldn't auto-display when show=False
    out = rep.plot_scores(show=False);      assert out is not None
    figs = rep.plot_training_curves(show=False);  assert isinstance(figs, tuple) and len(figs) == 2


def test_fold_is_second_column(toy_data, tmp_path):
    X, y = toy_data
    rep = cvbench(X, y, models="LR", n_folds=2, save_dir=tmp_path)
    for df in (rep.get_train_dataframe(), rep.get_test_dataframe()):
        assert list(df.columns[:2]) == ["model", "fold"]


def test_no_disk_writes_when_save_dir_false(toy_data, tmp_path):
    X, y = toy_data
    cwd = os.getcwd()
    os.chdir(tmp_path)  # so a stray timestamped dir would land here and be caught
    try:
        rep = cvbench(X, y, models="LR", n_folds=2, save_dir=False)
    finally:
        os.chdir(cwd)
    assert list(tmp_path.iterdir()) == []
    assert not rep.get_predictions_dataframe().empty  # still collected in memory


def test_results_directory_layout(toy_data, tmp_path):
    X, y = toy_data
    n_folds = 2
    cvbench(X, y, models=["LR", "meanMLP"], n_folds=n_folds, epochs=2, patience=2,
            save_dir=tmp_path)

    for name in ("cvbench_meta.json", "cvbench_train.csv",
                 "cvbench_test.csv", "cvbench_predictions.csv"):
        assert (tmp_path / name).exists(), f"missing {name}"
    # samples.csv is conditional on sample_ids= being passed, which it was not here
    assert not (tmp_path / "fold_records/sample_order.csv").exists()

    for fold in range(n_folds):
        fold_dir = tmp_path / "fold_records" / f"fold_{fold:02d}"
        assert (fold_dir / "indices.json").exists()
        assert (fold_dir / "checkpoints" / "meanMLP.pt").exists()
        assert (fold_dir / "checkpoints" / "LR.joblib").exists()


def test_fold_indices_are_disjoint_and_complete(toy_data, tmp_path):
    X, y = toy_data
    n_folds = 3
    cvbench(X, y, models="LR", n_folds=n_folds, save_dir=tmp_path)

    seen_test = []
    for fold in range(n_folds):
        with open(tmp_path / "fold_records" / f"fold_{fold:02d}" / "indices.json") as f:
            idx = json.load(f)
        train, val, test = set(idx["train"]), set(idx["val"]), set(idx["test"])
        assert not (train & val) and not (train & test) and not (val & test)
        assert train | val | test == set(range(len(y)))
        seen_test.extend(idx["test"])

    # every sample is tested exactly once across folds
    assert sorted(seen_test) == list(range(len(y)))


def test_checkpoints_can_be_disabled(toy_data, tmp_path):
    X, y = toy_data
    cvbench(X, y, models="LR", n_folds=2, save_dir=tmp_path, save_checkpoints=False)
    assert not (tmp_path / "fold_records" / "fold_00" / "checkpoints").exists()


def test_predictions_reproduce_logged_auc(toy_data, tmp_path):
    """The whole point of logging raw probabilities: metrics must be rederivable."""
    X, y = toy_data
    rep = cvbench(X, y, models=["LR", "meanMLP"], n_folds=2, epochs=3, patience=3,
                  save_dir=tmp_path)

    preds = rep.get_predictions_dataframe()
    test_df = rep.get_test_dataframe()

    for (model, fold), group in preds.groupby(["model", "fold"]):
        rederived = roc_auc_score(group["y_true"], group["p_1"])
        logged = test_df.loc[
            (test_df["model"] == model) & (test_df["fold"] == fold), "test_auc"
        ].iloc[0]
        assert rederived == pytest.approx(logged, abs=1e-9), f"{model} fold {fold}"


def test_each_sample_predicted_once_per_model(toy_data, tmp_path):
    X, y = toy_data
    rep = cvbench(X, y, models=["LR", "meanMLP"], n_folds=3, epochs=2, patience=2,
                  save_dir=tmp_path)
    preds = rep.get_predictions_dataframe()
    for model, group in preds.groupby("model"):
        assert sorted(group["sample_id"]) == list(range(len(y)))


def test_predictions_carry_true_labels(toy_data, tmp_path):
    """sample_id -> y_true in the predictions must match the input labels."""
    X, y = toy_data
    rep = cvbench(X, y, models="LR", n_folds=2, save_dir=tmp_path)
    preds = rep.get_predictions_dataframe()
    assert np.array_equal(y[preds["sample_id"].to_numpy()], preds["y_true"].to_numpy())


def test_string_sample_ids(toy_data, tmp_path):
    X, y = toy_data
    ids = np.array([f"sub-{i:03d}" for i in range(len(y))])
    rep = cvbench(X, y, models="LR", n_folds=2, sample_ids=ids, save_dir=tmp_path)

    preds = rep.get_predictions_dataframe()
    assert set(preds["sample_id"]) == set(ids)

    samples = pd.read_csv(tmp_path / "fold_records/sample_order.csv")
    assert list(samples.columns) == ["pos", "sample_id"]
    assert samples["sample_id"].tolist() == list(ids)


def test_duplicate_sample_ids_rejected(toy_data):
    X, y = toy_data
    ids = np.zeros(len(y), dtype=int)
    with pytest.raises(AssertionError, match="unique"):
        cvbench(X, y, models="LR", n_folds=2, sample_ids=ids, save_dir=False)


def test_wrong_length_sample_ids_rejected(toy_data):
    X, y = toy_data
    with pytest.raises(AssertionError, match="sample_ids"):
        cvbench(X, y, models="LR", n_folds=2, sample_ids=np.arange(3), save_dir=False)


def test_confusion_counts_are_consistent(toy_data, tmp_path):
    """Binary runs use the same cm_true{i}_pred{j} names as multiclass ones."""
    X, y = toy_data
    rep = cvbench(X, y, models="LR", n_folds=2, save_dir=tmp_path)
    test_df = rep.get_test_dataframe()

    for col in ("test_cm_true0_pred0", "test_cm_true0_pred1",
                "test_cm_true1_pred0", "test_cm_true1_pred1"):
        assert col in test_df.columns
    # the old binary names must be gone entirely
    assert not {"test_tn", "test_fp", "test_fn", "test_tp"} & set(test_df.columns)

    # counts must sum to the fold size and agree with the logged accuracy
    for _, row in test_df.iterrows():
        total = sum(row[f"test_cm_true{i}_pred{j}"] for i in range(2) for j in range(2))
        correct = row["test_cm_true0_pred0"] + row["test_cm_true1_pred1"]
        assert correct / total == pytest.approx(row["test_accuracy"])

    train_df = rep.get_train_dataframe()
    assert "train_cm_true1_pred1" in train_df.columns
    assert "val_cm_true1_pred1" in train_df.columns


def test_confusion_matches_predictions(toy_data, tmp_path):
    """cm_true1_pred1 is TP and cm_true0_pred0 is TN, per the documented mapping."""
    X, y = toy_data
    rep = cvbench(X, y, models="meanMLP", n_folds=2, epochs=3, patience=3, save_dir=tmp_path)
    preds = rep.get_predictions_dataframe()
    test_df = rep.get_test_dataframe().set_index("fold")

    for fold, group in preds.groupby("fold"):
        tp = int(((group["y_true"] == 1) & (group["y_pred"] == 1)).sum())
        tn = int(((group["y_true"] == 0) & (group["y_pred"] == 0)).sum())
        assert tp == test_df.loc[fold, "test_cm_true1_pred1"]
        assert tn == test_df.loc[fold, "test_cm_true0_pred0"]


def test_multiclass_confusion_columns(toy_data_multiclass, tmp_path):
    """Multiclass gets a full CxC grid of cm_true{i}_pred{j}, not binary names."""
    X, y = toy_data_multiclass
    C = len(np.unique(y))
    rep = cvbench(X, y, models="LR", n_folds=3, save_dir=tmp_path)
    test_df = rep.get_test_dataframe()

    cm_cols = [c for c in test_df.columns if c.startswith("test_cm_")]
    assert len(cm_cols) == C * C
    assert "test_cm_true0_pred1" in cm_cols
    # train log carries the same grid for both train and val
    train_cols = [c for c in rep.get_train_dataframe().columns if "cm_true" in c]
    assert len(train_cols) == 2 * C * C


def test_multiclass_confusion_matches_predictions(toy_data_multiclass, tmp_path):
    """Counts must be oriented [true, predicted] and agree with the raw predictions."""
    from sklearn.metrics import confusion_matrix

    X, y = toy_data_multiclass
    C = len(np.unique(y))
    rep = cvbench(X, y, models="LR", n_folds=3, save_dir=tmp_path)
    preds = rep.get_predictions_dataframe()
    test_df = rep.get_test_dataframe()

    for fold, g in preds.groupby("fold"):
        row = test_df[test_df["fold"] == fold].iloc[0]
        truth = confusion_matrix(g["y_true"], g["y_pred"], labels=list(range(C)))
        for i in range(C):
            for j in range(C):
                assert row[f"test_cm_true{i}_pred{j}"] == truth[i, j], (fold, i, j)


def test_pooled_confusion_covers_the_dataset(toy_data_multiclass, tmp_path):
    """Summed over folds, the annotated counts must add up to every sample, once."""
    X, y = toy_data_multiclass
    rep = cvbench(X, y, models="LR", n_folds=3, save_dir=tmp_path)
    fig = rep.plot_confusion(normalize=None, show=False)
    assert sum(float(t.get_text()) for t in fig.axes[0].texts) == len(y)


def test_plot_confusion_needs_cm_columns(toy_data_multiclass, tmp_path):
    X, y = toy_data_multiclass
    rep = cvbench(X, y, models="LR", n_folds=3, save_dir=tmp_path)
    bare = rep.get_test_dataframe().drop(
        columns=[c for c in rep.get_test_dataframe().columns if c.startswith("test_cm_")])
    with pytest.raises(KeyError, match="No test_cm_true"):
        Report(rep.get_train_dataframe(), bare, rep.get_predictions_dataframe(),
               rep.get_meta()).plot_confusion(show=False)


def test_plot_confusion(toy_data_multiclass, tmp_path):
    X, y = toy_data_multiclass
    rep = cvbench(X, y, models=["LR", "meanMLP"], n_folds=3, epochs=3, patience=3,
                  save_dir=tmp_path)

    for normalize in ("true", "pred", None):
        fig = rep.plot_confusion(normalize=normalize, show=False)
        assert fig is not None

    with pytest.raises(ValueError):
        rep.plot_confusion(normalize="nonsense", show=False)


def test_plot_confusion_binary(toy_data, tmp_path):
    X, y = toy_data
    rep = cvbench(X, y, models="LR", n_folds=2, save_dir=tmp_path)
    fig = rep.plot_confusion(show=False)
    assert fig is not None and len(fig.axes[0].texts) == 4   # a 2x2 grid


def test_samples_csv_has_no_label_column(toy_data, tmp_path):
    X, y = toy_data
    ids = np.array([f"sub-{i:03d}" for i in range(len(y))])
    cvbench(X, y, models="LR", n_folds=2, sample_ids=ids, save_dir=tmp_path)
    assert list(pd.read_csv(tmp_path / "fold_records/sample_order.csv").columns) == ["pos", "sample_id"]


def test_train_model_returns_three_dataframes(toy_data, dims):
    """train_model returns (train_log, test_log, predictions_log), each a DataFrame,
    with "model" already stamped by the model itself."""
    X, y = toy_data
    D, C = dims
    model = LR(input_size=D, output_size=C)
    loaders = [LR.prepare_dataloader(X, y) for _ in range(3)]
    train_log, test_log, predictions_log = model.train_model(*loaders)

    assert isinstance(train_log, pd.DataFrame) and "model" in train_log.columns
    assert isinstance(test_log, pd.DataFrame) and len(test_log) == 1
    assert isinstance(predictions_log, pd.DataFrame)
    assert {"model", "y_true", "y_pred"} <= set(predictions_log.columns)


def test_subclass_stamps_its_own_name(toy_data, dims):
    """A bundled model must report its own class name, not its parent's -- cvbench
    no longer patches this up after the fact."""
    X, y = toy_data
    D, C = dims
    model = MyLR(input_size=D, output_size=C)
    loaders = [MyLR.prepare_dataloader(X, y) for _ in range(3)]
    train_log, test_log, predictions_log = model.train_model(*loaders)

    assert set(train_log["model"]) == {"MyLR"}
    assert set(test_log["model"]) == {"MyLR"}
    assert set(predictions_log["model"]) == {"MyLR"}


def test_one_test_row_per_model_fold(toy_data, tmp_path):
    X, y = toy_data
    rep = cvbench(X, y, models="LR", n_folds=4, save_dir=tmp_path)
    test_df = rep.get_test_dataframe()
    assert len(test_df) == 4
    assert test_df["fold"].tolist() == [0, 1, 2, 3]


def test_folds_get_independent_inits(toy_data):
    """Each fold seeds with init_seed + fold_idx, so inits are independent draws."""
    X, y = toy_data
    rep = cvbench(X, y, models="meanMLP", n_folds=3, epochs=1, patience=1,
                  save_dir=False)
    # epoch-0 losses differ across folds only if the starting weights differ
    first_epochs = rep.get_train_dataframe().query("epoch == 0")["train_loss"]
    assert first_epochs.nunique() == len(first_epochs)


def test_same_init_seed_reproduces_results(toy_data, tmp_path):
    X, y = toy_data
    kwargs = dict(models="meanMLP", n_folds=2, epochs=5, patience=5, save_dir=False)
    a = cvbench(X, y, init_seed=7, **kwargs).get_test_dataframe()
    b = cvbench(X, y, init_seed=7, **kwargs).get_test_dataframe()
    assert np.allclose(a["test_auc"], b["test_auc"])


def test_different_init_seed_changes_results(toy_data):
    X, y = toy_data
    kwargs = dict(models="meanMLP", n_folds=2, epochs=5, patience=5, save_dir=False)
    a = cvbench(X, y, init_seed=7, **kwargs).get_train_dataframe()
    b = cvbench(X, y, init_seed=99, **kwargs).get_train_dataframe()
    # compare losses rather than AUC: AUC can saturate on the toy data and match by
    # coincidence. Splits are untouched, so any difference is initialisation alone.
    assert not np.allclose(a["val_loss"], b["val_loss"])


def test_init_seed_does_not_move_the_splits(toy_data, tmp_path):
    X, y = toy_data
    dirs = []
    for seed in (7, 99):
        d = tmp_path / f"seed_{seed}"
        cvbench(X, y, models="LR", n_folds=2, init_seed=seed, save_dir=d)
        dirs.append(d)

    for fold in range(2):
        loaded = [
            json.load(open(d / "fold_records" / f"fold_{fold:02d}" / "indices.json"))
            for d in dirs
        ]
        assert loaded[0] == loaded[1]


def test_cv_seed_moves_the_splits(toy_data, tmp_path):
    X, y = toy_data
    dirs = []
    for seed in (0, 1):
        d = tmp_path / f"cv_{seed}"
        cvbench(X, y, models="LR", n_folds=2, cv_seed=seed, save_dir=d)
        dirs.append(d)

    loaded = [
        json.load(open(d / "fold_records" / "fold_00" / "indices.json"))
        for d in dirs
    ]
    assert loaded[0]["test"] != loaded[1]["test"]


def test_random_state_is_rejected(toy_data):
    """The old single-seed kwarg must fail, not be silently ignored."""
    X, y = toy_data
    with pytest.raises(TypeError, match="random_state"):
        cvbench(X, y, models="LR", n_folds=2, random_state=3, save_dir=False)


def test_meta_is_readable_and_has_no_indices(toy_data, tmp_path):
    X, y = toy_data
    cvbench(X, y, models="LR", n_folds=2, save_dir=tmp_path)
    raw = (tmp_path / "cvbench_meta.json").read_text()
    assert "\n" in raw and "  " in raw, "meta.json should be pretty-printed"

    meta = json.loads(raw)
    assert "train_indices" not in meta and "test_indices" not in meta
    assert meta["status"] == "completed"
    assert meta["seeds"] == {"cv_seed": 42, "val_seed": 42, "init_seed": 42}
    assert meta["data"]["n_classes"] == 2
    assert meta["elapsed_seconds"] is not None


def test_meta_lists_custom_models(toy_data, tmp_path):
    """Regression: meta used to be built from `chosen`, silently dropping custom models."""
    X, y = toy_data
    rep = cvbench(X, y, models="LR", custom_models=MyLR, n_folds=2, save_dir=tmp_path)
    assert "MyLR" in rep.get_meta()["models"]
    assert rep.get_meta()["custom_models"] == ["MyLR"]
    assert set(rep.get_test_dataframe()["model"]) == {"LR", "MyLR"}
    # a module-level custom model is picklable, so it checkpoints like a bundled one
    assert (tmp_path / "fold_records" / "fold_00" / "checkpoints" / "MyLR.joblib").exists()


def test_unpicklable_model_warns_but_run_continues(toy_data, tmp_path):
    """
    A non-torch model defined inside a function can't be pickled by name. That must
    degrade to a warning, not abort the run -- the logs are the point, the weights
    are a convenience. (Torch models are unaffected: we save a state_dict, not the class.)
    """
    from ml4fmri.models import LR as _LR

    class LocalLR(_LR):  # function-local -> not importable -> unpicklable
        pass

    X, y = toy_data
    with pytest.warns(RuntimeWarning, match="Could not save checkpoint"):
        rep = cvbench(X, y, models=[], custom_models=LocalLR, n_folds=2, save_dir=tmp_path)

    assert not rep.get_test_dataframe().empty          # results still produced
    assert not rep.get_predictions_dataframe().empty   # predictions still logged
    assert (tmp_path / "cvbench_test.csv").exists()


def test_saved_csvs_match_in_memory(toy_data, tmp_path):
    """What lands on disk must equal what the Report holds."""
    X, y = toy_data
    rep = cvbench(X, y, models="LR", n_folds=2, save_dir=tmp_path)

    for name, frame in (("cvbench_test.csv", rep.get_test_dataframe()),
                        ("cvbench_predictions.csv", rep.get_predictions_dataframe())):
        pd.testing.assert_frame_equal(pd.read_csv(tmp_path / name), frame,
                                      check_dtype=False)


def test_reordering_dataloader_is_caught(toy_data):
    """A custom model that reorders test samples must fail loudly, not misalign IDs."""
    from ml4fmri.models import LR as _LR

    class ShufflingLR(_LR):
        @staticmethod
        def prepare_dataloader(data, labels, batch_size=64, shuffle=True):
            fnc, labels = _LR.prepare_dataloader(data, labels, batch_size, shuffle)
            order = np.arange(len(labels))[::-1]
            return fnc[order], labels[order]

    X, y = toy_data
    with pytest.raises(AssertionError, match="different order"):
        cvbench(X, y, models=[], custom_models=ShufflingLR, n_folds=2, save_dir=False)


def test_dropping_dataloader_is_caught(toy_data):
    from ml4fmri.models import LR as _LR

    class DroppingLR(_LR):
        @staticmethod
        def prepare_dataloader(data, labels, batch_size=64, shuffle=True):
            fnc, labels = _LR.prepare_dataloader(data, labels, batch_size, shuffle)
            return fnc[:-1], labels[:-1]

    X, y = toy_data
    with pytest.raises(AssertionError, match="add or drop"):
        cvbench(X, y, models=[], custom_models=DroppingLR, n_folds=2, save_dir=False)
