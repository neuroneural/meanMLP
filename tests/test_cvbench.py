import pandas as pd
from ml4fmri.cvreport import cvbench

def test_cvbench_smoke(toy_data):
    X, y = toy_data
    # Use your discovery inside cvbench; 'lite' defaults to a small set (e.g., meanMLP)
    rep = cvbench(X, y, models="lite", n_folds=2, val_ratio=0.2, random_state=0)
    tdf = rep.get_test_dataframe()
    trn = rep.get_train_dataframe()
    assert isinstance(tdf, pd.DataFrame) and not tdf.empty
    assert isinstance(trn, pd.DataFrame) and "epoch" in trn.columns

    # Plotting shouldn't auto-display when show=False
    out = rep.plot_scores(show=False);      assert out is not None
    figs = rep.plot_training_curves(show=False);  assert isinstance(figs, tuple) and len(figs) == 2
