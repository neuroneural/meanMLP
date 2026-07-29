![cvbench output example](https://github.com/neuroneural/meanMLP/blob/ml4fmri/assets/cvbench_example.png?raw=true)
# ml4fmri

A **one-line Python toolkit** for fMRI classification that allows you to benchmark multiple deep learning models for fMRI analysis on your data with a single function call. While designed for fMRI time series, it can work with any temporal classification task.

Originally based on the codebase behind the NeuroImage paper ["A simple but tough-to-beat baseline for fMRI time-series classification"](https://doi.org/10.1016/j.neuroimage.2024.120909).
This work was funded by the National Science Foundation grant 2112455, NIH grant R01MH123610, and in part by NIH grant R01MH129047.
## Use example

You can install the package directly from PyPI:

```bash
pip install ml4fmri
```

Check out the 👉 [**Colab tutorial**](https://colab.research.google.com/drive/1-WtiB3ne4dkiOg8lt7MyzDYADMnE1lTv?usp=sharing) for a more detailed guide and ways to modify the default behavior.


```python
# In Python, get fMRI time series DATA in shape (SAMPLES, TIME, FEATURES)
# and LABELS in shape (SAMPLES) (binary or multiclass)

from ml4fmri import cvbench  # runs CV experiments with implemented models on the given data

# Run cross-validation with all available models. See below for more info on available `models`
# Results are written to `save_dir` as the run proceeds; omit it to get a timestamped ./cvbench_YYYYmmdd_HHMMSS/ directory or set to False to keep everything in memory.
report = cvbench(DATA, LABELS, models='all', n_folds=5, save_dir='my_cvbench_run')

# Plot test AUC boxplots for all models
report.plot_scores()

# Access logs directly as variables
train_df = report.get_train_dataframe()
test_df  = report.get_test_dataframe()
pred_df  = report.get_predictions_dataframe()  # raw test-fold probabilities
meta     = report.get_meta()

# Inspect training curves
report.plot_training_curves()

# Confusion matrices per model, pooled across folds — shows *how* a model is wrong,
# which matters most for multiclass runs where one AUC hides the error structure
report.plot_confusion()
```

## Results on disk

Everything is written as the run proceeds.

```
my_cvbench_run/
├── cvbench_meta.json            # run configuration, seeds, environment, status
├── cvbench_samples.csv          # pos,sample_id
├── cvbench_train.csv            # model,fold,epoch,... per-epoch training log
├── cvbench_test.csv             # model,fold,... one row per (model, fold)
├── cvbench_predictions.csv      # model,fold,sample_id,y_true,y_pred,p_0,...,p_{C-1}
└── folds/
    ├── fold_00/
    │   ├── indices.json         # positional train/val/test indices
    │   └── checkpoints/         # best-validation weights per model
    │       ├── meanMLP.pt
    │       └── LR.joblib
    └── fold_01/ ...
```

- **`cvbench_predictions.csv`** – the raw probability for every test sample under every model, so any probability- or threshold-based metric can be recomputed afterwards.
- **`cvbench_test.csv`** – final test metrics per model and fold, plus confusion counts, training time and parameter count. Confusion counts are named `cm_true{i}_pred{j}` for any number of classes.
- **`cvbench_train.csv`** – train and validation metrics at every epoch; this is what `plot_training_curves()` draws.
- **`cvbench_meta.json`** – the run's configuration, seeds and timing, plus a `status` field recording whether it finished.
- **`cvbench_samples.csv`** – generated if you pass `sample_ids=` to `cvbench`; it maps row positions to your `sample_ids`, which is what links `indices.json` (positions) to the predictions (ids).

Set `save_checkpoints=False` to skip storing weights, or `save_dir=False` to keep results in memory only.


## Available Models
You can set `models` input in `cvbench` to:
- **'all'** - (default for non-CPU) run all models,
- **'lite'** (default for CPU) – use only faster models, better for quick tests  
- **'ts'** – run only time series models  
- **'fnc'** – run only FNC models; FNC data is derived from input time series  
- **'<model_name>'** – run only the specified model (e.g. 'meanMLP'); see below for full model list  
- **['<model_name_1>', '<model_name_2>']** – run only the listed models  

### Time Series Models

- **"meanMLP"** (Time Series)   
  A simple MLP model for time series classification, surprisingly good for fMRi time series.
  [Paper](https://doi.org/10.1016/j.neuroimage.2024.120909)  
  [Code and bib item](https://github.com/neuroneural/meanMLP/blob/ml4fmri/src/ml4fmri/models/meanMLP.py)

- **"LSTM"** (Time Series)  
  Standard LSTM recurrent network for sequence classification.  
  [Paper](https://doi.org/10.1162/neco.1997.9.8.1735)  
  [Code and bib item](https://github.com/neuroneural/meanMLP/blob/ml4fmri/src/ml4fmri/models/LSTM.py)

- **"meanLSTM"** (Time Series)  
  A variant of LSTM where outputs are mean-aggregated across time.  
  [Paper](https://doi.org/10.1016/j.neuroimage.2024.120909)  
  [Code and bib item](https://github.com/neuroneural/meanMLP/blob/ml4fmri/src/ml4fmri/models/meanLSTM.py)

- **"Transformer"** (Time Series)  
  Vanilla transformer encoder for modeling temporal dependencies in fMRI time series.  
  [Paper](https://doi.org/10.48550/arXiv.1706.03762)  
  [Code and bib item](https://github.com/neuroneural/meanMLP/blob/ml4fmri/src/ml4fmri/models/Transformer.py)

- **"meanTransformer"** (Time Series)  
  Transformer with temporal mean pooling for classification.  
  [Paper](https://doi.org/10.1016/j.neuroimage.2024.120909)  
  [Code and bib item](https://github.com/neuroneural/meanMLP/blob/ml4fmri/src/ml4fmri/models/meanTransformer.py)

- **"MILC"** (Time Series)  
  CNN+LSTM model for fMRI time series.  
  [Paper](https://doi.org/10.48550/arXiv.2007.16041)  
  [Code and bib item](https://github.com/neuroneural/meanMLP/blob/ml4fmri/src/ml4fmri/models/MILC.py)

- **"BolT"** (Time Series)
  Fused window Transformer for fMRI time series; slow but good, outperformed meanMLP on larger datasets.
  [Paper](https://doi.org/10.1016/j.media.2023.102841)
  [Code and bib item](https://github.com/neuroneural/meanMLP/blob/ml4fmri/src/ml4fmri/models/BolT.py)

- **"DICE"** (Time Series)
  LSTM-based connectivity estimator and classifier, works with time series.
  [Paper](https://doi.org/10.1016/j.neuroimage.2022.119737)
  [Code and bib item](https://github.com/neuroneural/meanMLP/blob/ml4fmri/src/ml4fmri/models/DICE.py)

- **"Glacier"** (Time Series)
  Transformer-based connectivity estimator and classifier, works with time series.
  [Paper](https://doi.org/10.1109/ICASSP49357.2023.10097126)
  [Code and bib item](https://github.com/neuroneural/meanMLP/blob/ml4fmri/src/ml4fmri/models/Glacier.py)

### FNC Models

- **"LR"** (FNC)  
  Logistic regression, wired for FNC classification.  
  [Paper](https://doi.org/10.1080/01621459.1944.10500699)  
  [Code and bib item](https://github.com/neuroneural/meanMLP/blob/ml4fmri/src/ml4fmri/models/LR.py)

- **"FBNetGen"** (Time Series + FNC)
  Estimates connectivity from time series, fuses it with FNC to do classification.
  [Paper](https://openreview.net/forum?id=oWFphg2IKon)
  [Code and bib item](https://github.com/neuroneural/meanMLP/blob/ml4fmri/src/ml4fmri/models/FBNetGen.py)

- **"BNT"** (FNC)
  Transformer/GNN model for FNC classification.
  [Paper](https://openreview.net/forum?id=1cJ1cbA6NLN)
  [Code and bib item](https://github.com/neuroneural/meanMLP/blob/ml4fmri/src/ml4fmri/models/BNT.py)

- **"BrainNetCNN"** (FNC)
  CNN classifier, designed for FNC-like matrices in mind.
  [Paper](https://doi.org/10.1016/j.neuroimage.2016.09.046)
  [Code and bib item](https://github.com/neuroneural/meanMLP/blob/ml4fmri/src/ml4fmri/models/BrainNetCNN.py)
