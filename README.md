![cvbench output example](https://github.com/neuroneural/meanMLP/blob/ml4fmri/assets/cvbench_example.png?raw=true)
# ml4fmri

A **one-line Python toolkit** for fMRI classification that allows you to benchmark multiple deep learning models for fMRI analysis on your data with a single function call. While designed for fMRI time series, it can work with any temporal classification task.

Originally based on the codebase behind the NeuroImage paper:  
["A simple but tough-to-beat baseline for fMRI time-series classification"](https://doi.org/10.1016/j.neuroimage.2024.120909)



## Use example

You can install the package directly from PyPI:

```bash
pip install ml4fmri
```


For a more detailed guide, check out the **Colab tutorial**:

👉 [Open in Google Colab](https://colab.research.google.com/drive/1JIaHvSxDTzLAqnMEXa_0l1wZzhJaAHmi?usp=sharing)


```python
# In Python, get fMRI time series data in shape (samples, time, n_features)
# and labels in shape (samples) (binary or multiclass)

from ml4fmri import cvbench  # runs CV experiments with implemented models on the given data

# Run cross-validation with all available models. See below for more info on available `models`
report = cvbench(data, labels, models='all', n_folds=5)

# Plot test AUC boxplots for all models
report.plot_scores()

# Save test and training dataframes as .csv, and metadata as .json
report.save()

# Access logs directly as variables
train_df = report.get_train_dataframe()
test_df  = report.get_test_dataframe()
meta     = report.get_meta()

# Inspect training curves
report.plot_training_curves()
```


## Available Models
You can set `models` input in `cvbench` to:
- **'all'** - run all models,
- **'lite'** (default) – use only faster models, better for quick tests  
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

- **"FBNetGen"** (Time Series + FNC)
  Connectivity estimator, fuses it with FNC to do classification.
  [Paper](https://openreview.net/forum?id=oWFphg2IKon)
  [Code and bib item](https://github.com/neuroneural/meanMLP/blob/ml4fmri/src/ml4fmri/models/FBNetGen.py)

### FNC Models

- **"LR"** (FNC)  
  Logistic regression, wired for FNC classification.  
  [Paper](https://doi.org/10.1080/01621459.1944.10500699)  
  [Code and bib item](https://github.com/neuroneural/meanMLP/blob/ml4fmri/src/ml4fmri/models/LR.py)

- **"BNT"** (FNC)
  Transformer/GNN model for FNC classification.
  [Paper](https://openreview.net/forum?id=1cJ1cbA6NLN)
  [Code and bib item](https://github.com/neuroneural/meanMLP/blob/ml4fmri/src/ml4fmri/models/BNT.py)

- **"BrainNetCNN"** (FNC)
  CNN classifier, designed for FNC-like matrices in mind.
  [Paper](https://doi.org/10.1016/j.neuroimage.2016.09.046)
  [Code and bib item](https://github.com/neuroneural/meanMLP/blob/ml4fmri/src/ml4fmri/models/BrainNetCNN.py)
