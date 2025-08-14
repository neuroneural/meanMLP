# ml4fmri
This repo contains implementation of several __deep learning models__ for fMRI data analysis, gathered and packaged together for the ease of experimentation. Originally based on the codebase behind the NeuroImage paper ["A simple but tough-to-beat baseline for fMRI time-series classification"](https://doi.org/10.1016/j.neuroimage.2024.120909).

TODO: Explain how to use, add tutorials. Add polyssifier-like functionality.

# Use example
```pip install ml4fmri```

```
# in python, get fMRI time series data in shape (samples, time, n_features)
# and labels in shape (samples) (binary or multiclass)
from ml4fmri import cvbench # runs CV experiments with implemented models on the given data
report = cvbench(data, labels, models='all', n_folds=5)
report.plot_scores()
```
