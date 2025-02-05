# meanMLP
This repo contains the meanMLP model implementation and the experimental setup from the NeuroImage paper ["A simple but tough-to-beat baseline for fMRI time-series classification"](https://doi.org/10.1016/j.neuroimage.2024.120909).

# 0. If you just want the meanMLP model source code
Go to `src/models/mlp.py`.
`meanMLP` and `default_HPs` is what you need.

You can also check the colab tutorial, it shows how to use the experiment framework and the model in minimalistic examples. <a href="https://colab.research.google.com/drive/1Lyzof8DakkZI4BPBR82xvmow7tAN1i3a?usp=sharing"><img alt="Colab" src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d0/Google_Colaboratory_SVG_Logo.svg/1600px-Google_Colaboratory_SVG_Logo.svg.png?20221103151432" width="50"></a>

# 1. Requirements
```bash
conda create -n mlp_nn python=3.12
conda activate mlp_nn
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
```

# 2. Reproducing the results
## 1. Figures 3 and 5: general and transfer classification comparisons
```bash
DATASETS=('fbirn' 'bsnip' 'cobre' 'abide_869' 'oasis' 'adni' 'hcp' 'ukb' 'ukb_age_bins' 'fbirn_roi' 'abide_roi' 'hcp_roi_752')
MODELS=('mlp' 'lstm' 'pe_transformer' 'milc' 'dice' 'bolT' 'glacier' 'bnt' 'fbnetgen' 'brainnetcnn' 'lr')
for dataset in "${DATASETS[@]}"; do 
    for model in "${MODELS[@]}"; do 
        PYTHONPATH=. python scripts/run_experiments.py mode=exp dataset=$dataset model=$model prefix=general ++model.default_HP=True
    done; 
done
```

## 2. Figures 6 and 7: reshuffling experiments and additional data pre-processing tests
```bash
DATASETS=('hcp' 'hcp_roi_752' 'hcp_schaefer' 'hcp_non_mni_2' 'hcp_mni_3' 'ukb')
MODELS=('mlp' 'lstm' 'mean_lstm' 'pe_transformer' 'mean_pe_transformer')

for model in "${MODELS[@]}"; do 
    PYTHONPATH=. python scripts/run_experiments.py mode=exp dataset='hcp_time' model=$model prefix=additional ++model.default_HP=True
    for dataset in "${DATASETS[@]}"; do 
        PYTHONPATH=. python scripts/run_experiments.py mode=exp dataset=$dataset model=$model prefix=additional ++model.default_HP=True
        PYTHONPATH=. python scripts/run_experiments.py mode=exp dataset=$dataset model=$model prefix=additional ++model.default_HP=True permute=Multiple
    done; 
done
```
## 3. Plotting the results
Plotting scripts can be found at `scripts/plot_figures.ipynb`.


# `scripts/run_experiments.py` options:
## Required:
- `mode`: 
    - `tune` - tune mode: run multiple experiments with different hyperparams
    - `exp` - experiment mode: run experiments with the best hyperparams found in the `tune` mode, or with default hyperparams if `default_HPs` is set to `True`

- `model`: model for the experiment. Models' config files can be found at `src/conf/model`, and their sourse code is located at `src/models`

| Name | `script_name` | Description | Trainable Params (on ICA data) |
|-------------------|---------------------|-------------------------------------------------------------------------------------------------------|------------------|
| meanMLP           | `mlp`                 | Presented model, `TS` model                                                                                     | 9282             |
| LSTM              | `lstm`                | Classic LSTM model for classification, `TS` model                                                        | 446042           |
| meanLSTM          | `mean_lstm`           | LSTM with LSTM output embeddings averaging, `TS` model                                                 | 446042           |
| Transformer       | `pe_transformer`      | BERT-inspired model, uses transformer encoder, `TS` model                                                | 6137098          |
| meanTransformer   | `mean_pe_transformer` | Transformer with encoder output averaging, `TS` model                                               | 6137098          |
| MILC              | `milc`                | `TS` model, [MILC paper](https://arxiv.org/abs/2007.16041)                                                | 1116643          |
| DICE              | `dice`                | `TS` model, [DICE paper](https://www.sciencedirect.com/science/article/pii/S1053811922008588?via%3Dihub)  | 818171           |
| BolT              | `bolT`                | `TS` model, [BolT paper](https://www.sciencedirect.com/science/article/pii/S1361841523001019)             | 675785           |
| Glacier           | `glacier`             | `TS` model, [Glacier paper](https://ieeexplore.ieee.org/document/10097126)                                | 865571           |
| BNT               | `bnt`                 | `FNC` model, [BNT paper](https://arxiv.org/abs/2210.06681)                                                | 670930           |
| FBNetGen          | `fbnetgen`            | `TS+FNC` model, [FBNetGen paper](https://arxiv.org/abs/2205.12465)                                        | 131334           |
| BrainNetCNN       | `brainnetcnn`         | `FNC` model, [BrainNetCNN paper](https://www.sciencedirect.com/science/article/pii/S1053811916305237)     | 274717           |
| LR                | `lr`                  | Logistic Regression, `FNC` model                                                                         | 2758             |

- `dataset`: dataset for the experiments. Datasets' config files can be found at `src/conf/dataset`, and their loading scripts are located at `src/datasets`.

| `script_name`          | Category            | Parcellation         | # Classes | Description                                            |
|-------------------|---------------------|----------------------|-----------|--------------------------------------------------------|
| `fbirn`             | Schizophrenia        | ICA                  | 2         | ICA FBIRN dataset                                      |
| `cobre`             | Schizophrenia        | ICA                  | 2         | ICA COBRE dataset                                      |
| `bsnip`             | Schizophrenia        | ICA                  | 2         | ICA BSNIP dataset                                      |
| `abide`             | Autism               | ICA                  | 2         | ICA ABIDE dataset (not used in the paper)              |
| `abide_869`         | Autism               | ICA                  | 2         | ICA ABIDE extended dataset                             |
| `oasis`             | Alzheimer            | ICA                  | 2         | ICA OASIS dataset                                      |
| `adni`              | Alzheimer            | ICA                  | 2         | ICA ADNI dataset                                       |
| `hcp`               | Sex                  | ICA                  | 2         | ICA HCP dataset                                        |
| `ukb`               | Sex                  | ICA                  | 2         | ICA UKB dataset with `sex` labels                      |
| `ukb_age_bins`      | Sex X Age bins       | ICA                  | 20        | ICA UKB dataset with `sex X age bins` labels           |
| `fbirn_roi`         | Schizophrenia        | Schaefer 200 ROIs    | 2         | Schaefer 200 ROIs FBIRN dataset                        |
| `abide_roi`         | Autism               | Schaefer 200 ROIs    | 2         | Schaefer 200 ROIs ABIDE dataset                        |
| `hcp_roi_752`       | Sex                  | Schaefer 200 ROIs    | 2         | Schaefer 200 ROIs HCP dataset                          |
| `hcp_non_mni_2`     | Sex                  | Desikan/Killiany ROIs| 2       | Deskian/Killiany ROIs HCP dataset in ORIG space        |
| `hcp_mni_3`         | Sex                  | Desikan/Killiany ROIs| 2       | Deskian/Killiany ROIs HCP dataset in MNI space         |
| `hcp_schaefer`      | Sex                  | Schaefer 200 ROIs    | 2       | Noisy Schaefer 200 ROIs HCP dataset                    |
| `hcp_time`          | Time Direction       | ICA                  | 2       | ICA HCP dataset with normal/inversed time direction    |


## Optional
- `prefix`: custom prefix for the project name
    - default prefix is UTC time
    - appears in the name of logs directory and the name of WandB project
    - `exp` mode runs with custom prefix will use HPs from `tune` mode runs with the same prefix
        - unless model.default_HP is set to `True`
- `permute`: whether TS models should be trained on time-reshuffled data
    - set to `permute=Multiple` to reshuffle on every new epoch
- `wandb_silent`: whether wandb logger should run silently (default: `True`)
- `wandb_offline`: whether wandb logger should only log results locally (default: `False`)

