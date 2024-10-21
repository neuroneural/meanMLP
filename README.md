# 0. If you just want the Mean MLP model source code
Go to `src/models/mlp.py`.
`MeanMLP` and `default_HPs` is what you need.

You can also check the colab tutorial. <a href="https://colab.research.google.com/drive/1Lyzof8DakkZI4BPBR82xvmow7tAN1i3a?usp=sharing"><img alt="Colab" src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d0/Google_Colaboratory_SVG_Logo.svg/1600px-Google_Colaboratory_SVG_Logo.svg.png?20221103151432" width="50"></a>

# 1. Requirements
```bash
conda create -n mlp_nn python=3.9
conda activate mlp_nn
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
```

# 2. Reproducing the results
## 1. Figures 3 and 4: general and transfer classification comparisons
```bash
DATASETS=('fbirn' 'bsnip' 'cobre' 'abide_869' 'oasis' 'adni' 'hcp' 'ukb' 'ukb_age_bins' 'fbirn_roi' 'abide_roi' 'hcp_roi_752')
MODELS=('mlp' 'lstm' 'pe_transformer' 'milc' 'dice' 'bolT' 'glacier' 'bnt' 'fbnetgen' 'brainnetcnn' 'lr')
for dataset in "${DATASETS[@]}"; do 
    for model in "${MODELS[@]}"; do 
        PYTHONPATH=. python scripts/run_experiments.py mode=exp dataset=$dataset model=$model prefix=general ++model.default_HP=True
    done; 
done
```

## 2. Figures 5 and 6: reshuffling experiments and additional data pre-processing tests
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
Data loading scripts rely on fetching the results from WandB. If you set WandB offline mode while running the experiments, you'll need rewire the script and load the csv files from the experiment folders in `assets/logs`.


# `scripts/run_experiments.py` options:
## Required:
- `mode`: 
    - `tune` - tune mode: run multiple experiments with different hyperparams
    - `exp` - experiment mode: run experiments with the best hyperparams found in the `tune` mode, or with default hyperparams `default_HPs` is set to `True`

- `model`: model for the experiment. Models' config files can be found at `src/conf/model`, and their sourse code is located at `src/models`
    - `mlp` - our hero, TS model
    - `lstm` - classic LSTM model for classification, TS model
    - `mean_lstm` - `lstm` with LSTM output embeddings averaging, TS model
    - `pe_transformer` - BERT-inspired model, uses transformer endocder, TS model
    - `mean_pe_transformer` - `pe_transformer` with encoder output averaging, TS model

    - `dice` - TS model, https://www.sciencedirect.com/science/article/pii/S1053811922008588?via%3Dihub
    - `milc` - TS model, https://arxiv.org/abs/2007.16041 
    - `bolT` - TS model, https://www.sciencedirect.com/science/article/pii/S1361841523001019
    - `glacier` - TS model, https://ieeexplore.ieee.org/document/10097126

    - `bnt` - FNC model, https://arxiv.org/abs/2210.06681
    - `fbnetgen` - TS+FNC model, https://arxiv.org/abs/2205.12465
    - `brainnetcnn` - FNC model, https://www.sciencedirect.com/science/article/pii/S1053811916305237
    - `lr` - Logistic Regression, FNC model

- `dataset`: dataset for the experiments. Datasets' config files can be found at `src/conf/dataset`, and their loading scripts are located at `src/datasets`.
    - `fbirn` - ICA FBIRN dataset
    - `cobre` - ICA COBRE dataset
    - `bsnip` - ICA BSNIP dataset
    - `abide` - ICA ABIDE dataset (not used in the paper)
    - `abide_869` - ICA ABIDE extended dataset
    - `oasis` - ICA OASIS dataset
    - `adni` - ICA ADNI dataset
    - `hcp` - ICA HCP dataset
    - `ukb` - ICA UKB dataset with `sex` labels
    - `ukb_age_bins` - ICA UKB dataset with `sex X age bins` labels

    - `fbirn_roi` - Schaefer 200 ROIs FBIRN dataset
    - `abide_roi` - Schaefer 200 ROIs ABIDE dataset
    - `hcp_roi_752` - Schaefer 200 ROIs HCP dataset

    - `hcp_non_mni_2` - Deskian/Killiany ROIs HCP dataset in ORIG space
    - `hcp_mni_3` - Deskian/Killiany ROIs HCP dataset in MNI space
    - `hcp_schaefer` - Noisy Schaefer 200 ROIs HCP dataset
    - `hcp_time` - ICA HCP dataset with normal/inversed time direcion

## Optional
- `prefix`: custom prefix for the project
    - default prefix is UTC time
    - appears in the name of logs directory and the name of WandB project
    - `exp` mode runs with custom prefix will use HPs from `tune` mode runs with the same prefix
        - unless model.default_HP is set to `True`
- `permute`: whether TS models should be trained on time-reshuffled data
    - set to `permute=Multiple` to reshuffle on every new epoch
- `wandb_silent`: whether wandb logger should run silently (default: `True`)
- `wandb_offline`: whether wandb logger should only log results locally (default: `False`)

