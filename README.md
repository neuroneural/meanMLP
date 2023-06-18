# Requirements

```bash
conda create -n mlp_nn python=3.9
conda activate mlp_nn
- if mac user:
    conda install pytorch::pytorch torchvision torchaudio -c pytorch
- else:
    conda install pytorch torchvision torchaudio pytorch-cuda=11.3 -c pytorch -c nvidia
pip install -r requirements.txt
```

# Examples:
```
PYTHONPATH=. python scripts/run_experiments.py mode=tune dataset=fbirn model=mlp prefix=test wandb_offline=True
PYTHONPATH=. python scripts/run_experiments.py mode=exp dataset=fbirn model=mlp prefix=test wandb_offline=True
```

# `scripts/run_experiments.py` options:
## Required:
- `mode`: 
    - `tune` - tune mode: run multiple experiments with different hyperparams
    - `exp` - experiment mode: run experiments with best hyperparams found in the `tune` mode

- `model`: model for the experiment. Models' config files can be found at `src/conf/model`, and their sourse code is located at `src/models`
    - `mlp` - our hero, TS model
    - `lstm` - classic LSTM model for classification, TS model (not used in the paper)
    - `mean_lstm` - `lstm` with LSTM output embeddings averaging, TS model
    - `transformer` - BERT-inspired model, uses transformer endocder, TS model (not used in the paper)
    - `mean_transformer` - `tansformer` with encoder output averaging, TS model

    - `dice` - TS model, https://www.sciencedirect.com/science/article/pii/S1053811922008588?via%3Dihub
    - `milc` - TS model, https://arxiv.org/abs/2007.16041 
        - not implemented yet, the authors` training script was used with dataloaders replaced with our scripts

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
    - `time_fbirn` - ICA FBIRN dataset, labeled according to time direction in the sample

    - `fbirn_roi` - Schaefer 200 ROIs FBIRN dataset
    - `abide_roi` - Schaefer 200 ROIs ABIDE dataset
    - `hcp_roi` - Schaefer 200 ROIs HCP dataset

    - `hcp_non_mni` - Deskian/Killiany ROIs HCP dataset (not used in the paper)

## Optional
- `prefix`: custom prefix for the project
    - default prefix is UTC time
    - appears in the name of logs directory and the name of WandB project
    - `exp` mode runs with custom prefix will use HPs from `tune` mode runs with the same prefix
- `use_additional_test_ds`: whether trained models whould be tested on compatible datasets (default: `False`)
    - not implemented
- `permute`: whether TS models should be trained on time-reshuffled data (default: `False`) 
    - not used in the paper
- `wandb_silent`: whether wandb logger should run silently (default: `True`)
- `wandb_offline`: whether wandb logger should only log results locally (default: `False`)

