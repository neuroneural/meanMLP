# pylint: disable=invalid-name, missing-function-docstring
""" MLP model module """
from random import uniform, randint

import torch
from torch import nn

from omegaconf import OmegaConf, DictConfig


def get_model(cfg: DictConfig, model_cfg: DictConfig):
    return meanMLP(model_cfg)


def default_HPs(cfg: DictConfig):
    model_cfg = {
        "dropout": 0.49,
        "hidden_size": 160,
        "num_layers": 0,
        "lr": 0.005,
        "input_size": cfg.dataset.data_info.main.data_shape[
            2
        ],  # data_shape: [batch_size, time_length, input_feature_size]
        "output_size": cfg.dataset.data_info.main.n_classes,
    }
    return OmegaConf.create(model_cfg)


def random_HPs(cfg: DictConfig):
    model_cfg = {
        "dropout": uniform(0.1, 0.9),
        "hidden_size": randint(32, 256),
        "num_layers": randint(0, 4),
        "lr": 10 ** uniform(-4, -3),
        "input_size": cfg.dataset.data_info.main.data_shape[
            2
        ],  # data_shape: [batch_size, time_length, input_feature_size]
        "output_size": cfg.dataset.data_info.main.n_classes,
    }
    return OmegaConf.create(model_cfg)


class ResidualBlock(nn.Module):
    """Residual block"""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor):
        return self.block(x) + x


class meanMLP(nn.Module):
    """
    meanMLP model for fMRI data.
    Expected input shape: [batch_size, time_length, input_feature_size].
    Output: [batch_size, n_classes]

    Hyperparameters expected in model_cfg:
        dropout: float
        hidden_size: int
        num_layers: int
    Data info expected in model_cfg:
        input_size: int - input_feature_size
        output_size: int - n_classes
    """

    def __init__(self, model_cfg: DictConfig):
        super().__init__()

        input_size = model_cfg.input_size
        output_size = model_cfg.output_size
        dropout = model_cfg.dropout
        hidden_size = model_cfg.hidden_size
        num_layers = model_cfg.num_layers

        # input block
        layers = [
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        ]
        # inter blocks: default HPs model has none of them
        for _ in range(num_layers):
            layers.append(
                nn.Sequential(
                    ResidualBlock(
                        nn.Sequential(
                            nn.Linear(hidden_size, hidden_size),
                            nn.LayerNorm(hidden_size),
                        )
                    ),
                    nn.ReLU(),
                    nn.Dropout(p=dropout),
                ),
            )

        # output block
        layers.append(
            nn.Linear(hidden_size, output_size),
        )

        self.fc = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, introspection=False):
        bs, tl, fs = x.shape  # [batch_size, time_length, input_feature_size]

        fc_output = self.fc(x.view(-1, fs))
        fc_output = fc_output.view(bs, tl, -1)

        logits = fc_output.mean(1)

        if introspection:
            predictions = torch.argmax(logits, axis=-1)
            return fc_output, predictions

        return logits
