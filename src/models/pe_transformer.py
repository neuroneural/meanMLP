# pylint: disable=invalid-name, missing-function-docstring, no-member
""" Transformer model module """
from random import uniform, randint
import math

from torch import nn
import torch

from omegaconf import OmegaConf, DictConfig


def get_model(cfg: DictConfig, model_cfg: DictConfig):
    return PETransformer(model_cfg)


def default_HPs(cfg: DictConfig):
    model_cfg = {
        "dropout": 0.46,
        "head_hidden_size": 100,
        "num_heads": 5,
        "num_layers": 2,
        "lr": 5e-5,
        "input_size": cfg.dataset.data_info.main.data_shape[2],
        "output_size": cfg.dataset.data_info.main.n_classes,
    }
    return OmegaConf.create(model_cfg)


def random_HPs(cfg: DictConfig):
    model_cfg = {
        "dropout": uniform(0.1, 0.9),
        "head_hidden_size": 2 * randint(2, 64),
        "num_heads": randint(1, 3),
        "num_layers": randint(1, 3),
        "lr": 10 ** uniform(-5, -2),
        "input_size": cfg.dataset.data_info.main.data_shape[2],
        "output_size": cfg.dataset.data_info.main.n_classes,
    }
    return OmegaConf.create(model_cfg)


class PETransformer(nn.Module):
    """
    Transformer model for fMRI data.
    Expected input shape: [batch_size, time_length, input_feature_size].
    Output: [batch_size, n_classes]

    Hyperparameters expected in model_cfg:
        dropout: float
        head_hidden_size: int
        num_heads: int
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
        head_hidden_size = model_cfg.head_hidden_size
        num_heads = model_cfg.num_heads
        num_layers = model_cfg.num_layers

        hidden_size = head_hidden_size * num_heads

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.input_embed = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)

        self.fc = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        input_embed = self.input_embed(x)
        input_embed = self.pos_encoder(input_embed)

        tf_output = self.transformer(input_embed)
        tf_output = tf_output[:, 0, :]

        fc_output = self.fc(tf_output)
        return fc_output


class PositionalEncoding(nn.Module):
    """Positional encoding module"""

    def __init__(self, input_feature_size, max_seq_length=2048):
        super().__init__()

        # Create a matrix of shape (max_seq_length, input_feature_size)
        pe = torch.zeros(max_seq_length, input_feature_size)

        # Calculate the positional encoding
        position = torch.arange(0, max_seq_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, input_feature_size, 2, dtype=torch.float32)
            * -(math.log(10000.0) / input_feature_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1).transpose(
            0, 1
        )  # Transpose for (batch_size, max_seq_length, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # expects data of size [batch_size, time_length, input_feature_size]
        return x + self.pe[:, : x.size(1), :]
