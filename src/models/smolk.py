# pylint: disable=invalid-name, missing-function-docstring
""" SMoLK model module """
from random import uniform, randint

import torch
from torch import nn
from torch.nn import functional as F

from omegaconf import OmegaConf, DictConfig


def get_model(cfg: DictConfig, model_cfg: DictConfig):
    return LearnedFilters(model_cfg)


def default_HPs(cfg: DictConfig):
    model_cfg = {
        "num_kernels": 24,
        "lr": 0.005,
        "input_size": cfg.dataset.data_info.main.data_shape[
            2
        ],  # data_shape: [batch_size, time_length, input_feature_size]
        "output_size": cfg.dataset.data_info.main.n_classes,
    }
    return OmegaConf.create(model_cfg)


# def random_HPs(cfg: DictConfig):
#     model_cfg = {
#         "dropout": uniform(0.1, 0.9),
#         "hidden_size": randint(32, 256),
#         "num_layers": randint(0, 4),
#         "lr": 10 ** uniform(-4, -3),
#         "input_size": cfg.dataset.data_info.main.data_shape[
#             2
#         ],  # data_shape: [batch_size, time_length, input_feature_size]
#         "output_size": cfg.dataset.data_info.main.n_classes,
#     }
#     return OmegaConf.create(model_cfg)

class LearnedFilters(nn.Module):
    def __init__(self, model_cfg):
        super(LearnedFilters, self).__init__()
        self.input_channels = model_cfg.input_size
        self.num_kernels = model_cfg.num_kernels
        self.conv1 = nn.Conv1d(1, model_cfg.num_kernels, 40, stride=1, bias=True)
        self.conv2 = nn.Conv1d(1, model_cfg.num_kernels, 10, stride=1, bias=True)
        self.conv3 = nn.Conv1d(1, model_cfg.num_kernels, 3, stride=1, bias=True)

        self.linear = nn.Linear(model_cfg.input_size*model_cfg.num_kernels*3, model_cfg.output_size)
    
    def forward(self, x):
        bs, tl, fs = x.shape  # [batch_size, time_length, input_feature_size]
        x = torch.swapaxes(x, 1, 2) # [batch_size, input_feature_size, time_length]
        x = x.reshape(bs*fs, 1, tl) # [batch_size*input_feature_size, 1, time_length]

        c1 = F.leaky_relu(self.conv1(x)).mean(dim=-1).reshape(bs, fs*self.num_kernels)
        c2 = F.leaky_relu(self.conv2(x)).mean(dim=-1).reshape(bs, fs*self.num_kernels)
        c3 = F.leaky_relu(self.conv3(x)).mean(dim=-1).reshape(bs, fs*self.num_kernels)
        
        aggregate = torch.cat([c1,c2,c3], dim=1)
        aggregate = self.linear(aggregate)
        
        return aggregate

