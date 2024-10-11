import torch
import torch.nn as nn

import os
import numpy as np
from omegaconf import OmegaConf, DictConfig
import math

import torch.nn.functional as F

def default_HPs(cfg: DictConfig):
    model_cfg = {
        "dropout": 0.49,
        "hidden_size": 160,
        "num_layers": 0,
        "lr": 0.005,
        "input_size": cfg.dataset.data_info.main.data_shape[
            2
        ],  # data_shape: [batch_size, time_length, input_feature_size]
        "time_points": cfg.dataset.data_info.main.data_shape[
            1
        ],  # data_shape: [batch_size, time_length, input_feature_size]
        "output_size": cfg.dataset.data_info.main.n_classes,

        'pre_training': 'None',
        "fMRI_twoD": False,
        "deep": False,
        'exp': 'NPT',
        'gain': 0.1,
        'temperature': 0.25,
        'script_ID': 1,
        'cv_Set': 1,
        'start_CV': 0,
        'teststart_ID': 1,
        'job_ID': 1,
        'ntrials': 10,
        'sample_number': 0,
        'num_frame_stack': 1,
        'no_downsample': True,
        'num_processes': 8,
        'linear': True,
        'use_multiple_predictors': False,
        'lr': 1e-4,
        'batch_size': 32, 
        'epochs': 300,
        'cuda_id': 1,
        'seed': 42,
        'encoder_type': "Nature",
        'model_type': "graph_the_works",
        'feature_size': 2,
        "fully_connected": False,
        'feature_size_pre_training': 32,
        'fMRI_feature_size': 1024,
        "patience": 15,
        "entropy_threshold": 0.6,
        "color": False,
        "wandb_proj": "curl-atari-neurips-scratch",
        "num_rew_evals": 10,
        "checkpoint-index": -1,
        "naff_fc_size": 2048,
        "pred_offset": 1,
        'sequence_length': 100,
        'steps_start': 0,
        'steps_end': 99,
        'steps_step': 4,
        'gru_size': 256,
        'lstm_size': 100,
        'lstm_size_within_window': 8,
        'fMRI_lstm_size': 100,
        'gru_layers': 1,
        'lstm_layers': 1,
        "collect_mode": "random_agent",
        "beta": 1.0,
        "train_encoder": True,
        'probe_lr': 3e-6,
        "probe_collect_mode": "random_agent",
        'num_runs': 1
    }

    return OmegaConf.create(model_cfg)

def get_model(cfg: DictConfig, model_cfg: DictConfig):
    return Glacier(model_cfg)


def get_criterion(cfg: DictConfig, model_cfg: DictConfig):
    return RegCEloss(model_cfg)

class RegCEloss:
    """Cross-entropy loss with model regularization"""

    def __init__(self, model_cfg):
        self.ce_loss = nn.CrossEntropyLoss()

    def __call__(self, logits, target, model, device):

        reg = 1e-7
        reg_loss = torch.zeros(1).to(device)

        ce_loss = self.ce_loss(logits, target)

        for name, param in model.gta_embed.named_parameters():
            if 'bias' not in name:
                reg_loss += (reg * torch.norm(param,p=1))

        for name, param in model.gta_attend.named_parameters():
            if 'bias' not in name:
                reg_loss += (reg * torch.norm(param,p=1))

        loss = ce_loss + reg_loss
        return loss
    
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class NatureCNN(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        self.final_conv_size = model_cfg.input_size ** 2
        self.main = nn.Sequential(
            Flatten(),
            nn.Linear(self.final_conv_size, 64),
            nn.ReLU(),
            nn.Linear(64, model_cfg.output_size)
        )

    def forward(self, inputs, fmaps=False):
        out = self.main(inputs)
        return out


class Glacier(nn.Module):
    """Glacier model. Cleaner than the original, but still a mess"""

    def __init__(self, model_cfg):

        super().__init__()

        n_regions=model_cfg.input_size

        self.encoder = NatureCNN(model_cfg)
        self.n_regions = n_regions
        self.n_regions_after = n_regions
        self.time_points = model_cfg.time_points
        self.division = 1 # note this value. is is supposed to be greater than one for it to make any effect 
        # in the original code, it is hardcoded into forward. but in fact its useless, so don't bother with it
        self.n_heads = 2
        self.n_heads_temporal = 2
        self.embedding_size = 48
        self.attention_embedding = self.embedding_size * self.n_heads
        self.upscale = 0.05
        self.upscale2 = 0.5
        self.embedder_output_dim = self.embedding_size * 1
        self.attention_embedding_temporal = self.embedding_size
        self.temperature = 2

        self.up_sample = nn.Linear(
                self.embedding_size, 
                self.embedding_size*self.n_heads
            )

        self.gta_embed = nn.Linear(
                self.n_regions * self.n_regions, 
                round(self.upscale * self.n_regions * self.n_regions)
            )

        self.gta_norm = nn.Sequential(
                nn.BatchNorm1d(round(self.upscale * self.n_regions * self.n_regions)), 
                nn.ReLU()
            )
        self.gta_attend = nn.Sequential(
                nn.Linear(
                    round(self.upscale * self.n_regions * self.n_regions), 
                    round(self.upscale2  * self.n_regions * self.n_regions)
                ),
                nn.ReLU(),
                nn.Linear(
                    round(self.upscale2 * self.n_regions * self.n_regions), 
                    1
                )
            )
        self.gta_dropout = nn.Dropout(0.35)


        self.multihead_attn = nn.MultiheadAttention(self.attention_embedding, self.n_heads)

        #################################### MHA 2 ###############################################
        self.position_embeddings_rois = nn.Parameter(torch.zeros(1, self.n_regions, self.embedder_output_dim * self.n_heads))
        self.position_embeddings_rois_dropout = nn.Dropout(0.1)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.time_points, self.embedder_output_dim))
        self.embedder = nn.Sequential(
            nn.Linear(1, self.embedder_output_dim),
            nn.Sigmoid(),
        )

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedder_output_dim, 
            nhead=self.n_heads_temporal,
            dim_feedforward=100,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, 
            num_layers=1
        )

        
        self.relu = torch.nn.ReLU()
        self.HS = torch.nn.Hardsigmoid()
        self.HW = torch.nn.Hardswish()
        self.selu = torch.nn.SELU()
        self.celu = torch.nn.CELU()
        self.tanh = torch.nn.Tanh()
        self.softplus = torch.nn.Softplus(threshold=20)

    def gta_attention(self, x):
        x_readout = x.mean(1, keepdim=True)
        x_readout = (x*x_readout)
        a = x_readout.shape[0]
        b = x_readout.shape[1]
        x_readout = x_readout.reshape(-1,x_readout.shape[2])
        x_embed = self.gta_norm(self.gta_embed(x_readout))
        x_graphattention = (self.gta_attend(x_embed).squeeze()).reshape(a, b)

        x_graphattention = self.HW(x_graphattention.reshape(a, b)) # You can use hard sigmoid here as well. It might decrease the classification performance, which can be fixed by changing tuning other hyper parameters.

        return (x * (x_graphattention.unsqueeze(-1))).mean(1), x_graphattention

    def multi_head_attention(self, outputs):

        outputs = outputs.permute(1, 0, 2)
        attn_output, attn_output_weights = self.multihead_attn(outputs, outputs, outputs)
        attn_output = attn_output.permute(1,0,2)

        return attn_output, attn_output_weights

    def forward(self, input):
        B = input.shape[0] # = batch_size
        W = input.shape[1] # = time_length
        R = input.shape[2] # = feature_size

        inputs = input.permute(0, 2, 1).contiguous() # [B, R, W]


        inputs = inputs.reshape(-1,1)
        inputs = self.embedder(inputs)
        inputs = inputs.reshape(B,R,W,-1)

        ########################## transformer encoder - start#################################
        ########################## Non iterative################################# This is faster but can go out of memory

        inputs = inputs.reshape(B * R, W, -1)
        inputs = inputs.permute(1,0,2).contiguous()
        inputs = self.transformer_encoder(inputs) # to get temporal weights, you need to edit 
        # the return statement of the forward functions of transformer encoder 
        # and transformer encoder layer to return weight matrix along with emebddings. 
        # and edit the function call to multi head attention 
        # in the forward function of transformer encoder layer by setting need_weights=True
        inputs = inputs.permute(1,0,2).contiguous()

        ########################## Non iterative#################################
        ########################## transformer encoder - end#################################

        inputs = self.up_sample(inputs)
        inputs = inputs.reshape(B, R, W, -1)
        ########################## transformer encoder#################################

        inputs = inputs.permute(2,0,1,3).contiguous()

        inputs = inputs.reshape(W*B,R,-1)

        inputs = self.position_embeddings_rois_dropout(inputs + self.position_embeddings_rois)
        
        _ , attn_weights = self.multi_head_attention(inputs)

        attn_weights = attn_weights.reshape(W,B,R,R)

        attn_weights = attn_weights.permute(1, 0, 2, 3).contiguous()


        attn_weights = attn_weights.reshape(B, W, -1)


        FC, _ = self.gta_attention(attn_weights) #FC_time_weights is the temporal attention weights to create single FC matrix

        FC = FC.squeeze().reshape(B,R,R)
        # FC_sum =  torch.mean(attn_weights,dim=1).squeeze().reshape(B,R,R) # can use the sum of all FC matrices if don't want to use attention based mean


        FC_logits = self.encoder((FC.unsqueeze(1)))


        return FC_logits#, FC, "temporal_FC"