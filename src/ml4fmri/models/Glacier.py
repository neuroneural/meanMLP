# pylint: disable=invalid-name, missing-function-docstring
"""
Glacier model module.

Transformer-based connectivity estimator and classifier, works with any time series.

@INPROCEEDINGS{glacier,
    author={Mahmood, Usman and Fu, Zening and Calhoun, Vince and Plis, Sergey},
    booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
    title={Glacier: Glass-Box Transformer for Interpretable Dynamic Neuroimaging}, 
    year={2023},
    pages={1-5},
    doi={10.1109/ICASSP49357.2023.10097126}
}

"""

import torch
from torch import nn
from torch.nn.functional import cross_entropy
from types import SimpleNamespace

from .helper_functions import (
    basic_handle_batch,
    basic_dataloader,
    basic_Adam_optimizer,
    BasicTrainer,
)


class Glacier(nn.Module):
    """
    TIME SERIES MODEL

    Glacier model from https://doi.org/10.1109/icassp49357.2023.10097126.
    Original: https://github.com/UsmanMahmood27/Glacier.
    Expected input shape: [batch_size, time_length, input_feature_size].
    Output: logits [batch_size, n_classes]; loss_load = {"logits": logits, "model": self}.
    """

    def __init__(self, input_size: int, output_size: int):
        super().__init__()

        # ---- Minimal config (kept as a dict, pruned of unused keys) ----
        model_cfg = {
            "input_size": input_size,
            "output_size": output_size,

            # hyperparams
            "embedding_size": 48,      # per-time scalar -> embedding dim
            "n_heads": 2,              # spatial MHA heads (across ROIs)
            "n_heads_temporal": 2,     # temporal Transformer heads
            "transformer_ff": 100,     # feed-forward size in Transformer layer
            "transformer_dropout": 0.1,
            "gta_upscale": 0.05,       # GTA bottleneck ratios
            "gta_upscale2": 0.5,
        }
        model_cfg = SimpleNamespace(**model_cfg)

        self.lr = 2e-4          # learning rate used in the paper. Defined like this in every model for reference.
        self.reg_param = 1e-7   # regularization strength for GlacierLoss
        self.criterion = GlacierLoss(self.reg_param)

        # Aliases
        C = model_cfg.input_size
        E = model_cfg.embedding_size
        H_spatial = model_cfg.n_heads
        H_temporal = model_cfg.n_heads_temporal

        # ---- Final classifier on FC matrices ----
        self.encoder = NatureCNN(model_cfg)  # flattens CxC -> MLP -> output_size

        # ---- Embedding / temporal encoder ----
        # Map each scalar x[t, roi] -> R^E
        self.embedder = nn.Sequential(
            nn.Linear(1, E),
            nn.Sigmoid(),
        )

        # Temporal transformer (over sequence length W), batch_first to avoid permutes
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=E,
            nhead=H_temporal,
            dim_feedforward=model_cfg.transformer_ff,
            dropout=model_cfg.transformer_dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        # Project temporal embeddings to concatenated per-head space for spatial MHA
        self.up_sample = nn.Linear(E, E * H_spatial)

        # Per-ROI positional embedding for spatial attention (broadcast over batches & timesteps)
        self.position_embeddings_rois = nn.Parameter(torch.zeros(1, C, E * H_spatial))
        self.position_embeddings_rois_dropout = nn.Dropout(0.1)

        # Spatial multi-head attention across ROIs (per time step)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=E * H_spatial,
            num_heads=H_spatial,
            batch_first=True,  # (N, S, E)
        )

        # ---- Global Temporal Attention (GTA) over time on flattened CxC ----
        self.upscale = model_cfg.gta_upscale
        self.upscale2 = model_cfg.gta_upscale2
        FC_dim = C * C

        self.HW = nn.Hardswish()
        self.gta_embed = nn.Linear(FC_dim, round(self.upscale * FC_dim))
        self.gta_norm = nn.Sequential(
            nn.BatchNorm1d(round(self.upscale * FC_dim)),
            nn.ReLU(),
        )
        self.gta_attend = nn.Sequential(
            nn.Linear(round(self.upscale * FC_dim), round(self.upscale2 * FC_dim)),
            nn.ReLU(),
            nn.Linear(round(self.upscale2 * FC_dim), 1),
        )
        self.gta_dropout = nn.Dropout(0.35)

    # ---- GTA pooling over time ----
    def gta_attention(self, x):
        """
        x: [B, W, C*C]  (flattened FC per time)
        Returns: FC_pooled [B, C*C]
        """
        # readout gate emphasizing time steps correlated with mean over time
        x_readout = x.mean(1, keepdim=True)   # [B, 1, C*C]
        x_readout = x * x_readout             # [B, W, C*C]

        B, W, F = x_readout.shape
        x_embed = self.gta_embed(x_readout.reshape(B * W, F))
        x_embed = self.gta_norm(x_embed)
        gate = self.gta_attend(x_embed).view(B, W)   # [B, W]
        gate = self.HW(gate)
        gate = self.gta_dropout(gate)

        # weighted mean over time
        return (x * gate.unsqueeze(-1)).mean(1)

    # ---- Spatial multi-head attention over ROIs at a single time ----
    def multi_head_attention(self, x):
        """
        x: [N, C, E*H] with batch_first=True (N = W*B)
        returns:
            attn_output: [N, C, E*H]
            attn_weights: [N, C, C] (averaged over heads)
        """
        attn_output, attn_weights = self.multihead_attn(x, x, x, need_weights=True, average_attn_weights=True)
        return attn_output, attn_weights

    def forward(self, input):
        """
        input: [B, W, C]
        """
        B, W, C = input.shape

        # 1) Scalar -> embedding per ROI
        # [B, W, C] -> [B*C, W, 1] -> embed -> [B*C, W, E]
        x = input.permute(0, 2, 1).contiguous().view(B * C, W, 1)
        x = self.embedder(x)  # [B*C, W, E]

        # 2) Temporal transformer per ROI across time (batch_first)
        x = self.transformer_encoder(x)  # [B*C, W, E]

        # 3) Prepare for spatial attention (per time step)
        # reshape back to [B, C, W, E] then to [B, W, C, E]
        x = x.view(B, C, W, -1).permute(0, 2, 1, 3).contiguous()  # [B, W, C, E]
        x = self.up_sample(x)  # [B, W, C, E*H_spatial]

        # Add per-ROI positional embeddings, then merge (B,W) as batch
        pos = self.position_embeddings_rois  # [1, C, E*H]
        x = self.position_embeddings_rois_dropout(x + pos)  # broadcast over B,W
        x = x.view(B * W, C, -1)  # [B*W, C, E*H]

        # 4) Spatial attention per time step -> attention weights (FC per time)
        _, attn_weights = self.multi_head_attention(x)  # attn_weights: [B*W, C, C]
        attn_weights = attn_weights.view(B, W, C, C)    # [B, W, C, C]

        # 5) GTA temporal pooling over flattened FCs
        attn_flat = attn_weights.view(B, W, C * C)      # [B, W, C*C]
        FC = self.gta_attention(attn_flat)              # [B, C*C]

        # 6) Classifier over FC
        FC_square = FC.view(B, C, C)                    # [B, C, C]
        logits = self.encoder(FC_square.unsqueeze(1))   # [B, output_size]

        return logits, {"logits": logits, "model": self}

    # ==== Helper functions for model training and evaluation ====

    def compute_loss(self, loss_load, targets):
        """Standard loss computation routine for models."""
        return self.criterion(loss_load, targets)

    def handle_batch(self, batch):
        """
        Standard batch handling routine for models. 
        Returns loss for backprop and a dictionary of classification metrics and losses for logs.
        
        Args:
            batch (tuple): A batch containing the time series data and labels as a tuple.
        Returns
        -------
        loss : Tensor
            Loss for backpropagation.
        batch_log : dict
            Dictionary of classification metrics and losses for logs.
        """

        loss, batch_log = basic_handle_batch(self, batch)
        return loss, batch_log
    
    def get_optimizer(self, lr=None):
        """
        Standard optimizer getter routine for models.
        """
        if lr is None:
            lr = self.lr
        return basic_Adam_optimizer(self, lr)

    @staticmethod
    def prepare_dataloader(data,
                           labels,
                           batch_size: int = 64,
                           shuffle: bool = True):
        """
        Returns torch DataLoader that produces appropriate batches for the model (can pass to `handle_batch`)
        Args:
            data (array-like): Time series data of shape (B, T, D).
            labels (array-like): Class labels for the data.
            batch_size (int, optional): Batch size for the DataLoader. Defaults to 64.
            shuffle (bool, optional): Whether to shuffle batching in the DataLoader. Defaults to True.
        
        Returns:
            DataLoader: A PyTorch DataLoader generating the batches of time series data and labels. 
        """
        return basic_dataloader(data,
                                labels,
                                type="TS",
                                batch_size=batch_size,
                                shuffle=shuffle)
    
    def train_model(self,
              train_loader,
              val_loader,
              test_loader,
              epochs: int = 200,
              lr: float = None,
              device: str = None,
              patience: int = 30,
        ):
        """
        Standard model training routine.
        Args:
            train_loader (DataLoader): DataLoader for the training set. Used for training.
            val_loader (DataLoader): DataLoader for the validation set. Used during training to find most generalizable model.
            test_loader (DataLoader): DataLoader for the test set.
            epochs (int, optional): Number of training epochs. Defaults to 200.
            lr (float, optional): Optimizer learning rate (default: use model's self.lr).
            device (str, optional): Device to train the model on: "cuda", "mps", or "cpu". Default: auto-detect (cuda -> mps -> cpu).
            patience (int, optional): Early stopping patience (in epochs). Defaults to 30.
        """

        trainer = BasicTrainer(
            model=self,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=epochs,
            lr=lr,
            device=device,
            patience=patience,
        )
        
        train_logs, test_logs = trainer.run()
        return train_logs, test_logs


class GlacierLoss:
    """Cross-entropy + L1 regularization"""

    def __init__(self, reg_param: float = 1e-7):
        self.reg_param = reg_param

    def __call__(self, loss_load, targets):
        logits = loss_load["logits"]
        model = loss_load["model"]
        device = logits.device

        ce_loss = cross_entropy(logits, targets)

        # L1 on all non-bias params of GTA blocks
        reg_loss = torch.zeros((), device=device)
        for module in (model.gta_embed, model.gta_attend):
            for name, p in module.named_parameters():
                if p.requires_grad and "bias" not in name:
                    reg_loss = reg_loss + p.abs().sum()

        loss = ce_loss + self.reg_param * reg_loss
        return loss, {
            "ce_loss": float(ce_loss.detach().cpu().item()),
            "reg_loss": float(reg_loss.detach().cpu().item()),
        }
    

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class NatureCNN(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        C = model_cfg.input_size
        out = model_cfg.output_size
        self.final_conv_size = C * C
        self.main = nn.Sequential(
            Flatten(),
            nn.Linear(self.final_conv_size, 64),
            nn.ReLU(),
            nn.Linear(64, out),
        )

    def forward(self, inputs, fmaps: bool = False):
        return self.main(inputs)
