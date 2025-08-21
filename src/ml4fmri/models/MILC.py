# pylint: disable=invalid-name, missing-function-docstring
""" 
MILC model module.

CNN+LSTM model for fMRI data, can work with any time series.

@inproceedings{wholeMILC,
    author = {Mahmood, Usman and Rahman, Md Mahfuzur and Fedorov, Alex and Lewis, Noah and Fu, Zening and Calhoun, Vince D. and Plis, Sergey M.},
    title = {Whole MILC: Generalizing Learned Dynamics Across Tasks, Datasets, and Populations},
    year = {2020},
    isbn = {978-3-030-59727-6},
    publisher = {Springer-Verlag},
    address = {Berlin, Heidelberg},
    url = {https://doi.org/10.1007/978-3-030-59728-3_40},
    doi = {10.1007/978-3-030-59728-3_40},
    booktitle = {Medical Image Computing and Computer Assisted Intervention – MICCAI 2020: 23rd International Conference, Lima, Peru, October 4–8, 2020, Proceedings, Part VII},
    pages = {407–417},
    numpages = {11},
    keywords = {Resting state fMRI., Deep learning, Self-supervised, Transfer learning},
    location = {Lima, Peru}
}
"""

from sys import path
import torch
from torch import nn
from torch.nn.functional import cross_entropy, softmax
from types import SimpleNamespace
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .helper_functions import basic_handle_batch, basic_dataloader, BasicTrainer, zscore_np
    
class MILCLoss:
    """Cross-entropy loss with model regularization"""

    def __init__(self, reg_param: float = 1e-6):

        self.reg_param = reg_param

    def __call__(self, loss_load, targets):
        logits = loss_load["logits"]
        model = loss_load["model"]
        device = logits.device

        ce_loss = cross_entropy(logits, targets)

        # L1 reg
        reg_loss = torch.zeros((), device=device)
        for module in (model.lstm, model.attn):
            for name, p in module.named_parameters():
                if "bias" not in name:
                    reg_loss = reg_loss + p.abs().sum()

        loss = ce_loss + self.reg_param * reg_loss
        return loss, {
            "ce_loss": float(ce_loss.detach().cpu().item()),
            "reg_loss": float(reg_loss.detach().cpu().item()),
        }
    
class MILC(nn.Module):
    """
    TIME SERIES MODEL
    MILC model for fMRI data from https://doi.org/10.48550/arXiv.2007.16041.
    Orig implementation: https://github.com/UsmanMahmood27/MILC.
    Expected input shape: [batch_size, n_windows, feature_size, window_size] (see prepare_dataloader for details)
    Output: [batch_size, n_classes]
    """

    def __init__(self,
                 input_size: int,
                 output_size: int
        ):
        super().__init__()

        self.lr = 2e-4  # learning rate used in the paper. Defined like this in every model for reference.
        self.eps = 1e-5 # epsilon for Adam optimizer
        self.reg_param = 1e-5  # regularization parameter, used in loss
        self.criterion = MILCLoss(self.reg_param)

        model_cfg = {
            "data_params": {
                "input_size": input_size,
                "output_size": output_size,
                "window_size": 20,
                "window_shift": 10,
            },
            "lstm": {
                "input_feature_size": 256, 
                "hidden_size": 200, 
                "n_layers": 1
            },
            "reg_param": 1e-5,
        }
        def dict_to_namespace(d):
            """Convert nested dict to nested SimpleNamespace"""
            if isinstance(d, dict):
                return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
            return d
        
        model_cfg = dict_to_namespace(model_cfg)


        self.encoder = NatureOneCNN(model_cfg)
        self.lstm = LSTM(model_cfg)
        self.attn = nn.Sequential(
            nn.Linear(2 * model_cfg.lstm.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.decoder = nn.Sequential(
            nn.Linear(model_cfg.lstm.hidden_size, model_cfg.lstm.hidden_size),
            nn.ReLU(),
            nn.Linear(model_cfg.lstm.hidden_size, model_cfg.data_params.output_size),
        )

        self.init_weight()

    def init_weight(self):
        for name, param in self.decoder.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param, gain=0.65)
        for name, param in self.attn.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param, gain=0.65)

    def get_attention(self, x):
        B, N, H = x.shape

        # last window embedding per sample: (B, 1, H) so it can broadcast across N
        x_last = x[:, -1:, :]                       # (B, 1, H)

        # Concatenate [x_i, x_last] for every i in a vectorized way: (B, N, 2H)
        pairs = torch.cat([x, x_last.expand(-1, N, -1)], dim=-1)

        # MLP over last dim (nn.Linear handles (..., in_features))
        weights = self.attn(pairs).squeeze(-1)      # (B, N)

        # Normalize across windows
        weights = softmax(weights, dim=1)           # (B, N)

        # Weighted sum across windows -> (B, H)
        # (B,N,H) * (B,N,1) -> sum over N
        attended = (x * weights.unsqueeze(-1)).sum(dim=1)
        return attended

    def forward(self, x):
        bs, nw, fs, ws = x.shape  # [batch_size, n_windows, feature_size, window_size]

        encoder_output = self.encoder(x.view(-1, fs, ws))
        encoder_output = encoder_output.view(bs, nw, -1)

        lstm_output = self.lstm(encoder_output)

        attention_output = self.get_attention(lstm_output)

        logits = self.decoder(attention_output)
    
        loss_load = {
            "logits": logits,
            "model": self,
        }
        return logits, loss_load

    #### Helper functions for model training and evaluation ####

    def compute_loss(self, loss_load, targets):
        """
        Standard loss computation routine for models.
        Args:
            loss_load (dict): Forward's second output for the batch.
            targets (torch.Tensor): True labels for the batch.
        
        Returns
        -------
        loss : Tensor
            Loss for backpropagation.
        logs : dict
            Dictionary containing the loss components for logs.
        """
        loss, loss_log = self.criterion(loss_load, targets)

        return loss, loss_log
    
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
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=self.eps)
        return optimizer

    @staticmethod
    def prepare_dataloader(data,
                           labels,
                           batch_size: int = 64,
                           shuffle: bool = True,
                           window_size: int = 20,
                           window_shift: int = 10,
                           zscore: bool = True):
        """
        Returns torch DataLoader that produces appropriate batches for the model (can pass to handle_batch)
        Args:
            data (array-like): Time series data of shape (B, T, D).
            labels (array-like): Class labels for the data.
            batch_size (int, optional): Batch size for the DataLoader. Defaults to 64.
            shuffle (bool, optional): Whether to shuffle batching in the DataLoader. Defaults to True.
        
        Returns:
            DataLoader: A PyTorch DataLoader generating the batches of time series data and labels. 
        """
        if zscore:
            data = zscore_np(data, axis=1)

        # transform data into overlapping windows of shape (Batch, (tau)Windows, Feature_size, (W)Window_size) 
        win_data = sliding_window_view(data, window_shape=window_size, axis=1)  # (B, Tau, D, W)
        win_data = win_data[:, ::window_shift, :, :]                            # (B, tau, D, W)

        return basic_dataloader(win_data,
                                labels,
                                type="TS",
                                batch_size=batch_size,
                                shuffle=shuffle,
                                zscore=False)
    
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
    

class LSTM(nn.Module):
    """Bidirectional LSTM for classifying subjects."""

    def __init__(self, model_cfg):
        super().__init__()
        input_size = model_cfg.lstm.input_feature_size
        self.hidden_size = model_cfg.lstm.hidden_size
        n_layers = model_cfg.lstm.n_layers

        self.lstm = nn.LSTM(
            input_size,
            self.hidden_size // 2,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True,
        )

        self.init_weight()

    def init_weight(self):
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_normal_(param, gain=0.65)  # 0.65 is default gain

    def forward(self, x):
        lstm_output, _ = self.lstm(x)
        return lstm_output
    
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class NatureOneCNN(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()

        # original feature size is treated as a number of channels of a 1D image
        feature_size = model_cfg.data_params.input_size
        # 1D image has size of the window time length
        window_size = model_cfg.data_params.window_size
        # encoder output is passed to LSTM
        output_size = model_cfg.lstm.input_feature_size

        # calculate CNN layers output
        cnn_output_size = window_size - 3
        cnn_output_size = cnn_output_size - 3
        cnn_output_size = cnn_output_size - 2
        final_conv_size = 200 * cnn_output_size

        self.cnn = nn.Sequential(
            self.init_module(nn.Conv1d(feature_size, 64, 4, stride=1)),
            nn.ReLU(),
            self.init_module(nn.Conv1d(64, 128, 4, stride=1)),
            nn.ReLU(),
            self.init_module(nn.Conv1d(128, 200, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            self.init_module(nn.Linear(final_conv_size, output_size)),
        )

    def init_module(self, module):
        # weight_init
        nn.init.orthogonal_(module.weight.data, gain=nn.init.calculate_gain("relu"))
        # bias_init
        nn.init.constant_(module.bias.data, 0)

        return module

    def forward(self, inputs):
        return self.cnn(inputs)