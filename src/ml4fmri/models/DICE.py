# pylint: disable=invalid-name, missing-function-docstring
""" 
DICE model module.

LSTM-based connectivity estimator and classifier, works with time series.

@article{dice,
    title = {Through the looking glass: Deep interpretable dynamic directed connectivity in resting {fMRI}},
    journal = {NeuroImage},
    volume = {264},
    pages = {119737},
    year = {2022},
    issn = {1053-8119},
    doi = {10.1016/j.neuroimage.2022.119737},
    url = {https://www.sciencedirect.com/science/article/pii/S1053811922008588},
    author = {Usman Mahmood and Zening Fu and Satrajit Ghosh and Vince Calhoun and Sergey Plis},
}
"""

import torch
from torch import nn
from torch.nn.functional import cross_entropy
from types import SimpleNamespace

from .helper_functions import basic_handle_batch, basic_dataloader, basic_Adam_optimizer, BasicTrainer

class DICELoss:
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
        for module in (model.gta_embed, model.gta_attend):
            for name, p in module.named_parameters():
                if "bias" not in name:
                    reg_loss = reg_loss + p.abs().sum()

        loss = ce_loss + self.reg_param * reg_loss
        return loss, {
            "ce_loss": float(ce_loss.detach().cpu().item()),
            "reg_loss": float(reg_loss.detach().cpu().item()),
        }

class DICE(nn.Module):
    """
    TIME SERIES MODEL
    DICE model for fMRI data from https://doi.org/10.1016/j.neuroimage.2022.119737.
    Orig implementation: https://github.com/UsmanMahmood27/DICE.
    Expected input shape: [batch_size, time_length, input_feature_size].
    Output: [batch_size, n_classes]
    """

    def __init__(self,
                 input_size: int,
                 output_size: int
        ):
        super().__init__()

        self.lr = 2e-4  # learning rate used in the paper. Defined like this in every model for reference.
        self.reg_param = 1e-6  # regularization parameter, used in loss
        self.criterion = DICELoss(self.reg_param)

        model_cfg = {
            "lstm": {
                "bidirectional": True,
                "num_layers": 1,
                "hidden_size": 50,
            },
            "clf": {
                "hidden_size": 64,
                "num_layers": 0,
            },
            "MHAtt": {
                "n_heads": 1,
                "head_hidden_size": 48,
                "dropout": 0.0,
            },
            "scheduler": {
                "patience": 4,
                "factor": 0.5,
            },
            "reg_param": 1e-6,
            "lr": 2e-4,
            "input_size": input_size,
            "output_size": output_size,
        }
        
        def dict_to_namespace(d):
            """Convert nested dict to nested SimpleNamespace"""
            if isinstance(d, dict):
                return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
            return d
        
        model_cfg = dict_to_namespace(model_cfg)

        lstm_hidden_size = model_cfg.lstm.hidden_size
        lstm_num_layers = model_cfg.lstm.num_layers
        bidirectional = model_cfg.lstm.bidirectional

        self.lstm_output_size = (
            lstm_hidden_size * 2 if bidirectional else lstm_hidden_size
        )

        clf_hidden_size = model_cfg.clf.hidden_size
        clf_num_layers = model_cfg.clf.num_layers

        MHAtt_n_heads = model_cfg.MHAtt.n_heads
        MHAtt_hidden_size = MHAtt_n_heads * model_cfg.MHAtt.head_hidden_size
        MHAtt_dropout = model_cfg.MHAtt.dropout

        # LSTM - first block
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Classifier - last block
        clf = [
            nn.Linear(input_size**2, clf_hidden_size),
            nn.ReLU(),
        ]
        for _ in range(clf_num_layers):
            clf.append(nn.Linear(clf_hidden_size, clf_hidden_size))
            clf.append(nn.ReLU())
        clf.append(
            nn.Linear(clf_hidden_size, output_size),
        )
        self.clf = nn.Sequential(*clf)

        # Multihead attention - second block
        self.key_layer = nn.Sequential(
            nn.Linear(
                self.lstm_output_size,
                MHAtt_hidden_size,
            ),
        )
        self.value_layer = nn.Sequential(
            nn.Linear(
                self.lstm_output_size,
                MHAtt_hidden_size,
            ),
        )
        self.query_layer = nn.Sequential(
            nn.Linear(
                self.lstm_output_size,
                MHAtt_hidden_size,
            ),
        )
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=MHAtt_hidden_size,
            num_heads=MHAtt_n_heads,
            dropout=MHAtt_dropout,
            batch_first=True,
        )

        # Global Temporal Attention - third block
        self.upscale = 0.05
        self.upscale2 = 0.5

        self.HW = torch.nn.Hardswish()
        self.gta_embed = nn.Sequential(
            nn.Linear(
                input_size**2,
                round(self.upscale * input_size**2),
            ),
        )
        self.gta_norm = nn.Sequential(
            nn.BatchNorm1d(round(self.upscale * input_size**2)),
            nn.ReLU(),
        )
        self.gta_attend = nn.Sequential(
            nn.Linear(
                round(self.upscale * input_size**2),
                round(self.upscale2 * input_size**2),
            ),
            nn.ReLU(),
            nn.Linear(round(self.upscale2 * input_size**2), 1),
        )

        self.init_weight()

    def init_weight(self):
        for module in (
            self.lstm,
            self.clf,
            self.query_layer,
            self.key_layer,
            self.value_layer,
            self.multihead_attn,
        ):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.kaiming_normal_(param, mode="fan_in")

        for module in (
            self.gta_embed,
            self.gta_attend,
        ):
            for name, param in module.named_parameters():
                if "weight" in name and param.dim() > 1:
                    nn.init.kaiming_normal_(param, mode="fan_in")
       
    def gta_attention(self, x, node_axis=1):
        # x.shape: [batch_size; time_length; input_feature_size * input_feature_size]
        x_readout = x.mean(node_axis, keepdim=True)
        x_readout = x * x_readout

        a = x_readout.shape[0]
        b = x_readout.shape[1]
        x_readout = x_readout.reshape(-1, x_readout.shape[2])
        x_embed = self.gta_norm(self.gta_embed(x_readout))
        x_graphattention = (self.gta_attend(x_embed).squeeze()).reshape(a, b)
        x_graphattention = self.HW(x_graphattention.reshape(a, b))
        return (x * (x_graphattention.unsqueeze(-1))).mean(node_axis)

    def multi_head_attention(self, x):
        # x.shape: [time_length * batch_size; input_feature_size; lstm_hidden_size]
        key = self.key_layer(x)
        value = self.value_layer(x)
        query = self.query_layer(x)

        attn_output, attn_output_weights = self.multihead_attn(key, value, query)

        return attn_output, attn_output_weights

    def forward(self, x):
        B, T, C = x.shape  # [batch_size, time_length, input_feature_size]

        # 1. pass input to LSTM; treat each channel as an independent single-feature time series
        x = x.permute(0, 2, 1)  # x.shape: [batch_size; input_feature_size; time_length]
        x = x.reshape(B * C, T, 1)  # x.shape: [batch_size * n_channels; time_length; 1]
        ##########################
        lstm_output, _ = self.lstm(x)
        # lstm_output.shape: [batch_size * input_feature_size; time_length; lstm_hidden_size]
        ##########################
        lstm_output = lstm_output.reshape(B, C, T, self.lstm_output_size)
        # lstm_output.shape: [batch_size; input_feature_size; time_length; lstm_hidden_size]

        # 2. pass lstm_output at each time point to multihead attention to reveal spatial connctions
        lstm_output = lstm_output.permute(2, 0, 1, 3)
        # lstm_output.shape: [time_length; batch_size; input_feature_size; lstm_hidden_size]
        lstm_output = lstm_output.reshape(T * B, C, self.lstm_output_size)
        # lstm_output.shape: [time_length * batch_size; input_feature_size; lstm_hidden_size]
        ##########################
        _, attn_weights = self.multi_head_attention(lstm_output)
        # attn_weights.shape: [time_length * batch_size; input_feature_size; input_feature_size]
        ##########################
        attn_weights = attn_weights.reshape(T, B, C, C)
        # attn_weights.shape: [time_length; batch_size; input_feature_size; input_feature_size]
        attn_weights = attn_weights.permute(1, 0, 2, 3)
        # attn_weights.shape: [batch_size; time_length; input_feature_size; input_feature_size]

        # 3. pass attention weights to a global temporal attention to obrain global graph
        attn_weights = attn_weights.reshape(B, T, -1)
        # attn_weights.shape: [batch_size; time_length; input_feature_size * input_feature_size]
        ##########################
        FC = self.gta_attention(attn_weights)
        # FC.shape: [batch_size; input_feature_size * input_feature_size]
        ##########################

        # 4. Pass learned graph to the classifier to get predictions
        logits = self.clf(FC)
        # logits.shape: [batch_size; n_classes]

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