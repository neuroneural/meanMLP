# pylint: disable=invalid-name, missing-function-docstring, no-member
""" 
Transformer model module.

Transformer-encoder based classifier, applicable to any time series.
Was used in:

@article{meanMLP,
    title = {A simple but tough-to-beat baseline for fMRI time-series classification},
    journal = {NeuroImage},
    volume = {303},
    pages = {120909},
    year = {2024},
    issn = {1053-8119},
    doi = {10.1016/j.neuroimage.2024.120909},
    url = {https://www.sciencedirect.com/science/article/pii/S1053811924004063},
    author = {Pavel Popov and Usman Mahmood and Zening Fu and Carl Yang and Vince Calhoun and Sergey Plis},
}
"""
import math
import torch
from torch import nn

from .helper_functions import basic_ce_loss, basic_handle_batch, basic_dataloader, basic_Adam_optimizer, BasicTrainer


class Transformer(nn.Module):
    """
    TIME SERIES MODEL
    Transformer model for fMRI data from https://doi.org/10.1016/j.neuroimage.2024.120909.
    Expected input shape: [batch_size, time_length, input_feature_size].
    Output: [batch_size, n_classes], loss_load = {"logits": logits}.
    """
    
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 head_hidden_size: int = 20,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 lr: float = 0.002
    ):
        """
        Initialize Transformer model.
        Args:
            input_size (int): Size of the vector at each time step in the input time series.
            output_size (int): Number of classes for classification.
            head_hidden_size (int, hyperparameter): Size of each attention head. Defaults to 20.
            num_heads (int, hyperparameter): Number of attention heads. Defaults to 3.
            num_layers (int, hyperparameter): Number of transformer encoder layers. Defaults to 2.
            dropout (float, hyperparameter): Dropout rate for classifier. Defaults to 0.3.
            lr (float, hyperparameter): Learning rate for the optimizer. Defaults to 0.002.
                Common to all models. Isn't used by the model per se, presented as a reference LR from the paper.
        """

        super().__init__()
        self.lr = lr  # learning rate used in the paper. Defined like this in every model for reference.

        hidden_size = head_hidden_size * num_heads

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.input_embed = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)

        self.clf = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        input_embed = self.input_embed(x)
        input_embed = self.pos_encoder(input_embed)

        tf_output = self.transformer(input_embed)
        tf_output = tf_output[:, 0, :]

        logits = self.clf(tf_output)

        return logits, {"logits": logits}


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

        loss, loss_log = basic_ce_loss(loss_load["logits"], targets)
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
