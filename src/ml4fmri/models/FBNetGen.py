# pylint: disable=invalid-name, missing-function-docstring, missing-class-docstring, unused-argument, too-few-public-methods, no-member, too-many-arguments, line-too-long, too-many-instance-attributes, too-many-locals
"""
FBNetGen model module

Works simultaneously with time series and FNC data.

@inproceedings{fbnetgen,
    title={{FBNETGEN}: Task-aware {GNN}-based f{MRI} Analysis via Functional Brain Network Generation},
    author={Xuan Kan and Hejie Cui and Joshua Lukemire and Ying Guo and Carl Yang},
    booktitle={Medical Imaging with Deep Learning},
    year={2022},
    url={https://openreview.net/forum?id=oWFphg2IKon}
}
"""

from types import SimpleNamespace
import numpy as np

from torch.nn.functional import softmax, cross_entropy
from torch import nn, optim
from torch.nn import Conv1d, MaxPool1d, Linear, GRU
from torch import nn
import torch

from .helper_functions import basic_handle_batch, basic_dataloader, BasicTrainer

class FBNetGen(nn.Module):
    """
    TIME SERIES + FNC model
    FBNetGen model for fMRI data from https://doi.org/10.48550/arXiv.2205.12465
    Original code: https://github.com/Wayfear/BrainNetworkTransformer

    Expected input shape: [batch_size, time_length, input_feature_size] for time series, [batch_size, input_feature_size, input_feature_size] for FNC.
    Output: [batch_size, n_classes], loss_load = {"logits": logits}
    """
    def __init__(self,
                 input_size,
                 output_size,
        ):
        """
        Initialize FBNetGen model.
        Args:
            input_size (int): Size of the vector at each time step in the input time series. \
                Common to all models, for FNC models it is the number of nodes (width or length) in the FNC matrix.
            output_size (int): Number of classes for classification. \
                Common to all models.
        """

        super().__init__()

        self.lr = 1e-4 # learning rate used in the paper. Defined like this in every model for reference.
        self.weight_decay = 1e-4 # weight decay used by the optimizer

        # FBNetGen loss configuration
        group_loss = True
        sparsity_loss = True
        sparsity_loss_weight = 0.1
        self.criterion = FBNetGenLoss(group_loss, sparsity_loss, sparsity_loss_weight)

        model_cfg = {
            "extractor_type": "gru",
            "embedding_size": 16,
            "window_size": 4,
            "cnn_pool_size": 16,
            "graph_generation": "product",  # product or linear
            "num_gru_layers": 4,
            "dropout": 0.5,
            # loss
            "group_loss": True,
            "sparsity_loss": True,
            "sparsity_loss_weight": 1e-4,
            # data shape
            "timeseries_sz": None, # some FBNetGen modules need time length to initialize (extractor_type = 'cnn')
            "node_sz": input_size,
            "node_feature_sz": input_size,
            "output_size": output_size,
            # optimizer
            "optimizer": {"lr": self.lr, "weight_decay": self.weight_decay},
        }

        def dict_to_namespace(d):
            """Convert nested dict to nested SimpleNamespace"""
            if isinstance(d, dict):
                return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
            return d
        
        model_cfg = dict_to_namespace(model_cfg)

        assert model_cfg.extractor_type in ["cnn", "gru"]
        assert model_cfg.graph_generation in ["linear", "product"]

        self.graph_generation = model_cfg.graph_generation
        if model_cfg.extractor_type == "cnn":
            self.extract = ConvKRegion(
                out_size=model_cfg.embedding_size,
                kernel_size=model_cfg.window_size,
                time_series=model_cfg.timeseries_sz,
            )
        elif model_cfg.extractor_type == "gru":
            self.extract = GruKRegion(
                out_size=model_cfg.embedding_size,
                kernel_size=model_cfg.window_size,
                layers=model_cfg.num_gru_layers,
            )
        if self.graph_generation == "linear":
            self.emb2graph = Embed2GraphByLinear(
                model_cfg.embedding_size, roi_num=model_cfg.node_sz
            )
        elif self.graph_generation == "product":
            self.emb2graph = Embed2GraphByProduct(
                model_cfg.embedding_size, roi_num=model_cfg.node_sz
            )

        self.predictor = GNNPredictor(
            model_cfg.node_feature_sz,
            roi_num=model_cfg.node_sz,
            n_classes=model_cfg.output_size,
        )

    def forward(self, time_series, node_feature):
        """
        Args:
            time_series: fMRI time series, shape (B, T, D)
            node_feature: FNC matrices derived from time series, shape (B, D, D)
        """
        time_series = time_series.transpose(1, 2)

        x = self.extract(time_series)
        x = softmax(x, dim=-1)
        m = self.emb2graph(x)

        logits = self.predictor(m, node_feature)
        loss_load = {
            "logits": logits,
            "learned_matrix": m
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
        optimizer = optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=self.weight_decay,
        )

        return optimizer

    @staticmethod
    def prepare_dataloader(data,
                           labels,
                           batch_size: int = 64,
                           shuffle: bool = True,
                           window_size = 4):
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
        # FBNetGen needs time series length to be divisible by window_size hyperparam
        T = data.shape[1]
        if T % window_size != 0:
            rem = T % window_size
            data = data[:, :-rem, :]

        return basic_dataloader(data,
                                labels,
                                type="ALL",
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
    

class FBNetGenLoss:
    def __init__(self, group_loss=True, sparsity_loss=True, sparsity_loss_weight=0.1):
        self.group_loss = group_loss
        self.sparsity_loss = sparsity_loss
        self.sparsity_loss_weight = sparsity_loss_weight

    def __call__(self, loss_load, target):
        logits = loss_load["logits"]
        learned_matrix = loss_load["learned_matrix"]

        ce_loss = cross_entropy(logits, target)
        loss_log = {"ce_loss": ce_loss.detach().cpu().item()}
        loss = 2 * ce_loss

        if self.group_loss:
            group_loss = 2 * self.intra_loss(target, learned_matrix) + self.inner_loss(target, learned_matrix)
            loss_log["group_loss"] = group_loss.detach().cpu().item()
            loss += group_loss

        if self.sparsity_loss:
            sparsity_loss = learned_matrix.abs().mean()
            loss_log["sparsity_loss"] = sparsity_loss.detach().cpu().item()
            loss += self.sparsity_loss_weight * sparsity_loss

        return loss, loss_log

    def inner_loss(self, label, matrices):
        loss = 0

        if torch.sum(label == 0) > 1:
            loss += torch.mean(torch.var(matrices[label == 0], dim=0))

        if torch.sum(label == 1) > 1:
            loss += torch.mean(torch.var(matrices[label == 1], dim=0))

        return loss

    def intra_loss(self, label, matrices):
        a, b = None, None

        if torch.sum(label == 0) > 0:
            a = torch.mean(matrices[label == 0], dim=0)

        if torch.sum(label == 1) > 0:
            b = torch.mean(matrices[label == 1], dim=0)

        if a is not None and b is not None:
            return 1 - torch.mean(torch.pow(a - b, 2))

        return 0
    

class GruKRegion(nn.Module):
    def __init__(self, kernel_size=8, layers=4, out_size=8, dropout=0.5):
        super().__init__()
        self.gru = GRU(
            kernel_size, kernel_size, layers, bidirectional=True, batch_first=True
        )

        self.kernel_size = kernel_size

        self.linear = nn.Sequential(
            nn.Dropout(dropout),
            Linear(kernel_size * 2, kernel_size),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(kernel_size, out_size),
        )

    def forward(self, raw):
        b, k, _ = raw.shape

        x = raw.reshape(b * k, -1, self.kernel_size)

        x, _ = self.gru(x)

        x = x[:, -1, :]

        x = x.view((b, k, -1))

        x = self.linear(x)

        return x


class ConvKRegion(nn.Module):
    def __init__(self, k=1, out_size=8, kernel_size=8, pool_size=16, time_series=512):
        super().__init__()
        self.conv1 = Conv1d(
            in_channels=k, out_channels=32, kernel_size=kernel_size, stride=2
        )

        output_dim_1 = (time_series - kernel_size) // 2 + 1

        self.conv2 = Conv1d(in_channels=32, out_channels=32, kernel_size=8)
        output_dim_2 = output_dim_1 - 8 + 1
        self.conv3 = Conv1d(in_channels=32, out_channels=16, kernel_size=8)
        output_dim_3 = output_dim_2 - 8 + 1
        self.max_pool1 = MaxPool1d(pool_size)
        output_dim_4 = output_dim_3 // pool_size * 16
        self.in0 = nn.InstanceNorm1d(time_series)
        self.in1 = nn.BatchNorm1d(32)
        self.in2 = nn.BatchNorm1d(32)
        self.in3 = nn.BatchNorm1d(16)

        self.linear = nn.Sequential(
            Linear(output_dim_4, 32),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(32, out_size),
        )

    def forward(self, x):
        b, k, d = x.shape

        x = torch.transpose(x, 1, 2)

        x = self.in0(x)

        x = torch.transpose(x, 1, 2)
        x = x.contiguous()

        x = x.view((b * k, 1, d))

        x = self.conv1(x)

        x = self.in1(x)
        x = self.conv2(x)

        x = self.in2(x)
        x = self.conv3(x)

        x = self.in3(x)
        x = self.max_pool1(x)

        x = x.view((b, k, -1))

        x = self.linear(x)

        return x


class Embed2GraphByProduct(nn.Module):
    def __init__(self, input_dim, roi_num=264):
        super().__init__()

    def forward(self, x):
        return x @ x.transpose(1, 2)


class GNNPredictor(nn.Module):
    def __init__(self, node_input_dim, roi_num=360, n_classes=2):
        super().__init__()
        inner_dim = roi_num
        self.roi_num = roi_num
        self.gcn = nn.Sequential(
            nn.Linear(node_input_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(inner_dim, inner_dim),
        )
        self.bn1 = torch.nn.BatchNorm1d(inner_dim)

        self.gcn1 = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn2 = torch.nn.BatchNorm1d(inner_dim)
        self.gcn2 = nn.Sequential(
            nn.Linear(inner_dim, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 8),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn3 = torch.nn.BatchNorm1d(inner_dim)

        self.fcn = nn.Sequential(
            nn.Linear(8 * roi_num, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, n_classes),
        )

    def forward(self, m, node_feature):
        bz = m.shape[0]

        x = torch.einsum("ijk,ijp->ijp", m, node_feature)

        x = self.gcn(x)

        x = x.reshape((bz * self.roi_num, -1))
        x = self.bn1(x)
        x = x.reshape((bz, self.roi_num, -1))

        x = torch.einsum("ijk,ijp->ijp", m, x)

        x = self.gcn1(x)

        x = x.reshape((bz * self.roi_num, -1))
        x = self.bn2(x)
        x = x.reshape((bz, self.roi_num, -1))

        x = torch.einsum("ijk,ijp->ijp", m, x)

        x = self.gcn2(x)

        x = self.bn3(x)

        x = x.view(bz, -1)

        return self.fcn(x)
    


class Embed2GraphByLinear(nn.Module):
    def __init__(self, input_dim, roi_num=360):
        super().__init__()

        self.fc_out = nn.Linear(input_dim * 2, input_dim)
        self.fc_cat = nn.Linear(input_dim, 1)

        def encode_onehot(labels):
            classes = set(labels)
            classes_dict = {
                c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)
            }
            labels_onehot = np.array(
                list(map(classes_dict.get, labels)), dtype=np.int32
            )
            return labels_onehot

        off_diag = np.ones([roi_num, roi_num])
        rel_rec = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec).cuda()
        self.rel_send = torch.FloatTensor(rel_send).cuda()

    def forward(self, x):
        batch_sz, region_num, _ = x.shape
        receivers = torch.matmul(self.rel_rec, x)

        senders = torch.matmul(self.rel_send, x)
        x = torch.cat([senders, receivers], dim=2)
        x = torch.relu(self.fc_out(x))
        x = self.fc_cat(x)

        x = torch.relu(x)

        m = torch.reshape(x, (batch_sz, region_num, region_num, -1))
        return m
