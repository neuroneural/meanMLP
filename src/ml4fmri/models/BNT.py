# pylint: disable=invalid-name, missing-function-docstring, missing-class-docstring, unused-argument, too-few-public-methods, no-member, too-many-arguments, line-too-long, too-many-instance-attributes
"""
BNT model module

Works with FNC data.

@inproceedings{bnt,
    title={Brain Network Transformer},
    author={Xuan Kan and Wei Dai and Hejie Cui and Zilong Zhang and Ying Guo and Carl Yang},
    booktitle={Advances in Neural Information Processing Systems},
    editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
    year={2022},
    url={https://openreview.net/forum?id=1cJ1cbA6NLN}
}
"""


import numpy as np
import math
from torch.nn.functional import softmax
from torch.nn import Parameter, TransformerEncoderLayer, functional as F
from torch import nn, optim
import torch

from types import SimpleNamespace

from .helper_functions import basic_ce_loss, basic_handle_batch, basic_dataloader, BasicTrainer

class BNT(nn.Module):
    """
    FNC MODEL
    BrainNetworkTransformer for fMRI data from https://doi.org/10.48550/arXiv.2210.06681
    Original implementation: https://github.com/Wayfear/BrainNetworkTransformer
    """
    def __init__(self,
                 input_size,
                 output_size,
        ):
        super().__init__()

        self.lr = 1e-4  # learning rate used in the paper. Defined like this in every model for reference.
        self.weight_decay = 1e-4  # weight decay used by optimizer

        model_cfg = {
            "sizes": [360, 100],  # Note: The input node size should not be included here
            "pooling": [False, True],
            "pos_encoding": "none",  # 'identity', 'none'
            "orthogonal": True,
            "freeze_center": True,
            "project_assignment": True,
            "pos_embed_dim": 360,
            # data shape
            "node_sz": input_size,
            "node_feature_sz": input_size,
            "output_size": output_size,
            # optimizer
            "optimizer": {"lr": 1e-4, "weight_decay": 1e-4},
        }

        def dict_to_namespace(d):
            """Convert nested dict to nested SimpleNamespace"""
            if isinstance(d, dict):
                return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
            return d
        
        model_cfg = dict_to_namespace(model_cfg)

        self.attention_list = nn.ModuleList()
        forward_dim = model_cfg.node_feature_sz

        self.pos_encoding = model_cfg.pos_encoding
        if self.pos_encoding == "identity":
            self.node_identity = nn.Parameter(
                torch.zeros(model_cfg.node_sz, model_cfg.pos_embed_dim),
                requires_grad=True,
            )
            forward_dim = model_cfg.node_sz + model_cfg.pos_embed_dim
            nn.init.kaiming_normal_(self.node_identity)

        sizes = model_cfg.sizes
        sizes[0] = model_cfg.node_sz
        in_sizes = [model_cfg.node_sz] + sizes[:-1]
        do_pooling = model_cfg.pooling
        self.do_pooling = do_pooling

        self.attention_list = nn.ModuleList()
        curr_feat = forward_dim  # may be 53 or 53+pos_embed_dim

        for index, size in enumerate(sizes):
            enc = TransPoolingEncoder(
                input_feature_size=curr_feat,        # previous stage's output feature width
                input_node_num=in_sizes[index],
                hidden_size=1024,
                output_node_num=size,
                pooling=do_pooling[index],
                orthogonal=model_cfg.orthogonal,
                freeze_center=model_cfg.freeze_center,
                project_assignment=model_cfg.project_assignment,
                n_head=4,                            # keep fixed heads
            )
            self.attention_list.append(enc)
            curr_feat = enc.out_feature_size        # <- adopt padded width for next stage

            self.final_feature_size = curr_feat
            self.dim_reduction = nn.Sequential(
                nn.Linear(self.final_feature_size, 8), nn.LeakyReLU(inplace=True)
            )

        self.fc = nn.Sequential(
            nn.Linear(8 * sizes[-1], 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, model_cfg.output_size),
        )

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]

    def get_cluster_centers(self) -> torch.Tensor:
        """
        Get the cluster centers, as computed by the encoder.

        :return: [number of clusters, hidden dimension] Tensor of dtype float
        """
        return self.dec.get_cluster_centers()

    def loss(self, assignments):
        """
        Compute KL loss for the given assignments. Note that not all encoders contain a pooling layer.
        Inputs: assignments: [batch size, number of clusters]
        Output: KL loss
        """
        decs = list(filter(lambda x: x.is_pooling_enabled(), self.attention_list))
        assignments = list(filter(lambda x: x is not None, assignments))
        loss_all = None

        for index, assignment in enumerate(assignments):
            if loss_all is None:
                loss_all = decs[index].loss(assignment)
            else:
                loss_all += decs[index].loss(assignment)
        return loss_all
    
    def forward(self, node_feature: torch.tensor):
        (
            bz,
            _,
            _,
        ) = node_feature.shape

        if self.pos_encoding == "identity":
            pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)
            node_feature = torch.cat([node_feature, pos_emb], dim=-1)

        assignments = []

        for _, atten in enumerate(self.attention_list):
            node_feature, assignment = atten(node_feature)
            assignments.append(assignment)

        node_feature = self.dim_reduction(node_feature)

        node_feature = node_feature.reshape((bz, -1))

        logits = self.fc(node_feature)
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
                                type="FNC",
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

class TransPoolingEncoder(nn.Module):
    """
    Transformer encoder with Pooling mechanism.
    Input : (B, input_node_num, input_feature_size)
    Output: (B, output_node_num, eff_input_size)  # NOTE: eff width (padded) is kept
    """
    def __init__(self,
                 input_feature_size,
                 input_node_num,
                 hidden_size,
                 output_node_num,
                 pooling=True,
                 orthogonal=True,
                 freeze_center=False,
                 project_assignment=True,
                 n_head=4):
        super().__init__()
        self.n_head = n_head
        self.input_size = input_feature_size
        self.eff_input_size = math.ceil(self.input_size / n_head) * n_head  # padded width
        self.pad = self.eff_input_size - self.input_size
        self.out_feature_size = self.eff_input_size  # expose for downstream

        # Run transformer at the padded width
        self.transformer = InterpretableTransformerEncoder(
            d_model=self.eff_input_size,      # ← was input_feature_size
            nhead=n_head,
            dim_feedforward=hidden_size,
            batch_first=True,
        )

        self.pooling = pooling
        if pooling:
            encoder_hidden_size = 32
            # DEC works at the padded width, too
            self.encoder = nn.Sequential(
                nn.Linear(self.eff_input_size * input_node_num, encoder_hidden_size),
                nn.LeakyReLU(inplace=True),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(inplace=True),
                nn.Linear(encoder_hidden_size, self.eff_input_size * input_node_num),
            )
            self.dec = DEC(
                cluster_number=output_node_num,
                hidden_dimension=self.eff_input_size,  # ← was input_feature_size
                encoder=self.encoder,
                orthogonal=orthogonal,
                freeze_center=freeze_center,
                project_assignment=project_assignment,
            )

    def is_pooling_enabled(self):
        return self.pooling

    def forward(self, x):  # x: (B, N, input_size)
        if self.pad:
            # pad last dim: (left=0, right=self.pad)
            x = F.pad(x, (0, self.pad))                 # (B, N, eff_input_size)
        x = self.transformer(x)                         # (B, N, eff_input_size)
        if self.pooling:
            x, assignment = self.dec(x)                 # (B, out_nodes, eff_input_size)
            return x, assignment
        return x, None

    def get_attention_weights(self):
        return self.transformer.get_attention_weights()

    def loss(self, assignment):
        return self.dec.loss(assignment)

    
class DEC(nn.Module):
    def __init__(
        self,
        cluster_number: int,
        hidden_dimension: int,
        encoder: torch.nn.Module,
        alpha: float = 1.0,
        orthogonal=True,
        freeze_center=True,
        project_assignment=True,
    ):
        """
        Module which holds all the moving parts of the DEC algorithm, as described in
        Xie/Girshick/Farhadi; this includes the AutoEncoder stage and the ClusterAssignment stage.

        :param cluster_number: number of clusters
        :param hidden_dimension: hidden dimension, output of the encoder
        :param encoder: encoder to use
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        """
        super(DEC, self).__init__()
        self.encoder = encoder
        self.hidden_dimension = hidden_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.assignment = ClusterAssignment(
            cluster_number,
            self.hidden_dimension,
            alpha,
            orthogonal=orthogonal,
            freeze_center=freeze_center,
            project_assignment=project_assignment,
        )

        self.loss_fn = nn.KLDivLoss(reduction='batchmean')

    def forward(self, batch: torch.Tensor):
        """
        Compute the cluster assignment using the ClusterAssignment after running the batch
        through the encoder part of the associated AutoEncoder module.

        :param batch: [batch size, embedding dimension] FloatTensor
        :return: [batch size, number of clusters] FloatTensor
        """
        node_num = batch.size(1)
        batch_size = batch.size(0)

        # [batch size, embedding dimension]
        flattened_batch = batch.view(batch_size, -1)
        encoded = self.encoder(flattened_batch)
        # [batch size * node_num, hidden dimension]
        encoded = encoded.view(batch_size * node_num, -1)
        # [batch size * node_num, cluster_number]
        assignment = self.assignment(encoded)
        # [batch size, node_num, cluster_number]
        assignment = assignment.view(batch_size, node_num, -1)
        # [batch size, node_num, hidden dimension]
        encoded = encoded.view(batch_size, node_num, -1)
        # Multiply the encoded vectors by the cluster assignment to get the final node representations
        # [batch size, cluster_number, hidden dimension]
        node_repr = torch.bmm(assignment.transpose(1, 2), encoded)
        return node_repr, assignment

    def target_distribution(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
        Xie/Girshick/Farhadi; this is used the KL-divergence loss function.

        :param batch: [batch size, number of clusters] Tensor of dtype float
        :return: [batch size, number of clusters] Tensor of dtype float
        """
        weight = (batch**2) / torch.sum(batch, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def loss(self, assignment):
        flat_q = assignment.view(-1, assignment.size(-1))       # q
        p = self.target_distribution(flat_q).detach()           # p (no grad)
        return self.loss_fn(flat_q.log(), p)

    def get_cluster_centers(self) -> torch.Tensor:
        """
        Get the cluster centers, as computed by the encoder.

        :return: [number of clusters, hidden dimension] Tensor of dtype float
        """
        return self.assignment.get_cluster_centers()


class ClusterAssignment(nn.Module):
    def __init__(
        self,
        cluster_number: int,
        embedding_dimension: int,
        alpha: float = 1.0,
        cluster_centers = None,
        orthogonal=True,
        freeze_center=True,
        project_assignment=True,
    ) -> None:
        """
        Module to handle the soft assignment, for a description see in 3.1.1. in Xie/Girshick/Farhadi,
        where the Student's t-distribution is used measure similarity between feature vector and each
        cluster centroid.

        :param cluster_number: number of clusters
        :param embedding_dimension: embedding dimension of feature vectors
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        :param cluster_centers: clusters centers to initialise, if None then use Xavier uniform
        """
        super(ClusterAssignment, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.project_assignment = project_assignment
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)

        else:
            initial_cluster_centers = cluster_centers

        if orthogonal:
            orthogonal_cluster_centers = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float
            )
            orthogonal_cluster_centers[0] = initial_cluster_centers[0]
            for i in range(1, cluster_number):
                project = 0
                for j in range(i):
                    project += self.project(
                        initial_cluster_centers[j], initial_cluster_centers[i]
                    )
                initial_cluster_centers[i] -= project
                orthogonal_cluster_centers[i] = initial_cluster_centers[i] / torch.norm(
                    initial_cluster_centers[i], p=2
                )

            initial_cluster_centers = orthogonal_cluster_centers

        self.cluster_centers = Parameter(
            initial_cluster_centers, requires_grad=(not freeze_center)
        )

    @staticmethod
    def project(u, v):
        return (torch.dot(u, v) / torch.dot(u, u)) * u

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the soft assignment for a batch of feature vectors, returning a batch of assignments
        for each cluster.

        :param batch: FloatTensor of [batch size, embedding dimension]
        :return: FloatTensor [batch size, number of clusters]
        """

        if self.project_assignment:
            assignment = batch @ self.cluster_centers.T
            # prove
            assignment = torch.pow(assignment, 2)

            norm = torch.norm(self.cluster_centers, p=2, dim=-1)
            soft_assign = assignment / norm
            return softmax(soft_assign, dim=-1)

        else:
            norm_squared = torch.sum(
                (batch.unsqueeze(1) - self.cluster_centers) ** 2, 2
            )
            numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
            power = float(self.alpha + 1) / 2
            numerator = numerator**power
            return numerator / torch.sum(numerator, dim=1, keepdim=True)

    def get_cluster_centers(self) -> torch.Tensor:
        """
        Get the cluster centers.

        :return: FloatTensor [number of clusters, embedding dimension]
        """
        return self.cluster_centers


class InterpretableTransformerEncoder(TransformerEncoderLayer):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        batch_first=False,
        norm_first=False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            device,
            dtype,
        )
        self.attention_weights = None

    def _sa_block(
        self,
        x,
        attn_mask,
        key_padding_mask,
        is_causal: bool = False,
    ):
        x, weights = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        self.attention_weights = weights
        return self.dropout1(x)

    def get_attention_weights(self):
        return self.attention_weights

    