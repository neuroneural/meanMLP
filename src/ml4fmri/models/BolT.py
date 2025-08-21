# pylint: disable=invalid-name, missing-function-docstring
"""
BolT model module.

Sliding window transformer model for fMRI time series analysis. 
Fairly good, one of the few models that outperformed meanMLP on large datasets.

@article{bolT,
    title = {BolT: Fused window transformers for fMRI time series analysis},
    journal = {Medical Image Analysis},
    volume = {88},
    pages = {102841},
    year = {2023},
    issn = {1361-8415},
    doi = {10.1016/j.media.2023.102841},
    url = {https://www.sciencedirect.com/science/article/pii/S1361841523001019},
    author = {Hasan A. Bedel and Irmak Sivgin and Onat Dalmaz and Salman U.H. Dar and Tolga Çukur},
}

"""

import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn.functional import cross_entropy
from torch.nn.init import trunc_normal_

from .helper_functions import (
    basic_handle_batch,
    basic_dataloader,
    basic_Adam_optimizer,
    BasicTrainer,
)


class BolT(nn.Module):
    """
    TIME SERIES MODEL
    BolT model for fMRI data from https://doi.org/10.1016/j.media.2023.102841.
    Original: https://github.com/icon-lab/BolT
    This version focuses on classification only (analysis/interpretability code removed).

    Expected input shape: [batch_size, time_length, input_feature_size].
    Output: logits [batch_size, n_classes], loss_load = {"logits": logits, "cls": cls}.
    """

    def __init__(self, input_size, output_size):
        """
        Initialize BolT model.
        Majority of hyperparameters are defined in the hyperParams variable.

        Args:
            input_size (int): Size of the vector at each time step in the input time series.
            output_size (int): Number of classes for classification.
        """
        super().__init__()
        self.lr = 2e-4 # learning rate used in the paper. Defined like this in every model for reference.
        self.lambdaCons = 1  # used in loss calculation

        hyperParams = {
            # oneCycleLR hyperparameters from the original code (not used here but kept for reference)
            "lr": 2e-4,
            "minLr": 2e-5,
            "maxLr": 4e-4,
            # BolT core
            "nOfLayers": 4,
            "dim": input_size,
            "numHeads": 36,
            "headDim": 20,
            "windowSize": 20,
            "shiftCoeff": 2.0 / 5.0,
            "fringeCoeff": 2,
            "focalRule": "expand",
            "mlpRatio": 1.0,
            "attentionBias": True,
            "drop": 0.1,
            "attnDrop": 0.1,
            # Loss param
            "lambdaCons": 1,
        }
        hyperParams = SimpleNamespace(**hyperParams)
        self.hyperParams = hyperParams

        dim = input_size
        n_classes = output_size

        self.clsToken = nn.Parameter(torch.zeros(1, 1, dim))

        shiftSize = int(hyperParams.windowSize * hyperParams.shiftCoeff)
        self.shiftSize = shiftSize

        # Build the stack of window-attention blocks
        blocks = []
        for i in range(hyperParams.nOfLayers):
            if hyperParams.focalRule == "expand":
                receptiveSize = hyperParams.windowSize + math.ceil(
                    hyperParams.windowSize * 2 * i * hyperParams.fringeCoeff * (1 - hyperParams.shiftCoeff)
                )
            else:  # "fixed"
                receptiveSize = hyperParams.windowSize + math.ceil(
                    hyperParams.windowSize * 2 * 1 * hyperParams.fringeCoeff * (1 - hyperParams.shiftCoeff)
                )

            blocks.append(
                BolTransformerBlock(
                    dim=dim,
                    numHeads=hyperParams.numHeads,
                    headDim=hyperParams.headDim,
                    windowSize=hyperParams.windowSize,
                    receptiveSize=receptiveSize,
                    shiftSize=shiftSize,
                    mlpRatio=hyperParams.mlpRatio,
                    attentionBias=hyperParams.attentionBias,
                    drop=hyperParams.drop,
                    attnDrop=hyperParams.attnDrop,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        self.encoder_postNorm = nn.LayerNorm(dim)
        self.classifierHead = nn.Linear(dim, n_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize CLS token (simple normal, as in the original)
        torch.nn.init.normal_(self.clsToken, std=1.0)

    def forward(self, x):
        # x: [B, T, C]
        B, T, _ = x.shape
        nW = (T - self.hyperParams.windowSize) // self.shiftSize + 1
        cls = self.clsToken.expand(B, nW, -1)

        for block in self.blocks:
            x, cls = block(x, cls)

        cls = self.encoder_postNorm(cls)
        logits = self.classifierHead(cls.mean(dim=1))
        return logits, {"logits": logits, "cls": cls}

    # ==== Helper functions for model training and evaluation ====

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
        ce_loss = cross_entropy(loss_load["logits"], targets)

        cls = loss_load["cls"]
        clsLoss = torch.mean(torch.square(cls - cls.mean(dim=1, keepdims=True)))

        loss = ce_loss + clsLoss * self.lambdaCons
        loss_log = {
            "CE_loss": float(ce_loss.detach().cpu().item()),
            "cls_loss": float(clsLoss.detach().cpu().item()),
        }
        return loss, loss_log

    def handle_batch(self, batch):
        """
        Standard batch handling routine for models.
        Returns loss for backprop and a dictionary of classification metrics and losses for logs.

        Args:
            batch (tuple): A batch containing the time series data and labels.

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
    def prepare_dataloader(data, labels, batch_size: int = 64, shuffle: bool = True):
        """
        Returns torch DataLoader that produces appropriate batches for the model (can pass to `handle_batch`).

        Args:
            data (array-like): Time series data of shape (B, T, D).
            labels (array-like): Class labels for the data.
            batch_size (int, optional): Batch size for the DataLoader. Defaults to 64.
            shuffle (bool, optional): Whether to shuffle batching in the DataLoader. Defaults to True.

        Returns
        -------
            DataLoader
        """
        return basic_dataloader(
            data,
            labels,
            type="TS",
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def train_model(
        self,
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
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation set.
            test_loader (DataLoader): DataLoader for the test set.
            epochs (int, optional): Number of training epochs. Defaults to 200.
            lr (float, optional): Optimizer learning rate (default: use model's self.lr).
            device (str, optional): Device: "cuda", "mps", or "cpu" (auto-detect if None).
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


# ==== BolT submodules and utilities ====

def windowBoldSignal(boldSignal, windowLength, stride):
    """
    Vectorized windowing via Tensor.unfold.

    boldSignal: (B, N, T)
    return:
        windowed: (B, nW, N, windowLength)
        samplingEndPoints: list[int]  # end indices for each window
    """
    x = boldSignal.unfold(dimension=2, size=windowLength, step=stride)  # (B, N, nW, W)
    x = x.permute(0, 2, 1, 3).contiguous()  # (B, nW, N, W)
    B, nW, N, W = x.shape
    samplingEndPoints = list(range(W, W + nW * stride, stride))
    return x, samplingEndPoints


class FeedForward(nn.Module):
    def __init__(self, dim, mult=1, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        activation = nn.GELU()
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim,
        windowSize,
        receptiveSize,
        numHeads,
        headDim=20,
        attentionBias=True,
        qkvBias=True,
        attnDrop=0.0,
        projDrop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.windowSize = windowSize  # N
        self.receptiveSize = receptiveSize  # M
        self.numHeads = numHeads
        head_dim = headDim
        self.scale = head_dim ** -0.5
        self.attentionBias = attentionBias

        # relative position bias
        maxDisparity = windowSize - 1 + (receptiveSize - windowSize) // 2
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(2 * maxDisparity + 1, numHeads)
        )  # (2*maxDisp+1, nH)

        self.cls_bias_sequence_up = nn.Parameter(torch.zeros((1, numHeads, 1, receptiveSize)))
        self.cls_bias_sequence_down = nn.Parameter(torch.zeros(1, numHeads, windowSize, 1))
        self.cls_bias_self = nn.Parameter(torch.zeros((1, numHeads, 1, 1)))

        # precompute pairwise relative position indices for window tokens
        coords_x = torch.arange(self.windowSize)  # N
        coords_x_ = torch.arange(self.receptiveSize) - (self.receptiveSize - self.windowSize) // 2  # M
        relative_coords = coords_x[:, None] - coords_x_[None, :]  # (N, M)
        relative_coords[:, :] += maxDisparity  # shift to start from 0
        relative_position_index = relative_coords  # (N, M)
        self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Linear(dim, head_dim * numHeads, bias=qkvBias)
        self.kv = nn.Linear(dim, 2 * head_dim * numHeads, bias=qkvBias)

        self.attnDrop = nn.Dropout(attnDrop)
        self.proj = nn.Linear(head_dim * numHeads, dim)
        self.projDrop = nn.Dropout(projDrop)

        # init learned biases
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        trunc_normal_(self.cls_bias_sequence_up, std=0.02)
        trunc_normal_(self.cls_bias_sequence_down, std=0.02)
        trunc_normal_(self.cls_bias_self, std=0.02)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, x_, mask, nW):
        """
        x  : (B*nW, 1+windowSize, C)     # queries
        x_ : (B*nW, 1+receptiveSize, C)  # keys/values
        mask: (mask_left, mask_right), each (maskCount, 1+windowSize, 1+receptiveSize), bool
        nW : number of windows

        returns: (B*nW, 1+windowSize, C)
        """
        # shapes / unpack
        BnW, n_tok, C = x.shape  # n_tok = 1 + N
        _, m_tok, _ = x_.shape   # m_tok = 1 + M
        N = n_tok - 1
        M = m_tok - 1
        H = self.numHeads
        B = BnW // nW

        # projections & split heads
        q = self.q(x)  # (B*nW, n_tok, H*d)
        k, v = self.kv(x_).chunk(2, dim=-1)
        d = (self.q.out_features // H)  # per-head dim
        q = q.view(BnW, n_tok, H, d).permute(0, 2, 1, 3).contiguous()   # (B*nW, H, n_tok, d)
        k = k.view(BnW, m_tok, H, d).permute(0, 2, 1, 3).contiguous()   # (B*nW, H, m_tok, d)
        v = v.view(BnW, m_tok, H, d).permute(0, 2, 1, 3).contiguous()   # (B*nW, H, m_tok, d)

        # attention logits
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B*nW, H, n_tok, m_tok)

        # relative and CLS biases
        if self.attentionBias:
            rel = self.relative_position_bias_table[self.relative_position_index.view(-1)]
            rel = rel.view(N, M, H).permute(2, 0, 1).contiguous()  # (H, N, M)
            rel = rel.to(dtype=attn.dtype, device=attn.device)
            attn[:, :, 1:, 1:] = attn[:, :, 1:, 1:] + rel.unsqueeze(0)  # (1, H, N, M)
            attn[:, :, :1, :1] = attn[:, :, :1, :1] + self.cls_bias_self.to(attn.dtype)
            attn[:, :, :1, 1:] = attn[:, :, :1, 1:] + self.cls_bias_sequence_up.to(attn.dtype)
            attn[:, :, 1:, :1] = attn[:, :, 1:, :1] + self.cls_bias_sequence_down.to(attn.dtype)

        # masking first/last windows (broadcasted boolean masks, no .repeat)
        mask_left, mask_right = mask  # (maskCount, n_tok, m_tok), bool
        attn = attn.view(B, nW, H, n_tok, m_tok)
        maskCount = min(mask_left.shape[0], attn.shape[1])
        if maskCount > 0:
            neg_inf = max_neg_value(attn)
            attn[:, :maskCount].masked_fill_(mask_left[:maskCount].unsqueeze(0).unsqueeze(2), neg_inf)
            attn[:, -maskCount:].masked_fill_(mask_right[-maskCount:].unsqueeze(0).unsqueeze(2), neg_inf)
        attn = attn.view(BnW, H, n_tok, m_tok)

        # softmax, dropout
        attn = self.softmax(attn)
        attn = self.attnDrop(attn)

        # apply attention to values, merge heads, project
        out = torch.matmul(attn, v)                   # (B*nW, H, n_tok, d)
        out = out.permute(0, 2, 1, 3).contiguous()    # (B*nW, n_tok, H, d)
        out = out.view(BnW, n_tok, H * d)             # (B*nW, n_tok, H*d)
        out = self.proj(out)
        out = self.projDrop(out)
        return out


class FusedWindowTransformer(nn.Module):
    def __init__(
        self,
        dim,
        windowSize,
        shiftSize,
        receptiveSize,
        numHeads,
        headDim,
        mlpRatio,
        attentionBias,
        drop,
        attnDrop,
    ):
        super().__init__()
        self.attention = WindowAttention(
            dim=dim,
            windowSize=windowSize,
            receptiveSize=receptiveSize,
            numHeads=numHeads,
            headDim=headDim,
            attentionBias=attentionBias,
            attnDrop=attnDrop,
            projDrop=drop,
        )
        self.mlp = FeedForward(dim=dim, mult=mlpRatio, dropout=drop)
        self.attn_norm = nn.LayerNorm(dim)
        self.mlp_norm = nn.LayerNorm(dim)
        self.shiftSize = shiftSize

    def forward(self, x, cls, windowX, windowX_, mask, nW):
        """
        x : (B, T, C)
        cls : (B, nW, C)
        windowX  : (B*nW, 1+windowSize, C)
        windowX_ : (B*nW, 1+receptiveSize, C)
        mask : (mask_left, mask_right)
        nW : number of windows

        returns:
            xTrans  : (B, T, C)
            clsTrans: (B, nW, C)
        """
        # window attention over (B*nW, ·, ·)
        B, T, C = x.shape
        windowXTrans = self.attention(self.attn_norm(windowX), self.attn_norm(windowX_), mask, nW)
        clsTrans = windowXTrans[:, :1]   # (B*nW, 1, C)
        xTrans = windowXTrans[:, 1:]     # (B*nW, windowSize, C)

        clsTrans = clsTrans.view(B, nW, 1, C).squeeze(2)   # (B, nW, C)
        xTrans   = xTrans.view(B, nW, xTrans.size(1), C)   # (B, nW, windowSize, C)

        # overlap-add windows back into sequence
        xTrans = self.gatherWindows(xTrans, x.shape[1], self.shiftSize)

        # residual + MLP
        clsTrans = clsTrans + cls
        xTrans = xTrans + x
        xTrans = xTrans + self.mlp(self.mlp_norm(xTrans))
        clsTrans = clsTrans + self.mlp(self.mlp_norm(clsTrans))
        return xTrans, clsTrans

    def gatherWindows(self, windowedX, dynamicLength, shiftSize):
        """
        windowedX : (B, nW, windowLength, C)
        returns   : (B, dynamicLength, C)
        """
        B, nW, windowLength, C = windowedX.shape
        device = windowedX.device
        dtype = windowedX.dtype

        destination = torch.zeros((B, dynamicLength, C), device=device, dtype=dtype)
        scalerDestination = torch.zeros_like(destination)

        # indices of each window laid out on the time axis
        # shape: (nW, windowLength)
        idx = (
            torch.arange(windowLength, device=device).unsqueeze(0)
            + torch.arange(nW, device=device).unsqueeze(1) * shiftSize
        )
        idx = idx.view(1, nW, windowLength, 1).expand(B, nW, windowLength, C)  # (B, nW, windowLength, C)

        # Use reshape (safe for non-contiguous) instead of view
        src = windowedX.reshape(B, nW * windowLength, C)
        idx = idx.reshape(B, nW * windowLength, C)

        destination.scatter_add_(dim=1, index=idx, src=src)

        scalerSrc = torch.ones((B, nW * windowLength, C), device=device, dtype=dtype)
        scalerDestination.scatter_add_(dim=1, index=idx, src=scalerSrc)

        destination = destination / scalerDestination.clamp_min_(1)
        return destination


class BolTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        numHeads,
        headDim,
        windowSize,
        receptiveSize,
        shiftSize,
        mlpRatio=1.0,
        drop=0.0,
        attnDrop=0.0,
        attentionBias=True,
    ):
        super().__init__()
        assert (receptiveSize - windowSize) % 2 == 0

        self.transformer = FusedWindowTransformer(
            dim=dim,
            windowSize=windowSize,
            shiftSize=shiftSize,
            receptiveSize=receptiveSize,
            numHeads=numHeads,
            headDim=headDim,
            mlpRatio=mlpRatio,
            attentionBias=attentionBias,
            drop=drop,
            attnDrop=attnDrop,
        )

        self.windowSize = windowSize
        self.receptiveSize = receptiveSize
        self.shiftSize = shiftSize
        self.remainder = (self.receptiveSize - self.windowSize) // 2

        # Precompute masks for non-matching query/key positions (boolean, broadcasted later)
        maskCount = self.remainder // shiftSize + 1
        mask_left = torch.zeros(maskCount, self.windowSize + 1, self.receptiveSize + 1, dtype=torch.bool)
        mask_right = torch.zeros_like(mask_left)
        for i in range(maskCount):
            if self.remainder > 0:
                mask_left[i, :, 1 : 1 + self.remainder - self.shiftSize * i] = True
                if (-self.remainder + self.shiftSize * i) > 0:
                    mask_right[maskCount - 1 - i, :, -self.remainder + self.shiftSize * i :] = True

        self.register_buffer("mask_left", mask_left)
        self.register_buffer("mask_right", mask_right)

    def forward(self, x, cls):
        """
        x   : (B, dynamicLength, C)
        cls : (B, nW, C)

        returns:
            fusedX_trans : (B, dynamicLength, C)
            cls_trans    : (B, nW, C)
        """
        B, _, C = x.shape
        device = x.device

        # Update Z to the exact covered length by windows
        Z = self.windowSize + self.shiftSize * (cls.shape[1] - 1)
        x = x[:, :Z]

        # Pad for receptive keys/values
        x_ = torch.cat(
            [torch.zeros((B, self.remainder, C), device=device), x, torch.zeros((B, self.remainder, C), device=device)],
            dim=1,
        )  # (B, remainder+Z+remainder, C)

        # Window the sequences (B, nW, C, window/receptive) -> transpose to (B, nW, window/receptive, C)
        windowedX, _ = windowBoldSignal(x.transpose(2, 1), self.windowSize, self.shiftSize)
        windowedX = windowedX.transpose(2, 3)

        windowedX_, _ = windowBoldSignal(x_.transpose(2, 1), self.receptiveSize, self.shiftSize)
        windowedX_ = windowedX_.transpose(2, 3)

        nW = windowedX.shape[1]

        # Insert CLS token as first position in each window; then flatten windows into batch
        xcls  = torch.cat([cls.unsqueeze(2),  windowedX],  dim=2)  # (B, nW, 1+W, C)
        xcls  = xcls.view(B * nW, xcls.size(2), C)                 # (B*nW, 1+W, C)
        xcls_ = torch.cat([cls.unsqueeze(2),  windowedX_], dim=2)  # (B, nW, 1+R, C)
        xcls_ = xcls_.view(B * nW, xcls_.size(2), C)               # (B*nW, 1+R, C)

        masks = (self.mask_left, self.mask_right)

        # Fused attention + MLP
        xTrans, clsTrans = self.transformer(x, cls, xcls, xcls_, masks, nW)
        return xTrans, clsTrans
