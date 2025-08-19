# pylint: disable=invalid-name, missing-function-docstring
""" 
BolT model module
"""

import torch
from torch import nn
from torch.nn.functional import cross_entropy

import math
import numpy as np
from einops import rearrange, repeat
from torch.nn.init import trunc_normal_

from .helper_functions import basic_handle_batch, basic_dataloader, basic_Adam_optimizer, BasicTrainer

from types import SimpleNamespace

class BolT(nn.Module):
    """
    TIME SERIES MODEL
    BolT model for fMRI data from https://doi.org/10.1016/j.media.2023.102841.
    No analysis features of the original implementation at https://github.com/icon-lab/BolT, just classification.
    Expected input shape: [batch_size, time_length, input_feature_size].
    Output: [batch_size, n_classes], loss_load = {"logits": logits, "cls": cls}.
    """
    def __init__(self, 
                 input_size, 
                 output_size,
    ):
        """
        Initialize BolT model.
        Majority of hyperparameters are defined in the hyperParams variable in the beginning of __init__.
        Args:
            input_size (int): Size of the vector at each time step in the input time series.
            output_size (int): Number of classes for classification.
        """
        super().__init__()
        self.lr = 2e-4

        hyperParams = {
            # oneCycleLR hyperparameters from the original code
            "lr" : 2e-4,
            "minLr" : 2e-5,
            "maxLr" : 4e-4,

            # FOR BOLT
            "nOfLayers" : 4,
            "dim" : input_size,

            "numHeads" : 36,
            "headDim" : 20,

            "windowSize" : 20,
            "shiftCoeff" : 2.0/5.0,            
            "fringeCoeff" : 2, # fringeSize = fringeCoeff * (windowSize) * 2 * (1-shiftCoeff)
            "focalRule" : "expand",

            "mlpRatio" : 1.0,
            "attentionBias" : True,
            "drop" : 0.1,
            "attnDrop" : 0.1,

            # Loss param
            "lambdaCons" : 1, # used in loss calculation
        }
        hyperParams = SimpleNamespace(**hyperParams)
        self.hyperParams = hyperParams

        dim = input_size
        nOfClasses = output_size


        self.lambdaCons = hyperParams.lambdaCons # for loss calculation

        self.inputNorm = nn.LayerNorm(dim)

        self.clsToken = nn.Parameter(torch.zeros(1, 1, dim))

        self.blocks = []

        shiftSize = int(hyperParams.windowSize * hyperParams.shiftCoeff)
        self.shiftSize = shiftSize
        self.receptiveSizes = []

        for i, layer in enumerate(range(hyperParams.nOfLayers)):
            
            if(hyperParams.focalRule == "expand"):
                receptiveSize = hyperParams.windowSize + math.ceil(hyperParams.windowSize * 2 * i * hyperParams.fringeCoeff * (1-hyperParams.shiftCoeff))
            elif(hyperParams.focalRule == "fixed"):
                receptiveSize = hyperParams.windowSize + math.ceil(hyperParams.windowSize * 2 * 1 * hyperParams.fringeCoeff * (1-hyperParams.shiftCoeff))

            # print("receptiveSize per window for layer {} : {}".format(i, receptiveSize))

            self.receptiveSizes.append(receptiveSize)

            self.blocks.append(BolTransformerBlock(
                dim = dim,
                numHeads = hyperParams.numHeads,
                headDim= hyperParams.headDim,
                windowSize = hyperParams.windowSize,
                receptiveSize = receptiveSize,
                shiftSize = shiftSize,
                mlpRatio = hyperParams.mlpRatio,
                attentionBias = hyperParams.attentionBias,
                drop = hyperParams.drop,
                attnDrop = hyperParams.attnDrop
            ))

        self.blocks = nn.ModuleList(self.blocks)


        self.encoder_postNorm = nn.LayerNorm(dim)
        self.classifierHead = nn.Linear(dim, nOfClasses)

        # for token painting
        self.last_numberOfWindows = None

        # for analysis only
        self.tokens = []


        self.initializeWeights()

    def initializeWeights(self):
        # a bit arbitrary
        torch.nn.init.normal_(self.clsToken, std=1.0)

    def calculateFlops(self, T):

        windowSize = self.hyperParams.windowSize
        shiftSize = self.shiftSize
        focalSizes = self.focalSizes
 
        macs = []

        nW = (T-windowSize) // shiftSize  + 1

        # C = 400 # for schaefer atlas
        C = self.hyperParams.dim
        H = self.hyperParams.numHeads
        D = self.hyperParams.headDim

        for l, focalSize in enumerate(focalSizes):

            mac = 0

            # MACS from attention calculation

                # projection in
            mac += nW * (1+windowSize) * C * H * D * 3

                # attention, softmax is omitted
            
            mac += 2 * nW * H * D * (1+windowSize) * (1+focalSize) 

                # projection out

            mac += nW * (1+windowSize) * C * H * D
            mac += 2 * (T+nW) * C * C

            macs.append(mac)

        return macs, np.sum(macs) * 2 # FLOPS = 2 * MAC

    def forward(self, x):
        B, T, _ = x.shape
        nW = (T - self.hyperParams.windowSize) // self.shiftSize + 1
        cls = self.clsToken.expand(B, nW, -1)  # no .repeat allocation of content

        for block in self.blocks:
            x, cls = block(x, cls, analysis=False)

        cls = self.encoder_postNorm(cls)
        logits = self.classifierHead(cls.mean(dim=1))
        return logits, {"logits": logits, "cls": cls}


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
        ce_loss = cross_entropy(loss_load["logits"], targets)

        cls = loss_load["cls"]
        clsLoss = torch.mean(torch.square(cls - cls.mean(dim=1, keepdims=True)))

        loss = ce_loss + clsLoss * self.lambdaCons
        loss_log = {
            "CE_loss": float(ce_loss.detach().cpu().item()),
            "cls_loss": float(clsLoss.detach().cpu().item())
            }
        
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
    


def windowBoldSignal(boldSignal, windowLength, stride):
    x = boldSignal.unfold(dimension=2, size=windowLength, step=stride)  # (B, N, nW, W)
    x = x.permute(0, 2, 1, 3).contiguous()                              # (B, nW, N, W)
    B, nW, N, W = x.shape
    samplingEndPoints = list(range(W, W + nW * stride, stride))
    return x, samplingEndPoints

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 1,
        dropout = 0.,
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim
        activation = nn.GELU()

        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            activation
        )

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out),
        )

    def forward(self, x):
        return self.net(x)


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


class WindowAttention(nn.Module):

    def __init__(self, dim, windowSize, receptiveSize, numHeads, headDim=20, attentionBias=True, qkvBias=True, attnDrop=0., projDrop=0.):

        super().__init__()
        self.dim = dim
        self.windowSize = windowSize  # N
        self.receptiveSize = receptiveSize # M
        self.numHeads = numHeads
        head_dim = headDim
        self.scale = head_dim ** -0.5

        self.attentionBias = attentionBias

        # define a parameter table of relative position bias
        
        maxDisparity = windowSize - 1 + (receptiveSize - windowSize)//2


        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(2*maxDisparity+1, numHeads))  # maxDisparity, nH

        self.cls_bias_sequence_up = nn.Parameter(torch.zeros((1, numHeads, 1, receptiveSize)))
        self.cls_bias_sequence_down = nn.Parameter(torch.zeros(1, numHeads, windowSize, 1))
        self.cls_bias_self = nn.Parameter(torch.zeros((1, numHeads, 1, 1)))

        # get pair-wise relative position index for each token inside the window
        coords_x = torch.arange(self.windowSize) # N
        coords_x_ = torch.arange(self.receptiveSize) - (self.receptiveSize - self.windowSize)//2 # M
        relative_coords = coords_x[:, None] - coords_x_[None, :]  # N, M
        relative_coords[:, :] += maxDisparity  # shift to start from 0
        relative_position_index = relative_coords  # (N, M)
        self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Linear(dim, head_dim * numHeads, bias=qkvBias)
        self.kv = nn.Linear(dim, 2 * head_dim * numHeads, bias=qkvBias)

        self.attnDrop = nn.Dropout(attnDrop)
        self.proj = nn.Linear(head_dim * numHeads, dim)


        self.projDrop = nn.Dropout(projDrop)

        # prep the biases
        trunc_normal_(self.relative_position_bias_table, std=.02)
        trunc_normal_(self.cls_bias_sequence_up, std=.02)
        trunc_normal_(self.cls_bias_sequence_down, std=.02)
        trunc_normal_(self.cls_bias_self, std=.02)
        
        self.softmax = nn.Softmax(dim=-1)


        # for token painting
        self.attentionMaps = None # shape = (#windows * nH, 1+windowSize, 1+receptiveSize)
        self.attentionGradients = None # shape = (#windows * nH, 1+windowSize, 1+receptiveSize)
        self.nW = None

    def save_attention_maps(self, attentionMaps):
        self.attentionMaps = attentionMaps

    def save_attention_gradients(self, grads):
        self.attentionGradients = grads

    def averageJuiceAcrossHeads(self, cam, grad):

        """
            Hacked from the original paper git repo ref: https://github.com/hila-chefer/Transformer-MM-Explainability
            cam : (numberOfHeads, n, m)
            grad : (numberOfHeads, n, m)
        """

        #cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
        #grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
        cam = grad * cam
        cam = cam.clamp(min=0).mean(dim=0)
        return cam


    def getJuiceFlow(self, shiftSize): # NOTE THAT, this functions assumes there is only one subject to analyze. So if you want to keep using this implementation, generate relevancy maps one by one for each subject

        # infer the dynamic length
        dynamicLength = self.windowSize + (self.nW - 1) * shiftSize

        targetAttentionMaps = self.attentionMaps # (nW, h, n, m) 
        targetAttentionGradients = self.attentionGradients #self.attentionGradients # (nW h n m)        

        globalJuiceMatrix = torch.zeros((self.nW + dynamicLength, self.nW + dynamicLength)).to(targetAttentionMaps.device)
        normalizerMatrix = torch.zeros((self.nW + dynamicLength, self.nW + dynamicLength)).to(targetAttentionMaps.device)


        # aggregate(by averaging) the juice from all the windows
        for i in range(self.nW):

            # average the juices across heads
            window_averageJuice = self.averageJuiceAcrossHeads(targetAttentionMaps[i], targetAttentionGradients[i]) # of shape (1+windowSize, 1+receptiveSize)
            
            # now broadcast the juice to the global juice matrix.

            # set boundaries for overflowing focal attentions
            L = (self.receptiveSize-self.windowSize)//2

            overflow_left = abs(min(i*shiftSize - L, 0))
            overflow_right = max(i*shiftSize + self.windowSize + L - dynamicLength, 0)

            leftMarker_global = i*shiftSize - L + overflow_left
            rightMarker_global = i*shiftSize + self.windowSize + L - overflow_right
            
            leftMarker_window = overflow_left
            rightMarker_window = self.receptiveSize - overflow_right

                # first the cls it self
            globalJuiceMatrix[i, i] += window_averageJuice[0,0]
            normalizerMatrix[i, i] += 1
                # cls to bold tokens
            globalJuiceMatrix[i, self.nW + leftMarker_global : self.nW + rightMarker_global] += window_averageJuice[0, 1+leftMarker_window:1+rightMarker_window]
            normalizerMatrix[i, self.nW + leftMarker_global : self.nW + rightMarker_global] += torch.ones_like(window_averageJuice[0, 1+leftMarker_window:1+rightMarker_window])
                # bold tokens to cls
            globalJuiceMatrix[self.nW + i*shiftSize : self.nW + i*shiftSize + self.windowSize, i] += window_averageJuice[1:, 0]
            normalizerMatrix[self.nW + i*shiftSize : self.nW + i*shiftSize + self.windowSize, i] += torch.ones_like(window_averageJuice[1:, 0])
                # bold tokens to bold tokens
            globalJuiceMatrix[self.nW + i*shiftSize : self.nW + i*shiftSize + self.windowSize, self.nW + leftMarker_global : self.nW + rightMarker_global] += window_averageJuice[1:, 1+leftMarker_window:1+rightMarker_window]
            normalizerMatrix[self.nW + i*shiftSize : self.nW + i*shiftSize + self.windowSize, self.nW + leftMarker_global : self.nW + rightMarker_global] += torch.ones_like(window_averageJuice[1:, 1+leftMarker_window:1+rightMarker_window])

        # to prevent divide by zero for those non-existent attention connections
        normalizerMatrix[normalizerMatrix == 0] = 1

        globalJuiceMatrix = globalJuiceMatrix / normalizerMatrix

        return globalJuiceMatrix

    def forward(self, x, x_, mask, nW, analysis=False):
        """
        x  : (B*nW, 1+windowSize, C)   # queries
        x_ : (B*nW, 1+receptiveSize, C)# keys/values
        mask: (mask_left, mask_right), each (maskCount, 1+windowSize, 1+receptiveSize), bool
        nW : number of windows
        returns: (B*nW, 1+windowSize, C)
        """
        # ----- shapes / unpack -----
        BnW, n_tok, C = x.shape       # n_tok = 1 + N
        _,    m_tok, _ = x_.shape     # m_tok = 1 + M
        N = n_tok - 1                 # windowSize
        M = m_tok - 1                 # receptiveSize
        H = self.numHeads
        d = (self.q.out_features // H)  # per-head dim
        B = BnW // nW

        # ----- projections & split heads -----
        q = self.q(x)                       # (B*nW, n_tok, H*d)
        k, v = self.kv(x_).chunk(2, dim=-1) # (B*nW, m_tok, H*d) each
        q = rearrange(q, "b n (h d) -> b h n d", h=H)  # (B*nW, H, n_tok, d)
        k = rearrange(k, "b m (h d) -> b h m d", h=H)  # (B*nW, H, m_tok, d)
        v = rearrange(v, "b m (h d) -> b h m d", h=H)

        # ----- attention logits -----
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B*nW, H, n_tok, m_tok)

        # relative position bias only for non-CLS rows/cols
        # self.relative_position_index: (N, M)
        # self.relative_position_bias_table: (2*maxDisp+1, H)
        rel = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, M, H).permute(2, 0, 1).contiguous()  # (H, N, M)

        # Make sure dtypes match
        rel = rel.to(dtype=attn.dtype, device=attn.device)
        attn[:, :, 1:, 1:] = attn[:, :, 1:, 1:] + rel.unsqueeze(0)            # (1, H, N, M)
        attn[:, :, :1, :1] = attn[:, :, :1, :1] + self.cls_bias_self.to(attn.dtype)
        attn[:, :, :1, 1:] = attn[:, :, :1, 1:] + self.cls_bias_sequence_up.to(attn.dtype)
        attn[:, :, 1:, :1] = attn[:, :, 1:, :1] + self.cls_bias_sequence_down.to(attn.dtype)

        # ----- masking first/last windows (no .repeat, just broadcast) -----
        mask_left, mask_right = mask  # (maskCount, n_tok, m_tok), bool
        # reshape to (B, nW, H, n_tok, m_tok) so we can apply per-window masks
        attn = rearrange(attn, "(b nW) h n m -> b nW h n m", b=B, nW=nW)

        maskCount = min(mask_left.shape[0], attn.shape[1])
        if maskCount > 0:
            neg_inf = max_neg_value(attn)  # -finfo.max in attn dtype

            # First maskCount windows: apply left mask
            # mask_left slice: (maskCount, n_tok, m_tok) -> (1, maskCount, 1, n_tok, m_tok)
            attn[:, :maskCount].masked_fill_(mask_left[:maskCount].unsqueeze(0).unsqueeze(2), neg_inf)

            # Last maskCount windows: apply right mask
            attn[:, -maskCount:].masked_fill_(mask_right[-maskCount:].unsqueeze(0).unsqueeze(2), neg_inf)

        # back to (B*nW, H, n_tok, m_tok)
        attn = rearrange(attn, "b nW h n m -> (b nW) h n m")

        # ----- softmax, optional analysis hook, dropout -----
        attn = self.softmax(attn)
        if analysis:
            # Save attention maps (detach) and register gradient hook
            self.save_attention_maps(attn.detach())
            handle = attn.register_hook(self.save_attention_gradients)
            self.nW = nW
            self.handle = handle

        attn = self.attnDrop(attn)

        # ----- apply attention to values, merge heads, project -----
        out = torch.matmul(attn, v)                    # (B*nW, H, n_tok, d)
        out = rearrange(out, "b h n d -> b n (h d)")   # (B*nW, n_tok, H*d)
        out = self.proj(out)                           # (B*nW, n_tok, C)
        out = self.projDrop(out)
        return out




class FusedWindowTransformer(nn.Module):

    def __init__(self, dim, windowSize, shiftSize, receptiveSize, numHeads, headDim, mlpRatio, attentionBias, drop, attnDrop):
        
        super().__init__()


        self.attention = WindowAttention(dim=dim, windowSize=windowSize, receptiveSize=receptiveSize, numHeads=numHeads, headDim=headDim, attentionBias=attentionBias, attnDrop=attnDrop, projDrop=drop)
        
        self.mlp = FeedForward(dim=dim, mult=mlpRatio, dropout=drop)

        self.attn_norm = nn.LayerNorm(dim)
        self.mlp_norm = nn.LayerNorm(dim)

        self.shiftSize = shiftSize

    def getJuiceFlow(self):  
        return self.attention.getJuiceFlow(self.shiftSize)

    def forward(self, x, cls, windowX, windowX_, mask, nW, analysis=False):
        """

            Input: 

            x : (B, T, C)
            cls : (B, nW, C)
            windowX: (B, 1+windowSize, C)
            windowX_ (B, 1+windowReceptiveSize, C)
            mask : (B, 1+windowSize, 1+windowReceptiveSize)
            nW : number of windows

            analysis : Boolean, it is set True only when you want to analyze the model, otherwise not important 

            Output:

            xTrans : (B, T, C)
            clsTrans : (B, nW, C)

        """

        # WINDOW ATTENTION
        windowXTrans = self.attention(self.attn_norm(windowX), self.attn_norm(windowX_), mask, nW, analysis=analysis) # (B*nW, 1+windowSize, C)
        clsTrans = windowXTrans[:,:1] # (B*nW, 1, C)
        xTrans = windowXTrans[:,1:] # (B*nW, windowSize, C)
        
        clsTrans = rearrange(clsTrans, "(b nW) l c -> b (nW l) c", nW=nW)
        xTrans = rearrange(xTrans, "(b nW) l c -> b nW l c", nW=nW)
        # FUSION
        xTrans = self.gatherWindows(xTrans, x.shape[1], self.shiftSize)
        
        # residual connections
        clsTrans = clsTrans + cls
        xTrans = xTrans + x

        # MLP layers
        xTrans = xTrans + self.mlp(self.mlp_norm(xTrans))
        clsTrans = clsTrans + self.mlp(self.mlp_norm(clsTrans))

        return xTrans, clsTrans

    def gatherWindows(self, windowedX, dynamicLength, shiftSize):
        
        """
        Input:
            windowedX : (batchSize, nW, windowLength, C)
            scatterWeights : (windowLength, )
        
        Output:
            destination: (batchSize, dynamicLength, C)
        
        """

        batchSize = windowedX.shape[0]
        windowLength = windowedX.shape[2]
        nW = windowedX.shape[1]
        C = windowedX.shape[-1]
        
        device = windowedX.device


        destination = torch.zeros((batchSize, dynamicLength,  C)).to(device)
        scalerDestination = torch.zeros((batchSize, dynamicLength, C)).to(device)

        indexes = torch.tensor([[j+(i*shiftSize) for j in range(windowLength)] for i in range(nW)]).to(device)
        indexes = indexes[None, :, :, None].repeat((batchSize, 1, 1, C)) # (batchSize, nW, windowSize, featureDim)

        src = rearrange(windowedX, "b n w c -> b (n w) c")
        indexes = rearrange(indexes, "b n w c -> b (n w) c")

        destination.scatter_add_(dim=1, index=indexes, src=src)


        scalerSrc = torch.ones((windowLength)).to(device)[None, None, :, None].repeat(batchSize, nW, 1, C) # (batchSize, nW, windowLength, featureDim)
        scalerSrc = rearrange(scalerSrc, "b n w c -> b (n w) c")

        scalerDestination.scatter_add_(dim=1, index=indexes, src=scalerSrc)

        destination = destination / scalerDestination


        return destination

    

class BolTransformerBlock(nn.Module):

    def __init__(self, dim, numHeads, headDim, windowSize, receptiveSize, shiftSize, mlpRatio=1.0, drop=0.0, attnDrop=0.0, attentionBias=True):

        assert((receptiveSize-windowSize)%2 == 0)

        super().__init__()
        self.transformer = FusedWindowTransformer(dim=dim, windowSize=windowSize, shiftSize=shiftSize, receptiveSize=receptiveSize, numHeads=numHeads, headDim=headDim, mlpRatio=mlpRatio, attentionBias=attentionBias, drop=drop, attnDrop=attnDrop)

        self.windowSize = windowSize
        self.receptiveSize = receptiveSize
        self.shiftSize = shiftSize

        self.remainder = (self.receptiveSize - self.windowSize) // 2

        # create mask here for non matching query and key pairs
        maskCount = self.remainder // shiftSize + 1
        mask_left  = torch.zeros(maskCount, self.windowSize+1, self.receptiveSize+1, dtype=torch.bool)
        mask_right = torch.zeros_like(mask_left)
        for i in range(maskCount):
            if self.remainder > 0:
                mask_left[i, :, 1:1+self.remainder-self.shiftSize*i] = True
                if (-self.remainder + self.shiftSize*i) > 0:
                    mask_right[maskCount-1-i, :, -self.remainder + self.shiftSize*i:] = True

        self.register_buffer("mask_left", mask_left)
        self.register_buffer("mask_right", mask_right)


    def getJuiceFlow(self):
        return self.transformer.getJuiceFlow()
    
    def forward(self, x, cls, analysis=False):
        """
        Input:
            x : (batchSize, dynamicLength, c)
            cls : (batchSize, nW, c)
        
            analysis : Boolean, it is set True only when you want to analyze the model, not important otherwise 


        Output:
            fusedX_trans : (batchSize, dynamicLength, c)
            cls_trans : (batchSize, nW, c)

        """

        B, Z, C = x.shape
        device = x.device

        #update z, incase some are dropped during windowing
        Z = self.windowSize + self.shiftSize * (cls.shape[1]-1)
        x = x[:, :Z]

        # form the padded x to be used for focal keys and values
        x_ = torch.cat([torch.zeros((B, self.remainder,C),device=device), x, torch.zeros((B, self.remainder,C), device=device)], dim=1) # (B, remainder+Z+remainder, C) 

        # window the sequences
        windowedX, _ = windowBoldSignal(x.transpose(2,1), self.windowSize, self.shiftSize) # (B, nW, C, windowSize)         
        windowedX = windowedX.transpose(2,3) # (B, nW, windowSize, C)

        windowedX_, _ = windowBoldSignal(x_.transpose(2,1), self.receptiveSize, self.shiftSize) # (B, nW, C, receptiveSize)
        windowedX_ = windowedX_.transpose(2,3) # (B, nW, receptiveSize, C)

        
        nW = windowedX.shape[1] # number of windows
    
        xcls = torch.cat([cls.unsqueeze(dim=2), windowedX], dim = 2) # (B, nW, 1+windowSize, C)
        xcls = rearrange(xcls, "b nw l c -> (b nw) l c") # (B*nW, 1+windowSize, C) 
       
        xcls_ = torch.cat([cls.unsqueeze(dim=2), windowedX_], dim=2) # (B, nw, 1+receptiveSize, C)
        xcls_ = rearrange(xcls_, "b nw l c -> (b nw) l c") # (B*nW, 1+receptiveSize, C)

        masks = [self.mask_left, self.mask_right]

        # pass to fused window transformer
        fusedX_trans, cls_trans = self.transformer(x, cls, xcls, xcls_, masks, nW, analysis) # (B*nW, 1+windowSize, C)


        return fusedX_trans, cls_trans