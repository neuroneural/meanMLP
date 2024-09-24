# pylint: disable=invalid-name, missing-function-docstring
""" BolT model module from https://github.com/icon-lab/BolT"""
from random import uniform, randint

import torch
from torch import nn, randperm as rp
import numpy as np
import pandas as pd
import wandb
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_
import math
import time
from tqdm import tqdm
from apto.utils.report import get_classification_report
from omegaconf import OmegaConf, DictConfig

from torch.utils.data import DataLoader
from omegaconf import open_dict
import gc
from src.trainer import BasicTrainer

def get_model(cfg: DictConfig, model_cfg: DictConfig):
    return BolT(model_cfg)


def default_HPs(cfg: DictConfig):
    model_cfg = {
        "weightDecay" : 0,

        "lr" : 2e-4,
        "minLr" : 2e-5,
        "maxLr" : 4e-4,

        # FOR BOLT
        "nOfLayers" : 4,
        # "dim" : 400,
        "dim" : cfg.dataset.data_info.main.data_shape[
            2
        ],

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
        "lambdaCons" : 1,

        # extra for ablation study
        "pooling" : "cls", # ["cls", "gmp"]         

        "input_size": cfg.dataset.data_info.main.data_shape[
            2
        ],  # data_shape: [batch_size, time_length, input_feature_size]
        "output_size": cfg.dataset.data_info.main.n_classes,
    }
    return OmegaConf.create(model_cfg)

def get_criterion(cfg: DictConfig, model_cfg: DictConfig):
    return bolTLoss(model_cfg)

class bolTLoss:
    """Cross-entropy loss with model regularization"""

    def __init__(self, model_cfg):
        self.ce_loss = nn.CrossEntropyLoss()

        self.lambdaCons = model_cfg.lambdaCons

    def __call__(self, logits, target, cls):
        clsLoss = torch.mean(torch.square(cls - cls.mean(dim=1, keepdims=True)))

        cross_entropy_loss = self.ce_loss(logits, target)

        return cross_entropy_loss + clsLoss * self.lambdaCons

def get_scheduler(cfg: DictConfig, model_cfg: DictConfig, optimizer):
    divFactor = model_cfg.maxLr / model_cfg.lr
    finalDivFactor = model_cfg.lr / model_cfg.minLr
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, model_cfg.maxLr, cfg.mode.max_epochs * cfg.dataloader.train.n_batches, div_factor=divFactor, final_div_factor=finalDivFactor, pct_start=0.3)
    return scheduler

def get_trainer(
    cfg, model_cfg, dataloaders, model, criterion, optimizer, scheduler, logger
):
    return BolTTrainer(cfg, model_cfg, dataloaders, model, criterion, optimizer, scheduler, logger)

class BolTTrainer(BasicTrainer):
    """BolTTrainer trainer. Different from BasicTrainer at the scheduler step position"""
    
    def __init__(self, cfg, model_cfg, dataloaders, model, criterion, optimizer, scheduler, logger):
        super().__init__(cfg, model_cfg, dataloaders, model, criterion, optimizer, scheduler, logger)

    def run_epoch(self, ds_name):
        """Run single epoch and monitor OutOfMemoryError"""
        impatience = 0
        while True:
            try:
                metrics = self.run_epoch_for_real(ds_name)
            except torch.cuda.OutOfMemoryError as e:
                if impatience > 5:
                    raise torch.cuda.OutOfMemoryError(
                        "Can't fix CUDA out of memory exception"
                    ) from e

                impatience += 1
                print("CUDA OOM encountered, reducing batch_size and cleaning memory")

                # run garbage collector and empty cache
                gc.collect()
                torch.cuda.empty_cache()

                # reduce batch_size
                batch_size = self.cfg.mode.batch_size // 2
                dataloader_info = {}
                for key, dl in self.dataloaders.items():
                    dataset = dl.dataset
                    self.dataloaders[key] = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        num_workers=0,
                        shuffle=key == "train",
                    )
                    dataloader_info[key] = {
                        "n_samples": len(dl.dataset),
                        "batch_size": dl.batch_size,
                        "n_batches": math.ceil(len(dl.dataset)/dl.batch_size)
                    }
                with open_dict(self.cfg):
                    self.cfg.mode.batch_size = batch_size
                    self.cfg.dataloader = dataloader_info

                self.scheduler = get_scheduler(self.cfg, self.model_cfg, self.optimizer)
                    
                # try to run the epoch again
                continue

            # no errors encountered, exiting loop
            break

        return metrics
    
    def run_epoch_for_real(self, ds_name):
        """Run single epoch on `ds_name` dataloder"""
        is_train_dataset = ds_name == "train"

        all_scores, all_targets = [], []
        total_loss, total_size = 0.0, 0

        self.model.train(is_train_dataset)
        start_time = time.time()

        with torch.set_grad_enabled(is_train_dataset):
            for data, target in self.dataloaders[ds_name]:
                # permute TS data if needed
                if is_train_dataset and self.permute:
                    for i, sample in enumerate(data):
                        data[i] = sample[rp(sample.shape[0]), :]

                data, target = data.to(self.device), target.to(self.device)
                total_size += data.shape[0]

                logits, cls = self.model(data)
                loss = self.criterion(logits, target, cls)
                score = torch.softmax(logits, dim=-1)

                all_scores.append(score.cpu().detach().numpy())
                all_targets.append(target.cpu().detach().numpy())
                total_loss += loss.sum().item()

                if is_train_dataset:
                    self.do_update(loss)

        average_time = (time.time() - start_time) / total_size
        average_loss = total_loss / total_size

        y_test = np.hstack(all_targets)
        y_score = np.vstack(all_scores)
        y_pred = np.argmax(y_score, axis=-1).astype(np.int32)

        report = get_classification_report(
            y_true=y_test, y_pred=y_pred, y_score=y_score, beta=0.5
        )

        metrics = {
            ds_name + "_accuracy": report["precision"].loc["accuracy"],
            ds_name + "_score": report["auc"].loc["weighted"],
            ds_name + "_average_loss": average_loss,
            ds_name + "_average_time": average_time,
        }

        return metrics

class BolT(nn.Module):
    def __init__(self, hyperParams):

        super().__init__()

        dim = hyperParams.dim
        # nOfClasses = details.nOfClasses
        nOfClasses = hyperParams.output_size

        self.hyperParams = hyperParams

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

            print("receptiveSize per window for layer {} : {}".format(i, receptiveSize))

            self.receptiveSizes.append(receptiveSize)

            self.blocks.append(BolTransformerBlock(
                dim = hyperParams.dim,
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
        C = self.hyperParams.input_size
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


            # MACS from MLP layer (2 layers with expand ratio = 1)

            mac += 2 * (T+nW) * C * C
        
            macs.append(mac)

        return macs, np.sum(macs) * 2 # FLOPS = 2 * MAC


    def forward(self, roiSignals, analysis=False):
        
        """
            Input : 
            
                roiSignals : (batchSize, N, dynamicLength)

                analysis : Boolean, it is set True only when you want to analyze the model, not important otherwise 
            
            Output:

                logits : (batchSize, #ofClasses)


        """
        # roiSignals is actually [batch_size, time_length, input_feature_size] in our code

        # roiSignals = roiSignals.permute((0,2,1))

        batchSize = roiSignals.shape[0]
        T = roiSignals.shape[1] # dynamicLength

        nW = (T-self.hyperParams.windowSize) // self.shiftSize  + 1
        cls = self.clsToken.repeat(batchSize, nW, 1) # (batchSize, #windows, C)
        
        # record nW and dynamicLength, need in case you want to paint those tokens later
        self.last_numberOfWindows = nW
        
        if(analysis):
            self.tokens.append(torch.cat([cls, roiSignals], dim=1))

        for block in self.blocks:
            roiSignals, cls = block(roiSignals, cls, analysis)
            
            if(analysis):
                self.tokens.append(torch.cat([cls, roiSignals], dim=1))
        
        """
            roiSignals : (batchSize, dynamicLength, featureDim)
            cls : (batchSize, nW, featureDim)
        """

        cls = self.encoder_postNorm(cls)

        if(self.hyperParams.pooling == "cls"):
            logits = self.classifierHead(cls.mean(dim=1)) # (batchSize, #ofClasses)
        elif(self.hyperParams.pooling == "gmp"):
            logits = self.classifierHead(roiSignals.mean(dim=1))

        torch.cuda.empty_cache()

        return logits, cls
    


def windowBoldSignal(boldSignal, windowLength, stride):
    
    """
        boldSignal : (batchSize, N, T)
        output : (batchSize, (T-windowLength) // stride, N, windowLength )
    """

    T = boldSignal.shape[2]

    # NOW WINDOWING 
    windowedBoldSignals = []
    samplingEndPoints = []

    for windowIndex in range((T - windowLength)//stride + 1):
        
        sampledWindow = boldSignal[:, :, windowIndex * stride  : windowIndex * stride + windowLength]
        samplingEndPoints.append(windowIndex * stride + windowLength)

        sampledWindow = torch.unsqueeze(sampledWindow, dim=1)
        windowedBoldSignals.append(sampledWindow)
        

    windowedBoldSignals = torch.cat(windowedBoldSignals, dim=1)

    return windowedBoldSignals, samplingEndPoints

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
            Input:

            x: base BOLD tokens with shape of (B*num_windows, 1+windowSize, C), the first one is cls token
            x_: receptive BOLD tokens with shape of (B*num_windows, 1+receptiveSize, C), again the first one is cls token
            mask: (mask_left, mask_right) with shape (maskCount, 1+windowSize, 1+receptiveSize)
            nW: number of windows
            analysis : Boolean, it is set True only when you want to analyze the model, not important otherwise 

            Output:

            transX : attended BOLD tokens from the base of the window, shape = (B*num_windows, 1+windowSize, C), the first one is cls token

        """


        B_, N, C = x.shape
        _, M, _ = x_.shape
        N = N-1
        M = M-1

        B = B_ // nW 

        mask_left, mask_right = mask

        # linear mapping
        q = self.q(x) # (batchSize * #windows, 1+N, C)
        k, v = self.kv(x_).chunk(2, dim=-1) # (batchSize * #windows, 1+M, C)

        # head seperation
        q = rearrange(q, "b n (h d) -> b h n d", h=self.numHeads)
        k = rearrange(k, "b m (h d) -> b h m d", h=self.numHeads)
        v = rearrange(v, "b m (h d) -> b h m d", h=self.numHeads)

        attn = torch.matmul(q , k.transpose(-1, -2)) * self.scale # (batchSize*#windows, h, n, m)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, M, -1)  # N, M, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, N, M
       

        if(self.attentionBias):
            attn[:, :, 1:, 1:] = attn[:, :, 1:, 1:] + relative_position_bias.unsqueeze(0)
            attn[:, :, :1, :1] = attn[:, :, :1, :1] + self.cls_bias_self
            attn[:, :, :1, 1:] = attn[:, :, :1, 1:] + self.cls_bias_sequence_up
            attn[:, :, 1:, :1] = attn[:, :, 1:, :1] + self.cls_bias_sequence_down
        
        # mask the not matching queries and tokens here
        maskCount = mask_left.shape[0]
        # repate masks for batch and heads
        mask_left = repeat(mask_left, "nM nn mm -> b nM h nn mm", b=B, h=self.numHeads)
        mask_right = repeat(mask_right, "nM nn mm -> b nM h nn mm", b=B, h=self.numHeads)

        mask_value = max_neg_value(attn) 


        attn = rearrange(attn, "(b nW) h n m -> b nW h n m", nW = nW)        
        
        # make sure masks do not overflow
        maskCount = min(maskCount, attn.shape[1])
        mask_left = mask_left[:, :maskCount]
        mask_right = mask_right[:, -maskCount:]

        attn[:, :maskCount].masked_fill_(mask_left==1, mask_value)
        attn[:, -maskCount:].masked_fill_(mask_right==1, mask_value)
        attn = rearrange(attn, "b nW h n m -> (b nW) h n m")


        attn = self.softmax(attn) # (b, h, n, m)

        if(analysis):
            self.save_attention_maps(attn.detach()) # save attention
            handle = attn.register_hook(self.save_attention_gradients) # save it's gradient
            self.nW = nW
            self.handle = handle

        attn = self.attnDrop(attn)

        x = torch.matmul(attn, v) # of shape (b_, h, n, d)

        x = rearrange(x, 'b h n d -> b n (h d)')

        x = self.proj(x)
        x = self.projDrop(x)
        
        return x



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
        mask_left = torch.zeros(maskCount, self.windowSize+1, self.receptiveSize+1)
        mask_right = torch.zeros(maskCount, self.windowSize+1, self.receptiveSize+1)

        for i in range(maskCount):
            if(self.remainder > 0):
                mask_left[i, :, 1:1+self.remainder-shiftSize*i] = 1
                if(-self.remainder+shiftSize*i > 0):
                    mask_right[maskCount-1-i, :, -self.remainder+shiftSize*i:] = 1

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