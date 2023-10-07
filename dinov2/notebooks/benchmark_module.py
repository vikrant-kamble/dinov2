import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from lightly.utils.benchmarking.knn import knn_predict

import tqdm
import os

from functools import wraps

def log_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if os.environ.get("DEBUG") is not None:
            print(f"Entering {func.__name__}")
        result = func(*args, **kwargs)
        if os.environ.get("DEBUG") is not None:
            print(f"Exiting {func.__name__}")
        return result
    return wrapper

# code for kNN prediction from here:
# https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
# @rank_zero_only
# def knn_predict(feature, feature_bank, feature_labels, classes: int, knn_k: int, knn_t: float):
#     """Helper method to run kNN predictions on features based on a feature bank

#     Args:
#         feature: Tensor of shape [N, D] consisting of N D-dimensional features
#         feature_bank: Tensor of a database of features used for kNN
#         feature_labels: Labels for the features in our feature_bank
#         classes: Number of classes (e.g. 10 for CIFAR-10)
#         knn_k: Number of k neighbors used for kNN
#         knn_t: 

#     """
#     # compute cos similarity between each feature vector and feature bank ---> [B, N]
#     sim_matrix = torch.mm(feature, feature_bank)
#     # [B, K]
#     sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
#     # [B, K]
#     sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
#     # we do a reweighting of the similarities 
#     sim_weight = (sim_weight / knn_t).exp()
#     # counts for each class
#     one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
#     # [B*K, C]
#     one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
#     # weighted score ---> [B, C]
#     pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
#     pred_labels = pred_scores.argsort(dim=-1, descending=True)
#     return pred_labels


class BenchmarkModule(pl.LightningModule):
    """A PyTorch Lightning Module for automated kNN callback
    
    At the end of every training epoch we create a feature bank by inferencing
    the backbone on the dataloader passed to the module. 
    At every validation step we predict features on the validation data.
    After all predictions on validation data (validation_epoch_end) we evaluate
    the predictions on a kNN classifier on the validation data using the 
    feature_bank features from the train data.
    We can access the highest accuracy during a kNN prediction using the 
    max_accuracy attribute.
    """
    def __init__(self, dataloader_kNN, gpus, classes, knn_k=200, knn_t=0.1):
        
        super().__init__()
        self.backbone = None

        # These will only be filled on rank zero
        self.dataloader_kNN = None
        self.gpus = None
        self.classes = None
        self.knn_k = None
        self.knn_t = None
        
        self.max_accuracy = -1
        
        self._setup(dataloader_kNN, gpus, classes, knn_k, knn_t)
            
    @rank_zero_only
    def _setup(self, dataloader_kNN, gpus, classes, knn_k, knn_t):
        
        self.dataloader_kNN = dataloader_kNN
        self.gpus = gpus
        self.classes = classes
        self.knn_k = knn_k
        self.knn_t = knn_t
        
        self._reset()
    
    @log_function
    def on_fit_start(self):
        assert self.backbone is not None, "You have to assign a backbone with self.backbone = backbone"
    
    @rank_zero_only
    def _reset(self):
        
        self.feature_bank = []
        self.targets_bank = []
        self.outputs = []
    
    @log_function
    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        
        images, targets = batch
        feature = self.backbone(images).squeeze()
        feature = F.normalize(feature, dim=1)         
        pred_labels = knn_predict(feature.cpu(), self.feature_bank, self.targets_bank, self.classes, self.knn_k, self.knn_t)
        num = images.size(0)
        top1 = (pred_labels[:, 0] == targets.cpu()).float().sum().item()
        self.outputs.append((num, top1))
    
    @log_function
    @rank_zero_only
    def on_validation_epoch_start(self):
        
        assert len(self.feature_bank) == 0
        assert len(self.targets_bank) == 0
        assert len(self.outputs) == 0
        
        feature_bank = []
        targets_bank = []
        
        for data in tqdm.tqdm(self.dataloader_kNN, total=len(self.dataloader_kNN)):
            img, target = data                
            if self.gpus > 0:
                img = img.cuda()
                target = target.cuda()
            feature = self.backbone(img).squeeze()
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            targets_bank.append(target)

        self.feature_bank = torch.cat(feature_bank, dim=0).t().cpu().contiguous()
        self.targets_bank = torch.cat(targets_bank, dim=0).t().cpu().contiguous()
    
    @log_function
    @rank_zero_only
    def on_validation_epoch_end(self):
        
        total_num = 0
        total_top1 = 0.

        for (num, top1) in self.outputs:
            total_num += num
            total_top1 += top1
        acc = float(total_top1 / total_num)
        
        if acc > self.max_accuracy:
            self.max_accuracy = acc

        self.log('kNN_accuracy', acc * 100.0, prog_bar=True, rank_zero_only=True)
                
        self._reset()
