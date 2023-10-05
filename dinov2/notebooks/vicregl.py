#!/usr/bin/env python
# coding: utf-8

# In[11]:


# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.
import os
import pytorch_lightning as pl
import torch
import torchvision
from torch import nn

from knn import knn_eval

from lightly.loss import VICRegLLoss

## The global projection head is the same as the Barlow Twins one
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.models.modules.heads import VicRegLLocalProjectionHead
from lightly.transforms.vicregl_transform import VICRegLTransform
from lightly.transforms import utils
from lightly.data import LightlyDataset

from benchmark_module import BenchmarkModule


from pytorch_lightning.loggers.neptune import NeptuneLogger

from pytorch_lightning.callbacks import LearningRateMonitor

# In[3]:

# class KNNCallback(pl.Callback):
#     def on_train_epoch_start(self, trainer, model):
#         acc = knn_eval(
#             model=model, 
#             train_dir="/data/2m/val/", 
#             log_dir=".", 
#             batch_size_per_device=32, 
#             num_workers=8, 
#             accelerator='gpu', 
#             devices=1, 
#             num_classes=len(dataset.dataset.classes),
#         #     strategy="ddp_notebook"
#         )
        
#         trainer.log("val/top_1", acc)


class VICRegL(BenchmarkModule):
    def __init__(self, dataloader_kNN, classes):
        
        super().__init__(dataloader_kNN, 1, classes, knn_k=16, knn_t=0.1)
        
#         resnet = torchvision.models.resnet18()
#         self.backbone = nn.Sequential(*list(resnet.children())[:-2])
#         out_dim = 512 # resnet18
        
        convnext = torchvision.models.convnext_small()
        convnext.classifier = nn.Identity()
        self.backbone = convnext.features
        
        out_dim = 768 # convnext
        
        self.projection_head = BarlowTwinsProjectionHead(out_dim, 2048, 2048)
        self.local_projection_head = VicRegLLocalProjectionHead(out_dim, 128, 128)
        self.average_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.criterion = VICRegLLoss()
        
    def forward(self, x):
        return self.backbone(x)
    
    def get_features(self, x):
        return self.average_pool(self.backbone(x))
    
    def _project(self, x):
        x = self.backbone(x)
        y = self.average_pool(x).flatten(start_dim=1)
        z = self.projection_head(y)
        y_local = x.permute(0, 2, 3, 1).contiguous()  # (B, D, W, H) to (B, W, H, D)
        z_local = self.local_projection_head(y_local)
        return z.contiguous(), z_local.contiguous()

    def training_step(self, batch, batch_index):
        views_and_grids = batch[0]
        views = views_and_grids[: len(views_and_grids) // 2]
        grids = views_and_grids[len(views_and_grids) // 2 :]
        features = [self._project(view) for view in views]
        loss = self.criterion(
            global_view_features=features[:2],
            global_view_grids=grids[:2],
            local_view_features=features[2:],
            local_view_grids=grids[2:],
        ).contiguous()
        
        self.log('train/batch/loss', loss.detach().cpu())
        
        return loss
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=5e-5)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6)
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


transform = VICRegLTransform()

val_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=utils.IMAGENET_NORMALIZE["mean"],
            std=utils.IMAGENET_NORMALIZE["std"],
        ),
    ]
)


def split_dataset(dataset, frac):
    num_train = int(frac * len(dataset))
    indices = list(range(len(dataset)))  
    one_indices, two_indices = indices[:num_train], indices[num_train:]

    # Create Samplers and DataLoaders
    split1_sampler = torch.utils.data.SubsetRandomSampler(one_indices)
    split2_sampler = torch.utils.data.SubsetRandomSampler(two_indices)
    
    return split1_sampler, split2_sampler


# In[4]:


import pickle

if os.path.exists("dataset_cache.pkl"):
    
    with open("dataset_cache.pkl", "rb") as fp:
        dataset, val_dataset = pickle.load(fp)

else:
    
    val_dataset = torchvision.datasets.ImageFolder("/data/2m/val", transform=val_transform)
    
    dataset = LightlyDataset("/data/2m/train", transform=transform)
    
    with open("dataset_cache.pkl", "wb+") as fp:
        pickle.dump([dataset, val_dataset], fp)


# train_sampler, _ = split_dataset(dataset, 0.99)
        
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    drop_last=True,
    num_workers=16,
#     sampler=train_sampler
)

knn_train_sampler, knn_val_sampler = split_dataset(val_dataset, 0.8)


knn_train_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=32,
#     shuffle=True,
    drop_last=False,
    num_workers=16,
    sampler=knn_train_sampler
)

knn_val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=32,
#     shuffle=True,
    drop_last=False,
    num_workers=16,
    sampler=knn_val_sampler
)


# In[6]:


neptune_logger = NeptuneLogger(
    project="cape/dinov2",  
    tags=["training", "vicregl"],  # optional
)


# In[7]:


lr_monitor = LearningRateMonitor(logging_interval='step')


# In[8]:
model = VICRegL(knn_train_dataloader, dataset.dataset.classes)

accelerator = "gpu" # if torch.cuda.is_available() else "cpu"

trainer = pl.Trainer(
    max_epochs=10, 
    devices='auto', 
    accelerator=accelerator, 
    precision='16-mixed', 
    logger=neptune_logger,
    callbacks=[lr_monitor]
)

trainer.fit(model=model, train_dataloaders=dataloader, val_dataloaders=knn_val_dataloader)


# In[9]:

try:
    trainer.logger.experiment.stop()
except AttributeError:
    pass

# In[10]:


import torch


# In[11]:


torch.cuda.empty_cache()

# In[29]:



# knn_eval(
#     model=model, 
#     train_dir="/data/2m/val/", 
#     val_dir="/data/2m/val", 
#     log_dir=".", 
#     batch_size_per_device=32, 
#     num_workers=8, 
#     accelerator='gpu', 
#     devices=1, 
#     num_classes=len(dataset.dataset.classes),
# #     strategy="ddp_notebook"
# )


# In[ ]:




