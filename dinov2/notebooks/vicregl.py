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

from neptune.utils import stringify_unsupported
from pytorch_lightning.loggers.neptune import NeptuneLogger

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.tuner.tuning import Tuner

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
# class VICRegL(pl.LightningModule):
    def __init__(self, dataloader_kNN, classes, lr, wd, knn_k, knn_t=0.1):
        
        super().__init__(dataloader_kNN, 1, classes, knn_k=knn_k, knn_t=knn_t)
#         super().__init__()
        
        resnet = torchvision.models.resnet101()
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
#         out_dim = 512 # resnet18
        out_dim = 2048 # resnet101
        
#         convnext = torchvision.models.convnext_small()
#         convnext.classifier = nn.Identity()
#         convnext.avgpool = nn.Identity()
#         self.backbone = convnext.features
        
#         out_dim = 768 # convnext
        
        self.projection_head = BarlowTwinsProjectionHead(out_dim, 2048, 2048)
        self.local_projection_head = VicRegLLocalProjectionHead(out_dim, 128, 128)
        self.average_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.criterion = VICRegLLoss()
        
        self.lr = lr
        self.wd = wd
    
    def forward(self, x):
        x = self.backbone(x)
        y = self.average_pool(x).flatten(start_dim=1)
        z = self.projection_head(y)
        y_local = x.permute(0, 2, 3, 1)  # (B, D, W, H) to (B, W, H, D)
        z_local = self.local_projection_head(y_local)
        return z, z_local

    def training_step(self, batch, batch_index):
        views_and_grids = batch[0]
        views = views_and_grids[: len(views_and_grids) // 2]
        grids = views_and_grids[len(views_and_grids) // 2 :]
        features = [self.forward(view) for view in views]
        loss = self.criterion(
            global_view_features=features[:2],
            global_view_grids=grids[:2],
            local_view_features=features[2:],
            local_view_grids=grids[2:],
        )
        
        self.log('train/batch/loss', loss.detach().cpu())
        
        return loss
    
#     def forward(self, x):
#         return self.backbone(x)
    
#     def get_features(self, x):
#         return self.average_pool(self.backbone(x))
    
#     def _project(self, x):
#         x = self.backbone(x)
#         y = self.average_pool(x).flatten(start_dim=1)
#         z = self.projection_head(y)
#         y_local = x.permute(0, 2, 3, 1).contiguous()  # (B, D, W, H) to (B, W, H, D)
#         z_local = self.local_projection_head(y_local)
#         return z.contiguous(), z_local.contiguous()

#     def training_step(self, batch, batch_index):
#         views_and_grids = batch[0]
#         views = views_and_grids[: len(views_and_grids) // 2]
#         grids = views_and_grids[len(views_and_grids) // 2 :]
#         features = [self._project(view) for view in views]
#         loss = self.criterion(
#             global_view_features=features[:2],
#             global_view_grids=grids[:2],
#             local_view_features=features[2:],
#             local_view_grids=grids[2:],
#         ).contiguous()
        
#         self.log('train/batch/loss', loss.detach().cpu())
        
#         return loss
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr, total_steps=self.trainer.estimated_stepping_batches
        )
        
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


def split_dataset(full_dataset, frac):
    train_size = int(frac * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    return train_dataset, test_dataset


def get_datasets():
    
    import pickle
    
    train_transform = VICRegLTransform()
    
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
    
    if os.path.exists("dataset_cache.pkl"):

        with open("dataset_cache.pkl", "rb") as fp:
            dataset, val_dataset = pickle.load(fp)

    else:

        val_dataset = torchvision.datasets.ImageFolder("/data/2m/val", transform=val_transform)

        dataset = LightlyDataset("/data/2m/train", transform=train_transform)

        with open("dataset_cache.pkl", "wb+") as fp:
            pickle.dump([dataset, val_dataset], fp)
    
    return dataset, val_dataset


if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, required=False, default=0.0, 
                        help='learning rate. If not provided, the LR finder will be used')

    parser.add_argument('--epochs', type=int, required=True, 
                        help='number of epochs')

    parser.add_argument('--batch_size', type=int, required=True, 
                        help='batch size')

    parser.add_argument('--wd', type=float, required=True,
                        help='weight decay')
    parser.add_argument('--train_batches', type=int, required=False, default=1.0,
                        help='Fraction of train batches or number of train batches to use. If not provided, use entire dataset.')
    parser.add_argument('--k', type=int, required=True,
                        help='Number of neighbors to consider in the KNN',
                        default=20)

    args = parser.parse_args()
    
    dataset, val_dataset = get_datasets()
        
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=6,
        pin_memory=True
    #     sampler=train_sampler
    )

    knn_train_dataset, knn_val_dataset = split_dataset(val_dataset, 0.8)


    knn_train_dataloader = torch.utils.data.DataLoader(
        knn_train_dataset,
        batch_size=args.batch_size,
    #     shuffle=True,
        drop_last=True,
        num_workers=6,
        pin_memory=True
    )

    knn_val_dataloader = torch.utils.data.DataLoader(
        knn_val_dataset,
        batch_size=2,
    #     shuffle=True,
        drop_last=True,
        num_workers=6,
        pin_memory=True
    )

    neptune_logger = NeptuneLogger(
        project="cape/dinov2",  
        tags=["training", "vicregl"],  # optional
    )
    neptune_logger.experiment["args"] = stringify_unsupported(args)
    
    lr_monitor = LearningRateMonitor(logging_interval='step', log_momentum=True)

    model = VICRegL(knn_train_dataloader, dataset.dataset.classes, args.lr, args.wd, args.k)
    
    checkpoint_callback = ModelCheckpoint(
        every_n_epochs=1,
        dirpath="checkpoints",
        filename="vicreg-{epoch:02d}-{kNN_accuracy:.2f}",
        save_on_train_epoch_end=True,
        save_top_k=-1
    )
    
    trainer = pl.Trainer(
        max_epochs=args.epochs, 
        devices='auto', 
        accelerator='auto', 
        precision='16-mixed', 
        logger=neptune_logger,
        callbacks=[lr_monitor, checkpoint_callback],
        strategy="auto",
        limit_train_batches=args.train_batches,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0
    )
    
    # Create a Tuner
    if args.lr == 0.0:
        tuner = Tuner(trainer)

        # finds learning rate automatically
        # sets hparams.lr or hparams.learning_rate to that learning rate
        tuner.lr_find(model, train_dataloaders=dataloader, val_dataloaders=knn_val_dataloader)
    
    model.active = True
    
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




