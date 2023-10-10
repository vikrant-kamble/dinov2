from pathlib import Path

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms as T

from lightly.data import LightlyDataset
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.utils.benchmarking import KNNClassifier, MetricCallback
from lightly.utils.dist import print_rank_zero
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.models.modules.heads import VicRegLLocalProjectionHead
from lightly.loss import VICRegLLoss

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn


class VICRegLMinimal(pl.LightningModule):
    def __init__(self):
        
#         super().__init__(dataloader_kNN, 1, classes, knn_k=knn_k, knn_t=knn_t)
        super().__init__()
        
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
    
    def forward(self, x):
        x = self.backbone(x)
        y = self.average_pool(x).flatten(start_dim=1)
        z = self.projection_head(y)
        y_local = x.permute(0, 2, 3, 1)  # (B, D, W, H) to (B, W, H, D)
        z_local = self.local_projection_head(y_local)
        return z#, z_local
    
#     def forward(self, x):
#         x = self.backbone(x)
#         y = self.average_pool(x).flatten(start_dim=1)
#         z = self.projection_head(y)
#         y_local = x.permute(0, 2, 3, 1).contiguous()  # (B, D, W, H) to (B, W, H, D)
#         z_local = self.local_projection_head(y_local)
#         return z.contiguous(), z_local.contiguous()

def knn_eval(
    model: LightningModule,
    train_dir: Path,
    log_dir: Path,
    batch_size_per_device: int,
    num_workers: int,
    accelerator: str,
    devices: int,
    num_classes: int,
    strategy="ddp_find_unused_parameters_true"
) -> None:
    """Runs KNN evaluation on the given model.

    Parameters follow InstDisc [0] settings.

    The most important settings are:
        - Num nearest neighbors: 200
        - Temperature: 0.1

    References:
       - [0]: InstDict, 2018, https://arxiv.org/abs/1805.01978
    """
    print_rank_zero("Running KNN evaluation...")

    # Setup training data.
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(
                mean=IMAGENET_NORMALIZE["mean"],
                std=IMAGENET_NORMALIZE["std"],
            ),
        ]
    )
    train_dataset = LightlyDataset(input_dir=str(train_dir), transform=transform)
    
    # Determine indices for train/val split 
    num_train = int(0.8 * len(train_dataset))
    indices = list(range(len(train_dataset)))  
    train_indices, val_indices = indices[:num_train], indices[num_train:]

    # Create Samplers and DataLoaders
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_device,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        sampler=train_sampler,
    )

    # Setup validation data.
#     val_dataset = LightlyDataset(input_dir=str(val_dir), transform=transform)
    val_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_device,
        shuffle=False,
        num_workers=num_workers,
        sampler=valid_sampler
    )

    classifier = KNNClassifier(
        model=model,
        num_classes=num_classes,
        feature_dtype=torch.float16,
    )

    # Run KNN evaluation.
    metric_callback = MetricCallback()
    trainer = Trainer(
        max_epochs=1,
        accelerator=accelerator,
        devices=devices,
#         logger=TensorBoardLogger(save_dir=str(log_dir), name="knn_eval"),
        callbacks=[
            DeviceStatsMonitor(),
            metric_callback,
        ],
        strategy=strategy,
        num_sanity_val_steps=0,
    )
    trainer.fit(
        model=classifier,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    for metric in ["val_top1", "val_top5"]:
        print_rank_zero(f"knn {metric}: {max(metric_callback.val_metrics[metric])}")
    
    return max(metric_callback.val_metrics["val_top1"])


if __name__ == "__main__":
    
    import argparse
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint')

    args = parser.parse_args()
    
    model = VICRegLMinimal.load_from_checkpoint(args.checkpoint)
    model.eval()
    
    knn_eval(
        model=model, 
        train_dir="/data/2m/val/", 
#         val_dir="/data/2m/val", 
        log_dir=".", 
        batch_size_per_device=64, 
        num_workers=1, 
        accelerator='gpu', 
        devices=1, 
        num_classes=6,
    )
    