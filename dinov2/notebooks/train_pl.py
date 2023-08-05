import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import torchmetrics

import sys
sys.path.append("/cnvrg/")

from omegaconf import OmegaConf
from dinov2.configs import dinov2_default_config
from dinov2.eval.setup import build_model_for_eval
from dinov2.eval.linear import create_linear_input

# Define constants (adjust these to your dataset)
IMG_SIZE = 224      # Image size expected by your backbone model

# 1. Dataset Preparation
class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=512, invert=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform = transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE),  # Resize and crop the image to a 224x224 square
            transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            transforms.GaussianBlur((5, 5), sigma=(0.1, 0.5)),
            transforms.ToTensor(),  # Convert the image to a tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize the image with mean and standard deviation
        ])
        
        self.val_transform = transform = transforms.Compose([
            transforms.CenterCrop(IMG_SIZE),  # Resize and crop the image to a 224x224 square
            transforms.ToTensor(),  # Convert the image to a tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize the image with mean and standard deviation
        ])
        
        # whether invert train with val (because we screw up preparing RG dataset)
        self.invert = invert

    def setup(self, stage=None):
        
        if self.invert:
            print("SWAPPING train and val")
            self.train_data = ImageFolder(os.path.join(self.data_dir, 'val'), transform=self.transform)
            self.val_data = ImageFolder(os.path.join(self.data_dir, 'train'), transform=self.transform)
        
        else:
            
            self.train_data = ImageFolder(os.path.join(self.data_dir, 'train'), transform=self.transform)
            self.val_data = ImageFolder(os.path.join(self.data_dir, 'val'), transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=12)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=12)

# 2. Model Definition
class ImageClassifier(pl.LightningModule):
    def __init__(self, backbone_model, n_classes):
        super().__init__()
        self.backbone = backbone_model
        self.backbone.eval()  # Freeze the backbone model
        for param in self.backbone.parameters():
            param.requires_grad = False
        
#         self.classifier = nn.Linear(in_features=1024, out_features=n_classes)
        
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.Dropout(0.25),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.Dropout(0.25),
            nn.GELU(),
            nn.Linear(256, n_classes)
        )
        
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes, top_k=1)
        
        self.training_loss = []
        self.valid_loss = []
        self.valid_acc = []

    def forward(self, x):
#         x = self.backbone(x)
        
        features = self.backbone.get_intermediate_layers(
                    x, 1, return_class_token=True
        )
        x = create_linear_input(features, use_n_blocks=1, use_avgpool=True)
        
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y, label_smoothing=0.1)
        self.training_loss.append(loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat.softmax(dim=-1), y)
        
        self.valid_loss.append(loss)
        self.valid_acc.append(acc)
        
        return {'loss': loss, 'acc': acc}
    
    def on_validation_epoch_end(self):
        avg_loss_val = torch.stack(self.valid_loss).mean()
        avg_acc = torch.stack(self.valid_acc).mean()
        
        self.log('val_loss', avg_loss_val, prog_bar=True, sync_dist=True)
        self.log('val_acc', avg_acc, prog_bar=True, sync_dist=True)
        
        # training_loss is not defined when testing at the beginning of training
        if self.training_loss:
            avg_loss_train = torch.stack(self.training_loss).mean()
            self.log('train_loss', avg_loss_train, prog_bar=True, sync_dist=True)
        
        self.training_loss.clear()
        self.valid_loss.clear()
        self.valid_acc.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.001)
        
#         scheduler = torch.optim.lr_scheduler.OneCycleLR(
#             optimizer, max_lr=1e-3, total_steps=self.trainer.estimated_stepping_batches
#         )
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser(description='Run Pythorch Lightning training')
    parser.add_argument(
        '--data', 
        help='Path to an imagenet dataset with train and val directories',
        required=True
    )
    parser.add_argument(
        '--config', 
        help='Path to the Dinov2 configuration corresponding to the checkpoint',
        required=True
    )
    parser.add_argument(
        '--checkpoint', 
        help='Path to the checkpoint for the dinov2 model',
        required=True
    )
    parser.add_argument(
        '--n_classes', 
        help='Number of classes',
        required=True,
        type=int
    )
    
    parser.add_argument('--swap', action='store_true', help="Swap train and val")
    parser.add_argument('--no-swap', dest='swap', action='store_false')
    parser.set_defaults(swap=True)

    args = parser.parse_args()
    
    default_cfg = OmegaConf.create(dinov2_default_config)
    cfg = OmegaConf.load(args.config)

    pretrained_weights = args.checkpoint
    model = build_model_for_eval(cfg, pretrained_weights, cuda=False)

    
    # 3. Training
    data_dir = args.data
    data_module = ImageNetDataModule(data_dir, batch_size=512, invert=args.swap)
    classifier_model = ImageClassifier(model, args.n_classes)
    
    neptune_logger = NeptuneLogger(
        api_key = os.getenv("NEPTUNE_API_TOKEN"),
        project = os.environ.get('NEPTUNE_PROJECT'),
        tags = ['mlp', 'giacomo'],
        log_model_checkpoints=False  # otherwise Neptune saves _every_ checkpoint for a total of 100 Gb
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    trainer = pl.Trainer(
        max_epochs=100, 
        accelerator="gpu", 
        devices=4, 
        log_every_n_steps=10, 
        logger=neptune_logger,
        callbacks=[lr_monitor]
    )
    trainer.fit(classifier_model, datamodule=data_module)
