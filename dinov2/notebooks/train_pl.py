import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
import torchmetrics

import sys
sys.path.append("/cnvrg/")

from omegaconf import OmegaConf
from dinov2.configs import dinov2_default_config

from dinov2.eval.setup import build_model_for_eval

default_cfg = OmegaConf.create(dinov2_default_config)
cfg = OmegaConf.load("30epochswd01_config.yaml")

pretrained_weights = "30epochswd01_87499.pth"
model = build_model_for_eval(cfg, pretrained_weights, cuda=False)

# Define constants (adjust these to your dataset)
NUM_CLASSES = 6  # Number of classes in your dataset
IMG_SIZE = 224      # Image size expected by your backbone model

# 1. Dataset Preparation
class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=512):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform = transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE),  # Resize and crop the image to a 224x224 square
            transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
            transforms.ToTensor(),  # Convert the image to a tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize the image with mean and standard deviation
        ])
        
        self.val_transform = transform = transforms.Compose([
            transforms.CenterCrop(IMG_SIZE),  # Resize and crop the image to a 224x224 square
            transforms.ToTensor(),  # Convert the image to a tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize the image with mean and standard deviation
        ])

    def setup(self, stage=None):
        self.train_data = ImageFolder(os.path.join(self.data_dir, 'val'), transform=self.transform)
        self.val_data = ImageFolder(os.path.join(self.data_dir, 'train'), transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=12)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=12)

# 2. Model Definition
class ImageClassifier(pl.LightningModule):
    def __init__(self, backbone_model):
        super().__init__()
        self.backbone = backbone_model
        self.backbone.eval()  # Freeze the backbone model
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.classifier = nn.Linear(in_features=1024, out_features=NUM_CLASSES)
        
        self.accuracy = torchmetrics.Accuracy()
        
        self.training_loss = []
        self.valid_loss = []
        self.valid_acc = []

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        self.training_loss.append(loss)
        return loss
    
    def on_training_epoch_end(self):
        avg_loss = torch.stack(self.training_loss).mean()
        self.log('train_loss', avg_loss, prog_bar=True, sync_dist=True)
        self.training_loss.clear()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat.softmax(dim=-1), y)
        
        self.valid_loss.append(loss)
        self.valid_acc.append(acc)
        
        return {'loss': loss, 'acc': acc}
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.valid_loss).mean()
        avg_acc = torch.stack(self.valid_acc).mean()
        self.log('val_loss', avg_loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', avg_acc, prog_bar=True, sync_dist=True)
        
        self.valid_loss.clear()
        self.valid_acc.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=0.001)
        
#         scheduler = torch.optim.lr_scheduler.OneCycleLR(
#             optimizer, max_lr=1e-3, total_steps=self.trainer.estimated_stepping_batches
#         )
        return {"optimizer": optimizer}

    
if __name__ == "__main__":
    # 3. Training
    data_dir = '/data/rg_evaluation_imagenet/'
    data_module = ImageNetDataModule(data_dir, batch_size=512)
    classifier_model = ImageClassifier(model)
    
    neptune_logger = NeptuneLogger(
        api_key = os.getenv("NEPTUNE_API_TOKEN"),
        project = os.environ.get('NEPTUNE_PROJECT'),
        tags = ['mlp', 'giacomo']
    )
    
    trainer = pl.Trainer(max_epochs=10, accelerator="gpu", devices=4, log_every_n_steps=10, logger=neptune_logger)
    trainer.fit(classifier_model, datamodule=data_module)
