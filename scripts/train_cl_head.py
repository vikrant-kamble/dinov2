import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import torch
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torchmetrics
import albumentations as aug
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import numpy as np
import math
import sys
sys.path.append("/cnvrg/")

from omegaconf import OmegaConf
from dinov2.configs import dinov2_default_config
from dinov2.eval.setup import build_model_for_eval
from dinov2.eval.linear import create_linear_input

# Define constants (adjust these to your dataset)
IMG_SIZE = 224      # Image size expected by your backbone model

class Tr:
    def __init__(self, alb_t):
        self._alb_t = alb_t
    
    def __call__(self, item):
        
        return self._alb_t(image=np.array(item))['image']

    
def stratified_sampling(samples, n, classes, val_fraction):
    # n is the total number of samples we want in the end, with a val fraction of val_fraction

    # Build lookup table of class -> file
    idxs = {i: [] for i in range(len(classes))}

    for i, s in enumerate(samples):

        idxs[s[1]].append(i)
    
    # Build sample
    sample_train = []
    sample_val = []

    n_train = math.floor((n * (1 - val_fraction))/len(classes))
    n_val = math.floor((n * val_fraction)/len(classes))
    not_chosen_at_all = []
    for c in idxs:
        chosen_for_train = np.random.choice(idxs[c], min(n_train, len(idxs[c])), replace=False)
        not_chosen_for_train = [i for i in idxs[c] if i not in chosen_for_train]
        chosen_for_val = np.random.choice(not_chosen_for_train, min(n_val, len(not_chosen_for_train)), replace=False)
        sample_train.extend(chosen_for_train)
        sample_val.extend(chosen_for_val)
        not_chosen_at_all.extend([i for i in idxs[c] if i not in chosen_for_train and i not in chosen_for_val])
    # add the rest of the samples to the train set
    extension_for_train = np.random.choice(not_chosen_at_all, math.floor(n * (1 - val_fraction)) - len(sample_train), replace=False)
    sample_train.extend(extension_for_train)
    not_chosen_for_train_extension = [i for i in not_chosen_at_all if i not in extension_for_train]
    sample_val.extend(np.random.choice(not_chosen_for_train_extension, math.floor(n * val_fraction) - len(sample_val), replace=False))
    
    return np.array(sample_train), np.array(sample_val)

    
# 1. Dataset Preparation
class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=512, transform_kind='dinov2', val_fraction=0.2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        
        if transform_kind == 'dinov2':
            
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
        elif transform_kind == 'cape':
        
            self.val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(IMG_SIZE),  # Resize and crop the image to a 224x224 square
                transforms.ToTensor(),  # Convert the image to a tensor
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize the image with mean and standard deviation
            ])
            self.transform = Tr(aug.Compose(
                [
                    # resize every chip to NxN
                    aug.Resize(256, 256, interpolation=cv2.INTER_LINEAR),
                    aug.RandomCrop(IMG_SIZE, IMG_SIZE),
                    aug.Flip(),
                    aug.Transpose(),
                    aug.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4, p=0.5),
                    aug.OneOf(
                        [
                            aug.GaussianBlur(),
                            aug.MotionBlur(),
                            aug.Downscale(interpolation=cv2.INTER_NEAREST),
                            aug.ImageCompression(30, 100),
                        ],
                        p=0.25,
                    ),
                    aug.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            ))
        else:
            raise ValueError("Transform kind can only be cape or dinov2")
            
        # whether invert train with val (because we screw up preparing RG dataset)

    def setup(self, stage=None):

        self.train_val_data = ImageFolder(os.path.join(self.data_dir, 'train'), transform=self.transform)
        # take subset of training data in case we want to train/val on less data
        total_samples_num = len(self.train_val_data.samples)
        if args.subset > 0:
            total_samples_num = args.subset
        subset_train, subset_val = stratified_sampling(self.train_val_data.samples, total_samples_num, self.train_val_data.classes, args.valfraction)
        np.save(args.outpath+"/subset_val_indices.npy", subset_val)
        np.save(args.outpath + "/subset_train_indices.npy", subset_train)
        np.save(args.outpath + "/all_samples.npy", self.train_val_data.samples)


        dataset_train = torch.utils.data.Subset(
            self.train_val_data,
            subset_train
        )
        dataset_val = torch.utils.data.Subset(
            self.train_val_data,
            subset_val
        )

        self.train_data = dataset_train
        self.val_data = dataset_val
        self.test_data = ImageFolder(os.path.join(self.data_dir, 'val'), transform=self.val_transform)
    
    @property
    def n_classes(self):
        return len(self.train_val_data.classes)
    
    def train_dataloader(self):
        print(f"Using {len(self.train_data)} training data points")
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        print(f"Using {len(self.val_data)} validation data points")
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=8)

# 2. Model Definition
class ImageClassifier(pl.LightningModule):
    def __init__(self, backbone_model, n_classes):
        super().__init__()
        self.backbone = backbone_model
        self.backbone.eval()  # Freeze the backbone model
        for param in self.backbone.parameters():
            param.requires_grad = False
        
#         self.classifier = nn.Linear(in_features=2048, out_features=n_classes)
        
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.Dropout(0.3),
            nn.GELU(),
            nn.Linear(256, n_classes)
        )
        
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=n_classes, top_k=1)
        
        self.training_loss = []
        self.valid_loss = []
        self.valid_acc = []

    def forward(self, x, **kwargs):
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

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)

        # log 6 example images
        #sample_imgs = x[:6]
        #grid = torchvision.utils.make_grid(sample_imgs)
        #self.logger.experiment.add_image('example_images', grid, 0)

        # calculate acc
        test_acc = self.accuracy(y_hat.softmax(dim=-1), y)

        # log the outputs!
        self.log_dict({'test_loss': loss, 'test_acc': test_acc})
    
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
        
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               factor=0.5, patience=10, threshold=0.0001,
                                                               threshold_mode='rel',
                                                               verbose=True, min_lr=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


if __name__ == "__main__":
    
    import argparse
    pl.utilities.seed.seed_everything(seed=0, workers=True)

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
        '--outpath',
        help='Path to the output directory',
        required=True
    )

    parser.add_argument(
        '--subset', 
        help='How many data points to use. Use 0 to use them all',
        required=False,
        type=int,
        default=0
    )

    parser.add_argument(
        '--valfraction',
        help='What fraction of the training data to use for validation',
        required=False,
        type=float,
        default=0.2
    )
    
    parser.set_defaults(swap=False)

    args = parser.parse_args()
    
    default_cfg = OmegaConf.create(dinov2_default_config)
    cfg = OmegaConf.load(args.config)

    pretrained_weights = args.checkpoint
    model = build_model_for_eval(cfg, pretrained_weights, cuda=False)

    
    # 3. Training
    data_dir = args.data
    data_module = ImageNetDataModule(data_dir, batch_size=1024, transform_kind='dinov2', val_fraction=args.valfraction)
    data_module.setup()
    classifier_model = ImageClassifier(model, data_module.n_classes)
    
    neptune_logger = NeptuneLogger(
        api_key ="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIwODViNzIxOC1jMjUyLTRhMGMtOWYwNi1kYjgxMGFkM2FhMjAifQ==",
        project = "cape/dinov2",
        tags = ['mlp', 'franziska'],
        description=os.getenv('CNVRG_JOB_URL'),
        log_model_checkpoints=False  # otherwise Neptune saves _every_ checkpoint for a total of 100 Gb
    )
    # record the additional metadata
    neptune_logger.experiment['CNVRG_JOB_NAME'] = os.getenv('CNVRG_JOB_NAME', 'local')
    neptune_logger.experiment['CNVRG_JOB_URL'] = os.getenv('CNVRG_JOB_URL', 'local')
    # in order to continue writing into this neptune run,
    # we need to set the run_id to the one from the environment
    run_id = neptune_logger.experiment['sys/id'].fetch()
    # for non-master GPUs run_id will be None
    if run_id:
        os.environ["NEPTUNE_RUN_ID"] = run_id

    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        monitor="epoch",
        mode="max",
        dirpath=args.outpath,
        filename="dinohead-{epoch:02d}-{global_step}",
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu", 
        devices=1, 
        log_every_n_steps=10, 
        logger=neptune_logger,
        callbacks=[lr_monitor, checkpoint_callback]
    )
    
    neptune_logger.experiment["parameters"] = args
    
    trainer.fit(classifier_model, datamodule=data_module)
    classifier_model.eval()
    trainer.test(classifier_model, dataloaders=data_module.test_dataloader())
