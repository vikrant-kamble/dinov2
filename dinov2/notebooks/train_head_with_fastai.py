from fastai.distributed import *


from fastai.vision.all import *

from accelerate import notebook_launcher
from accelerate.utils import write_basic_config

write_basic_config()

set_seed(99, True)


import pandas as pd
import timm
from torch import nn
import torch
import functools
import json

import albumentations as aug
from albumentations.pytorch.transforms import ToTensorV2
import cv2

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import torch.distributed

from evaluate import ModelWrapper
from collections import Counter

import tqdm


class AlbumentationsTransform(RandTransform):
    "A transform handler for multiple `Albumentation` transforms"
    
    split_idx, order = None, 2
    
    def __init__(self, train_aug, valid_aug): 
        store_attr()
    
    def before_call(self, b, split_idx):
        self.idx = split_idx
    
    def encodes(self, img: PILImage):
        if self.idx == 0:
        
            aug_img = self.train_aug(image=np.array(img))['image']
        
        else:
        
            aug_img = self.valid_aug(image=np.array(img))['image']
        
        return PILImage.create(aug_img)


def get_train_transforms(size_initial=256, size_final=224):
    
    return aug.Compose(
            [
                # resize every chip to NxN
                aug.Resize(size_initial, size_initial, interpolation=cv2.INTER_LINEAR),
                aug.RandomCrop(size_final, size_final),
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
#                 aug.Normalize(mean=(0.5, 0.5, 0.5), std=(1.0, 1.0, 1.0)),
#                 ToTensorV2(),
            ]
        )


def get_valid_transforms(size_initial, size_final): 
    
    return aug.Compose([
        aug.Resize(size_initial, size_initial, interpolation=cv2.INTER_LINEAR),
        aug.CenterCrop(size_final, size_final, p=1.),
#         aug.Normalize(mean=(0.5, 0.5, 0.5), std=(1.0, 1.0, 1.0)),
#         ToTensorV2(),
    ], p=1.)


def get_weights(path, lbl_dict):
    
    # This is weird but we need to return the weigths for _all_ files
    # (including validation) even though fastai will use the weights only
    # for the training dataset
    files = get_image_files(path)
    
    train_labels = [lbl_dict[parent_label(files[idx])] for idx in GrandparentSplitter()(files)[0]]
    weights = {k: 1 / v for k, v in (dict(Counter(train_labels))).items()}
    
    all_labels = [lbl_dict[parent_label(f)] for f in files]
    all_weights = [weights[l] / len(weights) for l in all_labels]
    
    return all_weights        


# def label_func(fname, lbl_dict):
#     return lbl_dict[parent_label(fname)]

def get_items(path_tuple):
    
    train_path, val_path = path_tuple
    
    if os.path.exists(f"{train_path}/items.parquet") and os.path.exists(f"{val_path}/items.parquet"):
        print("Reading cached items")
        df = pd.concat(
            [
                pd.read_parquet(f"{train_path}/items.parquet"),
                pd.read_parquet(f"{val_path}/items.parquet")
            ]
        ).reset_index(drop=True)
        
        return df
    
    _paths = {
        'train': train_path,
        'val': val_path
    }
    
    train_votes = pd.read_parquet(f"{train_path}/chips/dataset.parquet")
    val_votes = pd.read_parquet(f"{val_path}/chips/dataset.parquet")

    rows = []

    for split, df in zip(['train', 'val'], [train_votes, val_votes]):
        
        groups = df.groupby(['attribute_geometry_id', 'imagery_source'])
        
        for _, grp_df in tqdm.tqdm(groups, total=len(groups)):

            rows.append(
                {
                    "label_counts": grp_df.geometry_labels.value_counts(),
                    "filename": f"{_paths[split]}/chips/{grp_df['filename'].iloc[0]}",
                    "is_valid": split == 'val'
                }
            )
        
        df.to_parquet(f"{_paths[split]}/items.parquet")

    return pd.DataFrame(rows)


def get_x(row):
    
    return PILImage.create(row['filename'])

def get_y(row):
    
    # Return majority voting for now
    return row['label_counts'].idxmax()


def get_dataloaders(config):
    
    # get_y = functools.partial(label_func, lbl_dict=lbl_dict)
    
#     item_tfms = [Resize(256)]
#     # Batch-level operations (executed on GPU)
#     batch_tfms = [
#         # "layer-agnostic" normalization
#         Normalize.from_stats(0.5, 1.0),
#         # Bunch of random transforms
#         *aug_transforms(
#             do_flip=True,  # Do flips with default probability of 0.5
#             flip_vert=True,  # including vertical flip
#             max_rotate=0.0,  # no rotation
#             max_zoom=0.0,  # no zoom
#             max_lighting=0.4,  # color augmentation
#             max_warp=0.0,  # no warp
#             size=256  # Then resize to 256x256
#         ),
#         # Random crop to 224 (it automatically becomes center crop during validation)
#         RandomCrop(224)
#     ]
    
    dblock = DataBlock(
        blocks    = (ImageBlock, CategoryBlock),
        get_items = get_items,
        get_x     = get_x,
        get_y     = get_y,
        splitter  = ColSplitter("is_valid"),
        item_tfms = [AlbumentationsTransform(get_train_transforms(256, 224), get_valid_transforms(256, 224))], 
        batch_tfms = Normalize.from_stats(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
    )
    
    if config['stratified_batches']:
        dls = dblock.weighted_dataloaders(
            (config['train_path'], config['val_path']),
            batch_size=config['batch_size'], 
            wgts=get_weights(path, lbl_dict)
        )
    else:
        dls = dblock.dataloaders(
            (config['train_path'], config['val_path']),
            batch_size=config['batch_size']
        )
    
    return dls


class BackboneWrapper(nn.Module):
    
    def __init__(self, backbone):
        super().__init__()
        
        self.backbone = backbone
    
    def forward(self, x):
        for i in range(4):
            x = self.backbone.downsample_layers[i](x)
            x = self.backbone.stages[i](x)
        return x

def add_fastai_head(backbone, config):
        
    backbone_wrapped = BackboneWrapper(backbone)
    
    return add_head(
        backbone_wrapped, 
        768, 
        config['n_classes'], 
        ps=[config['dropout_1'], config['dropout_2']], 
        lin_ftrs=[config['size_of_linear_layer']]
    )


def get_vicreg_model(checkpoint, config):
    
    import convnext
    
    backbone, embedding = convnext.__dict__[f"convnext_{config['convnext_arch']}"](
        drop_path_rate=0.1,
        layer_scale_init_value=0.0,
    )
        
    state_dict = torch.load(checkpoint, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
        state_dict = {
            key.replace("module.backbone.", ""): value
            for (key, value) in state_dict.items()
        }
    backbone.load_state_dict(state_dict, strict=False)
    
    if config['fastai_head']:
        # Remove unused parameters
        backbone.norm = nn.Identity()
        backbone.avgpool = nn.Identity()

        model = add_fastai_head(backbone, config)

        model[0].requires_grad_(False)
    
    else:
    
        head = nn.Linear(embedding, config['n_classes'])
        head.weight.data.normal_(mean=0.0, std=0.01)
        head.bias.data.zero_()
        # model = nn.Sequential(backbone, head)
        model = ModelWrapper(backbone, head)
        backbone.requires_grad_(False)
        head.requires_grad_(True)
    
    return model


def train(
    train_path,
    val_path,
    model_type, 
    checkpoint, 
    convnext_arch, 
    batch_size=256,
    lr=2e-3, 
    n_epochs=20, 
    dp1=0.2, 
    dp2=0.5, 
    size_of_linear_layer=512, 
    fastai_head=False,
    stratified_batches=False,
    run_id=0,
    callbacks=[],
    save_checkpoint=False
):
    
#     lbl_dict = get_lbl_dict(train_path)
    
    config = {
        "batch_size": batch_size,
        "lr": lr,
        "n_epochs": n_epochs,
        "dropout_1": dp1, 
        "dropout_2": dp2, 
        "size_of_linear_layer": size_of_linear_layer,
        "n_classes": None,
        "convnext_arch": convnext_arch,
        "fastai_head": fastai_head,
        "stratified_batches": stratified_batches,
        "train_path": train_path,
        "val_path": val_path
    }
    
    dls = get_dataloaders(config)
    
    config['n_classes'] = len(dls[0].vocab)
    
    print(json.dumps(config, indent=4))
    
    if model_type == 'vicregl':
        model = get_vicreg_model(checkpoint, config)
    else:
        model = timm.create_model(
            model_type, 
            num_classes=config['n_classes'], 
            pretrained=True
        )
        
    learn = Learner(
        dls, 
        model, 
        metrics=accuracy, 
        cbs=callbacks
    ).to_fp16()
    
#     res = learn.lr_find()
#     import pdb;pdb.set_trace()
# #     1/0
        
    with learn.distrib_ctx(sync_bn=True, in_notebook=False):
        learn.fine_tune(n_epochs, lr, freeze_epochs=0 if model_type=='vicregl' else 1)
    
        if save_checkpoint:

            root_name = f"{run_id}_{os.path.basename(data_path)}_{model_type}"
            
            # Make confusion matrices
            ci = ClassificationInterpretation.from_learner(learn)
            c = ci.confusion_matrix()

            # Accuracy
            acc = c.diagonal().sum() / c.sum()

            # Recall
            fig, sub = plt.subplots()
            row_sums = c.sum(axis=1)
            row_sums_matrix = np.tile(row_sums[:, np.newaxis], (1, c.shape[1])) 
            ConfusionMatrixDisplay(c / row_sums_matrix, display_labels=ci.vocab).plot(cmap='Blues', xticks_rotation='vertical', ax=sub)

            fig.suptitle(f"Precision\n(accuracy: {acc * 100:.1f} %)")
            fig.savefig(f"{root_name}_precision.png", bbox_inches='tight')

            # Precision
            fig, sub = plt.subplots()
            ConfusionMatrixDisplay(c / c.sum(axis=0), display_labels=ci.vocab).plot(cmap='Blues', xticks_rotation='vertical', ax=sub)
            fig.suptitle(f"Recall\n(accuracy: {acc * 100:.1f} %)")
            fig.savefig(f"{root_name}_recall.png", bbox_inches='tight')

        else:

            acc = learn.recorder.metrics[0].value.item()
    
    if save_checkpoint:
        learn.export(f"{root_name}")
    
    return acc

    
if __name__ == "__main__":
    
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', choices=['swin_base_patch4_window7_224_in22k', 'vicregl'], required=True)
    parser.add_argument('--train_data', type=Path, required=True, help="Path to root of Bedrock-like dataset")
    parser.add_argument('--val_data', type=Path, required=True, help="Path to root of Bedrock-like dataset")
    parser.add_argument('--arch', choices=['tiny', 'small', 'base', 'large', 'xlarge'], required=False, default='small')
    parser.add_argument('--checkpoint', type=Path, required=False)

    args = parser.parse_args()
        
    if args.model == 'vicregl' and not args.checkpoint:
        raise argparse.ArgumentError(args.checkpoint, '--checkpoint is required when model is vicregl')
      
    train(
        str(args.train_data.absolute()), 
        str(args.val_data.absolute()), 
        args.model, 
        args.checkpoint, 
        args.arch, 
        save_checkpoint=True
    )
