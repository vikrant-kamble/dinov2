# Import necessary modules
from tqdm import tqdm
import sys
sys.path.append("/cnvrg/")
from dinov2.models import vision_transformer as vits
from dinov2.configs import dinov2_default_config
from dinov2.eval.linear import create_linear_input
import os
import os.path
from PIL import Image

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import logging
import torch
from torchvision.datasets.folder import IMG_EXTENSIONS, DatasetFolder
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
import torchvision
from urllib.parse import urlparse
from omegaconf import OmegaConf


IMG_SIZE = 224
logger = logging.getLogger("dinov2")
def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    if urlparse(pretrained_weights).scheme:  # If it looks like an URL
        state_dict = torch.hub.load_state_dict_from_url(pretrained_weights, map_location="cpu")
    else:
        state_dict = torch.load(pretrained_weights, map_location="cpu")
    if checkpoint_key is not None and checkpoint_key in state_dict:
        logger.info(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    logger.info("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)
def default_loader(path: str) -> Any:

    if torchvision.get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)

class IndexedDatasetFolder(DatasetFolder):

    def __init__(
        self,
        root: str,
        loader: Callable[[str], Any],
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super().__init__(root,
            loader,
            extensions,
            transform,
            target_transform,
            is_valid_file)



    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        sample_name = path.split("/")[-1]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, sample_name


def get_dataloaders_for_embedding_creation(data_dir_train, data_dir_val):
    # Define transformations: random crop, random flip, convert to tensor, and normalize
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE),  # Resize and crop the image to a 224x224 square
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # Normalize the image with mean and standard deviation
    ])

    val_transform = transforms.Compose([
        transforms.CenterCrop(IMG_SIZE),  # Resize and crop the image to a 224x224 square
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # Normalize the image with mean and standard deviation
    ])

    # Load the dataset from directory and apply transformations

    train_dataset = IndexedDatasetFolder(root=data_dir_train, transform=val_transform, extensions=IMG_EXTENSIONS, loader=default_loader)
    val_dataset = IndexedDatasetFolder(root=data_dir_val, transform=val_transform, extensions=IMG_EXTENSIONS, loader=default_loader)


    # Create data loaders for train and validation sets
    # They provide an easy way to iterate over the dataset in mini-batches
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # Shuffle the training data
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)  # No need to shuffle validation data

    return {'train': train_loader, 'val': val_loader}, len(train_dataset.classes)  # Return loaders and number of classes in the dataset



def get_backbone_embeddings(model, dataloaders, device, saving_dir):
    model = model.to(device)
    model.eval()
    phase = "val"
    with tqdm(total=len(dataloaders[phase]), unit='batch') as p:
        for inputs, labels, sample_name in dataloaders[phase]:
            os.makedirs(os.path.join(saving_dir, "embeddings", "test", str(labels.item())), exist_ok=True)
            filename_out = os.path.join(saving_dir, "embeddings", "test", str(labels.item()), str(sample_name[0]).replace(".png",".t"))
            inputs = inputs.to(device)
            with torch.set_grad_enabled(False):  # Only calculate gradients in training phase
                features = model.get_intermediate_layers(
                    inputs, 1, return_class_token=True
                )
                outputs = create_linear_input(features, use_n_blocks=1, use_avgpool=True)
                torch.save(torch.squeeze(outputs), filename_out)
    phase = "train"
    os.makedirs(os.path.join(saving_dir, "embeddings", phase), exist_ok=True)
    with tqdm(total=len(dataloaders[phase]), unit='batch') as p:
        for inputs, labels, sample_name in dataloaders[phase]:
            os.makedirs(os.path.join(saving_dir, "embeddings", "train", str(labels.item())), exist_ok=True)
            filename_out = os.path.join(saving_dir, "embeddings", "train", str(labels.item()), str(sample_name[0]).replace(".png",".t"))
            inputs = inputs.to(device)
            with torch.set_grad_enabled(False):  # Only calculate gradients in training phase
                features = model.get_intermediate_layers(
                    inputs, 1, return_class_token=True
                )
                outputs = create_linear_input(features, use_n_blocks=1, use_avgpool=True)
                torch.save(torch.squeeze(outputs), filename_out)

def build_model(args, only_teacher=False, img_size=224):
    #args.arch = args.arch.removesuffix("_memeff")
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate=args.drop_path_rate,
            drop_path_uniform=args.drop_path_uniform,
        )
        embed_dim = student.embed_dim
    return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher=False):
    return build_model(cfg.student, only_teacher=only_teacher, img_size=cfg.crops.global_crops_size)
def build_model_for_eval(config, pretrained_weights, cuda=True):
    model, _ = build_model_from_cfg(config, only_teacher=True)
    load_pretrained_weights(model, pretrained_weights, "teacher")
    model.eval()
    if cuda:
        model.cuda()
    return model

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
        '--outpath',
        help='Path to the output directory',
        required=True
    )

    parser.set_defaults(swap=False)

    args = parser.parse_args()

    default_cfg = OmegaConf.create(dinov2_default_config)

    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(default_cfg, cfg)

    pretrained_weights = args.checkpoint
    model = build_model_for_eval(cfg, pretrained_weights, cuda=False)
    # add output directory if it doesn't exist
    if not os.path.exists(args.outpath):
        os.makedirs(args.outpath)

    # 3. Training
    data_dir = args.data

    data_dir_train = os.path.join(data_dir, "train")
    data_dir_test = os.path.join(data_dir, "val")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloaders, num_classes = get_dataloaders_for_embedding_creation(data_dir_train, data_dir_test)

    # run inference once and save the weights
    get_backbone_embeddings(model, dataloaders, device, args.outpath)


#python /cnvrg/scripts/dino_embedding_creation.py --data /data/dino_fixed_rg_evaluation_imagenet --config /data/dino_models/30epochswd01_config.yaml --checkpoint /data/dino_models/30epochswd01_87499.pth --outpath /cnvrg/output/rgevaluation_embeddings_no_aug