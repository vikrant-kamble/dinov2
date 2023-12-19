import os
import subprocess

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from omegaconf import OmegaConf
from pj_cape_foundation_eval.models.dino_embedding_creator import DinoEmbeddingCreator, build_model_for_eval
from slimtp.lib.common import make_directory
from slimtp.modules import AMDataset
from slimtp.pipelines import AMPreprocessing
from torch.utils.data import DataLoader

from dinov2.configs import dinov2_default_config


class DinoInferenceDataset(AMDataset):
    def __getitem__(self, index: int):
        item = self.get_df_data(index)
        sample_name = item["identifier"]
        x_clean = self.get_aug_item(item, self.t_clean)
        return x_clean, sample_name


@hydra.main(config_path="../slimtp_configs/living_area/configs", config_name="config")
def run(config):
    if config.get("download_cache", {}).get("use"):
        print("Downloading cache...")

        if os.path.exists(os.path.join(config.cache.local_cache_path, "cache.lmdb")):
            print("Cache already exists, skipping download.")
        else:
            cache_dir = config.cache.local_cache_path
            os.makedirs(cache_dir, exist_ok=True)
            os.system(f"gsutil -m cp -r {config.download_cache.gcs_path} {cache_dir}")

    ampreproc = AMPreprocessing(config)
    ampreproc.run()

    df = pd.read_hdf(config.data_index_path)
    df["cache_key"] = df["cache_key"].apply(eval)

    # Create Inference dataset
    amdataset = DinoInferenceDataset(config=config, df=df)

    # Create torch dataloader
    data_loader = DataLoader(amdataset, batch_size=config.batch_size, shuffle=False)

    # Download checkpoint model using gstuil
    dino_v2_backbone_path = config.dino_v2_backbone_path
    if not os.path.exists(dino_v2_backbone_path):
        subprocess.run(["gsutil", "cp", config.dinov2_backbone, dino_v2_backbone_path])

    # Download config file using gstuil
    dino_v2_config_path = config.dino_v2_config_path
    if not os.path.exists(dino_v2_config_path):
        subprocess.run(["gsutil", "cp", config.dinov2_config, dino_v2_config_path])

    # Create the model
    default_cfg = OmegaConf.create(dinov2_default_config)
    dino_v2_config = OmegaConf.load(dino_v2_config_path)
    dino_v2_config = OmegaConf.merge(default_cfg, dino_v2_config)

    backbone_model = build_model_for_eval(dino_v2_config, dino_v2_backbone_path, cuda=True)
    dino_model = DinoEmbeddingCreator(backbone_model=backbone_model)

    trainer = pl.Trainer(max_epochs=1, accelerator="gpu", devices=1, log_every_n_steps=10)
    _ = dino_model.eval()

    # Do inference
    results = trainer.predict(dino_model, dataloaders=data_loader)

    # parse inference results and save
    embeddings = torch.cat([r[0] for r in results], 0).cpu().numpy()
    identifiers = np.concatenate([list(r[1]) for r in results])
    embeddings_df = pd.DataFrame(
        embeddings, index=identifiers, columns=[f"emb_{i}" for i in range(embeddings.shape[1])]
    )

    # Join with the original dataframe
    df = df.join(embeddings_df, on="identifier")

    # Write to disk
    make_directory(config.tensors_path)
    df.to_hdf(config.tensors_path, key="data", mode="w")


if __name__ == "__main__":
    run()
