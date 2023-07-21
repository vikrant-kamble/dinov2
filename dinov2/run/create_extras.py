#!/usr/bin/env python

import argparse
from dinov2.data import datasets
import logging
import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description="Create or recreate extra files for a dataset")
    parser.add_argument("--dataset_class", type=str, required=True, help="Dataset class")
    parser.add_argument("--path", type=str, required=True, help="The path to the root of the dataset")
    args = parser.parse_args()

    logger.info(f"Using dataclass {args.dataset_class} with root {args.path}")

    try:
        Cl = getattr(datasets, args.dataset_class)
    except AttributeError:
        raise ValueError(f"Unknown dataset class: {args.dataset_class}")
    
    root = args.path
    extra = root

    logger.info("Creating extra files")

    for split in [Cl.Split.TRAIN, Cl.Split.VAL]:
        
        dataset = Cl(split=split, root=root, extra=extra)
        dataset.dump_extra()
    
    # Testing dataset
    logger.info("Testing dataset")
    for split in [Cl.Split.TRAIN, Cl.Split.VAL]:
        ds = Cl(split=split, root=root, extra=extra)

        logger.info(f"Split {split}")

        for i in tqdm.tqdm(range(len(ds)), total=len(ds)):
            _ = ds.get_target(i)
            _ = ds.get_image_data(i)
