import io
from typing import Any, Dict, List, Optional, Union

import mtp.cache.base
import numpy as np
import pandas as pd
import torch
from PIL import Image
from slimtp.lib.cache import get_cache
from slimtp.lib.common import merge_dict
from torch import Tensor
from torchvision import transforms

RESIZE_SIZE = 256
IMG_SIZE = 224


class DinoInferenceDataset:
    def __init__(self, config, df):
        self.config = config
        self.df = df.reset_index(drop=True)
        self.cache = self._get_cache(config)

    @staticmethod
    def _get_cache(config: Dict[str, Any]) -> Optional[mtp.cache.base.Cache]:
        cache_args = {
            "readonly": True,
            "lazy_init": True,
            "cloud": False,
        }
        return get_cache(**merge_dict(config["cache"], cache_args))  # cache in dataloader mode

    @property
    def t_clean(self) -> Any:
        return transforms.Compose(
            [
                transforms.Resize(RESIZE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _load_image(self, item):
        image_bytes = self.cache.get(item["cache_key"])
        image = Image.open(io.BytesIO(image_bytes))
        return image.convert("RGB")

    def get_aug_item(
        self, item: Union[Dict, pd.Series, pd.DataFrame], aug_comp: Any
    ) -> Union[torch.Tensor, np.ndarray, List[np.ndarray]]:
        """
        Get an input item with augmentation, view-awared.

        :param item: one item in dataset
        :param aug_comp: an augmentation composition
        :return: torch tensor of the augmented input. If aug_comp is None, then the raw image(s) are returned
            as numpy array or list of array.
        """

        def _apply_aug(image: np.ndarray) -> Union[torch.Tensor, np.ndarray]:
            return aug_comp(image)

        image = self._load_image(item)
        return _apply_aug(image)

    def get_df_data(self, index: int) -> Union[Dict, pd.Series, pd.DataFrame]:
        """
        Return df row corresponding to a given index.
        """
        return self.df.iloc[index]

    def __getitem__(self, index: int):
        item = self.get_df_data(index)
        sample_name = item["identifier"]
        x_clean = self.get_aug_item(item, self.t_clean)
        return x_clean, sample_name
