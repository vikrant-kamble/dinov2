import logging
from abc import ABC
from collections import OrderedDict
from typing import Any, Optional
import torch
from dinov2.utils.efficient_net import EfficientNet


class _BaseBackbone(ABC):
    """
    Base class for pre-trained models.
    """

    def __init__(self, model_name: str, layer_key_name: str = "model.backbone", **kwargs: Any) -> None:
        # pylint: disable=unused-argument
        super().__init__()
        self._name = model_name
        self._backbone: Optional[torch.nn.Module] = None
        self._layer_key_name = layer_key_name

    def get_features(self, add_pooling: bool = True, checkpoint_path: Optional[str] = None) -> torch.nn.Module:
        """
        Get the model computing the feature.

        :param add_pooling: whether add the pooling layer at the top
        :param checkpoint_path: checkpoint to load, none to skip.

        :return: the model computing the feature
        """
        backbone = self._backbone
        assert backbone is not None
        if add_pooling:
            backbone = torch.nn.Sequential(backbone, torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(start_dim=1))
        else:
            backbone = torch.nn.Sequential(backbone, torch.nn.Flatten(start_dim=1))
            backbone = backbone
        # load state dict if provided
        if checkpoint_path:
            loading_result = backbone.load_state_dict(self._extract_state_dict(checkpoint_path, self._layer_key_name), strict=False)
            print(loading_result)
        return backbone

    @property
    def model_name(self) -> str:
        return self._name

    def is_jit_script_exportable(self) -> bool:
        try:
            _ = torch.jit.script(self._backbone)
        # pylint: disable=W0703
        except Exception:
            return False
        return True

    @staticmethod
    def _extract_state_dict(
        checkpoint_path: str, layer_key_name: str = "model.backbone"
    ) -> "OrderedDict[str, torch.Tensor]":
        local_path = checkpoint_path
        # download if external model

        # read the state-dict
        state_dict = torch.load(local_path)["state_dict"]
        state_dict_sub = OrderedDict(
            [(k[len(f"{layer_key_name}.") :], v) for k, v in state_dict.items() if k.startswith(f"{layer_key_name}.")]
        )
        if len(state_dict_sub) == 0:
            logging.warning(
                "Can not load the state_dict from the backbone initialization checkpoint %s.", checkpoint_path
            )
        return state_dict_sub


class CapeBackbone(_BaseBackbone):
    """
    Implementation of CAPE pre-trained backbones.
    """

    AVAILABLE_MODELS = ["efficientnet-b" + str(i) for i in range(9)]

    def __init__(self, model_name: str, pretrained: bool = True, **kwargs: Any) -> None:
        super().__init__(model_name, **kwargs)
        # create backbone
        # pylint: disable=consider-iterating-dictionary
        in_channels = kwargs["in_channels"] if "in_channels" in kwargs.keys() else 3
        self._backbone = CapeBackbone._init_features(model_name, pretrained, in_channels=in_channels)

    @staticmethod
    def _init_features(model_name: str, pretrained: bool, in_channels: int = 3) -> torch.nn.Module:
        model = EfficientNet.from_pretrained(
            model_name,
            pretrained=pretrained,
            weight_name=f"imagenet-{model_name.split('-')[-1]}",
            in_channels=in_channels,
        ).features
        return model

    def __str__(self) -> str:
        return f'CapeBackbone(model_name={self.model_name}, imagenet-{self.model_name.split("-")[-1]})'
