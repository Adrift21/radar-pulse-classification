"""Model registry: maps a config name string to a model factory (Module C).

Keeps experiments/train.py architecture-agnostic. The training script
just calls ``build_model(cfg)``; adding ResNet-50 / ViT later is a
one-line registry entry, so the 9-experiment matrix shares one trainer.

Usage
-----
    from models.registry import build_model
    model = build_model(cfg)   # cfg.model.name -> class, cfg.model.kwargs
"""

from __future__ import annotations

from typing import Callable, Dict

import torch.nn as nn

from models.custom_cnn import CustomCNN
from models.resnet50 import ResNet50
from models.vit import ViTSmall

# name -> factory(num_classes, in_channels, **kwargs) -> nn.Module
_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    "custom_cnn": CustomCNN,
    "resnet50": ResNet50,
    "vit_small": ViTSmall,
}


def available_models() -> list[str]:
    return sorted(_REGISTRY)


def build_model(cfg) -> nn.Module:
    """Instantiate the model described by ``cfg.model``.

    All radar TF images are single-channel (in_channels=1) and there are
    ``cfg.data.num_classes`` classes; both are injected here so individual
    YAMLs don't have to repeat them.
    """
    name = cfg.model.name
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Available: {available_models()}.")
    factory = _REGISTRY[name]
    kwargs = dict(cfg.model.kwargs or {})
    return factory(
        num_classes=cfg.data.num_classes,
        in_channels=1,
        **kwargs,
    )
