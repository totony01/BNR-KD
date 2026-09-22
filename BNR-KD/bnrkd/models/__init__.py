"""Backbone registry.

Every architecture referenced by the paper is registered here. CIFAR backbones
are self-contained implementations; ImageNet / Transformer models are thin
wrappers around ``torchvision`` and ``timm`` (the latter is optional and only
imported when actually instantiated).
"""

from .resnet import Bottleneck, resnet8x4, resnet32x4, resnet50, resnet110x2
from .wrn import wrn_16_2, wrn_40_1, wrn_40_2
from .vgg import vgg8, vgg13
from .shufflenetv2 import shufflenetv2
from .mobilenetv2 import mobilenetv2
from .imagenet import resnet18, resnet34, resnet50_imagenet, swin_l, swin_t

# ---------------------------------------------------------------------------
# CIFAR backbones (32x32)
# ---------------------------------------------------------------------------
MODEL_ZOO = {
    # ResNet family
    'resnet8x4': resnet8x4,
    'resnet32x4': resnet32x4,
    'resnet110x2': resnet110x2,
    'resnet50': resnet50,
    # WideResNet family
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'wrn_16_2': wrn_16_2,
    # VGG family
    'vgg8': vgg8,
    'vgg13': vgg13,
    # Lightweight families
    'shufflenetv2': shufflenetv2,
    'mobilenetv2': mobilenetv2,
}

# ---------------------------------------------------------------------------
# ImageNet backbones (224x224) and Transformer backbones
# ---------------------------------------------------------------------------
IMAGENET_ZOO = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50_imagenet': resnet50_imagenet,
    'swin_t': swin_t,
    'swin_l': swin_l,
}

ALL_MODELS = {**MODEL_ZOO, **IMAGENET_ZOO}


def build_backbone(name, num_classes=100):
    """Instantiate a backbone by name."""
    if name not in ALL_MODELS:
        raise KeyError(f'unknown backbone {name!r}; available: {sorted(ALL_MODELS)}')
    return ALL_MODELS[name](num_classes=num_classes)


__all__ = ['MODEL_ZOO', 'IMAGENET_ZOO', 'ALL_MODELS', 'build_backbone',
           'Bottleneck', 'resnet8x4', 'resnet32x4', 'resnet50', 'resnet110x2',
           'wrn_40_1', 'wrn_40_2', 'wrn_16_2', 'vgg8', 'vgg13',
           'shufflenetv2', 'mobilenetv2',
           'resnet18', 'resnet34', 'resnet50_imagenet', 'swin_t', 'swin_l']
