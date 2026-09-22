"""ImageNet-scale and Transformer backbones.

Thin wrappers that expose the same ``forward(x, is_feat=True)`` interface as the
CIFAR backbones, so the distillation head can be reused unchanged.

* ``resnet18`` / ``resnet34`` / ``resnet50`` wrap ``torchvision.models``.
* ``swin_t`` / ``swin_l`` wrap ``timm`` (an optional dependency).

Both wrappers collect four stage outputs via forward hooks; the returned list is
``[stem, stage1, stage2, stage3, stage4]`` to match the five-entry convention
used elsewhere in this package.
"""
import torch
import torch.nn as nn


class TorchvisionResNet(nn.Module):
    """torchvision ResNet exposing intermediate stage features."""

    def __init__(self, arch='resnet34', num_classes=1000, pretrained=False):
        super().__init__()
        import torchvision.models as tvm
        weights = 'DEFAULT' if pretrained else None
        self.backbone = getattr(tvm, arch)(weights=weights)
        if num_classes != 1000:
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self._feats = {}

    def _collect(self, name):
        def hook(_m, _i, out):
            self._feats[name] = out
        return hook

    def forward(self, x, is_feat=False):
        b = self.backbone
        handles = [
            b.conv1.register_forward_hook(self._collect('stem')),
            b.layer1.register_forward_hook(self._collect('s1')),
            b.layer2.register_forward_hook(self._collect('s2')),
            b.layer3.register_forward_hook(self._collect('s3')),
            b.layer4.register_forward_hook(self._collect('s4')),
        ]
        logits = b(x)
        for h in handles:
            h.remove()
        if is_feat:
            feats = [self._feats['stem'], self._feats['s1'], self._feats['s2'],
                     self._feats['s3'], self._feats['s4']]
            return feats, logits
        return logits


class TimmBackbone(nn.Module):
    """``timm`` model wrapper exposing stage features.

    ``timm`` models report features via ``forward_features``, which returns a
    spatial map for convolutional architectures and a token sequence for
    Transformers. Both are converted to ``B x C x H x W`` here.
    """

    def __init__(self, arch='swin_tiny_patch4_window7_224', num_classes=1000,
                 pretrained=False):
        super().__init__()
        import timm
        self.backbone = timm.create_model(arch, pretrained=pretrained,
                                          num_classes=num_classes)

    def forward(self, x, is_feat=False):
        if not is_feat:
            return self.backbone(x)
        feats = self.backbone.forward_intermediates(x) if hasattr(
            self.backbone, 'forward_intermediates') else None
        if feats is None:
            out = self.backbone.forward_features(x)
            if out.dim() == 3:              # B x N x C -> B x C x h x w
                b, n, c = out.shape
                h = w = int(n ** 0.5)
                out = out.transpose(1, 2).reshape(b, c, h, w)
            feats = [out]
        elif isinstance(feats, tuple):
            feats = list(feats[0])
        logits = self.backbone(x)
        # pad / trim to a five-entry list for interface consistency
        while len(feats) < 5:
            feats.insert(0, feats[0])
        return feats[-5:], logits


def resnet18(num_classes=1000, **kw):
    return TorchvisionResNet('resnet18', num_classes, **kw)


def resnet34(num_classes=1000, **kw):
    return TorchvisionResNet('resnet34', num_classes, **kw)


def resnet50_imagenet(num_classes=1000, **kw):
    return TorchvisionResNet('resnet50', num_classes, **kw)


def swin_t(num_classes=1000, **kw):
    return TimmBackbone('swin_tiny_patch4_window7_224', num_classes, **kw)


def swin_l(num_classes=1000, **kw):
    return TimmBackbone('swin_large_patch4_window7_224', num_classes, **kw)
