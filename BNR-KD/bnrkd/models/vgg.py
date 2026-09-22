"""VGG backbones for CIFAR.

Follows the CIFAR configuration used in our experiments: an all-3x3
convolutional stack with five stages and no fully connected hidden layers.
``forward(x, is_feat=True)`` returns the five stage outputs so the distillation
head can attach at any depth.
"""
import torch.nn as nn

CFG = {
    'VGG8': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
              512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name='VGG13', num_classes=100):
        super().__init__()
        self.feature = self._make_layers(CFG[vgg_name])
        self.classifier = nn.Linear(512, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AdaptiveAvgPool2d((1, 1))]
        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False):
        # Collect one feature per pooling stage so that every VGG variant
        # exposes the same number of stages.
        feats = []
        for layer in self.feature:
            x = layer(x)
            if isinstance(layer, nn.MaxPool2d):
                feats.append(x)
        out = x.flatten(1)
        logits = self.classifier(out)
        if is_feat:
            while len(feats) < 4:      # pad for shallow configurations
                feats.insert(0, feats[0])
            return feats[:4] + [self.relu(out)], logits
        return logits


def vgg13(num_classes=100):
    return VGG('VGG13', num_classes)


def vgg8(num_classes=100):
    return VGG('VGG8', num_classes)
