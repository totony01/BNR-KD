"""CIFAR-style ResNet backbones.

Provides the ``resnet8x4`` / ``resnet32x4`` variants used in our experiments.
``forward(x, is_feat=True)`` returns per-stage feature maps, which allows the
distillation modules to hook onto any intermediate representation without
modifying the backbone itself.
"""
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Standard two-conv residual block with an optional projection shortcut."""

    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super().__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        preact = out + self.shortcut(x)
        out = F.relu(preact)
        if self.is_last:
            return out, preact
        return out


class ResNet(nn.Module):
    """ResNet for 32x32 inputs.

    Args:
        depth: total depth (8, 14, 20, 32, 44, 56, 110).
        num_filters: per-stage output channels, e.g. [32, 64, 128, 256].
        num_classes: number of output classes.
    """

    def __init__(self, depth, num_filters, num_classes=100):
        super().__init__()
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        self.num_channels = num_filters[3]
        self.conv1 = nn.Conv2d(3, num_filters[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.layer1 = self._make_layer(num_filters[0], num_filters[1], n, 1)
        self.layer2 = self._make_layer(num_filters[1], num_filters[2], n, 2)
        self.layer3 = self._make_layer(num_filters[2], num_filters[3], n, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_filters[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_planes, planes, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            layers.append(BasicBlock(
                in_planes if i == 0 else planes,
                planes,
                stride=stride if i == 0 else 1,
                is_last=(i == num_blocks - 1),
            ))
        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False):
        out = F.relu(self.bn1(self.conv1(x)))
        f0 = out
        out, _ = self.layer1(out)
        f1 = out
        out, _ = self.layer2(out)
        f2 = out
        out, _ = self.layer3(out)
        f3 = out
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        f4 = out
        logits = self.fc(out)
        if is_feat:
            return [f0, f1, f2, f3, f4], logits
        return logits


def resnet8x4(num_classes=100):
    return ResNet(8, [32, 64, 128, 256], num_classes)


def resnet32x4(num_classes=100):
    return ResNet(32, [32, 64, 128, 256], num_classes)


class Bottleneck(nn.Module):
    """Bottleneck residual block used by the deeper CIFAR ResNet variants."""

    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super().__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        preact = out + self.shortcut(x)
        out = F.relu(preact)
        if self.is_last:
            return out, preact
        return out


class ResNetBottleneck(nn.Module):
    """CIFAR ResNet built from bottleneck blocks (e.g. ResNet-50)."""

    def __init__(self, num_blocks, num_filters, num_classes=100):
        super().__init__()
        assert len(num_blocks) == 3, 'expected three bottleneck stages'
        assert len(num_filters) == 3, 'num_filters holds the three bottleneck widths'
        self.num_channels = num_filters[2] * Bottleneck.expansion

        self.conv1 = nn.Conv2d(3, num_filters[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.layer1 = self._make_layer(num_filters[0], num_filters[0], num_blocks[0], 1)
        self.layer2 = self._make_layer(num_filters[0] * Bottleneck.expansion,
                                       num_filters[1], num_blocks[1], 2)
        self.layer3 = self._make_layer(num_filters[1] * Bottleneck.expansion,
                                       num_filters[2], num_blocks[2], 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.num_channels, num_classes)

    def _make_layer(self, in_planes, planes, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            layers.append(Bottleneck(in_planes if i == 0 else planes * Bottleneck.expansion,
                                     planes, stride if i == 0 else 1,
                                     is_last=(i == num_blocks - 1)))
        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False):
        out = F.relu(self.bn1(self.conv1(x)))
        f0 = out
        out, _ = self.layer1(out); f1 = out
        out, _ = self.layer2(out); f2 = out
        out, _ = self.layer3(out); f3 = out
        out = self.avgpool(out).view(out.size(0), -1)
        f4 = out
        logits = self.fc(out)
        if is_feat:
            return [f0, f1, f2, f3, f4], logits
        return logits


def resnet110x2(num_classes=100):
    """Wide-and-deep CIFAR ResNet used as a large teacher."""
    return ResNet(110, [32, 64, 128, 256], num_classes)


def resnet50(num_classes=100):
    """CIFAR ResNet-50: bottleneck blocks in a 3/4/6 layout."""
    return ResNetBottleneck([3, 4, 6], [64, 128, 256], num_classes)
