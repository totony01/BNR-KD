"""ShuffleNetV2 backbone for CIFAR (32x32 inputs).

Lightweight channel-split network used as a heterogeneous student. The five
feature stages are returned by ``forward(x, is_feat=True)``.
"""
import torch
import torch.nn as nn


def channel_shuffle(x, groups):
    n, c, h, w = x.size()
    return x.view(n, groups, c // groups, h, w).permute(0, 2, 1, 3, 4).reshape(n, c, h, w)


class BasicBlock(nn.Module):
    """Channel-split block: one branch is passed through, the other is transformed."""

    def __init__(self, channels):
        super().__init__()
        half = channels // 2
        self.conv1 = nn.Conv2d(half, half, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(half)
        self.conv2 = nn.Conv2d(half, half, 3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(half)
        self.conv3 = nn.Conv2d(half, half, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(half)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        out = torch.relu(self.bn1(self.conv1(x2)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return torch.relu(torch.cat([x1, out], dim=1))


class DownBlock(nn.Module):
    """Spatial downsampling block that doubles the channel count."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = out_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, mid, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid)
        self.conv4 = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(in_channels)
        self.conv5 = nn.Conv2d(in_channels, mid, 3, stride=2, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(mid)

    def forward(self, x):
        branch1 = torch.relu(self.bn1(self.conv1(x)))
        branch1 = torch.relu(self.bn2(self.conv2(branch1)))
        branch1 = torch.relu(self.bn3(self.conv3(branch1)))
        branch2 = torch.relu(self.bn4(self.conv4(x)))
        branch2 = torch.relu(self.bn5(self.conv5(branch2)))
        return channel_shuffle(torch.cat([branch1, branch2], dim=1), 2)


class ShuffleNetV2(nn.Module):
    """CIFAR ShuffleNetV2.

    Args:
        net_size: 0.5, 1.0 or 1.5 width multiplier.
    """

    CONFIGS = {
        0.5: dict(out_channels=(24, 48, 96, 512), num_blocks=(3, 7, 3)),
        1.0: dict(out_channels=(40, 80, 160, 512), num_blocks=(3, 7, 3)),
        1.5: dict(out_channels=(48, 96, 192, 1024), num_blocks=(3, 7, 3)),
    }

    def __init__(self, net_size=1.0, num_classes=100):
        super().__init__()
        cfg = self.CONFIGS[net_size]
        out_channels = cfg['out_channels']
        num_blocks = cfg['num_blocks']

        self.conv1 = nn.Conv2d(3, 24, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(24, out_channels[0], num_blocks[0])
        self.layer2 = self._make_layer(out_channels[0], out_channels[1], num_blocks[1])
        self.layer3 = self._make_layer(out_channels[1], out_channels[2], num_blocks[2])
        self.conv2 = nn.Conv2d(out_channels[2], out_channels[3], 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels[3])
        self.fc = nn.Linear(out_channels[3], num_classes)

    def _make_layer(self, in_channels, out_channels, blocks):
        layers = [DownBlock(in_channels, out_channels)]
        for _ in range(blocks):
            layers.append(BasicBlock(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False):
        out = self.relu(self.bn1(self.conv1(x)))
        f0 = out
        out = self.layer1(out)
        f1 = out
        out = self.layer2(out)
        f2 = out
        out = self.layer3(out)
        f3 = out
        out = self.relu(self.bn2(self.conv2(out)))
        out = out.mean(dim=(2, 3))
        f4 = out
        logits = self.fc(out)
        if is_feat:
            return [f0, f1, f2, f3, f4], logits
        return logits


def shufflenetv2(num_classes=100):
    return ShuffleNetV2(net_size=1.0, num_classes=num_classes)
