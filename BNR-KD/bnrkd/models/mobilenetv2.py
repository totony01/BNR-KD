"""MobileNetV2 backbone for CIFAR (32x32 inputs).

Inverted-residual network with a configurable width multiplier. The 0.5x
variant is the lightweight student used in our heterogeneous experiments.
``forward(x, is_feat=True)`` returns five stage outputs.
"""
import torch.nn as nn


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden = int(round(inp * expand_ratio))
        self.use_residual = (stride == 1 and inp == oup)

        layers = []
        if expand_ratio != 1:
            layers += [nn.Conv2d(inp, hidden, 1, bias=False),
                       nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True)]
        layers += [
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, oup, 1, bias=False),
            nn.BatchNorm2d(oup),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        return x + out if self.use_residual else out


class MobileNetV2(nn.Module):
    """CIFAR MobileNetV2.

    Args:
        width_mult: channel multiplier (0.5 for the lightweight student).
    """

    def __init__(self, width_mult=1.0, num_classes=100):
        super().__init__()
        input_channel = _make_divisible(32 * width_mult)
        last_channel = _make_divisible(1280 * width_mult) if width_mult > 1.0 else 1280

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(input_channel), nn.ReLU6(inplace=True),
        )
        self.relu = nn.ReLU6(inplace=True)

        # (expand_ratio, out_channels, num_blocks, stride) per stage
        spec = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]
        self.features = nn.ModuleList()
        for t, c, n, s in spec:
            output_channel = _make_divisible(c * width_mult)
            blocks = []
            for i in range(n):
                blocks.append(InvertedResidual(input_channel, output_channel,
                                               s if i == 0 else 1, t))
                input_channel = output_channel
            self.features.append(nn.Sequential(*blocks))

        self.conv2 = nn.Sequential(
            nn.Conv2d(input_channel, last_channel, 1, bias=False),
            nn.BatchNorm2d(last_channel), nn.ReLU6(inplace=True),
        )
        self.fc = nn.Linear(last_channel, num_classes)

    def forward(self, x, is_feat=False):
        out = self.conv1(x)
        f0 = out
        feats = [f0]
        for i, stage in enumerate(self.features):
            out = stage(out)
            feats.append(out)
        out = self.conv2(out)
        out = out.mean(dim=(2, 3))
        logits = self.fc(out)
        if is_feat:
            # return five representative stages
            return [feats[0], feats[1], feats[3], feats[5], self.relu(out)], logits
        return logits


def mobilenetv2(num_classes=100, width_mult=0.5):
    return MobileNetV2(width_mult=width_mult, num_classes=num_classes)
