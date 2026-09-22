"""Noise purification network.

The denoiser predicts the noise component of a corrupted feature map. It is
built from :class:`StepAdaptiveReparamBlock`, a multi-branch block whose
branches are combined by diffusion-step-dependent gates. Because all branches
are linear up to BatchNorm, they can be folded into a single convolution at
inference time, which keeps the deployed cost low.

Also provided: a light linear autoencoder used to compress high-dimensional
teacher features before they enter the diffusion process.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

EMBEDDING_MAX_STEPS = 1280


# ---------------------------------------------------------------------------
# BatchNorm folding helpers
# ---------------------------------------------------------------------------
def fold_bn(weight, bias, bn):
    """Fold a BatchNorm layer into the convolution that precedes it."""
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    w = weight * scale.view(-1, *([1] * (weight.dim() - 1)))
    if bias is None:
        bias = torch.zeros_like(bn.running_mean)
    b = (bias - bn.running_mean) * scale + bn.bias
    return w, b


def identity_kernel(channels, bn, device, dtype):
    """Equivalent 3x3 kernel + bias of an identity map followed by BatchNorm."""
    k = torch.zeros(channels, channels, 3, 3, device=device, dtype=dtype)
    idx = torch.arange(channels, device=device)
    k[idx, idx, 1, 1] = 1.0
    return fold_bn(k, None, bn)


def identity_dw_kernel(channels, bn, device, dtype):
    """Depthwise counterpart of :func:`identity_kernel`."""
    k = torch.zeros(channels, 1, 3, 3, device=device, dtype=dtype)
    k[:, 0, 1, 1] = 1.0
    return fold_bn(k, None, bn)


def pad_k1x1(k1, b1):
    """Pad a 1x1 kernel to 3x3."""
    k = torch.zeros(k1.shape[0], k1.shape[1], 3, 3, device=k1.device, dtype=k1.dtype)
    k[:, :, 1, 1] = k1[:, :, 0, 0]
    return k, b1


def pad_dw_k1x1(k1, b1):
    k = torch.zeros(k1.shape[0], 1, 3, 3, device=k1.device, dtype=k1.dtype)
    k[:, 0, 1, 1] = k1[:, 0, 0, 0]
    return k, b1


def compose_bottleneck(w_a, b_a, w_b, b_b, w_c, b_c):
    """Compose 1x1 -> 3x3 -> 1x1 (all BN-folded) into one 3x3 kernel."""
    w_ab = torch.einsum('ompq,mi->oipq', w_b, w_a[:, :, 0, 0])
    b_ab = torch.einsum('ompq,m->o', w_b, b_a) + b_b
    w_final = torch.einsum('om,mipq->oipq', w_c[:, :, 0, 0], w_ab)
    b_final = torch.einsum('om,m->o', w_c[:, :, 0, 0], b_ab) + b_c
    return w_final, b_final


# ---------------------------------------------------------------------------
# Denoising block
# ---------------------------------------------------------------------------
class StepAdaptiveReparamBlock(nn.Module):
    """Time-step adaptive reparameterizable fusion block.

    Training form::

        h = ReLU( sum_i alpha_i(T) * BN_i( f_i(z_T + t_emb) ) )

    where the gates ``alpha_i(T)`` are produced by a small MLP from the step
    embedding. Two variants are provided:

    * ``use_dw=False`` -- branches are identity, 1x1 conv and standard 3x3
      conv; they fold into a single standard 3x3 convolution.
    * ``use_dw=True`` -- branches are identity, 1x1 depthwise and 3x3
      depthwise; they fold into a single 3x3 depthwise convolution.

    Setting ``reduction`` turns the 3x3 branch into a linear bottleneck, which
    lowers the parameter count while remaining exactly composable.
    """

    def __init__(self, channels, use_dw=False, emb_dim=32, reduction=None):
        super().__init__()
        self.channels = channels
        self.use_dw = use_dw
        self.reduction = reduction

        self.bn0 = nn.BatchNorm2d(channels)
        if use_dw:
            self.conv1 = nn.Conv2d(channels, channels, 1, groups=channels)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
            self.bn2 = nn.BatchNorm2d(channels)
        else:
            self.conv1 = nn.Conv2d(channels, channels, 1)
            self.bn1 = nn.BatchNorm2d(channels)
            if reduction:
                mid = max(channels // reduction, 1)
                self.conv2a = nn.Conv2d(channels, mid, 1)
                self.bn2a = nn.BatchNorm2d(mid)
                self.conv2b = nn.Conv2d(mid, mid, 3, padding=1)
                self.bn2b = nn.BatchNorm2d(mid)
                self.conv2c = nn.Conv2d(mid, channels, 1)
                self.bn2c = nn.BatchNorm2d(channels)
            else:
                self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(channels)

        self.time_emb = nn.Embedding(EMBEDDING_MAX_STEPS, emb_dim)
        self.gate = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, 3),
        )

    def _gating(self, t):
        t = t.long()
        return self.gate(self.time_emb(t)).softmax(-1)

    def _forward_train(self, x, alpha):
        o0 = self.bn0(x)
        o1 = self.bn1(self.conv1(x))
        if self.use_dw:
            o2 = self.bn2(self.conv2(x))
        elif self.reduction:
            o2 = self.bn2c(self.conv2c(self.bn2b(self.conv2b(self.bn2a(self.conv2a(x))))))
        else:
            o2 = self.bn2(self.conv2(x))
        out = (alpha[:, 0:1, None, None] * o0
               + alpha[:, 1:2, None, None] * o1
               + alpha[:, 2:3, None, None] * o2)
        return F.relu(out)

    def _fold(self, alpha):
        """Fold BN into each branch, scale by the gates, and merge to one conv."""
        dev, dtype = self.bn0.weight.device, self.bn0.weight.dtype
        if self.use_dw:
            k0, b0 = identity_dw_kernel(self.channels, self.bn0, dev, dtype)
            k1, b1 = pad_dw_k1x1(*fold_bn(self.conv1.weight, self.conv1.bias, self.bn1))
            k2, b2 = fold_bn(self.conv2.weight, self.conv2.bias, self.bn2)
        else:
            k0, b0 = identity_kernel(self.channels, self.bn0, dev, dtype)
            k1, b1 = pad_k1x1(*fold_bn(self.conv1.weight, self.conv1.bias, self.bn1))
            if self.reduction:
                k2, b2 = compose_bottleneck(
                    *fold_bn(self.conv2a.weight, self.conv2a.bias, self.bn2a),
                    *fold_bn(self.conv2b.weight, self.conv2b.bias, self.bn2b),
                    *fold_bn(self.conv2c.weight, self.conv2c.bias, self.bn2c))
            else:
                k2, b2 = fold_bn(self.conv2.weight, self.conv2.bias, self.bn2)
        w = alpha[0] * k0 + alpha[1] * k1 + alpha[2] * k2
        b = alpha[0] * b0 + alpha[1] * b1 + alpha[2] * b2
        return w, b

    def _forward_eval(self, x, alpha):
        if alpha.shape[0] == 1:
            w, b = self._fold(alpha[0])
            groups = self.channels if self.use_dw else 1
            return F.relu(F.conv2d(x, w, b, padding=1, groups=groups))
        return self._forward_train(x, alpha)

    def forward(self, x, t):
        alpha = self._gating(t)
        if alpha.dim() == 1:
            alpha = alpha.reshape(1, 3)
        if self.training:
            return self._forward_train(x, alpha)
        return self._forward_eval(x, alpha)


class NoisePredictionNet(nn.Module):
    """Lightweight noise predictor: two reparameterizable blocks + 1x1 head."""

    def __init__(self, channels_in, kernel_size=3, reduction=None, block2_dw=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.time_embedding = nn.Embedding(EMBEDDING_MAX_STEPS, channels_in)
        if kernel_size == 3:
            self.block1 = StepAdaptiveReparamBlock(channels_in, use_dw=False,
                                                   reduction=reduction)
            self.block2 = StepAdaptiveReparamBlock(channels_in, use_dw=block2_dw)
            self.pred = nn.Conv2d(channels_in, channels_in, 1)
        else:
            self.pred = nn.Sequential(
                nn.Conv2d(channels_in, channels_in * 4, 1),
                nn.BatchNorm2d(channels_in * 4), nn.ReLU(inplace=True),
                nn.Conv2d(channels_in * 4, channels_in, 1),
                nn.BatchNorm2d(channels_in),
                nn.Conv2d(channels_in, channels_in * 4, 1),
                nn.BatchNorm2d(channels_in * 4), nn.ReLU(inplace=True),
                nn.Conv2d(channels_in * 4, channels_in, 1),
            )

    def forward(self, noisy, t):
        if t.dtype != torch.long:
            t = t.type(torch.long)
        feat = noisy + self.time_embedding(t)[..., None, None]
        if self.kernel_size == 1:
            return self.pred(feat)
        feat = self.block1(feat, t)
        feat = self.block2(feat, t)
        return self.pred(feat)


class TimestepEstimator(nn.Module):
    """Predicts a normalized starting timestep from a student feature map.

    The output is a per-sample scalar in ``[0, 1]`` used to align the student
    feature with the noise level the denoiser was trained on.
    """

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        if kernel_size == 3:
            self.feat = nn.Sequential(
                nn.Conv2d(channels, channels // 8, 1),
                nn.BatchNorm2d(channels // 8), nn.ReLU(inplace=True),
                nn.Conv2d(channels // 8, channels // 8, 3, padding=1),
                nn.BatchNorm2d(channels // 8), nn.ReLU(inplace=True),
                nn.Conv2d(channels // 8, channels, 1),
                nn.BatchNorm2d(channels),
                nn.AdaptiveAvgPool2d(1),
            )
        else:
            self.feat = nn.Sequential(
                nn.Conv2d(channels, channels * 2, 1),
                nn.BatchNorm2d(channels * 2), nn.ReLU(inplace=True),
                nn.Conv2d(channels * 2, channels, 1),
                nn.BatchNorm2d(channels),
            )
        self.pred = nn.Linear(channels, 2)

    def forward(self, x):
        x = self.feat(x).flatten(1)
        return self.pred(x).softmax(1)[:, 0]


class LinearAutoEncoder(nn.Module):
    """Compresses teacher features to a lower-dimensional latent space."""

    def __init__(self, channels, latent_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, latent_channels, 1, padding=0),
            nn.BatchNorm2d(latent_channels),
        )
        self.decoder = nn.Sequential(nn.Conv2d(latent_channels, channels, 1, padding=0))

    def forward(self, x):
        hidden = self.encoder(x)
        return hidden, self.decoder(hidden)
