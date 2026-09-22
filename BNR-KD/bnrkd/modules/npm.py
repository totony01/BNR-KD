"""NPM -- Noise Purification Module (denoising direction).

NPM treats a student feature map as a noisy observation of the corresponding
teacher feature and learns to remove the discrepancy:

    student feature --> project --> estimate noise level --> reverse chain --> purified

An optional linear autoencoder compresses wide teacher features before they
enter the diffusion process, which reduces the cost of both the reverse chain
and the noise-prediction network.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .denoiser import NoisePredictionNet, LinearAutoEncoder, TimestepEstimator
from .diffusion import DDIMScheduler, DenoisingPipeline
from ..losses import diffusion_loss


class NoisePurificationModule(nn.Module):
    """Purifies student features toward the teacher representation.

    Args:
        student_channels: channels of the student feature map.
        teacher_channels: channels of the teacher feature map.
        kernel_size: 3 for spatial features, 1 for vector-like features.
        inference_steps: number of reverse steps at inference.
        num_train_timesteps: length of the training diffusion chain.
        use_ae: compress the teacher feature with a linear autoencoder.
        ae_channels: latent width; defaults to half of ``teacher_channels``.
        reduction: bottleneck ratio of the first denoising block.
        block2_dw: use depthwise branches in the second block.
    """

    def __init__(self, student_channels, teacher_channels, kernel_size=3,
                 inference_steps=5, num_train_timesteps=1000, use_ae=False,
                 ae_channels=None, loss_type='smooth_log', reduction=None,
                 block2_dw=True):
        super().__init__()
        self.use_ae = use_ae
        self.loss_type = loss_type
        self.inference_steps = inference_steps

        if use_ae:
            if ae_channels is None:
                ae_channels = teacher_channels // 2
            self.ae = LinearAutoEncoder(teacher_channels, ae_channels)
            teacher_channels = ae_channels

        self.trans = nn.Conv2d(student_channels, teacher_channels, 1)
        self.model = NoisePredictionNet(teacher_channels, kernel_size=kernel_size,
                                        reduction=reduction, block2_dw=block2_dw)
        self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps,
                                       clip_sample=False)
        self.noise_adapter = TimestepEstimator(teacher_channels, kernel_size)
        self.pipeline = DenoisingPipeline(self.model, self.scheduler,
                                          self.noise_adapter)
        self.proj = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, 1),
            nn.BatchNorm2d(teacher_channels),
        )

    def forward(self, student_feat, teacher_feat):
        """Returns ``(purified, teacher_latent, diffusion_loss, rec_loss)``."""
        student_feat = self.trans(student_feat)

        if self.use_ae:
            hidden, recon = self.ae(teacher_feat)
            rec_loss = F.mse_loss(teacher_feat, recon)
            teacher_feat = hidden.detach()
        else:
            rec_loss = None

        purified = self.pipeline(student_feat, num_inference_steps=self.inference_steps)
        purified = self.proj(purified)

        diff_loss = self._denoising_loss(teacher_feat)
        return purified, teacher_feat, diff_loss, rec_loss

    def _denoising_loss(self, clean_feat):
        """Sample a step, corrupt ``clean_feat`` and score the prediction."""
        noise = torch.randn_like(clean_feat)
        bs = clean_feat.shape[0]
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bs,),
                                  device=clean_feat.device).long()
        noisy = self.scheduler.add_noise(clean_feat, noise, timesteps)
        noise_pred = self.model(noisy, timesteps)
        return diffusion_loss(noise_pred, noise, timesteps=timesteps,
                              alphas_cumprod=self.scheduler.alphas_cumprod,
                              loss_type=self.loss_type)
