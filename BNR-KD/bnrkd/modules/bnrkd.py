"""BNR-KD -- Bidirectional Noise Regulation for Knowledge Distillation.

The framework regulates noise in two complementary directions over the same
feature space:

* **denoising (NPM)** projects a student feature toward the teacher
  representation, correcting the *direction* of the representation;
* **add-noise (KPM)** perturbs the teacher anchor to synthesise local views,
  acting as a regulariser inside a trustworthy neighbourhood.

Total objective for one feature level::

    L = L_CE + w * ( L_npm + L_align + L_kpm + L_rec )

where ``L_align`` is the clean alignment term, ``L_kpm`` the perturbation term
and ``w`` the distillation weight.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .npm import NoisePurificationModule
from .kpm import KnowledgePerturbationModule


class BidirectionalNoiseRegulation(nn.Module):
    """Full BNR-KD distillation head for a single teacher/student feature pair.

    Args:
        student_channels: channels of the student feature map.
        teacher_channels: channels of the teacher feature map.
        kpm_alpha / kpm_views / kpm_beta: KPM perturbation settings.
        use_ae: compress teacher features with a linear autoencoder.
        loss_type: denoising loss variant (``smooth_log`` / ``mse`` / ``huber``).
        reduction: bottleneck ratio of the first denoising block.
        block2_dw: use depthwise branches in the second denoising block.
    """

    def __init__(self, student_channels, teacher_channels, kernel_size=3,
                 inference_steps=5, num_train_timesteps=1000, use_ae=False,
                 ae_channels=None, loss_type='smooth_log', reduction=None,
                 block2_dw=True, kpm_alpha=0.1, kpm_views=3, kpm_beta=0.8):
        super().__init__()
        self.npm = NoisePurificationModule(
            student_channels, teacher_channels, kernel_size=kernel_size,
            inference_steps=inference_steps, num_train_timesteps=num_train_timesteps,
            use_ae=use_ae, ae_channels=ae_channels, loss_type=loss_type,
            reduction=reduction, block2_dw=block2_dw,
        )
        self.kpm = KnowledgePerturbationModule(alpha=kpm_alpha, num_views=kpm_views,
                                               beta=kpm_beta)

    def forward(self, student_feat, teacher_feat):
        """Returns ``(purified, teacher_latent, dict_of_losses)``."""
        purified, teacher_latent, diff_loss, rec_loss = self.npm(student_feat, teacher_feat)
        align_loss = self.kpm(purified, teacher_latent, F.mse_loss)
        losses = {'align': align_loss, 'diffusion': diff_loss}
        if rec_loss is not None:
            losses['reconstruction'] = rec_loss
        return purified, teacher_latent, losses
