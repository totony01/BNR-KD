"""Loss functions used by BNR-KD.

``diffusion_loss`` implements the denoising objective. The default
``smooth_log`` variant rescales the per-element error by the noise variance of
the sampled diffusion step: ``log(1 + MSE / sigma_t^2)`` with
``sigma_t^2 = 1 - alpha_bar_t``. Large errors behave like a log-SNR term
(strong gradient), while errors approaching zero degrade to plain MSE, so the
gradient stays bounded at both ends.
"""
import torch
import torch.nn.functional as F


def diffusion_loss(noise_pred, noise, timesteps=None, alphas_cumprod=None,
                   loss_type='smooth_log'):
    """Denoising loss between predicted and injected noise.

    Args:
        noise_pred: predicted noise, same shape as ``noise``.
        noise: ground-truth Gaussian noise.
        timesteps: sampled diffusion steps, required for ``smooth_log``.
        alphas_cumprod: cumulative alphas of the scheduler.
        loss_type: one of ``mse``, ``smooth_log``, ``huber``.
    """
    error = noise_pred - noise
    mse = error.pow(2).mean()
    if loss_type == 'mse':
        return mse
    if loss_type == 'huber':
        return F.huber_loss(noise_pred, noise, delta=0.1)
    if loss_type == 'smooth_log':
        per_sample = error.pow(2).flatten(1).sum(1) / error[0].numel()
        if alphas_cumprod is not None and timesteps is not None:
            sigma2 = 1.0 - alphas_cumprod[timesteps]
        else:
            sigma2 = noise.pow(2).flatten(1).sum(1).detach() / error[0].numel()
        return torch.log1p(per_sample / (sigma2 + 1e-8)).mean()
    raise ValueError(f'unknown loss_type: {loss_type}')


def kd_kl_loss(student_logits, teacher_logits, temperature=1.0):
    """Temperature-scaled KL divergence between student and teacher logits."""
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction='batchmean',
    ) * temperature ** 2


def feature_mse(student_feat, teacher_feat):
    return F.mse_loss(student_feat, teacher_feat)
