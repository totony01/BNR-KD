"""DDIM scheduler and the feature denoising pipeline.

The scheduler implements the standard linear-beta DDIM formulation
(forward noising plus a deterministic reverse step). The pipeline turns a
student feature map into a purified one by first matching its noise level and
then running a short reverse chain.
"""
import numpy as np
import torch


class DDIMScheduler:
    """Linear-beta DDIM scheduler.

    Args:
        num_train_timesteps: length of the training diffusion chain.
        beta_start / beta_end: endpoints of the linear beta schedule.
        clip_sample: whether to clip the predicted clean sample.
    """

    def __init__(self, num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02,
                 clip_sample=False):
        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps,
                                    dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.final_alpha_cumprod = torch.tensor(1.0)
        self.num_inference_steps = None
        self.timesteps = None

    # -- forward process ----------------------------------------------------
    def add_noise(self, original, noise, timesteps):
        """q(x_t | x_0) with cumulative alphas indexed by ``timesteps``."""
        self.alphas_cumprod = self.alphas_cumprod.to(original.device)
        timesteps = timesteps.to(original.device)
        sqrt_alpha = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus = (1.0 - self.alphas_cumprod[timesteps]) ** 0.5
        while sqrt_alpha.dim() < original.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)
        return sqrt_alpha * original + sqrt_one_minus * noise

    def add_noise_from_level(self, original, noise, level):
        """Blend with a continuous noise level (used by the adaptive adapter).

        ``level`` acts as the cumulative alpha directly, so a level close to 1
        keeps the original feature and a level close to 0 approaches pure noise.
        """
        alpha = level.flatten()
        while alpha.dim() < original.dim():
            alpha = alpha.unsqueeze(-1)
        return alpha * original + (1.0 - alpha) * noise

    # -- reverse process ----------------------------------------------------
    def set_timesteps(self, num_inference_steps, device=None):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        ts = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy()
        self.timesteps = torch.from_numpy(ts.astype(np.int64))
        if device is not None:
            self.timesteps = self.timesteps.to(device)

    def step(self, model_output, timestep, sample):
        """One deterministic DDIM update (eta = 0)."""
        prev_t = timestep - self.num_train_timesteps // self.num_inference_steps
        alpha_t = self.alphas_cumprod[timestep]
        alpha_prev = (self.alphas_cumprod[prev_t] if prev_t >= 0
                      else self.final_alpha_cumprod)
        beta_t = 1 - alpha_t

        pred_x0 = (sample - beta_t ** 0.5 * model_output) / alpha_t ** 0.5
        if self.clip_sample:
            pred_x0 = torch.clamp(pred_x0, -1, 1)
        direction = (1 - alpha_prev) ** 0.5 * model_output
        return alpha_prev ** 0.5 * pred_x0 + direction


class DenoisingPipeline:
    """Purify a feature map by matching its noise level and reversing the chain.

    ``feat`` is treated as a noisy observation. The adaptive adapter predicts a
    starting noise level per sample, the feature is blended with Gaussian noise
    accordingly, and a short DDIM chain produces the purified feature.
    """

    def __init__(self, model, scheduler, noise_adapter=None):
        self.model = model
        self.scheduler = scheduler
        self.noise_adapter = noise_adapter

    @torch.no_grad()
    def __call__(self, feat, num_inference_steps=5):
        if self.noise_adapter is not None:
            noise = torch.randn_like(feat)
            level = self.noise_adapter(feat)
            image = self.scheduler.add_noise_from_level(feat, noise, level)
        else:
            image = feat

        self.scheduler.set_timesteps(num_inference_steps * 2, device=feat.device)
        for t in self.scheduler.timesteps[len(self.scheduler.timesteps) // 2:]:
            noise_pred = self.model(image, t)
            image = self.scheduler.step(noise_pred, t, image)
        return image
