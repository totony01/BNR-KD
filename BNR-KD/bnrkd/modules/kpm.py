"""KPM -- Knowledge Perturbation Module (add-noise direction).

While NPM removes noise from the student feature, KPM injects controlled noise
into the teacher anchor to synthesise several "perspectives" of the same
knowledge. The student is aligned with both the clean anchor and the perturbed
views:

    L_kpm = d(purified, anchor) + beta * mean_i d(purified, anchor_i)

The extra term is *added* rather than blended into the clean term, so the
purification objective is never diluted. Perturbing the anchor is what keeps the
synthetic views inside a trustworthy neighbourhood: they vary in local
appearance but preserve the teacher's representation direction.
"""
import torch
import torch.nn as nn


class KnowledgePerturbationModule(nn.Module):
    """Generates perturbed views of the teacher anchor.

    Args:
        alpha: perturbation strength; each view is ``(1 - alpha) * anchor + alpha * noise``.
        num_views: number of synthetic views drawn per step.
        beta: weight of the aggregated perturbation term.
    """

    def __init__(self, alpha=0.1, num_views=3, beta=0.8):
        super().__init__()
        self.alpha = alpha
        self.num_views = num_views
        self.beta = beta

    @torch.no_grad()
    def perturb(self, anchor):
        """Return ``num_views`` perturbed copies of ``anchor``."""
        return [(1.0 - self.alpha) * anchor + self.alpha * torch.randn_like(anchor)
                for _ in range(self.num_views)]

    def forward(self, purified, anchor, distance_fn):
        """Augmented alignment loss for one feature level.

        Args:
            purified: output of NPM, aligned by gradient.
            anchor: clean teacher feature (no gradient).
            distance_fn: ``(pred, target) -> scalar`` distance, e.g. MSE.
        """
        base = distance_fn(purified, anchor)
        if self.num_views == 0 or self.alpha == 0.0:
            return base
        views = self.perturb(anchor)
        extra = sum(distance_fn(purified, v) for v in views) / len(views)
        return base + self.beta * extra
