"""Training engine for BNR-KD students.

The teacher is frozen and only the student plus the distillation head are
optimised. Features are taken from a configurable stage of each backbone, so
the same code covers homogeneous pairs (e.g. WRN-40-2 -> WRN-40-1) and
heterogeneous pairs (e.g. ResNet-32x4 -> ShuffleNetV2).
"""
import logging
import time

import torch
import torch.nn.functional as F

from .losses import kd_kl_loss

logger = logging.getLogger(__name__)


class Distiller:
    """Wraps teacher/student/head and runs the distillation loop.

    Args:
        teacher: frozen teacher network.
        student: student network to optimise.
        head: a :class:`~bnrkd.modules.bnrkd.BidirectionalNoiseRegulation` instance.
        feature_stage: index of the feature map used for distillation.
        kd_weight: weight of the distillation block relative to cross-entropy.
        logit_temperature: temperature used for the logit-level KL term.
    """

    def __init__(self, teacher, student, head, feature_stage=3, kd_weight=2.0,
                 logit_temperature=1.0):
        self.teacher = teacher
        self.student = student
        self.head = head
        self.feature_stage = feature_stage
        self.kd_weight = kd_weight
        self.logit_temperature = logit_temperature

    def _features(self, model, x):
        feats, logits = model(x, is_feat=True)
        return feats[self.feature_stage], logits

    def compute_loss(self, x, y):
        """Return ``(total_loss, parts)`` for one mini-batch."""
        with torch.no_grad():
            teacher_feat, teacher_logits = self._features(self.teacher, x)

        student_feat, student_logits = self._features(self.student, x)

        ce_loss = F.cross_entropy(student_logits, y)
        purified, teacher_latent, head_losses = self.head(student_feat, teacher_feat)

        parts = {
            'ce': ce_loss,
            'align': head_losses['align'],
            'diffusion': head_losses['diffusion'],
            'logit': kd_kl_loss(student_logits, teacher_logits, self.logit_temperature),
        }
        if 'reconstruction' in head_losses:
            parts['reconstruction'] = head_losses['reconstruction']

        distill = sum(parts[k] for k in ('align', 'diffusion', 'logit', 'reconstruction')
                      if k in parts)
        total = ce_loss + self.kd_weight * distill

        with torch.no_grad():
            parts['cosine'] = F.cosine_similarity(
                purified.flatten(1), teacher_latent.flatten(1)).mean()
        return total, parts

    @torch.no_grad()
    def evaluate(self, loader):
        self.student.eval()
        correct = total = 0
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            correct += (self.student(x).argmax(1) == y).sum().item()
            total += y.size(0)
        return 100.0 * correct / total

    def train(self, train_loader, test_loader, epochs=240, lr=0.05, momentum=0.9,
              weight_decay=5e-4, milestones=(150, 180, 210), log_interval=1,
              callback=None):
        """Run the full training schedule and return the best accuracy."""
        params = list(self.student.parameters()) + list(self.head.parameters())
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum,
                                    weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=list(milestones),
                                                         gamma=0.1)
        self.teacher.eval()
        best_acc = 0.0

        for epoch in range(epochs):
            self.student.train()
            self.head.train()
            t0 = time.time()
            running = {}

            for x, y in train_loader:
                x, y = x.cuda(), y.cuda()
                loss, parts = self.compute_loss(x, y)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
                optimizer.step()

                for k, v in parts.items():
                    running[k] = running.get(k, 0.0) + float(v)
            scheduler.step()

            acc = self.evaluate(test_loader)
            best_acc = max(best_acc, acc)
            if (epoch + 1) % log_interval == 0:
                n = len(train_loader)
                logger.info(
                    'epoch %d/%d | ce %.3f | align %.3f | kpm %.3f | cos %.3f | '
                    'acc %.2f | best %.2f | %.0fs',
                    epoch + 1, epochs, running['ce'] / n, running['align'] / n,
                    running['diffusion'] / n, running['cosine'] / n,
                    acc, best_acc, time.time() - t0)
            if callback is not None:
                callback(epoch, acc, best_acc)

        return best_acc
