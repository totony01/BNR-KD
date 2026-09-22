"""Run BNR-KD distillation on CIFAR.

Example (homogeneous pair):
    python tools/train_student.py --teacher-ckpt checkpoints/wrn40_2_cifar100.pth \\
        --teacher-arch wrn_40_2 --student-arch wrn_40_1 --dataset cifar100

Example (heterogeneous pair):
    python tools/train_student.py --teacher-ckpt checkpoints/resnet32x4_cifar100.pth \\
        --teacher-arch resnet32x4 --student-arch resnet8x4 --block2-dw 0
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bnrkd.data import build_dataloaders
from bnrkd.models import build_backbone
from bnrkd.modules import BidirectionalNoiseRegulation
from bnrkd.utils import count_parameters, set_seed, setup_logger
from bnrkd.trainer import Distiller


def parse_args():
    p = argparse.ArgumentParser(description='BNR-KD distillation')
    p.add_argument('--teacher-arch', default='wrn_40_2')
    p.add_argument('--student-arch', default='wrn_40_1')
    p.add_argument('--teacher-ckpt', required=True)
    p.add_argument('--dataset', default='cifar100', choices=['cifar10', 'cifar100'])
    p.add_argument('--data-root', default='./data')

    p.add_argument('--feature-stage', type=int, default=3,
                   help='index of the feature map used for distillation')
    p.add_argument('--use-ae', type=int, default=1,
                   help='compress teacher features with a linear autoencoder')
    p.add_argument('--ae-channels', type=int, default=0, help='0 = teacher_channels / 2')
    p.add_argument('--loss-type', default='smooth_log',
                   choices=['smooth_log', 'mse', 'huber'])
    p.add_argument('--inference-steps', type=int, default=5)
    p.add_argument('--reduction', type=int, default=0)
    p.add_argument('--block2-dw', type=int, default=1,
                   help='use depthwise branches in the second denoising block')

    p.add_argument('--kpm-alpha', type=float, default=0.1)
    p.add_argument('--kpm-views', type=int, default=3)
    p.add_argument('--kpm-beta', type=float, default=0.8)

    p.add_argument('--epochs', type=int, default=240)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.05)
    p.add_argument('--weight-decay', type=float, default=5e-4)
    p.add_argument('--kd-weight', type=float, default=2.0)
    p.add_argument('--logit-temperature', type=float, default=1.0)
    p.add_argument('--milestones', type=int, nargs='+', default=[150, 180, 210])
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', default='checkpoints/student_bnrkd.pth')
    return p.parse_args()


def infer_channels(model, stage, num_classes):
    """Return the channel width of a given feature stage via a dummy forward."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        feats, _ = model(torch.zeros(1, 3, 32, 32).cuda(), is_feat=True)
    if was_training:
        model.train()
    return feats[stage].shape[1]


def main():
    args = parse_args()
    logger = setup_logger('bnrkd.student', log_file=os.path.join(
        os.path.dirname(args.output), 'train_log.txt'))
    set_seed(args.seed)

    num_classes = 100 if args.dataset == 'cifar100' else 10
    train_loader, test_loader = build_dataloaders(
        args.dataset, args.data_root, args.batch_size, args.workers)

    teacher = build_backbone(args.teacher_arch, num_classes=num_classes).cuda()
    ckpt = torch.load(args.teacher_ckpt, map_location='cuda')
    teacher.load_state_dict(ckpt.get('state_dict', ckpt))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    logger.info('teacher %s loaded (stored acc %s)', args.teacher_arch,
                ckpt.get('acc', 'n/a'))

    student = build_backbone(args.student_arch, num_classes=num_classes).cuda()
    logger.info('student %s | %.2fM params', args.student_arch,
                count_parameters(student))

    t_ch = infer_channels(teacher, args.feature_stage, num_classes)
    s_ch = infer_channels(student, args.feature_stage, num_classes)
    logger.info('feature stage %d | teacher %d ch | student %d ch',
                args.feature_stage, t_ch, s_ch)

    head = BidirectionalNoiseRegulation(
        student_channels=s_ch, teacher_channels=t_ch,
        inference_steps=args.inference_steps, use_ae=bool(args.use_ae),
        ae_channels=args.ae_channels or None, loss_type=args.loss_type,
        reduction=args.reduction or None, block2_dw=bool(args.block2_dw),
        kpm_alpha=args.kpm_alpha, kpm_views=args.kpm_views, kpm_beta=args.kpm_beta,
    ).cuda()
    logger.info('BNR-KD head | %.3fM extra params', count_parameters(head))

    distiller = Distiller(teacher, student, head, feature_stage=args.feature_stage,
                          kd_weight=args.kd_weight,
                          logit_temperature=args.logit_temperature)

    best_state = {}

    def on_epoch(epoch, acc, best):
        if acc >= best:
            best_state.clear()
            best_state.update({k: v.detach().cpu()
                               for k, v in student.state_dict().items()})

    best_acc = distiller.train(train_loader, test_loader, epochs=args.epochs,
                               lr=args.lr, weight_decay=args.weight_decay,
                               milestones=args.milestones, callback=on_epoch)

    torch.save({'state_dict': best_state, 'acc': best_acc,
                'student_arch': args.student_arch}, args.output)
    logger.info('done. best accuracy %.2f%% -> %s', best_acc, args.output)


if __name__ == '__main__':
    main()
