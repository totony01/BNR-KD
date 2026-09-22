"""Train a teacher network on CIFAR.

Example:
    python tools/train_teacher.py --arch resnet32x4 --dataset cifar100 \\
        --epochs 240 --output checkpoints/resnet32x4_cifar100.pth
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bnrkd.data import build_dataloaders
from bnrkd.models import build_backbone
from bnrkd.utils import count_parameters, save_checkpoint, set_seed, setup_logger


def parse_args():
    p = argparse.ArgumentParser(description='Train a teacher on CIFAR')
    p.add_argument('--arch', default='resnet32x4', help='backbone name')
    p.add_argument('--dataset', default='cifar100', choices=['cifar10', 'cifar100'])
    p.add_argument('--data-root', default='./data')
    p.add_argument('--epochs', type=int, default=240)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.05)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight-decay', type=float, default=5e-4)
    p.add_argument('--milestones', type=int, nargs='+', default=[150, 180, 210])
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', default='checkpoints/teacher.pth')
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


def main():
    args = parse_args()
    logger = setup_logger('bnrkd.teacher')
    set_seed(args.seed)

    num_classes = 100 if args.dataset == 'cifar100' else 10
    train_loader, test_loader = build_dataloaders(
        args.dataset, args.data_root, args.batch_size, args.workers)

    model = build_backbone(args.arch, num_classes=num_classes).cuda()
    logger.info('%s | %.2fM params | %d classes',
                args.arch, count_parameters(model), num_classes)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=0.1)

    best = 0.0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            loss = F.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        acc = evaluate(model, test_loader)
        if acc > best:
            best = acc
            save_checkpoint({'state_dict': model.state_dict(), 'acc': best,
                             'arch': args.arch, 'epoch': epoch + 1}, args.output)
        logger.info('epoch %d/%d | loss %.4f | acc %.2f | best %.2f | %.0fs',
                    epoch + 1, args.epochs, total_loss / len(train_loader),
                    acc, best, time.time() - t0)

    logger.info('done. best accuracy %.2f%% -> %s', best, args.output)


if __name__ == '__main__':
    main()
