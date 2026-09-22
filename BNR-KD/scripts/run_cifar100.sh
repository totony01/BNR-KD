#!/usr/bin/env bash
# End-to-end recipe on CIFAR-100.
#
# Stage 1 trains the teacher to convergence.
# Stage 2 distils the student with BNR-KD.
#
# Usage:  bash scripts/run_cifar100.sh
set -e

DATA_ROOT=./data
CKPT_DIR=./checkpoints
mkdir -p "${CKPT_DIR}"

# ---------------------------------------------------------------- stage 1
# Teacher: ResNet-32x4 on CIFAR-100 (240 epochs).
python tools/train_teacher.py \
    --arch resnet32x4 \
    --dataset cifar100 \
    --data-root "${DATA_ROOT}" \
    --epochs 240 \
    --lr 0.05 \
    --batch-size 128 \
    --milestones 150 180 210 \
    --output "${CKPT_DIR}/resnet32x4_cifar100.pth"

# ---------------------------------------------------------------- stage 2
# Student: ResNet-8x4 distilled with BNR-KD.
# `--block2-dw 0` selects standard convolutions in the second denoising block,
# which works better when the student feature is wide (the default depthwise
# variant is preferred for narrow features).
python tools/train_student.py \
    --teacher-arch resnet32x4 \
    --student-arch resnet8x4 \
    --teacher-ckpt "${CKPT_DIR}/resnet32x4_cifar100.pth" \
    --dataset cifar100 \
    --data-root "${DATA_ROOT}" \
    --feature-stage 3 \
    --use-ae 0 \
    --loss-type mse \
    --kpm-alpha 0.1 \
    --kpm-views 3 \
    --kpm-beta 0.8 \
    --kd-weight 2.0 \
    --epochs 240 \
    --block2-dw 0 \
    --output "${CKPT_DIR}/resnet8x4_cifar100_bnrkd.pth"
