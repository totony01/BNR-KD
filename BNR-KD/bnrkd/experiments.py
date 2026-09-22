"""Registry of every experiment configuration reported in the paper.

Each entry records the teacher/student pair, the dataset, the reference
student baseline (trained without distillation) and the distillation settings
used for that pair. The registry drives ``tools/run_experiments.py`` and is
also convenient to import when assembling result tables.

Note on architecture names: ``resnet110x2`` / ``resnet50`` are CIFAR variants
(32x32 inputs, 6n+2 / 9n+2 depth). ImageNet pairs use the ``*_imagenet``
entries, which wrap ``torchvision`` models.
"""

# ---------------------------------------------------------------------------
# Shared distillation settings
# ---------------------------------------------------------------------------
CIFAR_DISTILL = dict(feature_stage=3, epochs=240, batch_size=128, lr=0.05,
                     milestones=(150, 180, 210), kd_weight=2.0)
IMAGENET_DISTILL = dict(feature_stage=3, epochs=90, batch_size=1024, lr=0.1,
                        milestones=(30, 60, 90), kd_weight=1.0)


# ---------------------------------------------------------------------------
# CIFAR-100, homogeneous architecture pairs
# ---------------------------------------------------------------------------
CIFAR100_HOMOGENEOUS = [
    dict(name='wrn40_2_wrn40_1', teacher='wrn_40_2', student='wrn_40_1',
         dataset='cifar100', baseline=71.80,
         teacher_ckpt='checkpoints/wrn40_2_cifar100.pth',
         loss_type='smooth_log', use_ae=True, block2_dw=True),
    dict(name='wrn40_2_wrn16_2', teacher='wrn_40_2', student='wrn_16_2',
         dataset='cifar100', baseline=73.26,
         teacher_ckpt='checkpoints/wrn40_2_cifar100.pth',
         loss_type='smooth_log', use_ae=True, block2_dw=True),
    dict(name='resnet32x4_resnet8x4', teacher='resnet32x4', student='resnet8x4',
         dataset='cifar100', baseline=72.50,
         teacher_ckpt='checkpoints/resnet32x4_cifar100.pth',
         loss_type='mse', use_ae=False, block2_dw=False),
    dict(name='vgg13_vgg8', teacher='vgg13', student='vgg8',
         dataset='cifar100', baseline=70.36,
         teacher_ckpt='checkpoints/vgg13_cifar100.pth',
         loss_type='smooth_log', use_ae=True, block2_dw=True),
]

# ---------------------------------------------------------------------------
# CIFAR-100, heterogeneous architecture pairs
# ---------------------------------------------------------------------------
CIFAR100_HETEROGENEOUS = [
    dict(name='resnet50_mobilenetv2', teacher='resnet50', student='mobilenetv2',
         dataset='cifar100', baseline=64.60,
         teacher_ckpt='checkpoints/resnet50_cifar100.pth',
         loss_type='mse', use_ae=False, block2_dw=False),
    dict(name='resnet32x4_shufflenetv2', teacher='resnet32x4', student='shufflenetv2',
         dataset='cifar100', baseline=71.82,
         teacher_ckpt='checkpoints/resnet32x4_cifar100.pth',
         loss_type='mse', use_ae=False, block2_dw=False),
    dict(name='resnet110x2_shufflenetv2', teacher='resnet110x2', student='shufflenetv2',
         dataset='cifar100', baseline=72.54,
         teacher_ckpt='checkpoints/resnet110x2_cifar100.pth',
         loss_type='mse', use_ae=False, block2_dw=False),
    dict(name='wrn40_2_mobilenetv2', teacher='wrn_40_2', student='mobilenetv2',
         dataset='cifar100', baseline=65.12,
         teacher_ckpt='checkpoints/wrn40_2_cifar100.pth',
         loss_type='smooth_log', use_ae=True, block2_dw=True),
]

# ---------------------------------------------------------------------------
# ImageNet
# ---------------------------------------------------------------------------
IMAGENET = [
    dict(name='resnet34_resnet18', teacher='resnet34', student='resnet18',
         dataset='imagenet', baseline=69.76, num_classes=1000,
         teacher_ckpt='checkpoints/resnet34_imagenet.pth',
         loss_type='mse', use_ae=False, block2_dw=False),
    dict(name='resnet50_mobilenetv2_imagenet', teacher='resnet50_imagenet',
         student='mobilenetv2', dataset='imagenet', baseline=70.13, num_classes=1000,
         teacher_ckpt='checkpoints/resnet50_imagenet.pth',
         loss_type='mse', use_ae=False, block2_dw=False),
    dict(name='swin_l_swin_t', teacher='swin_l', student='swin_t',
         dataset='imagenet', baseline=81.30, num_classes=1000,
         teacher_ckpt='checkpoints/swin_l_imagenet.pth',
         loss_type='mse', use_ae=False, block2_dw=False),
]

# ---------------------------------------------------------------------------
# Tiny-ImageNet
# ---------------------------------------------------------------------------
TINY_IMAGENET = [
    dict(name='tiny_resnet32x4_resnet8x4', teacher='resnet32x4', student='resnet8x4',
         dataset='tinyimagenet', baseline=52.41,
         teacher_ckpt='checkpoints/resnet32x4_tinyimagenet.pth',
         loss_type='mse', use_ae=False, block2_dw=False),
    dict(name='tiny_resnet32x4_shufflenetv2', teacher='resnet32x4',
         student='shufflenetv2', dataset='tinyimagenet', baseline=58.44,
         teacher_ckpt='checkpoints/resnet32x4_tinyimagenet.pth',
         loss_type='mse', use_ae=False, block2_dw=False),
]

# ---------------------------------------------------------------------------
# COCO object detection (backbone distillation, R101 -> R50)
# ---------------------------------------------------------------------------
COCO_DETECTION = [
    dict(name='retinanet_r101_r50', detector='RetinaNet',
         teacher='R101', student='R50', dataset='coco', baseline=37.4,
         metric='AP'),
    dict(name='fasterrcnn_r101_r50', detector='Faster R-CNN',
         teacher='R101', student='R50', dataset='coco', baseline=38.4,
         metric='AP'),
    dict(name='fcos_r101_r50', detector='FCOS',
         teacher='R101', student='R50', dataset='coco', baseline=38.5,
         metric='AP'),
]

# ---------------------------------------------------------------------------
# Ablations (all on WRN-40-2 -> WRN-40-1, CIFAR-100)
# ---------------------------------------------------------------------------
ABLATION_COMPONENTS = [
    dict(name='vanilla', npm=False, kpm=False, reference=71.80),
    dict(name='kpm_only', npm=False, kpm=True, reference=73.43),
    dict(name='npm_only', npm=True, kpm=False, reference=74.16),
    dict(name='bnrkd', npm=True, kpm=True, reference=74.82),
]

ABLATION_NPM = [
    dict(name='full', reference=74.16),
    dict(name='w/o_autoencoder', reference=72.87),
    dict(name='w/o_adaptive_reparam_block', reference=73.44),
    dict(name='w/o_noise_level_matching', reference=73.02),
    dict(name='smooth_log_to_mse', reference=73.93),
]

ABLATION_KPM = [
    dict(name='level_feature', reference=74.82),
    dict(name='level_logits', reference=74.24),
    dict(name='level_feature_logits', reference=74.52),
    dict(name='center_teacher', reference=74.82),
    dict(name='center_mixture', reference=74.26),
    dict(name='center_student', reference=73.71),
    dict(name='loss_kpm_only', reference=74.44),
    dict(name='loss_plus_alignment', reference=74.82),
    dict(name='loss_plus_kl', reference=74.19),
]

# ---------------------------------------------------------------------------
# Aggregated view
# ---------------------------------------------------------------------------
SUITES = {
    'cifar100_homogeneous': (CIFAR100_HOMOGENEOUS, CIFAR_DISTILL),
    'cifar100_heterogeneous': (CIFAR100_HETEROGENEOUS, CIFAR_DISTILL),
    'imagenet': (IMAGENET, IMAGENET_DISTILL),
    'tinyimagenet': (TINY_IMAGENET, CIFAR_DISTILL),
    'coco': (COCO_DETECTION, None),
}

ALL_PAIRS = (CIFAR100_HOMOGENEOUS + CIFAR100_HETEROGENEOUS + IMAGENET
             + TINY_IMAGENET)


def iter_pairs(suite=None):
    """Yield ``(pair_dict, shared_settings)`` for one or all suites."""
    names = [suite] if suite else list(SUITES)
    for key in names:
        pairs, shared = SUITES[key]
        for pair in pairs:
            yield pair, shared or {}


def summary():
    """Return a list of ``(suite, name, teacher, student, dataset)`` rows."""
    rows = []
    for key in SUITES:
        for pair, _ in iter_pairs(key):
            rows.append((key, pair['name'], pair.get('teacher', '-'),
                         pair.get('student', '-'), pair.get('dataset', '-')))
    return rows


if __name__ == '__main__':
    print(f'{"suite":24s} {"name":34s} {"teacher":20s} {"student":16s} dataset')
    for suite, name, t, s, d in summary():
        print(f'{suite:24s} {name:34s} {t:20s} {s:16s} {d}')
    print(f'\n{"total pairs":<24s} {len(ALL_PAIRS)}')
