# BNR-KD: Bidirectional Noise Regulation for Knowledge Distillation

Official implementation of **BNR-KD**, a knowledge distillation framework that
regulates noise in two complementary directions over the same feature space.

Instead of treating the teacher-student feature gap as something to be either
blindly aligned or blindly augmented, BNR-KD handles it from both sides at once:

* a **Noise Purification Module (NPM)** that *removes* the discrepancy, and
* a **Knowledge Perturbation Module (KPM)** that *injects* controlled noise.

The two directions are complementary: purification fixes the *direction* of the
student representation, and perturbation then generates local views inside a
trustworthy neighbourhood.

---

## Highlights

| | |
|---|---|
| **Two-directional** | Denoising and perturbation are unified in a single objective rather than applied in isolation. |
| **Direction-aware** | Purification aligns the student feature *direction* to the teacher; the residual magnitude is deliberately left untouched. |
| **Cheap at inference** | All extra modules are training-only. The multi-branch denoiser folds into a single convolution, and KPM disappears entirely, so the deployed model is just the student. |
| **Backbone-agnostic** | Works with ResNet, WideResNet, VGG, ShuffleNet, MobileNet and Transformer backbones — homogeneous or heterogeneous pairs. |
| **Plug-and-play** | The distillation head is a separate module; swap the backbone by changing one argument. |

---

## Method

### Overview

```
                    ┌──────────────────────── NPM (denoise) ────────────────────────┐
                    │                                                               │
  student feat ──► 1x1 proj ──► [noise-level adapter] ──► DDIM reverse chain ──► purified
                    │             ▲                             ▲                   │
                    │             │                             │                   │
                    │        step estimate             step-adaptive denoiser       │
                    └─────────────────────────────────────────────────────────────┘
                                                                                     │
                                                                                     ▼
  teacher feat ──► [optional linear AE] ──► teacher anchor ──► KPM: perturb ──► views
                                                    │                               │
                                                    └───────────► alignment ◄───────┘
```

### NPM — Noise Purification Module (denoising direction)

The student feature is treated as a noisy observation of the corresponding
teacher feature. NPM learns to reverse that corruption:

1. a 1×1 projection maps the student feature into the teacher's latent space;
2. a lightweight **noise-level adapter** predicts, per sample, how noisy the
   student feature currently is;
3. a short **DDIM reverse chain** (5 steps by default) produces the purified
   feature, which is aligned with the teacher anchor.

The denoiser is built from **step-adaptive reparameterizable blocks**. Each block
holds three parallel branches (identity, 1×1 conv, 3×3 conv) whose outputs are
weighted by diffusion-step-dependent gates. Because every branch is linear up to
BatchNorm, they fold into a *single* convolution at inference — the multi-branch
capacity is free at deployment time.

An optional **linear autoencoder** compresses wide teacher features before the
diffusion process, cutting the cost of both the reverse chain and the denoiser.

**Denoising objective.** The default `smooth_log` loss rescales the
per-element error by the noise variance of the sampled step:

```
L_diff = log(1 + MSE / sigma_t^2),    sigma_t^2 = 1 - alpha_bar_t
```

A large error behaves like a log-SNR term (strong gradient); an error
approaching zero degrades to plain MSE, so the gradient stays bounded at both
ends. `mse` and `huber` variants are also provided.

### KPM — Knowledge Perturbation Module (add-noise direction)

KPM perturbs the teacher anchor to synthesise several perspectives of the same
knowledge:

```
anchor_i = (1 - alpha) * anchor + alpha * eta_i,     eta_i ~ N(0, I)

L_kpm = d(purified, anchor) + beta * mean_i d(purified, anchor_i)
```

Two design choices matter:

* the perturbation is **additive**, not a convex blend. The clean alignment term
  keeps its full weight, so the purification objective is never diluted;
* the perturbation is centred on the **teacher anchor**, not on the student
  feature. This keeps the synthetic views inside a trustworthy neighbourhood —
  they vary locally while preserving the teacher's representation direction.

### Total objective

```
L = L_CE + w * ( L_align + L_diff + L_kpm + L_rec )
```

where `w` is `--kd-weight` and `L_rec` is present only when the autoencoder is
enabled.

---

## Repository layout

```
BNR-KD-release/
├── bnrkd/                          # installable package
│   ├── data.py                     # CIFAR loaders
│   ├── losses.py                   # denoising losses, KD losses
│   ├── trainer.py                  # Distiller: the training loop
│   ├── experiments.py              # registry of every reported experiment
│   ├── models/                     # backbones
│   │   ├── resnet.py               # ResNet-8x4 / 32x4 / 110x2 / 50 (CIFAR)
│   │   ├── wrn.py                  # WideResNet-40-2 / 40-1 / 16-2
│   │   ├── vgg.py                  # VGG-13 / VGG-8 (CIFAR)
│   │   ├── shufflenetv2.py         # ShuffleNetV2 (CIFAR)
│   │   ├── mobilenetv2.py          # MobileNetV2 (CIFAR, width multiplier)
│   │   └── imagenet.py             # torchvision ResNet / timm Swin wrappers
│   ├── modules/                    # the distillation machinery
│   │   ├── bnrkd.py                # top-level head: NPM + KPM
│   │   ├── npm.py                  # Noise Purification Module
│   │   ├── kpm.py                  # Knowledge Perturbation Module
│   │   ├── denoiser.py             # step-adaptive blocks, noise predictor, AE
│   │   └── diffusion.py            # DDIM scheduler and denoising pipeline
│   └── utils/
│       └── __init__.py             # logging, seeds, checkpoint helpers
├── tools/
│   ├── train_teacher.py            # stage 1: train the teacher
│   ├── train_student.py            # stage 2: BNR-KD distillation
│   └── run_experiments.py          # list / launch the reported configurations
├── scripts/
│   └── run_cifar100.sh             # end-to-end recipe
├── requirements.txt
└── LICENSE
```

---

## Installation

```bash
git clone <this-repo>
cd BNR-KD-release
pip install -r requirements.txt
```

A single CUDA GPU is enough for all CIFAR experiments. The defaults assume
`torch >= 1.13`. `timm` is only required for the Swin backbones.

---

## Data

CIFAR-10 / CIFAR-100 are downloaded automatically on first use into `--data-root`
(default `./data`). No manual preparation is required. If your machine has no
internet access, point `--data-root` at a directory that already contains the
extracted `cifar-100-python` folder.

ImageNet and Tiny-ImageNet follow the standard `train/` + `val/` folder layout
and are passed through `--data-root`.

---

## Quick start

### 1. Train the teacher

```bash
python tools/train_teacher.py \
    --arch resnet32x4 \
    --dataset cifar100 \
    --epochs 240 \
    --output checkpoints/resnet32x4_cifar100.pth
```

### 2. Distil the student with BNR-KD

```bash
python tools/train_student.py \
    --teacher-arch resnet32x4 \
    --student-arch resnet8x4 \
    --teacher-ckpt checkpoints/resnet32x4_cifar100.pth \
    --feature-stage 3 \
    --use-ae 0 \
    --loss-type mse \
    --kd-weight 2.0 \
    --block2-dw 0 \
    --output checkpoints/resnet8x4_cifar100_bnrkd.pth
```

### Or run both stages at once

```bash
bash scripts/run_cifar100.sh
```

### Homogeneous pair

```bash
python tools/train_student.py \
    --teacher-arch wrn_40_2 --student-arch wrn_40_1 \
    --teacher-ckpt checkpoints/wrn40_2_cifar100.pth \
    --feature-stage 3 --use-ae 1 --loss-type smooth_log \
    --kd-weight 2.0 \
    --output checkpoints/wrn40_1_cifar100_bnrkd.pth
```

---

## Experiment configurations

Every teacher-student pair reported in the paper is registered in
`bnrkd/experiments.py`. Inspect or launch them with `tools/run_experiments.py`:

```bash
# list all pairs
python tools/run_experiments.py --list

# print the commands for one suite without running them
python tools/run_experiments.py --suite cifar100_heterogeneous --print-cmd

# launch a single pair
python tools/run_experiments.py --pair vgg13_vgg8
```

### CIFAR-100, homogeneous pairs

| Teacher | Student | Student baseline | Distillation settings |
|---|---|---|---|
| WRN-40-2 | WRN-40-1 | 71.80 | `smooth_log`, AE on, dw on |
| WRN-40-2 | WRN-16-2 | 73.26 | `smooth_log`, AE on, dw on |
| ResNet-32x4 | ResNet-8x4 | 72.50 | `mse`, AE off, dw off |
| VGG13 | VGG8 | 70.36 | `smooth_log`, AE on, dw on |

### CIFAR-100, heterogeneous pairs

| Teacher | Student | Student baseline | Distillation settings |
|---|---|---|---|
| ResNet-50 | MobileNetV2 | 64.60 | `mse`, AE off, dw off |
| ResNet-32x4 | ShuffleNetV2 | 71.82 | `mse`, AE off, dw off |
| ResNet-110x2 | ShuffleNetV2 | 72.54 | `mse`, AE off, dw off |
| WRN-40-2 | MobileNetV2 | 65.12 | `smooth_log`, AE on, dw on |

### Tiny-ImageNet

| Teacher | Student | Student baseline |
|---|---|---|
| ResNet-32x4 | ResNet-8x4 | 52.41 |
| ResNet-32x4 | ShuffleNetV2 | 58.44 |

### ImageNet

| Teacher | Student | Student baseline | Distillation settings |
|---|---|---|---|
| ResNet-34 | ResNet-18 | 69.76 | 90 epochs, batch 1024 |
| ResNet-50 | MobileNetV2 | 70.13 | 90 epochs, batch 1024 |
| Swin-L | Swin-T | 81.30 | 90 epochs, batch 1024 |

### COCO object detection

Backbone distillation (R101 → R50); the detection head is trained normally.

| Detector | Teacher | Student |
|---|---|---|
| RetinaNet | R101 | R50 |
| Faster R-CNN | R101 | R50 |
| FCOS | R101 | R50 |

### Ablations

All ablations use the WRN-40-2 → WRN-40-1 pair on CIFAR-100.

**Component ablation (2×2).** Enabling NPM and KPM independently isolates the
contribution of each direction.

| NPM | KPM | Configuration |
|:---:|:---:|---|
| ✗ | ✗ | vanilla |
| ✗ | ✓ | KPM only |
| ✓ | ✗ | NPM only |
| ✓ | ✓ | BNR-KD |

**NPM components.**

| Variant |
|---|
| full |
| w/o autoencoder |
| w/o step-adaptive reparameterizable block |
| w/o noise-level matching |
| smooth-log → MSE |

**KPM components.**

| Group | Variant |
|---|---|
| Perturbation level | feature / logits / feature + logits |
| Perturbation centre | teacher / mixture / student |
| Loss components | KPM-MSE only / + direction alignment / + KL term |

---

## Backbone reference

| Name | Family | Params | Stage-3 shape | Notes |
|---|---|---|---|---|
| `resnet8x4` | ResNet (CIFAR) | 1.23 M | 256 × 8 × 8 | |
| `resnet32x4` | ResNet (CIFAR) | 7.43 M | 256 × 8 × 8 | |
| `resnet110x2` | ResNet (CIFAR) | 27.58 M | 256 × 8 × 8 | large teacher |
| `resnet50` | ResNet (CIFAR) | 8.64 M | 1024 × 8 × 8 | bottleneck |
| `wrn_40_1` | WideResNet | 0.57 M | 64 × 8 × 8 | |
| `wrn_40_2` | WideResNet | 2.26 M | 128 × 8 × 8 | |
| `wrn_16_2` | WideResNet | 0.70 M | 128 × 8 × 8 | |
| `vgg8` | VGG (CIFAR) | 3.96 M | 512 × 2 × 2 | |
| `vgg13` | VGG (CIFAR) | 9.46 M | 512 × 2 × 2 | |
| `shufflenetv2` | ShuffleNetV2 | 0.66 M | 160 × 4 × 4 | |
| `mobilenetv2` | MobileNetV2 | 0.82 M | 48 × 4 × 4 | width 0.5 |
| `resnet18` / `resnet34` | torchvision | — | — | ImageNet |
| `resnet50_imagenet` | torchvision | — | — | ImageNet |
| `swin_t` / `swin_l` | timm | — | — | ImageNet |

---

## Command-line reference

### `tools/train_student.py`

**Model / data**

| Flag | Default | Description |
|---|---|---|
| `--teacher-arch` | `wrn_40_2` | Teacher backbone name. |
| `--student-arch` | `wrn_40_1` | Student backbone name. |
| `--teacher-ckpt` | required | Path to the pre-trained teacher. |
| `--dataset` | `cifar100` | `cifar100` or `cifar10`. |
| `--data-root` | `./data` | Dataset directory. |
| `--feature-stage` | `3` | Index of the feature map used for distillation. Stage 3 is the last spatial map for ResNet and WRN. |

**NPM (denoising)**

| Flag | Default | Description |
|---|---|---|
| `--use-ae` | `1` | Compress teacher features with a linear autoencoder. Useful for wide features; turn off when teacher and student widths already match. |
| `--ae-channels` | `0` | Latent width; `0` means half of the teacher channels. |
| `--loss-type` | `smooth_log` | Denoising loss: `smooth_log`, `mse`, `huber`. |
| `--inference-steps` | `5` | Number of reverse steps. |
| `--reduction` | `0` | Bottleneck ratio of the first denoising block. |
| `--block2-dw` | `1` | Depthwise branches in the second block. Prefer `0` for wide student features. |

**KPM (perturbation)**

| Flag | Default | Description |
|---|---|---|
| `--kpm-alpha` | `0.1` | Perturbation strength. |
| `--kpm-views` | `3` | Number of synthetic views per step. |
| `--kpm-beta` | `0.8` | Weight of the aggregated perturbation term. |

**Optimisation**

| Flag | Default | Description |
|---|---|---|
| `--epochs` | `240` | Training epochs. |
| `--batch-size` | `128` | Mini-batch size. |
| `--lr` | `0.05` | Initial SGD learning rate. |
| `--weight-decay` | `5e-4` | SGD weight decay. |
| `--kd-weight` | `2.0` | Weight of the distillation block relative to cross-entropy. |
| `--logit-temperature` | `1.0` | Temperature of the logit-level KL term. |
| `--milestones` | `150 180 210` | Epochs at which the learning rate decays by 10×. |
| `--seed` | `42` | Random seed. |

---

## Notes and practical tips

**Feature stage.** Stage 3 (the last 8×8 spatial map) is what we distil by
default. On ResNet backbones this map is *not* normalised by a downstream
BatchNorm, so its scale differs between teacher and student. Two options:

* enable `--use-ae 1` — the autoencoder's BatchNorm normalises the teacher
  latent and the projection maps the student into the same space;
* or leave `--use-ae 0` and let the 1×1 projection learn the scale.

For WideResNet the feature already passes through BatchNorm + ReLU and sits at a
comparable scale for both networks.

**`--block2-dw`.** The depthwise variant is cheaper and works well when the
student feature is narrow. When the student feature is wide (ResNet-8x4 has 256
channels at stage 3) the standard-convolution variant is usually stronger.

**Additive vs. blended perturbation.** `--kpm-beta` controls how much the
synthetic views contribute. Note that the clean alignment term always keeps its
full weight — the perturbation is *added*, never blended in. Blending the two
dilutes the purification objective and the perturbation stops helping.

**Reproducibility.** Every reported number is a mean over three seeds
(`42 / 1 / 3407`). Pass `--seed` to reproduce an individual run.

---

## Citation

If you find this code useful, please cite:

```bibtex
@article{bnrkd2026,
  title   = {Bidirectional Noise-Regulated Knowledge Distillation},
  author  = {Anonymous},
  journal = {Under review},
  year    = {2026}
}
```

---

## License

Released under the MIT License. See [LICENSE](LICENSE).
