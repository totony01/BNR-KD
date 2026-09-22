"""List or launch the experiment configurations from the paper.

Examples:
    # show every teacher-student pair
    python tools/run_experiments.py --list

    # print the shell commands for one suite
    python tools/run_experiments.py --suite cifar100_heterogeneous --print-cmd

    # actually launch one pair
    python tools/run_experiments.py --pair vgg13_vgg8
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bnrkd.experiments import SUITES, iter_pairs, summary


def build_command(pair, shared, output_dir='checkpoints'):
    """Translate a registry entry into a `tools/train_student.py` invocation."""
    cmd = [sys.executable, 'tools/train_student.py',
           '--teacher-arch', pair['teacher'],
           '--student-arch', pair['student'],
           '--teacher-ckpt', pair.get('teacher_ckpt', ''),
           '--dataset', pair.get('dataset', 'cifar100'),
           '--feature-stage', str(shared.get('feature_stage', 3)),
           '--epochs', str(shared.get('epochs', 240)),
           '--batch-size', str(shared.get('batch_size', 128)),
           '--lr', str(shared.get('lr', 0.05)),
           '--kd-weight', str(shared.get('kd_weight', 2.0)),
           '--use-ae', '1' if pair.get('use_ae') else '0',
           '--block2-dw', '1' if pair.get('block2_dw', True) else '0',
           '--loss-type', pair.get('loss_type', 'mse'),
           '--output', os.path.join(output_dir, f"{pair['student']}_{pair['name']}.pth")]
    if shared.get('milestones'):
        cmd += ['--milestones'] + [str(m) for m in shared['milestones']]
    return cmd


def main():
    ap = argparse.ArgumentParser(description='BNR-KD experiment launcher')
    ap.add_argument('--list', action='store_true', help='list all pairs and exit')
    ap.add_argument('--suite', default=None, choices=sorted(SUITES),
                    help='restrict to one suite')
    ap.add_argument('--pair', default=None, help='run a single pair by name')
    ap.add_argument('--print-cmd', action='store_true',
                    help='print commands instead of running them')
    ap.add_argument('--output-dir', default='checkpoints')
    args = ap.parse_args()

    if args.list:
        print(f'{"suite":24s} {"name":34s} {"teacher":20s} {"student":16s} dataset')
        for suite, name, t, s, d in summary():
            print(f'{suite:24s} {name:34s} {t:20s} {s:16s} {d}')
        return

    selected = []
    for pair, shared in iter_pairs(args.suite):
        if args.pair and pair['name'] != args.pair:
            continue
        selected.append((pair, shared))

    if not selected:
        print('no matching pair found; use --list to inspect the registry')
        return

    for pair, shared in selected:
        cmd = build_command(pair, shared, args.output_dir)
        print('\n# suite pair:', pair['name'])
        print(' '.join(cmd))
        if not args.print_cmd:
            subprocess.run(cmd, check=False)


if __name__ == '__main__':
    main()
