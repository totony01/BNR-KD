"""Utility helpers: logging, seeds, model summary."""

import logging
import os
import random
import sys

import numpy as np
import torch


def setup_logger(name='bnrkd', log_file=None, level=logging.INFO):
    """Configure a console logger, optionally mirrored to ``log_file``."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s',
                            datefmt='%H:%M:%S')

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handler = logging.FileHandler(log_file)
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def set_seed(seed=42):
    """Make a run reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model, trainable_only=True):
    """Number of parameters, in millions."""
    params = (p for p in model.parameters() if p.requires_grad or not trainable_only)
    return sum(p.numel() for p in params) / 1e6


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, map_location='cpu'):
    return torch.load(path, map_location=map_location)
