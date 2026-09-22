"""BNR-KD: Bidirectional Noise Regulation for Knowledge Distillation."""

from .modules.bnrkd import BidirectionalNoiseRegulation
from .modules.npm import NoisePurificationModule
from .modules.kpm import KnowledgePerturbationModule

__all__ = [
    'BidirectionalNoiseRegulation',
    'NoisePurificationModule',
    'KnowledgePerturbationModule',
]

__version__ = '1.0.0'
