"""Lifeguard: Open-Source Life Preservation & Decision Intelligence Framework.

A standardized suite of datasets, tripartite objectives, structural heads,
and diagnostic evaluation tools for aligning autonomous AI models with human life preservation.
"""

from lifeguard.diagnostics import FactorialANOVA, LifeguardDiagnostic
from lifeguard.heads import ComplementHead, DualPredicateGate, When2CallGate
from lifeguard.loss import TripartiteLifeguardLoss, UnanchoredContrastiveLoss
from lifeguard.generator import LifeguardGenerator

__version__ = "1.0.0"
__all__ = [
    "TripartiteLifeguardLoss",
    "UnanchoredContrastiveLoss",
    "ComplementHead",
    "DualPredicateGate",
    "When2CallGate",
    "FactorialANOVA",
    "LifeguardDiagnostic",
    "LifeguardGenerator",
]
