"""Symmetry Coupling Metric package."""

from .core import Config, Metrics, compute_coupling, compute_coupling_batch, load_synonym_dictionary
from .on_the_edge_learning import OnTheEdgeLearning

__all__ = [
    "Config",
    "Metrics",
    "compute_coupling",
    "compute_coupling_batch",
    "load_synonym_dictionary",
    "OnTheEdgeLearning",
]
