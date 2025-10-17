"""Unit tests for the pure-Python OnTheEdgeLearning implementation."""

from __future__ import annotations

import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from symmetry_coupling.on_the_edge_learning import OnTheEdgeLearning


def test_gaussian_known_variance() -> None:
    model = OnTheEdgeLearning(theta_min=-2, theta_max=2, n_min=100, n_max=1000, n_grid=21, m_grid=15)
    model.run()

    large_n_idx = -1
    ep_large_n = [row[large_n_idx] for row in model.E_p]
    max_error = max(abs(value - 1.0) for value in ep_large_n)

    assert max_error < 0.05


def test_bernoulli_eigenvalue_collapse() -> None:
    class BernoulliModel(OnTheEdgeLearning):
        def compute_fisher_information(self, theta: float) -> float:  # type: ignore[override]
            return max(1.0 / (theta * (1.0 - theta)), self.epsilon_min)

    model = BernoulliModel(theta_min=0.1, theta_max=0.9, n_min=10, n_max=200, n_grid=19, m_grid=11)
    model.run()

    for row in model.E_max:
        assert all(math.isfinite(value) for value in row)
        assert all(value >= 1.0 - model.delta for value in row)
