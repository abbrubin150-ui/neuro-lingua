"""Tests covering the recursive optimizer history semantics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from edge_formalism.optimization import RecursiveOptimizer


@dataclass
class StubAnalyzer:
    offset: float = 0.0

    def compute_J(self, theta: float, n_value: float) -> float:  # noqa: D401 - simple stub
        del n_value
        return theta + self.offset

    def compute_gradient_magnitude(self, theta: float, n_value: float) -> float:
        del n_value
        return abs(theta) + self.offset

    def compute_E_p(self, theta: float, n_value: float) -> float:
        del theta, n_value
        return 1.0


GRID: List[Tuple[float, float]] = [(0.1, 10.0), (0.2, 20.0), (0.3, 30.0)]


def test_optimizer_history_persists_previous_cycle() -> None:
    analyzer = StubAnalyzer(offset=0.0)
    optimizer = RecursiveOptimizer(analyzer, GRID)
    first_cycle = optimizer.optimize_cycle()
    analyzer.offset = -0.2
    second_cycle = optimizer.optimize_cycle()

    assert optimizer.J_val_history[-1] == second_cycle
    assert optimizer.J_val_history[-2] == first_cycle
    assert not optimizer.is_monotonic_increasing()

    analyzer.offset = 0.1
    third_cycle = optimizer.optimize_cycle()
    assert optimizer.J_val_history[-1] == third_cycle
    assert optimizer.is_monotonic_increasing()


def test_should_stop_ignores_unpaired_weights() -> None:
    analyzer = StubAnalyzer(offset=0.0)
    optimizer = RecursiveOptimizer(analyzer, GRID, weights=[1.0, 1.0, 1.0, -100.0])
    optimizer.optimize_cycle()

    assert not optimizer.should_stop()
