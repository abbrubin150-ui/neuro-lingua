"""Recursive optimisation routine for the edge analyzer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

__all__ = ["RecursiveOptimizer"]


@dataclass
class RecursiveOptimizer:
    """Simple optimiser that records objective histories across cycles."""

    core_analyzer: any
    grid: Sequence[Tuple[float, float]]
    weights: Sequence[float] | None = None
    J_val_history: List[List[float]] = field(default_factory=list)

    def optimize_cycle(self) -> List[float]:
        """Evaluate the objective across the grid and persist history."""

        values = [self.core_analyzer.compute_J(theta, n_value) for theta, n_value in self.grid]
        self.J_val_history.append(values.copy())
        return values

    # Diagnostics --------------------------------------------------------------

    def cycle_summaries(self) -> List[Dict[str, float]]:
        summaries: List[Dict[str, float]] = []
        for values in self.J_val_history:
            if not values:
                summaries.append({"mean": 0.0, "min": 0.0, "max": 0.0})
            else:
                summaries.append({
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                })
        return summaries

    def is_monotonic_increasing(self) -> bool:
        if len(self.J_val_history) < 2:
            return True
        prev_values = self.J_val_history[-2]
        curr_values = self.J_val_history[-1]
        threshold = 1e-9
        return all(curr >= prev - threshold for curr, prev in zip(curr_values, prev_values))

    def should_stop(self) -> bool:
        if not self.J_val_history:
            return False
        latest = self.J_val_history[-1]
        if self.weights:
            total_weight = sum(self.weights) or 1.0
            weighted = sum(value * weight for value, weight in zip(latest, self.weights)) / total_weight
            return weighted <= 0.0
        return all(value <= 0.0 for value in latest)
