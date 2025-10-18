"""High level module that couples the analyzer with conversational heuristics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from .core import EdgeAnalyzerWithMBLT
from .mblt import ConfigFactory, load_spec
from .optimization import RecursiveOptimizer

__all__ = ["UnifiedEdgeModule"]


@dataclass
class UnifiedEdgeModule:
    spec_path: str
    evaluation_grid_size: int = 5
    analyzer: EdgeAnalyzerWithMBLT | None = field(default=None, init=False)
    optimizer: RecursiveOptimizer | None = field(default=None, init=False)
    _turn_history: List[Dict[str, float]] = field(default_factory=list, init=False)

    def initialize(self) -> None:
        spec = load_spec(self.spec_path)
        config = ConfigFactory(spec).build()
        self.analyzer = EdgeAnalyzerWithMBLT(config)

    # ------------------------------------------------------------------

    def _ensure_analyzer(self) -> EdgeAnalyzerWithMBLT:
        if self.analyzer is None:
            raise RuntimeError("UnifiedEdgeModule must be initialised before use")
        return self.analyzer

    def run_optimization(self, max_cycles: int = 10) -> Sequence[List[float]]:
        analyzer = self._ensure_analyzer()
        theta_values = [
            analyzer.config.grid.theta_min
            + i * analyzer.config.grid.theta_step
            for i in range(self.evaluation_grid_size)
        ]
        n_step = (analyzer.config.grid.n_max - analyzer.config.grid.n_min) / max(self.evaluation_grid_size - 1, 1)
        n_values = [analyzer.config.grid.n_min + i * n_step for i in range(self.evaluation_grid_size)]
        grid = [(theta, n_value) for theta in theta_values for n_value in n_values]
        self.optimizer = RecursiveOptimizer(analyzer, grid)
        cycles: List[List[float]] = []
        for _ in range(max_cycles):
            cycles.append(self.optimizer.optimize_cycle())
            if self.optimizer.should_stop():
                break
        return cycles

    # ------------------------------------------------------------------

    def analyze_conversation_turn(self, question: str, response: str, turn_index: int) -> Dict[str, float]:
        analyzer = self._ensure_analyzer()
        theta = self._estimate_theta(question, response)
        n_value = max(float(turn_index), 1.0)
        metrics = {
            "theta": theta,
            "n": n_value,
            "efficiency_E_p": analyzer.compute_E_p(theta, n_value),
            "efficiency_E_max": analyzer.compute_E_max(theta, n_value),
            "objective_J": analyzer.compute_J(theta, n_value),
            "gradient_magnitude": analyzer.compute_gradient_magnitude(theta, n_value),
            "on_edge": float(analyzer.is_on_edge(theta, n_value)),
        }
        self._turn_history.append(metrics)
        return metrics

    def _estimate_theta(self, question: str, response: str) -> float:
        analyzer = self._ensure_analyzer()
        if not response:
            return analyzer.config.grid.theta_min
        question_tokens = len(question.split()) or 1
        response_tokens = len(response.split())
        ratio = response_tokens / float(question_tokens)
        delta = ratio - 1.0
        return analyzer.config.grid.clamp_theta(delta)
