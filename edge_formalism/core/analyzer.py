"""Implementation of the information-geometric edge analyzer."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Tuple

from ..mblt.config import AnalyzerConfig

__all__ = ["EdgeAnalyzerWithMBLT"]


def _sigmoid(x: float, steepness: float) -> float:
    return 1.0 / (1.0 + math.exp(-steepness * x))


def _softplus(x: float) -> float:
    if x > 50.0:
        return x
    return math.log1p(math.exp(x))


def _gaussian_weights(step: float, bandwidth: float) -> List[Tuple[float, float]]:
    """Return offsets and weights for a small Gaussian stencil."""

    if bandwidth <= 0.0:
        return [(0.0, 1.0)]

    offsets = [-step, 0.0, step]
    denom = 2.0 * bandwidth * bandwidth
    weights = [math.exp(-(offset * offset) / max(denom, 1e-12)) for offset in offsets]
    total = sum(weights) or 1.0
    weights = [weight / total for weight in weights]
    return list(zip(offsets, weights))


@dataclass
class EdgeAnalyzerWithMBLT:
    """High level analytic wrapper around the MBLT configuration."""

    config: AnalyzerConfig

    def __post_init__(self) -> None:
        self._hyper = self.config.hyperparameters
        self._components = self.config.components
        self._theta_step = self.config.grid.theta_step
        self._stencil = _gaussian_weights(self._theta_step, self.config.grid.theta_bandwidth)

    # --- Component evaluation -------------------------------------------------

    def _normalise_theta(self, theta: float) -> float:
        return self.config.grid.clamp_theta(theta)

    def _normalise_n(self, n_value: float) -> float:
        return max(self.config.grid.n_min, min(n_value, self.config.grid.n_max))

    def compute_fisher(self, theta: float) -> float:
        theta = self._normalise_theta(theta)
        value = self._components.fisher(theta)
        return max(value, self.config.epsilon_min)

    def compute_covariance(self, theta: float, n_value: float) -> float:
        theta = self._normalise_theta(theta)
        n_value = self._normalise_n(n_value)
        value = self._components.covariance(theta, n_value)
        return max(value, self.config.epsilon_min)

    def compute_entropy(self, theta: float) -> float:
        theta = self._normalise_theta(theta)
        return self._components.entropy(theta)

    # --- Derived measures -----------------------------------------------------

    def compute_E_p(self, theta: float, n_value: float) -> float:
        fisher = self.compute_fisher(theta)
        covariance = self.compute_covariance(theta, n_value)
        return (n_value / float(self.config.dimension)) * fisher * covariance

    def compute_E_max(self, theta: float, n_value: float) -> float:
        fisher = self.compute_fisher(theta)
        covariance = self.compute_covariance(theta, n_value)
        value = fisher * covariance
        tau = max(self._hyper.tau, 1.0)
        scaled = tau * value
        if scaled > 50.0:
            lse = scaled
        else:
            lse = math.log(math.exp(scaled) + 1e-12)
        return n_value * max(lse / tau, self.config.epsilon_min)

    def _s_theta(self, theta: float) -> float:
        return math.sqrt(self.compute_fisher(theta))

    def compute_B(self, theta: float, n_value: float) -> float:
        n_value = max(n_value, 1.0)
        return 1.0 / (math.sqrt(n_value) * max(self._s_theta(theta), math.sqrt(self.config.epsilon_min)))

    def compute_V(self, theta: float, n_value: float) -> float:
        deviation = self.compute_E_p(theta, n_value) - 1.0
        smooth = math.exp(-self._hyper.alpha * deviation * deviation)
        gate = _sigmoid(deviation + self._hyper.delta, steepness=10.0)
        return smooth * gate

    def compute_Psi(self, theta: float, n_value: float) -> float:
        base_values: List[float] = []
        for offset, weight in self._stencil:
            shifted_theta = self._normalise_theta(theta + offset)
            entropy = self.compute_entropy(shifted_theta)
            coupling = self.compute_B(shifted_theta, n_value) * self.compute_V(shifted_theta, n_value)
            base_values.append(weight * entropy * coupling)
        return sum(base_values)

    def compute_log_n_derivative(self, theta: float, n_value: float) -> float:
        n_value = self._normalise_n(n_value)
        step = max(self.config.grid.n_step_fraction * n_value, 1e-6)
        n_plus = self._normalise_n(n_value + step)
        n_minus = self._normalise_n(max(n_value - step, self.config.grid.n_min))
        ep_plus = self.compute_E_p(theta, n_plus)
        ep_minus = self.compute_E_p(theta, n_minus)
        if n_plus == n_minus:
            return 0.0
        # derivative w.r.t log n: dE/dlogn = n * dE/dn
        derivative = (ep_plus - ep_minus) / (n_plus - n_minus)
        return derivative * n_value

    def compute_J(self, theta: float, n_value: float) -> float:
        psi = self.compute_Psi(theta, n_value)
        entropy = self.compute_entropy(theta)
        efficiency = self.compute_E_p(theta, n_value)
        derivative = self.compute_log_n_derivative(theta, n_value)
        penalty = self._hyper.lambda_E * (efficiency - 1.0) ** 2
        smooth_penalty = self._hyper.lambda_S * derivative * derivative
        soft_constraint = self._hyper.mu * (_softplus(1.0 - self._hyper.delta - efficiency) ** 2)
        return psi + self._hyper.lambda_H * entropy - penalty - smooth_penalty - soft_constraint

    # --- Diagnostics ----------------------------------------------------------

    def compute_J_gradient(self, theta: float, n_value: float, theta_ref: float | None = None) -> float:
        del theta_ref  # retained for API compatibility
        step = max(self._theta_step * 0.5, 1e-4)
        plus = self.compute_J(theta + step, n_value)
        minus = self.compute_J(theta - step, n_value)
        return (plus - minus) / (2.0 * step)

    def compute_gradient_magnitude(self, theta: float, n_value: float) -> float:
        return abs(self.compute_J_gradient(theta, n_value))

    def is_on_edge(self, theta: float, n_value: float) -> bool:
        return abs(self.compute_E_p(theta, n_value) - 1.0) <= self._hyper.delta

    # --- Aggregates -----------------------------------------------------------

    def evaluate_grid(self, theta_values: Iterable[float], n_values: Iterable[float]) -> Dict[Tuple[float, float], Dict[str, float]]:
        summary: Dict[Tuple[float, float], Dict[str, float]] = {}
        for theta in theta_values:
            for n_value in n_values:
                key = (theta, n_value)
                summary[key] = {
                    "E_p": self.compute_E_p(theta, n_value),
                    "E_max": self.compute_E_max(theta, n_value),
                    "Psi": self.compute_Psi(theta, n_value),
                    "J": self.compute_J(theta, n_value),
                }
        return summary
