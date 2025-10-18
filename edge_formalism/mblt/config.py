"""MBLT configuration factory."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Dict

__all__ = ["AnalyzerComponents", "GridConfig", "HyperParameters", "AnalyzerConfig", "ConfigFactory"]


@dataclass
class AnalyzerComponents:
    fisher: Callable[[float], float]
    covariance: Callable[[float, float], float]
    entropy: Callable[[float], float]


@dataclass
class GridConfig:
    theta_min: float = -2.0
    theta_max: float = 2.0
    theta_steps: int = 41
    n_min: float = 10.0
    n_max: float = 1000.0
    n_steps: int = 25
    theta_bandwidth: float = 0.1
    n_step_fraction: float = 0.05

    def clamp_theta(self, theta: float) -> float:
        return max(self.theta_min, min(theta, self.theta_max))

    @property
    def theta_step(self) -> float:
        steps = max(self.theta_steps - 1, 1)
        return (self.theta_max - self.theta_min) / steps


@dataclass
class HyperParameters:
    delta: float = 0.01
    lambda_H: float = 0.5
    lambda_E: float = 0.5
    lambda_S: float = 0.2
    mu: float = 0.1
    tau: float = 20.0
    alpha: float = 10.0
    eta: float = 1e-4


@dataclass
class AnalyzerConfig:
    components: AnalyzerComponents
    grid: GridConfig
    hyperparameters: HyperParameters
    dimension: int = 1
    epsilon_min: float = 1e-6


class ConfigFactory:
    """Construct an :class:`AnalyzerConfig` from a JSON spec."""

    def __init__(self, spec: Dict[str, Any]):
        self._spec = spec

    def build(self) -> AnalyzerConfig:
        components = self._build_components(self._spec.get("components", {}))
        grid = self._build_grid(self._spec.get("grid", {}))
        hyper = self._build_hyper(self._spec.get("hyperparameters", {}))
        dimension = int(self._spec.get("dimension", 1))
        epsilon = float(self._spec.get("epsilon_min", 1e-6))
        return AnalyzerConfig(components=components, grid=grid, hyperparameters=hyper, dimension=dimension, epsilon_min=epsilon)

    # --- Builders -------------------------------------------------------------

    def _build_components(self, section: Dict[str, Any]) -> AnalyzerComponents:
        kind = section.get("type", "gaussian")
        params = section.get("params", {})
        if kind == "gaussian":
            return self._build_gaussian(params)
        if kind == "bernoulli":
            return self._build_bernoulli(params)
        if kind == "misspecified_gaussian":
            return self._build_misspecified_gaussian(params)
        raise ValueError(f"Unknown component type: {kind}")

    def _build_grid(self, section: Dict[str, Any]) -> GridConfig:
        theta_min = float(section.get("theta_min", -2.0))
        theta_max = float(section.get("theta_max", 2.0))
        theta_steps = int(section.get("theta_steps", 41))
        n_min = float(section.get("n_min", 10.0))
        n_max = float(section.get("n_max", 1000.0))
        n_steps = int(section.get("n_steps", 25))
        theta_bandwidth = float(section.get("theta_bandwidth", (theta_max - theta_min) / max(theta_steps - 1, 1)))
        n_step_fraction = float(section.get("n_step_fraction", 0.05))
        return GridConfig(
            theta_min=theta_min,
            theta_max=theta_max,
            theta_steps=theta_steps,
            n_min=n_min,
            n_max=n_max,
            n_steps=n_steps,
            theta_bandwidth=max(theta_bandwidth, 0.0),
            n_step_fraction=max(n_step_fraction, 1e-4),
        )

    def _build_hyper(self, section: Dict[str, Any]) -> HyperParameters:
        return HyperParameters(
            delta=float(section.get("delta", 0.01)),
            lambda_H=float(section.get("lambda_H", 0.5)),
            lambda_E=float(section.get("lambda_E", 0.5)),
            lambda_S=float(section.get("lambda_S", 0.2)),
            mu=float(section.get("mu", 0.1)),
            tau=float(section.get("tau", 20.0)),
            alpha=float(section.get("alpha", 10.0)),
            eta=float(section.get("eta", 1e-4)),
        )

    # --- Component factories --------------------------------------------------

    def _build_gaussian(self, params: Dict[str, Any]) -> AnalyzerComponents:
        variance = float(params.get("variance", 1.0))
        bias_scale = float(params.get("bias_scale", 0.0))

        def fisher(theta: float) -> float:
            del theta
            return 1.0 / max(variance, 1e-6)

        def covariance(theta: float, n_value: float) -> float:
            n_value = max(n_value, 1.0)
            base = variance / n_value
            bias = bias_scale * theta / math.sqrt(n_value)
            return max(base + bias * bias, 0.0)

        def entropy(theta: float) -> float:
            return -math.log1p(theta * theta)

        return AnalyzerComponents(fisher=fisher, covariance=covariance, entropy=entropy)

    def _build_bernoulli(self, params: Dict[str, Any]) -> AnalyzerComponents:
        epsilon = float(params.get("epsilon", 1e-6))

        def fisher(theta: float) -> float:
            clipped = min(max(theta, epsilon), 1.0 - epsilon)
            value = 1.0 / max(clipped * (1.0 - clipped), epsilon)
            return value

        def covariance(theta: float, n_value: float) -> float:
            info = fisher(theta)
            n_value = max(n_value, 1.0)
            return 1.0 / (n_value * info)

        def entropy(theta: float) -> float:
            clipped = min(max(theta, epsilon), 1.0 - epsilon)
            return -(clipped * math.log(clipped) + (1.0 - clipped) * math.log(1.0 - clipped))

        return AnalyzerComponents(fisher=fisher, covariance=covariance, entropy=entropy)

    def _build_misspecified_gaussian(self, params: Dict[str, Any]) -> AnalyzerComponents:
        variance = float(params.get("variance", 1.0))
        mismatch = float(params.get("mismatch", 0.1))

        def fisher(theta: float) -> float:
            del theta
            return 1.0 / max(variance, 1e-6)

        def covariance(theta: float, n_value: float) -> float:
            n_value = max(n_value, 1.0)
            base = variance / n_value
            extra = mismatch / max(n_value, 1.0)
            return max(base + extra, 0.0)

        def entropy(theta: float) -> float:
            return -math.log1p(theta * theta)

        return AnalyzerComponents(fisher=fisher, covariance=covariance, entropy=entropy)
