"""Pure-Python implementation of the On-the-Edge learning heuristic.

The original prototype relied heavily on :mod:`numpy` and :mod:`scipy` for
matrix square-roots, FFT-based convolutions, and automatic differentiation.
Those dependencies are not available in this repository's execution
environment, so this module re-implements the required operations using plain
Python and the :mod:`math` standard library.  The implementation mirrors the
behaviour of the reference notebook closely enough for the bundled unit tests.

The class focuses on the one-dimensional setting used in the experiments.  As
such the Fisher information matrices are scalars which makes many of the
linear-algebra operations straightforward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import List


def _linspace(start: float, stop: float, num: int) -> List[float]:
    if num <= 0:
        raise ValueError("num must be positive")
    if num == 1:
        return [float(start)]
    step = (stop - start) / (num - 1)
    return [float(start + i * step) for i in range(num)]


def _exp(value: float) -> float:
    return math.exp(value)


def _log(value: float) -> float:
    if value <= 0:
        raise ValueError("log undefined for non-positive values")
    return math.log(value)


def _sigmoid(x: float, steepness: float) -> float:
    return 1.0 / (1.0 + math.exp(-steepness * x))


def _softplus(x: float) -> float:
    return math.log1p(math.exp(x))


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (q / 100.0)
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[int(pos)]
    fraction = pos - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _reflect_index(idx: int, length: int) -> int:
    if length <= 0:
        raise ValueError("length must be positive")
    if length == 1:
        return 0
    while idx < 0 or idx >= length:
        if idx < 0:
            idx = -idx - 1
        elif idx >= length:
            idx = 2 * length - idx - 1
    return idx


def _reflect_pad(signal: List[float], pad_width: int) -> List[float]:
    if pad_width <= 0:
        return signal[:]
    n = len(signal)
    padded: List[float] = [0.0] * (n + 2 * pad_width)
    for i in range(n + 2 * pad_width):
        source_idx = _reflect_index(i - pad_width, n)
        padded[i] = signal[source_idx]
    return padded


def _gaussian_kernel(sigma: float) -> List[float]:
    if sigma <= 0:
        return [1.0]
    radius = max(1, int(math.ceil(3.0 * sigma)))
    values = [math.exp(-float(x * x) / (2.0 * sigma * sigma)) for x in range(-radius, radius + 1)]
    total = sum(values)
    if total == 0.0:
        return [1.0 / (2 * radius + 1)] * (2 * radius + 1)
    return [value / total for value in values]


def _convolve_reflect(signal: List[float], kernel: List[float]) -> List[float]:
    radius = len(kernel) // 2
    padded = _reflect_pad(signal, radius)
    result: List[float] = [0.0] * len(signal)
    for i in range(len(signal)):
        acc = 0.0
        for k, weight in enumerate(kernel):
            acc += padded[i + k] * weight
        result[i] = acc
    return result


def _gradient(values: List[float], coords: List[float]) -> List[float]:
    n = len(values)
    if n != len(coords):
        raise ValueError("values and coords must be the same length")
    if n == 0:
        return []
    if n == 1:
        return [0.0]
    grad: List[float] = [0.0] * n
    for i in range(n):
        if i == 0:
            numerator = values[1] - values[0]
            denominator = coords[1] - coords[0]
        elif i == n - 1:
            numerator = values[-1] - values[-2]
            denominator = coords[-1] - coords[-2]
        else:
            numerator = values[i + 1] - values[i - 1]
            denominator = coords[i + 1] - coords[i - 1]
        if denominator == 0:
            grad[i] = 0.0
        else:
            grad[i] = numerator / denominator
    return grad


@dataclass
class OnTheEdgeLearning:
    """One-dimensional implementation of the edge-of-efficiency heuristic."""

    theta_min: float = -5.0
    theta_max: float = 5.0
    n_min: float = 10.0
    n_max: float = 1000.0
    n_grid: int = 100
    m_grid: int = 50
    epsilon_min: float = 1e-6
    delta: float = 0.01
    tau: float = 20.0
    alpha: float = 10.0
    lambda_H: float = 0.5
    lambda_E: float = 0.5
    lambda_S: float = 0.5
    mu: float = 0.5
    eta: float = 1e-4

    theta: List[float] = field(init=False)
    log_n: List[float] = field(init=False)
    n: List[float] = field(init=False)
    sigma_theta: float = field(init=False)

    I1: List[float] = field(init=False)
    H: List[float] = field(init=False)
    E_p: List[List[float]] = field(init=False)
    E_max: List[List[float]] = field(init=False)
    V: List[List[float]] = field(init=False)
    B: List[List[float]] = field(init=False)
    Psi: List[List[float]] = field(init=False)
    J: List[List[float]] = field(init=False)

    def __post_init__(self) -> None:
        self.theta = _linspace(self.theta_min, self.theta_max, self.n_grid)
        self.log_n = _linspace(_log(self.n_min), _log(self.n_max), self.m_grid)
        self.n = [_exp(value) for value in self.log_n]

        span = self.theta_max - self.theta_min
        self.sigma_theta = 1.06 * span / (self.n_grid ** 0.2)

        self.I1 = [0.0] * self.n_grid
        self.H = [0.0] * self.n_grid
        self.E_p = [[0.0 for _ in range(self.m_grid)] for _ in range(self.n_grid)]
        self.E_max = [[0.0 for _ in range(self.m_grid)] for _ in range(self.n_grid)]
        self.V = [[0.0 for _ in range(self.m_grid)] for _ in range(self.n_grid)]
        self.B = [[0.0 for _ in range(self.m_grid)] for _ in range(self.n_grid)]
        self.Psi = [[0.0 for _ in range(self.m_grid)] for _ in range(self.n_grid)]
        self.J = [[0.0 for _ in range(self.m_grid)] for _ in range(self.n_grid)]

    # --- Core modelling components -------------------------------------------------

    def compute_fisher_information(self, theta: float) -> float:
        value = 1.0 / (1.0 + theta * theta)
        return value if value >= self.epsilon_min else self.epsilon_min

    def compute_entropy(self, theta: float) -> float:
        return -_log(1.0 + theta * theta)

    def compute_estimator_covariance(self, theta: float, n_value: float) -> float:
        fisher = self.compute_fisher_information(theta)
        variance = 1.0 / (n_value * fisher)
        bias = 0.1 * theta / math.sqrt(max(n_value, 1.0))
        return variance + bias * bias

    # --- Composite measures -------------------------------------------------------

    def compute_efficiency_measures(self) -> None:
        """Populate ``E_p`` and the smoothed upper envelope ``E_max``.

        The original notebook applied a log-sum-exp (LSE) smoothing to the
        dominant eigenvalue of the Fisher-scaled covariance.  The scalar
        setting collapses the spectrum to a single value, but we still benefit
        from a small spatial neighbourhood when constructing the envelope: it
        avoids sharp transitions caused by discretisation artifacts and uses
        the ``tau`` temperature parameter that was previously unused in the
        port.
        """

        for i, theta_value in enumerate(self.theta):
            fisher = self.compute_fisher_information(theta_value)
            self.I1[i] = fisher

        for j, n_value in enumerate(self.n):
            products: List[float] = [0.0] * self.n_grid
            for i, theta_value in enumerate(self.theta):
                fisher = self.I1[i]
                sigma = self.compute_estimator_covariance(theta_value, n_value)
                product = fisher * sigma
                self.E_p[i][j] = n_value * product
                products[i] = product

            lse_values = self._local_lse_max(products)
            for i, envelope in enumerate(lse_values):
                self.E_max[i][j] = n_value * envelope

    def _local_lse_max(self, values: List[float], radius: int = 2) -> List[float]:
        if not values:
            return []
        smoothed: List[float] = [0.0] * len(values)
        for idx in range(len(values)):
            start = max(0, idx - radius)
            end = min(len(values), idx + radius + 1)
            window = values[start:end]
            smoothed[idx] = self._log_sum_exp(window)
        return smoothed

    def _log_sum_exp(self, window: List[float]) -> float:
        if not window:
            return self.epsilon_min
        max_value = max(window)
        if not math.isfinite(max_value):
            return max_value
        shifted = [math.exp(self.tau * (value - max_value)) for value in window]
        total = sum(shifted)
        if total == 0.0:
            return self.epsilon_min
        result = max_value + (math.log(total) / self.tau)
        return result if result > self.epsilon_min else self.epsilon_min

    def compute_variance_constraint(self) -> None:
        for i in range(self.n_grid):
            for j in range(self.m_grid):
                deviation = self.E_p[i][j] - 1.0
                smooth = math.exp(-self.alpha * deviation * deviation)
                gate = _sigmoid(deviation + self.delta, steepness=10.0)
                self.V[i][j] = smooth * gate

    def compute_cramer_rao_bound(self) -> None:
        for i, fisher in enumerate(self.I1):
            s_theta = math.sqrt(fisher)
            for j, n_value in enumerate(self.n):
                self.B[i][j] = 1.0 / (math.sqrt(n_value) * s_theta)

    def compute_convolution(self) -> None:
        kernel = _gaussian_kernel(self.sigma_theta)
        for j in range(self.m_grid):
            column = [self.H[i] * self.B[i][j] * self.V[i][j] for i in range(self.n_grid)]
            smoothed = _convolve_reflect(column, kernel)
            for i, value in enumerate(smoothed):
                self.Psi[i][j] = value

    def compute_derivative(self) -> List[List[float]]:
        gradients: List[List[float]] = []
        for row in self.E_p:
            grad_row = _gradient(row, self.log_n)
            gradients.append(self.ema_smooth(grad_row))
        return gradients

    def ema_smooth(self, row: List[float], beta: float = 0.6) -> List[float]:
        if not row:
            return []
        smoothed = row[:]
        for i in range(1, len(row)):
            smoothed[i] = beta * smoothed[i - 1] + (1.0 - beta) * row[i]
        return smoothed

    def compute_objective(self) -> None:
        derivative = self.compute_derivative()
        for i in range(self.n_grid):
            for j in range(self.m_grid):
                e_p = self.E_p[i][j]
                self.J[i][j] = (
                    self.Psi[i][j]
                    + self.lambda_H * self.H[i]
                    - self.lambda_E * (e_p - 1.0) ** 2
                    - self.lambda_S * derivative[i][j] ** 2
                    - self.mu * (_softplus(1.0 - self.delta - e_p) ** 2)
                )

    # --- Diagnostics ----------------------------------------------------------------

    def check_flat_region(self) -> bool:
        gradients: List[float] = []
        for j in range(self.m_grid):
            column = [self.J[i][j] for i in range(self.n_grid)]
            grad_column = _gradient(column, self.theta)
            gradients.extend(abs(value) for value in grad_column)
        return _percentile(gradients, 25.0) < self.eta

    # --- Execution -------------------------------------------------------------------

    def run(self) -> None:
        for i, theta_value in enumerate(self.theta):
            self.I1[i] = self.compute_fisher_information(theta_value)
            self.H[i] = self.compute_entropy(theta_value)
        self.compute_efficiency_measures()
        self.compute_variance_constraint()
        self.compute_cramer_rao_bound()
        self.compute_convolution()
        self.compute_objective()
        self.check_flat_region()

    # --- Optional visualisation ------------------------------------------------------

    def visualize(self) -> None:  # pragma: no cover - convenience helper
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("matplotlib is required for visualization") from exc

        theta = self.theta
        n_values = self.n

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        def contour(ax, data, title: str) -> None:
            contour_data = [row[:] for row in data]
            ax.contourf(theta, n_values, list(map(list, zip(*contour_data))), levels=20)
            ax.set_title(title)
            ax.set_xlabel("θ")
            ax.set_ylabel("n")

        contour(axes[0][0], self.E_p, "E_p(θ, n)")
        contour(axes[0][1], self.Psi, "Ψ(θ, n)")
        contour(axes[1][0], self.J, "J(θ, n)")

        edge_band = [
            [1.0 if 1.0 - self.delta <= self.E_max[i][j] <= 1.0 + self.delta else 0.0 for i in range(self.n_grid)]
            for j in range(self.m_grid)
        ]
        axes[1][1].contourf(theta, n_values, edge_band, levels=[0, 0.5, 1], colors=["lightcoral", "lightblue"])
        axes[1][1].set_title("Edge Band (1±δ)")
        axes[1][1].set_xlabel("θ")
        axes[1][1].set_ylabel("n")

        fig.tight_layout()
        plt.show()
