"""Tests for the MBLT-enabled edge analyzer."""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from edge_formalism.core import EdgeAnalyzerWithMBLT
from edge_formalism.mblt import ConfigFactory, load_spec
SPEC_DIR = PROJECT_ROOT / "data" / "edge_formalism"


def build_analyzer(spec_name: str) -> EdgeAnalyzerWithMBLT:
    spec_path = SPEC_DIR / f"{spec_name}.json"
    spec = load_spec(spec_path)
    config = ConfigFactory(spec).build()
    return EdgeAnalyzerWithMBLT(config)


def test_gaussian_efficiency_near_one() -> None:
    analyzer = build_analyzer("gaussian")
    for theta in [-1.5, -0.5, 0.0, 0.5, 1.5]:
        for n_value in [100.0, 250.0, 500.0, 750.0, 1000.0]:
            efficiency = analyzer.compute_E_p(theta, n_value)
            assert abs(efficiency - 1.0) < 0.05


def test_bernoulli_edge_band_conservative() -> None:
    analyzer = build_analyzer("bernoulli")
    delta = analyzer.config.hyperparameters.delta
    for theta in [0.1, 0.25, 0.5, 0.75, 0.9]:
        for n_value in [25.0, 100.0, 250.0, 400.0]:
            e_max = analyzer.compute_E_max(theta, n_value)
            assert e_max >= 1.0 - delta
            assert e_max > 0.0


def test_misspecified_objective_penalises_bias() -> None:
    gaussian = build_analyzer("gaussian")
    misspecified = build_analyzer("misspecified_gaussian")
    theta = 0.25
    n_value = 120.0
    baseline = gaussian.compute_J(theta, n_value)
    degraded = misspecified.compute_J(theta, n_value)
    assert degraded < baseline
