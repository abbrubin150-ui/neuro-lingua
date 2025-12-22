# Python Modules

This document describes the Python modules in the Neuro-Lingua project.

## Module Overview

| Module | Purpose | Location |
|--------|---------|----------|
| `edge_formalism` | Edge learning analysis and MBLT-aware optimization | `/edge_formalism/` |
| `symmetry_coupling` | Symmetry coupling metrics and on-the-edge learning | `/symmetry_coupling/` |
| `neurosync` | Real-time EEG-based consciousness assessment | `/neurosync/` |

## edge_formalism

Edge formalism toolkit for analyzing learning at the edge of stability.

**Main Components:**
- `EdgeAnalyzerWithMBLT` - MBLT-aware edge analyzer
- `RecursiveOptimizer` - Recursive optimization algorithms
- `UnifiedEdgeModule` - Unified edge learning module

**Usage:**
```python
from edge_formalism import EdgeAnalyzerWithMBLT, RecursiveOptimizer

analyzer = EdgeAnalyzerWithMBLT()
optimizer = RecursiveOptimizer()
```

## symmetry_coupling

Symmetry coupling metrics for measuring semantic relationships.

**Main Components:**
- `Config` - Configuration settings
- `Metrics` - Coupling metrics computation
- `compute_coupling` - Single pair coupling computation
- `compute_coupling_batch` - Batch coupling computation
- `OnTheEdgeLearning` - On-the-edge learning implementation

**Usage:**
```python
from symmetry_coupling import compute_coupling, OnTheEdgeLearning

coupling = compute_coupling(embedding1, embedding2)
edge_learner = OnTheEdgeLearning()
```

## neurosync

Real-time consciousness assessment framework using EEG analysis.

**Main Components:**
- `ConsciousnessMetrics` - Validated metrics (PCI, LZc, Phi*, CRS)
- `NeuralSynchronyAnalyzer` - Neural synchrony pattern analysis
- `NeuroSync` - Main analysis system

**Usage:**
```python
from neurosync import NeuroSync

neuros = NeuroSync(sampling_rate=256)
results = neuros.analyze_eeg(eeg_data)
neuros.export_results(results, 'analysis.json')
```

## Scripts

| Script | Purpose | Location |
|--------|---------|----------|
| `train_experiment.py` | Run training experiments | `/scripts/` |
| `build_corpus.py` | Build training corpus | `/scripts/data/` |

## Examples

Example usage scripts are in `/examples/`:
- `example_usage.py` - Basic usage demonstration

## Tests

Python tests are in `/tests/`:
- `test_edge_analyzer.py`
- `test_recursive_optimizer.py`
- `test_on_the_edge_learning.py`

Run tests with:
```bash
python -m pytest tests/test_*.py
```
