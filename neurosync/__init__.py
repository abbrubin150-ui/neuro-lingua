"""
NeuroSync: Real-time Consciousness Assessment Framework
========================================================

Open-source, empirically-validated EEG analysis toolkit for consciousness research

Main Components:
- ConsciousnessMetrics: Validated consciousness metrics (PCI, LZc, Phi*, CRS)
- NeuralSynchronyAnalyzer: Neural synchrony pattern analysis
- NeuroSync: Main analysis system

Example Usage:
```python
from neurosync import NeuroSync

# Initialize system
neuros = NeuroSync(sampling_rate=256)

# Analyze EEG data
results = neuros.analyze_eeg(eeg_data)

# Export results
neuros.export_results(results, 'analysis.json')
```

Version: 1.0
License: MIT
"""

from .neurosync import (
    ConsciousnessMetrics,
    NeuralSynchronyAnalyzer,
    NeuroSync,
    create_sample_eeg,
    run_demo
)

__version__ = "1.0.0"
__author__ = "Neuro-Lingua Research Team"

__all__ = [
    'ConsciousnessMetrics',
    'NeuralSynchronyAnalyzer',
    'NeuroSync',
    'create_sample_eeg',
    'run_demo'
]
