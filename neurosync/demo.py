#!/usr/bin/env python3
"""
Simple demonstration of NeuroSync capabilities
"""

from neurosync import ConsciousnessMetrics, create_sample_eeg
import numpy as np

def simple_demo():
    """Run a simple demonstration"""

    print("="*60)
    print("NeuroSync Simple Demo")
    print("="*60)

    # Initialize metrics calculator
    print("\n1. Initializing NeuroSync Metrics...")
    metrics = ConsciousnessMetrics(fs=256)
    print("   ✓ System initialized")

    # Generate sample data
    print("\n2. Generating sample EEG data...")
    eeg_data = create_sample_eeg(fs=256, duration=5, state='awake')
    print(f"   ✓ Generated {eeg_data.shape[0]} channels, {eeg_data.shape[1]} samples")

    # Analyze core metrics only (faster)
    print("\n3. Computing consciousness metrics...")
    print("   - Computing PCI...")
    pci = metrics.compute_PCI(eeg_data)
    print(f"     ✓ PCI = {pci:.3f}")

    print("   - Computing PCC...")
    pcc = metrics.compute_PCC(eeg_data)
    print(f"     ✓ PCC = {pcc:.3f}")

    print("   - Computing LZc...")
    lzc = metrics.compute_LZc(eeg_data)
    print(f"     ✓ LZc = {lzc:.3f}")

    print("   - Computing CRS...")
    crs = metrics.compute_CRS(eeg_data)
    print(f"     ✓ CRS = {crs:.1f}/100")

    # Summary
    print("\n" + "="*60)
    print("QUICK SUMMARY")
    print("="*60)
    print(f"Perturbational Complexity Index (PCI): {pci:.3f}")
    print(f"Phase Coherence Complexity (PCC): {pcc:.3f}")
    print(f"Lempel-Ziv Complexity (LZc): {lzc:.3f}")
    print(f"Consciousness Repertoire Score (CRS): {crs:.1f}/100")

    # Classification
    results = {
        'PCI': pci,
        'PCC': pcc,
        'LZc': lzc,
        'CRS': crs
    }
    state = metrics.classify_state(results)
    print(f"\nConsciousness State: {state}")

    print("\n✓ Demo complete!")

    return results

if __name__ == "__main__":
    simple_demo()
