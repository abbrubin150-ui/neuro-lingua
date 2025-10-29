import { describe, it, expect } from 'vitest';
import {
  computeEdgeDiagnostics,
  interpretEdgeDiagnostics,
  isEdgeDiagnosticsAvailable
} from '../src/lib/edgeDiagnostics';

describe('Edge Diagnostics', () => {
  it('should check if edge diagnostics are available', async () => {
    const available = await isEdgeDiagnosticsAvailable();
    expect(typeof available).toBe('boolean');
  });

  it('should compute edge diagnostics with default parameters', async () => {
    const available = await isEdgeDiagnosticsAvailable();
    if (!available) {
      console.warn('Skipping test: Python3 not available');
      return;
    }

    const diagnostics = await computeEdgeDiagnostics({
      nGrid: 20,
      mGrid: 15
    });

    expect(diagnostics).toHaveProperty('fisherInformationRange');
    expect(diagnostics).toHaveProperty('entropyRange');
    expect(diagnostics).toHaveProperty('averageEfficiency');
    expect(diagnostics).toHaveProperty('inFlatRegion');
    expect(diagnostics).toHaveProperty('edgeBandPercentage');

    expect(Array.isArray(diagnostics.fisherInformationRange)).toBe(true);
    expect(diagnostics.fisherInformationRange.length).toBe(2);
    expect(Array.isArray(diagnostics.entropyRange)).toBe(true);
    expect(diagnostics.entropyRange.length).toBe(2);
    expect(typeof diagnostics.averageEfficiency).toBe('number');
    expect(typeof diagnostics.inFlatRegion).toBe('boolean');
    expect(typeof diagnostics.edgeBandPercentage).toBe('number');
  }, 10000); // 10s timeout for Python execution

  it('should interpret edge diagnostics', () => {
    const diagnostics = {
      fisherInformationRange: [0.1, 1.0] as [number, number],
      entropyRange: [-2.3, -0.01] as [number, number],
      averageEfficiency: 1.01,
      inFlatRegion: true,
      edgeBandPercentage: 95.5
    };

    const interpretation = interpretEdgeDiagnostics(diagnostics);

    expect(interpretation).toContain('near optimal statistical efficiency');
    expect(interpretation).toContain('flat region');
    expect(interpretation).toContain('edge band');
  });

  it('should warn about inefficient models', () => {
    const diagnostics = {
      fisherInformationRange: [0.1, 1.0] as [number, number],
      entropyRange: [-2.3, -0.01] as [number, number],
      averageEfficiency: 0.75,
      inFlatRegion: false,
      edgeBandPercentage: 30.0
    };

    const interpretation = interpretEdgeDiagnostics(diagnostics);

    expect(interpretation).toContain('inefficient');
    expect(interpretation).toContain('not in flat region');
    expect(interpretation).toContain('poorly calibrated');
  });

  it('should handle custom parameter ranges', async () => {
    const available = await isEdgeDiagnosticsAvailable();
    if (!available) {
      console.warn('Skipping test: Python3 not available');
      return;
    }

    const diagnostics = await computeEdgeDiagnostics({
      thetaMin: -1.0,
      thetaMax: 1.0,
      nMin: 50,
      nMax: 500,
      nGrid: 15,
      mGrid: 10
    });

    expect(diagnostics.averageEfficiency).toBeGreaterThan(0);
    expect(diagnostics.edgeBandPercentage).toBeGreaterThanOrEqual(0);
    expect(diagnostics.edgeBandPercentage).toBeLessThanOrEqual(100);
  }, 10000);
});
