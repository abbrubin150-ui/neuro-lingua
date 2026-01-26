import { describe, expect, it } from 'vitest';
import {
  clipByGlobalNorm,
  clipByPerParameter,
  clipGradients,
  type GradientClipConfig
} from '../../src/training/GradientClipping';

describe('GradientClipping', () => {
  describe('clipByGlobalNorm', () => {
    it('does not clip when global norm is below threshold', () => {
      const matGrads = [[[0.1, 0.2], [0.3, 0.1]]];
      const vecGrads = [[0.1, 0.2]];
      const result = clipByGlobalNorm(matGrads, vecGrads, 10.0);

      expect(result.wasClipped).toBe(false);
      expect(result.scaleFactor).toBe(1);
      // Values should be unchanged
      expect(matGrads[0][0][0]).toBeCloseTo(0.1);
      expect(matGrads[0][0][1]).toBeCloseTo(0.2);
    });

    it('clips when global norm exceeds threshold', () => {
      // Create gradients with known norm
      const matGrads = [[[3.0, 4.0]]]; // norm = sqrt(9+16) = 5
      const vecGrads: number[][] = [];
      const result = clipByGlobalNorm(matGrads, vecGrads, 2.5);

      expect(result.wasClipped).toBe(true);
      expect(result.globalNorm).toBeCloseTo(5.0);
      expect(result.scaleFactor).toBeCloseTo(0.5);

      // Gradients should be scaled down
      expect(matGrads[0][0][0]).toBeCloseTo(1.5);
      expect(matGrads[0][0][1]).toBeCloseTo(2.0);
    });

    it('preserves gradient direction when clipping', () => {
      const matGrads = [[[6.0, 8.0]]]; // norm = 10
      const vecGrads: number[][] = [];
      clipByGlobalNorm(matGrads, vecGrads, 5.0);

      // Ratio should be preserved: 6/8 = 0.75
      expect(matGrads[0][0][0] / matGrads[0][0][1]).toBeCloseTo(0.75);
    });

    it('handles zero gradients', () => {
      const matGrads = [[[0, 0]]];
      const vecGrads = [[0, 0]];
      const result = clipByGlobalNorm(matGrads, vecGrads, 1.0);

      expect(result.wasClipped).toBe(false);
      expect(result.scaleFactor).toBe(1);
    });

    it('clips vector gradients as part of global norm', () => {
      // mat: [[3,0]], vec: [4] => norm = sqrt(9+16) = 5
      const matGrads = [[[3.0, 0.0]]];
      const vecGrads = [[4.0]];
      const result = clipByGlobalNorm(matGrads, vecGrads, 2.5);

      expect(result.wasClipped).toBe(true);
      expect(result.scaleFactor).toBeCloseTo(0.5);
      expect(matGrads[0][0][0]).toBeCloseTo(1.5);
      expect(vecGrads[0][0]).toBeCloseTo(2.0);
    });

    it('handles empty gradient arrays', () => {
      const result = clipByGlobalNorm([], [], 1.0);
      expect(result.wasClipped).toBe(false);
      expect(result.globalNorm).toBe(0);
    });
  });

  describe('clipByPerParameter', () => {
    it('clips individual matrices that exceed norm', () => {
      const matGrads = [
        [[1.0, 0.0]], // norm = 1 (not clipped)
        [[3.0, 4.0]]  // norm = 5 (clipped to 2.5)
      ];
      const vecGrads: number[][] = [];
      const result = clipByPerParameter(matGrads, vecGrads, 2.5);

      expect(result.numClipped).toBe(1);
      // First matrix unchanged
      expect(matGrads[0][0][0]).toBeCloseTo(1.0);
      // Second matrix scaled: 3*0.5=1.5, 4*0.5=2.0
      expect(matGrads[1][0][0]).toBeCloseTo(1.5);
      expect(matGrads[1][0][1]).toBeCloseTo(2.0);
    });

    it('clips individual vectors that exceed norm', () => {
      const matGrads: number[][][] = [];
      const vecGrads = [
        [0.5, 0.5], // norm ~= 0.707 (not clipped)
        [3.0, 4.0]  // norm = 5 (clipped to 1.0)
      ];
      const result = clipByPerParameter(matGrads, vecGrads, 1.0);

      expect(result.numClipped).toBe(1);
      // First vector unchanged
      expect(vecGrads[0][0]).toBeCloseTo(0.5);
      // Second vector scaled
      expect(vecGrads[1][0]).toBeCloseTo(0.6);
      expect(vecGrads[1][1]).toBeCloseTo(0.8);
    });

    it('does not clip when all norms are below threshold', () => {
      const matGrads = [[[0.1, 0.2]]];
      const vecGrads = [[0.3]];
      const result = clipByPerParameter(matGrads, vecGrads, 100.0);

      expect(result.numClipped).toBe(0);
    });
  });

  describe('clipGradients (unified)', () => {
    it('applies global_norm mode correctly', () => {
      const matGrads = [[[6.0, 8.0]]]; // norm = 10
      const vecGrads: number[][] = [];
      const config: GradientClipConfig = { mode: 'global_norm', maxNorm: 5.0 };

      const stats = clipGradients(matGrads, vecGrads, config);

      expect(stats.wasClipped).toBe(true);
      expect(stats.globalNormBefore).toBeCloseTo(10.0);
      expect(stats.globalNormAfter).toBeCloseTo(5.0);
      expect(stats.scaleFactor).toBeCloseTo(0.5);
      expect(stats.numParamsClipped).toBe(0); // Not in per-param mode
    });

    it('applies per_parameter mode correctly', () => {
      const matGrads = [
        [[1.0, 0.0]],
        [[6.0, 8.0]]
      ];
      const vecGrads: number[][] = [];
      const config: GradientClipConfig = { mode: 'per_parameter', maxNorm: 5.0 };

      const stats = clipGradients(matGrads, vecGrads, config);

      expect(stats.wasClipped).toBe(true);
      expect(stats.numParamsClipped).toBe(1);
      expect(stats.scaleFactor).toBe(1); // Not in global mode
    });

    it('applies both mode: global then per-parameter', () => {
      // Two matrices, each with norm 5 => global norm = sqrt(50) ~= 7.07
      const matGrads = [
        [[3.0, 4.0]], // norm = 5
        [[3.0, 4.0]]  // norm = 5
      ];
      const vecGrads: number[][] = [];
      const config: GradientClipConfig = {
        mode: 'both',
        maxNorm: 5.0,       // Global norm clip
        perParamMaxNorm: 3.0 // Per-parameter clip
      };

      const stats = clipGradients(matGrads, vecGrads, config);

      expect(stats.wasClipped).toBe(true);
      // Global norm was ~7.07, clipped to 5.0
      // Then per-param clips individual matrices to 3.0
      expect(stats.globalNormAfter).toBeLessThanOrEqual(5.0 + 0.001);
    });

    it('returns correct stats when no clipping needed', () => {
      const matGrads = [[[0.1, 0.1]]];
      const vecGrads = [[0.1]];
      const config: GradientClipConfig = { mode: 'global_norm', maxNorm: 100.0 };

      const stats = clipGradients(matGrads, vecGrads, config);

      expect(stats.wasClipped).toBe(false);
      expect(stats.scaleFactor).toBe(1);
      expect(stats.numParamsClipped).toBe(0);
      expect(stats.globalNormBefore).toBeCloseTo(stats.globalNormAfter);
    });
  });
});
