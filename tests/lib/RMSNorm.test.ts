import { describe, it, expect, beforeEach } from 'vitest';
import { RMSNorm, rmsNorm, batchRMSNorm } from '../../src/lib/RMSNorm';

describe('RMSNorm', () => {
  describe('Standalone rmsNorm function', () => {
    it('normalizes vector to unit RMS', () => {
      const x = [1, 2, 3, 4];
      const gamma = [1, 1, 1, 1]; // No scaling

      const output = rmsNorm(x, gamma);

      // Compute RMS of output (should be close to 1.0)
      const sumSquares = output.reduce((sum, val) => sum + val * val, 0);
      const rms = Math.sqrt(sumSquares / output.length);

      expect(rms).toBeCloseTo(1.0, 5);
    });

    it('applies gamma scaling correctly', () => {
      const x = [2, 2, 2, 2];
      const gamma = [2, 3, 4, 5];

      const output = rmsNorm(x, gamma);

      // After normalization, x becomes [1, 1, 1, 1], then scaled by gamma
      expect(output[0]).toBeCloseTo(2.0, 5);
      expect(output[1]).toBeCloseTo(3.0, 5);
      expect(output[2]).toBeCloseTo(4.0, 5);
      expect(output[3]).toBeCloseTo(5.0, 5);
    });

    it('handles zero vector gracefully with epsilon', () => {
      const x = [0, 0, 0, 0];
      const gamma = [1, 1, 1, 1];

      // Should not crash or return NaN
      const output = rmsNorm(x, gamma, 1e-6);

      expect(output.every((val) => isFinite(val))).toBe(true);
      expect(output.every((val) => val === 0)).toBe(true);
    });

    it('is scale-invariant up to gamma', () => {
      const x1 = [1, 2, 3];
      const x2 = [10, 20, 30]; // 10x scaled
      const gamma = [1, 1, 1];

      const output1 = rmsNorm(x1, gamma);
      const output2 = rmsNorm(x2, gamma);

      // Should produce same normalized output (scale-invariant)
      for (let i = 0; i < output1.length; i++) {
        expect(output1[i]).toBeCloseTo(output2[i], 5);
      }
    });
  });

  describe('RMSNorm class', () => {
    let layer: RMSNorm;

    beforeEach(() => {
      layer = new RMSNorm(4);
    });

    it('initializes gamma to ones', () => {
      const state = layer.exportState();
      expect(state.gamma).toEqual([1, 1, 1, 1]);
    });

    it('performs forward pass correctly', () => {
      const x = [1, 2, 3, 4];
      const output = layer.forward(x);

      // Verify normalization
      const sumSquares = output.reduce((sum, val) => sum + val * val, 0);
      const rms = Math.sqrt(sumSquares / output.length);
      expect(rms).toBeCloseTo(1.0, 5);
    });

    it('throws error for mismatched input dimension', () => {
      const wrongSize = [1, 2, 3]; // Should be size 4

      expect(() => layer.forward(wrongSize)).toThrow(/dimension/i);
    });

    it('computes gradients in backward pass', () => {
      const x = [1, 2, 3, 4];
      const gradOutput = [0.1, 0.2, 0.3, 0.4];

      layer.forward(x);
      const gradInput = layer.backward(gradOutput);

      expect(gradInput.length).toBe(4);
      expect(gradInput.every((val) => isFinite(val))).toBe(true);
    });

    it('accumulates gamma gradients', () => {
      const x = [2, 2, 2, 2];
      const gradOutput = [1, 1, 1, 1];

      layer.forward(x);
      layer.backward(gradOutput);

      const state = layer.exportState();
      // Gamma gradient should be non-zero
      // (can't easily check exact value without exposing private field)
      expect(state.gamma).toEqual([1, 1, 1, 1]); // Gamma unchanged before update
    });

    it('updates gamma parameters with learning rate', () => {
      const x = [1, 2, 3, 4];
      const gradOutput = [0.1, 0.1, 0.1, 0.1];

      layer.forward(x);
      layer.backward(gradOutput);
      layer.updateParameters(0.1);

      const state = layer.exportState();
      // Gamma should have changed after update
      expect(state.gamma).not.toEqual([1, 1, 1, 1]);
    });

    it('zeros gradients after zeroGradients()', () => {
      const x = [1, 2, 3, 4];
      const gradOutput = [0.1, 0.1, 0.1, 0.1];

      layer.forward(x);
      layer.backward(gradOutput);
      layer.zeroGradients();

      // After zeroing, next backward should start fresh
      layer.forward(x);
      layer.backward(gradOutput);
      layer.updateParameters(0.1);

      // Should only reflect one backward pass worth of gradients
      const state = layer.exportState();
      expect(state.gamma.some((g) => g !== 1)).toBe(true);
    });

    it('resets to initial state', () => {
      const x = [1, 2, 3, 4];
      const gradOutput = [0.1, 0.1, 0.1, 0.1];

      // Train for a bit
      for (let i = 0; i < 5; i++) {
        layer.forward(x);
        layer.backward(gradOutput);
        layer.updateParameters(0.1);
        layer.zeroGradients();
      }

      layer.reset();

      const state = layer.exportState();
      expect(state.gamma).toEqual([1, 1, 1, 1]);
    });

    it('reports correct parameter count', () => {
      expect(layer.getParameterCount()).toBe(4); // Only gamma, no beta
    });
  });

  describe('Serialization', () => {
    it('exports and imports state correctly', () => {
      const layer1 = new RMSNorm(3);
      const x = [1, 2, 3];
      const gradOutput = [0.1, 0.2, 0.3];

      // Train layer1
      for (let i = 0; i < 10; i++) {
        layer1.forward(x);
        layer1.backward(gradOutput);
        layer1.updateParameters(0.1);
        layer1.zeroGradients();
      }

      // Export state
      const state = layer1.exportState();

      // Create new layer from state
      const layer2 = RMSNorm.loadState(state, 3);

      // Should produce same output
      const output1 = layer1.forward(x);
      const output2 = layer2.forward(x);

      for (let i = 0; i < output1.length; i++) {
        expect(output1[i]).toBeCloseTo(output2[i], 10);
      }
    });

    it('preserves epsilon in serialization', () => {
      const customEpsilon = 1e-8;
      const layer1 = new RMSNorm(4, customEpsilon);

      const state = layer1.exportState();
      expect(state.epsilon).toBe(customEpsilon);

      const layer2 = RMSNorm.loadState(state, 4);
      const state2 = layer2.exportState();
      expect(state2.epsilon).toBe(customEpsilon);
    });
  });

  describe('Batch processing', () => {
    it('processes batch of vectors', () => {
      const batch = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
      ];
      const gamma = [1, 1, 1];

      const output = batchRMSNorm(batch, gamma);

      expect(output.length).toBe(3);
      expect(output[0].length).toBe(3);

      // Each output should be normalized
      output.forEach((vec) => {
        const sumSquares = vec.reduce((sum, val) => sum + val * val, 0);
        const rms = Math.sqrt(sumSquares / vec.length);
        expect(rms).toBeCloseTo(1.0, 5);
      });
    });

    it('applies gamma consistently across batch', () => {
      const batch = [
        [2, 2, 2],
        [3, 3, 3]
      ];
      const gamma = [2, 3, 4];

      const output = batchRMSNorm(batch, gamma);

      // Each row should have same pattern of scaling
      expect(output[0][1] / output[0][0]).toBeCloseTo(gamma[1] / gamma[0], 5);
      expect(output[1][1] / output[1][0]).toBeCloseTo(gamma[1] / gamma[0], 5);
    });
  });

  describe('Numerical stability', () => {
    it('handles very small values', () => {
      const x = [1e-10, 2e-10, 3e-10, 4e-10];
      const gamma = [1, 1, 1, 1];

      const output = rmsNorm(x, gamma);

      expect(output.every((val) => isFinite(val))).toBe(true);
      expect(output.every((val) => !isNaN(val))).toBe(true);
    });

    it('handles very large values', () => {
      const x = [1e10, 2e10, 3e10, 4e10];
      const gamma = [1, 1, 1, 1];

      const output = rmsNorm(x, gamma);

      expect(output.every((val) => isFinite(val))).toBe(true);
      const sumSquares = output.reduce((sum, val) => sum + val * val, 0);
      const rms = Math.sqrt(sumSquares / output.length);
      expect(rms).toBeCloseTo(1.0, 5);
    });

    it('handles mixed positive and negative values', () => {
      const x = [-2, -1, 1, 2];
      const gamma = [1, 1, 1, 1];

      const output = rmsNorm(x, gamma);

      // RMS should ignore sign
      const sumSquares = output.reduce((sum, val) => sum + val * val, 0);
      const rms = Math.sqrt(sumSquares / output.length);
      expect(rms).toBeCloseTo(1.0, 5);

      // Signs should be preserved
      expect(output[0]).toBeLessThan(0);
      expect(output[1]).toBeLessThan(0);
      expect(output[2]).toBeGreaterThan(0);
      expect(output[3]).toBeGreaterThan(0);
    });
  });

  describe('Comparison with LayerNorm behavior', () => {
    it('produces similar normalization effect', () => {
      const x = [1, 2, 3, 4, 5];
      const gamma = [1, 1, 1, 1, 1];

      const output = rmsNorm(x, gamma);

      // RMSNorm should still center distribution around reasonable values
      const mean = output.reduce((sum, val) => sum + val, 0) / output.length;
      const variance =
        output.reduce((sum, val) => sum + (val - mean) * (val - mean), 0) / output.length;

      // Variance should be controlled (not too large)
      expect(variance).toBeLessThan(2.0);
      expect(variance).toBeGreaterThan(0.1);
    });

    it('has fewer parameters than LayerNorm', () => {
      const dimension = 128;
      const rmsNormLayer = new RMSNorm(dimension);

      // RMSNorm: only gamma (128 params)
      // LayerNorm would have: gamma + beta (256 params)
      expect(rmsNormLayer.getParameterCount()).toBe(dimension);
      expect(rmsNormLayer.getParameterCount()).toBeLessThan(dimension * 2);
    });
  });

  describe('Edge cases', () => {
    it('handles single-element vector', () => {
      const x = [5];
      const gamma = [2];

      const output = rmsNorm(x, gamma);

      // RMS of [5] is 5, normalized is [1], scaled is [2]
      expect(output[0]).toBeCloseTo(2.0, 5);
    });

    it('handles two-element vector', () => {
      const x = [3, 4]; // RMS = 5/sqrt(2)
      const gamma = [1, 1];

      const output = rmsNorm(x, gamma);

      const sumSquares = output.reduce((sum, val) => sum + val * val, 0);
      const rms = Math.sqrt(sumSquares / output.length);
      expect(rms).toBeCloseTo(1.0, 5);
    });

    it('handles all-same values', () => {
      const x = [7, 7, 7, 7];
      const gamma = [1, 1, 1, 1];

      const output = rmsNorm(x, gamma);

      // All values should be equal after normalization
      expect(output[0]).toBeCloseTo(output[1], 10);
      expect(output[1]).toBeCloseTo(output[2], 10);
      expect(output[2]).toBeCloseTo(output[3], 10);

      // And should have unit RMS
      const sumSquares = output.reduce((sum, val) => sum + val * val, 0);
      const rms = Math.sqrt(sumSquares / output.length);
      expect(rms).toBeCloseTo(1.0, 5);
    });
  });
});
