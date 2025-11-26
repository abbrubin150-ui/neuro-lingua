import { describe, it, expect } from 'vitest';
import {
  quantizeArray,
  quantizeMatrix,
  dequantizeArray,
  dequantizeMatrix,
  calculateQuantizationError,
  serializeQuantizedWeights,
  deserializeQuantizedWeights
} from '../../src/compression/quantization';

describe('Quantization', () => {
  describe('quantizeArray', () => {
    it('should quantize a simple array to int8', () => {
      const arr = [0.5, -0.5, 1.0, -1.0, 0.0];
      const quantized = quantizeArray(arr);

      expect(quantized.values).toBeInstanceOf(Int8Array);
      expect(quantized.values.length).toBe(arr.length);
      expect(quantized.params.scale).toBeGreaterThan(0);
    });

    it('should handle empty arrays', () => {
      const arr: number[] = [];
      const quantized = quantizeArray(arr);

      expect(quantized.values.length).toBe(0);
      expect(quantized.originalShape).toEqual([0]);
    });

    it('should use symmetric quantization', () => {
      const arr = [1.0, -1.0, 0.5, -0.5];
      const quantized = quantizeArray(arr);

      // Zero point should be 0 for symmetric quantization
      expect(quantized.params.zeroPoint).toBe(0);
    });

    it('should clamp values to int8 range [-128, 127]', () => {
      const arr = [1000, -1000, 0];
      const quantized = quantizeArray(arr);

      for (let i = 0; i < quantized.values.length; i++) {
        expect(quantized.values[i]).toBeGreaterThanOrEqual(-128);
        expect(quantized.values[i]).toBeLessThanOrEqual(127);
      }
    });
  });

  describe('quantizeMatrix', () => {
    it('should quantize a 2D matrix', () => {
      const matrix = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
      ];

      const quantized = quantizeMatrix(matrix);

      expect(quantized.values.length).toBe(6);
      expect(quantized.originalShape).toEqual([2, 3]);
    });

    it('should handle empty matrices', () => {
      const matrix: number[][] = [];
      const quantized = quantizeMatrix(matrix);

      expect(quantized.values.length).toBe(0);
      expect(quantized.originalShape).toEqual([0, 0]);
    });
  });

  describe('dequantizeArray', () => {
    it('should approximately reconstruct original array', () => {
      const original = [0.5, -0.5, 1.0, -1.0, 0.0];
      const quantized = quantizeArray(original);
      const reconstructed = dequantizeArray(quantized);

      expect(reconstructed.length).toBe(original.length);

      // Check that values are close (within quantization error)
      for (let i = 0; i < original.length; i++) {
        expect(Math.abs(original[i] - reconstructed[i])).toBeLessThan(0.1);
      }
    });

    it('should preserve zero values exactly', () => {
      const original = [0, 0, 0];
      const quantized = quantizeArray(original);
      const reconstructed = dequantizeArray(quantized);

      for (let i = 0; i < reconstructed.length; i++) {
        expect(Math.abs(reconstructed[i])).toBeLessThan(1e-6);
      }
    });
  });

  describe('dequantizeMatrix', () => {
    it('should reconstruct matrix with correct shape', () => {
      const original = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
      ];

      const quantized = quantizeMatrix(original);
      const reconstructed = dequantizeMatrix(quantized);

      expect(reconstructed.length).toBe(2);
      expect(reconstructed[0].length).toBe(3);
    });

    it('should approximately preserve values', () => {
      const original = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
      ];

      const quantized = quantizeMatrix(original);
      const reconstructed = dequantizeMatrix(quantized);

      for (let i = 0; i < original.length; i++) {
        for (let j = 0; j < original[i].length; j++) {
          expect(Math.abs(original[i][j] - reconstructed[i][j])).toBeLessThan(0.05);
        }
      }
    });
  });

  describe('calculateQuantizationError', () => {
    it('should calculate MSE correctly', () => {
      const original = [1.0, 2.0, 3.0, 4.0];
      const quantized = quantizeArray(original);
      const error = calculateQuantizationError(original, quantized);

      expect(error).toBeGreaterThanOrEqual(0);
      expect(error).toBeLessThan(0.1); // Should be small for this range
    });

    it('should return zero error for perfect reconstruction', () => {
      const original = [0, 0, 0];
      const quantized = quantizeArray(original);
      const error = calculateQuantizationError(original, quantized);

      expect(error).toBeLessThan(1e-6);
    });
  });

  describe('Serialization', () => {
    it('should serialize and deserialize quantized weights', () => {
      const original = [1.0, 2.0, 3.0, 4.0, 5.0];
      const quantized = quantizeArray(original);

      const serialized = serializeQuantizedWeights(quantized);
      const deserialized = deserializeQuantizedWeights(serialized);

      expect(deserialized.values).toBeInstanceOf(Int8Array);
      expect(deserialized.values.length).toBe(quantized.values.length);
      expect(deserialized.params.scale).toBe(quantized.params.scale);
      expect(deserialized.originalShape).toEqual(quantized.originalShape);
    });

    it('should preserve quantization after round-trip', () => {
      const original = [0.5, -0.5, 1.0, -1.0];
      const quantized = quantizeArray(original);
      const serialized = serializeQuantizedWeights(quantized);
      const deserialized = deserializeQuantizedWeights(serialized);

      const reconstructed = dequantizeArray(deserialized);

      for (let i = 0; i < original.length; i++) {
        expect(Math.abs(original[i] - reconstructed[i])).toBeLessThan(0.1);
      }
    });
  });

  describe('Compression Ratio', () => {
    it('should achieve ~4x compression for float32 to int8', () => {
      const size = 1000;
      const arr = Array(size)
        .fill(0)
        .map(() => Math.random());

      const quantized = quantizeArray(arr);

      // Original: 1000 floats × 4 bytes = 4000 bytes
      // Quantized: 1000 int8s × 1 byte = 1000 bytes
      // Expected ratio: ~4x

      const originalBytes = size * 4;
      const quantizedBytes = quantized.values.length * 1;
      const ratio = originalBytes / quantizedBytes;

      expect(ratio).toBeCloseTo(4, 0);
    });
  });
});
