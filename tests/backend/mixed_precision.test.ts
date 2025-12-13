/**
 * Tests for Mixed Precision Training Support
 *
 * Verifies:
 * 1. FP32 to FP16 conversion accuracy
 * 2. FP16 to FP32 conversion accuracy
 * 3. Special values handling (NaN, Infinity, zero)
 * 4. Dynamic loss scaling
 * 5. Overflow/underflow detection
 * 6. Mixed precision tensor management
 * 7. Numerical stability of operations
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  float32ToFloat16Bits,
  float16BitsToFloat32,
  float32ArrayToFloat16,
  float16ArrayToFloat32,
  isSafeForFP16,
  checkFP16Safety,
  DynamicLossScaler,
  MixedPrecisionTensor,
  MixedPrecisionOps,
  DEFAULT_MIXED_PRECISION_CONFIG
} from '../../src/backend/mixed_precision';

describe('FP16 Conversion', () => {
  describe('float32ToFloat16Bits', () => {
    it('should convert zero correctly', () => {
      const bits = float32ToFloat16Bits(0);
      expect(bits).toBe(0x0000);
    });

    it('should convert negative zero correctly', () => {
      const bits = float32ToFloat16Bits(-0);
      expect(bits).toBe(0x8000);
    });

    it('should convert 1.0 correctly', () => {
      const bits = float32ToFloat16Bits(1.0);
      expect(bits).toBe(0x3c00);
    });

    it('should convert -1.0 correctly', () => {
      const bits = float32ToFloat16Bits(-1.0);
      expect(bits).toBe(0xbc00);
    });

    it('should convert 0.5 correctly', () => {
      const bits = float32ToFloat16Bits(0.5);
      expect(bits).toBe(0x3800);
    });

    it('should handle NaN', () => {
      const bits = float32ToFloat16Bits(NaN);
      expect(bits).toBe(0x7e00);
    });

    it('should handle positive infinity', () => {
      const bits = float32ToFloat16Bits(Infinity);
      expect(bits).toBe(0x7c00);
    });

    it('should handle negative infinity', () => {
      const bits = float32ToFloat16Bits(-Infinity);
      expect(bits).toBe(0xfc00);
    });

    it('should overflow large values to infinity', () => {
      const bits = float32ToFloat16Bits(100000);
      expect(bits).toBe(0x7c00); // Positive infinity
    });
  });

  describe('float16BitsToFloat32', () => {
    it('should convert zero correctly', () => {
      expect(float16BitsToFloat32(0x0000)).toBe(0);
    });

    it('should convert 1.0 correctly', () => {
      expect(float16BitsToFloat32(0x3c00)).toBe(1.0);
    });

    it('should convert 0.5 correctly', () => {
      expect(float16BitsToFloat32(0x3800)).toBe(0.5);
    });

    it('should convert NaN correctly', () => {
      expect(Number.isNaN(float16BitsToFloat32(0x7e00))).toBe(true);
    });

    it('should convert infinity correctly', () => {
      expect(float16BitsToFloat32(0x7c00)).toBe(Infinity);
      expect(float16BitsToFloat32(0xfc00)).toBe(-Infinity);
    });
  });

  describe('roundtrip conversion', () => {
    it('should preserve common values', () => {
      const testValues = [0, 1, -1, 0.5, -0.5, 2.0, 100, 0.001];

      for (const v of testValues) {
        const bits = float32ToFloat16Bits(v);
        const recovered = float16BitsToFloat32(bits);
        // Allow small error due to FP16 precision
        expect(Math.abs(recovered - v)).toBeLessThan(Math.abs(v) * 0.01 + 0.001);
      }
    });
  });

  describe('array conversion', () => {
    it('should convert Float32Array to Float16 Uint16Array', () => {
      const fp32 = new Float32Array([1.0, 0.5, -1.0, 0]);
      const fp16 = float32ArrayToFloat16(fp32);

      expect(fp16.length).toBe(4);
      expect(fp16[0]).toBe(0x3c00); // 1.0
      expect(fp16[1]).toBe(0x3800); // 0.5
    });

    it('should convert Float16 back to Float32', () => {
      const fp32Original = new Float32Array([1.0, 0.5, -1.0, 0]);
      const fp16 = float32ArrayToFloat16(fp32Original);
      const fp32Recovered = float16ArrayToFloat32(fp16);

      expect(fp32Recovered.length).toBe(4);
      expect(fp32Recovered[0]).toBeCloseTo(1.0, 2);
      expect(fp32Recovered[1]).toBeCloseTo(0.5, 2);
      expect(fp32Recovered[2]).toBeCloseTo(-1.0, 2);
      expect(fp32Recovered[3]).toBe(0);
    });
  });
});

describe('FP16 Safety Checks', () => {
  describe('isSafeForFP16', () => {
    it('should accept normal values', () => {
      expect(isSafeForFP16(1.0)).toBe(true);
      expect(isSafeForFP16(100)).toBe(true);
      expect(isSafeForFP16(0.001)).toBe(true);
    });

    it('should accept zero', () => {
      expect(isSafeForFP16(0)).toBe(true);
    });

    it('should reject values too large', () => {
      expect(isSafeForFP16(100000)).toBe(false);
    });

    it('should reject values too small (subnormal)', () => {
      expect(isSafeForFP16(1e-10)).toBe(false);
    });
  });

  describe('checkFP16Safety', () => {
    it('should detect no issues for safe array', () => {
      const values = [1.0, 0.5, -1.0, 2.0, 100];
      const result = checkFP16Safety(values);

      expect(result.hasOverflow).toBe(false);
      expect(result.hasUnderflow).toBe(false);
      expect(result.overflowCount).toBe(0);
      expect(result.underflowCount).toBe(0);
    });

    it('should detect overflow', () => {
      const values = [1.0, 100000, 2.0];
      const result = checkFP16Safety(values);

      expect(result.hasOverflow).toBe(true);
      expect(result.overflowCount).toBe(1);
    });

    it('should detect underflow', () => {
      const values = [1.0, 1e-10, 2.0];
      const result = checkFP16Safety(values);

      expect(result.hasUnderflow).toBe(true);
      expect(result.underflowCount).toBe(1);
    });

    it('should report max and min absolute values', () => {
      const values = [1.0, 100, 0.5, 0.001];
      const result = checkFP16Safety(values);

      expect(result.maxAbs).toBe(100);
      expect(result.minAbs).toBe(0.001);
    });
  });
});

describe('DynamicLossScaler', () => {
  let scaler: DynamicLossScaler;

  beforeEach(() => {
    scaler = new DynamicLossScaler();
  });

  describe('initialization', () => {
    it('should initialize with default config', () => {
      expect(scaler.getScale()).toBe(DEFAULT_MIXED_PRECISION_CONFIG.initialLossScale);
    });

    it('should accept custom config', () => {
      const customScaler = new DynamicLossScaler({ initialLossScale: 1024 });
      expect(customScaler.getScale()).toBe(1024);
    });
  });

  describe('loss scaling', () => {
    it('should scale loss correctly', () => {
      const loss = 0.5;
      const scaledLoss = scaler.scaleLoss(loss);
      expect(scaledLoss).toBe(loss * scaler.getScale());
    });

    it('should unscale gradients correctly', () => {
      const gradients = [0.1, 0.2, 0.3];
      const scale = scaler.getScale();
      const scaled = gradients.map((g) => g * scale);

      const unscaled = scaler.unscaleGradients(scaled);

      expect(unscaled[0]).toBeCloseTo(0.1, 5);
      expect(unscaled[1]).toBeCloseTo(0.2, 5);
      expect(unscaled[2]).toBeCloseTo(0.3, 5);
    });

    it('should unscale gradient matrix correctly', () => {
      const gradients = [
        [0.1, 0.2],
        [0.3, 0.4]
      ];
      const scale = scaler.getScale();
      const scaled = gradients.map((row) => row.map((g) => g * scale));

      const unscaled = scaler.unscaleGradientMatrix(scaled);

      expect(unscaled[0][0]).toBeCloseTo(0.1, 5);
      expect(unscaled[1][1]).toBeCloseTo(0.4, 5);
    });
  });

  describe('overflow detection', () => {
    it('should detect NaN gradients', () => {
      const gradients = [0.1, NaN, 0.3];
      const valid = scaler.checkAndUpdateScale(gradients);

      expect(valid).toBe(false);
      expect(scaler.getState().lastOverflow).toBe(true);
    });

    it('should detect Inf gradients', () => {
      const gradients = [0.1, Infinity, 0.3];
      const valid = scaler.checkAndUpdateScale(gradients);

      expect(valid).toBe(false);
    });

    it('should scale down on overflow', () => {
      const initialScale = scaler.getScale();
      const gradients = [NaN];

      scaler.checkAndUpdateScale(gradients);

      expect(scaler.getScale()).toBeLessThan(initialScale);
    });

    it('should not scale below minimum', () => {
      const customScaler = new DynamicLossScaler({
        initialLossScale: 2,
        scaleBackoffFactor: 0.5,
        minLossScale: 1
      });

      // Trigger multiple overflows
      for (let i = 0; i < 10; i++) {
        customScaler.checkAndUpdateScale([NaN]);
      }

      expect(customScaler.getScale()).toBeGreaterThanOrEqual(1);
    });
  });

  describe('scale growth', () => {
    it('should scale up after successful steps', () => {
      const customScaler = new DynamicLossScaler({
        initialLossScale: 1000,
        scaleGrowthInterval: 3,
        scaleGrowthFactor: 2
      });

      const initialScale = customScaler.getScale();

      // 3 successful steps
      for (let i = 0; i < 3; i++) {
        customScaler.checkAndUpdateScale([0.1, 0.2]);
      }

      expect(customScaler.getScale()).toBe(initialScale * 2);
    });

    it('should reset successful steps on overflow', () => {
      const customScaler = new DynamicLossScaler({
        initialLossScale: 1000,
        scaleGrowthInterval: 5
      });

      // 3 successful steps
      for (let i = 0; i < 3; i++) {
        customScaler.checkAndUpdateScale([0.1]);
      }

      // Overflow resets counter
      customScaler.checkAndUpdateScale([NaN]);

      expect(customScaler.getState().successfulSteps).toBe(0);
    });
  });

  describe('state management', () => {
    it('should reset state', () => {
      scaler.checkAndUpdateScale([NaN]); // Trigger overflow
      scaler.reset();

      expect(scaler.getScale()).toBe(DEFAULT_MIXED_PRECISION_CONFIG.initialLossScale);
      expect(scaler.getState().overflowCount).toBe(0);
    });

    it('should export and import state', () => {
      scaler.checkAndUpdateScale([0.1]);
      scaler.checkAndUpdateScale([NaN]);

      const exported = scaler.exportState();

      const newScaler = new DynamicLossScaler();
      newScaler.importState(exported);

      expect(newScaler.getState().overflowCount).toBe(1);
    });
  });
});

describe('MixedPrecisionTensor', () => {
  it('should store master data', () => {
    const data = new Float32Array([1.0, 2.0, 3.0]);
    const tensor = new MixedPrecisionTensor(data, [3]);

    expect(tensor.getMasterData()).toBe(data);
    expect(tensor.size).toBe(3);
    expect(tensor.shape).toEqual([3]);
  });

  it('should create FP16 working copy on demand', () => {
    const data = new Float32Array([1.0, 2.0, 3.0]);
    const tensor = new MixedPrecisionTensor(data, [3]);

    const working = tensor.getWorkingData();

    expect(working instanceof Uint16Array).toBe(true);
    expect(working.length).toBe(3);
  });

  it('should cache working copy', () => {
    const data = new Float32Array([1.0, 2.0, 3.0]);
    const tensor = new MixedPrecisionTensor(data, [3]);

    const working1 = tensor.getWorkingData();
    const working2 = tensor.getWorkingData();

    expect(working1).toBe(working2); // Same reference
  });

  it('should invalidate working copy on master update', () => {
    const data = new Float32Array([1.0, 2.0, 3.0]);
    const tensor = new MixedPrecisionTensor(data, [3]);

    const working1 = tensor.getWorkingData();

    // Update master
    tensor.updateMaster(new Float32Array([4.0, 5.0, 6.0]));

    const working2 = tensor.getWorkingData();

    // Should be different array
    expect(working1).not.toBe(working2);
    // New values
    expect(float16BitsToFloat32(working2[0])).toBeCloseTo(4.0, 2);
  });

  it('should report memory usage', () => {
    const data = new Float32Array([1.0, 2.0, 3.0, 4.0]);
    const tensor = new MixedPrecisionTensor(data, [4]);

    // Before accessing working copy
    let memory = tensor.getMemoryUsage();
    expect(memory.master).toBe(16); // 4 * 4 bytes
    expect(memory.working).toBe(0);

    // After accessing working copy
    tensor.getWorkingData();
    memory = tensor.getMemoryUsage();
    expect(memory.master).toBe(16);
    expect(memory.working).toBe(8); // 4 * 2 bytes
  });
});

describe('MixedPrecisionOps', () => {
  describe('matvec', () => {
    it('should compute matrix-vector multiplication', () => {
      const matrix = [
        [1, 2],
        [3, 4]
      ];
      const vector = [1, 1];

      const result = MixedPrecisionOps.matvec(matrix, vector);

      expect(result).toEqual([3, 7]);
    });
  });

  describe('softmax', () => {
    it('should compute stable softmax', () => {
      const logits = [1.0, 2.0, 3.0];
      const result = MixedPrecisionOps.softmax(logits);

      // Should sum to 1
      const sum = result.reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1.0, 5);

      // Should be monotonically increasing
      expect(result[0]).toBeLessThan(result[1]);
      expect(result[1]).toBeLessThan(result[2]);
    });

    it('should handle large values without overflow', () => {
      const logits = [1000, 1001, 1002];
      const result = MixedPrecisionOps.softmax(logits);

      const sum = result.reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1.0, 5);
      expect(Number.isFinite(result[0])).toBe(true);
    });
  });

  describe('layerNorm', () => {
    it('should normalize input', () => {
      const input = [1, 2, 3, 4];
      const gamma = [1, 1, 1, 1];
      const beta = [0, 0, 0, 0];

      const result = MixedPrecisionOps.layerNorm(input, gamma, beta);

      // Mean should be ~0
      const mean = result.reduce((a, b) => a + b, 0) / result.length;
      expect(mean).toBeCloseTo(0, 5);

      // Variance should be ~1
      const variance =
        result.reduce((a, b) => a + b * b, 0) / result.length - mean * mean;
      expect(variance).toBeCloseTo(1, 2);
    });

    it('should apply scale and shift', () => {
      const input = [0, 0, 0, 0];
      const gamma = [2, 2, 2, 2];
      const beta = [1, 1, 1, 1];

      const result = MixedPrecisionOps.layerNorm(input, gamma, beta);

      // All zeros normalized are still zeros, then scaled and shifted
      expect(result[0]).toBeCloseTo(1, 5);
    });
  });

  describe('clipGradients', () => {
    it('should not clip small gradients', () => {
      const gradients = [0.1, 0.2, 0.3];
      const result = MixedPrecisionOps.clipGradients(gradients);

      expect(result[0]).toBeCloseTo(0.1, 5);
    });

    it('should clip large gradients', () => {
      const gradients = [10000, 10000, 10000];
      const maxNorm = 100;
      const result = MixedPrecisionOps.clipGradients(gradients, maxNorm);

      const norm = Math.sqrt(result.reduce((a, b) => a + b * b, 0));
      expect(norm).toBeLessThanOrEqual(maxNorm + 0.01);
    });
  });
});
