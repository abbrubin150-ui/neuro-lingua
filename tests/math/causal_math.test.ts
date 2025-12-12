/**
 * Causal Math Utilities Test Suite
 *
 * Tests for mathematical foundations of the causal inference system.
 * Covers:
 * - Statistical functions (sigmoid, softmax, mean, variance)
 * - Propensity score estimation
 * - AIPW estimator
 * - Outcome model fitting
 * - Quantization/dequantization
 * - Statistical testing
 * - Random number generation
 *
 * @module causal_math.test
 */

import { describe, it, expect } from 'vitest';
import {
  sigmoid,
  logSumExp,
  stableSoftmax,
  mean,
  variance,
  stdDev,
  covariance,
  estimatePropensityScores,
  predictPropensity,
  computeAIPWEstimate,
  computeIPWEstimate,
  fitOutcomeModel,
  predictOutcome,
  createUniformQuantization,
  createEntropyQuantization,
  quantize,
  batchQuantize,
  learnDequantizationMappings,
  dequantize,
  adaptQuantization,
  testATESignificance,
  bootstrapATEConfidenceInterval,
  estimatePower,
  minimumDetectableEffect,
  createSeededRandom,
  randomNormal,
  randomMultivariateNormal,
  dotProduct,
  transpose,
  matrixVectorMultiply,
  matrixMultiply,
  invertMatrix,
  normalCDF,
  normalQuantile
} from '../../src/math/causal_math';
import { WebGPUBackend } from '../../src/backend/webgpu';

// ============================================================================
// Basic Statistical Functions
// ============================================================================

describe('Basic Statistical Functions', () => {
  describe('sigmoid', () => {
    it('should return 0.5 for input 0', () => {
      expect(sigmoid(0)).toBeCloseTo(0.5, 10);
    });

    it('should approach 1 for large positive input', () => {
      expect(sigmoid(10)).toBeCloseTo(1, 4);
      expect(sigmoid(100)).toBeCloseTo(1, 10);
    });

    it('should approach 0 for large negative input', () => {
      expect(sigmoid(-10)).toBeCloseTo(0, 4);
      expect(sigmoid(-100)).toBeCloseTo(0, 10);
    });

    it('should be numerically stable', () => {
      // Should not overflow or underflow
      expect(isFinite(sigmoid(1000))).toBe(true);
      expect(isFinite(sigmoid(-1000))).toBe(true);
    });

    it('should be monotonically increasing', () => {
      const values = [-5, -2, 0, 2, 5];
      for (let i = 1; i < values.length; i++) {
        expect(sigmoid(values[i])).toBeGreaterThan(sigmoid(values[i - 1]));
      }
    });
  });

  describe('logSumExp', () => {
    it('should return -Infinity for empty array', () => {
      expect(logSumExp([])).toBe(-Infinity);
    });

    it('should return the value for single element', () => {
      expect(logSumExp([5])).toBeCloseTo(5, 10);
    });

    it('should compute log(e^a + e^b) correctly', () => {
      // log(e^0 + e^0) = log(2)
      expect(logSumExp([0, 0])).toBeCloseTo(Math.log(2), 10);
    });

    it('should be numerically stable for large values', () => {
      const result = logSumExp([1000, 1001]);
      expect(isFinite(result)).toBe(true);
      expect(result).toBeCloseTo(1001 + Math.log(1 + Math.exp(-1)), 5);
    });
  });

  describe('stableSoftmax', () => {
    it('should return probabilities summing to 1', () => {
      const probs = stableSoftmax([1, 2, 3]);
      const sum = probs.reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1, 10);
    });

    it('should preserve relative ordering', () => {
      const probs = stableSoftmax([1, 2, 3]);
      expect(probs[2]).toBeGreaterThan(probs[1]);
      expect(probs[1]).toBeGreaterThan(probs[0]);
    });

    it('should be stable for large values', () => {
      const probs = stableSoftmax([1000, 1001, 1002]);
      expect(probs.every(isFinite)).toBe(true);
      expect(probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 10);
    });

    it('should avoid overflow near exponent limits', () => {
      const logits = [1e4, 1e4 - 1, 1e4 - 2];
      const probs = stableSoftmax(logits);
      expect(probs.every((p) => p >= 0 && p <= 1)).toBe(true);
      expect(probs).toMatchObject([
        expect.closeTo(0.66524096, 6),
        expect.closeTo(0.24472847, 6),
        expect.closeTo(0.09003057, 6)
      ]);
      expect(probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
    });

    it('should avoid underflow for extremely negative logits', () => {
      const logits = [-1e4, -1e4 - 1, -1e4 - 2];
      const probs = stableSoftmax(logits);
      expect(probs.every((p) => p >= 0 && p <= 1)).toBe(true);
      expect(probs).toMatchObject([
        expect.closeTo(0.66524096, 6),
        expect.closeTo(0.24472847, 6),
        expect.closeTo(0.09003057, 6)
      ]);
      expect(probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
    });

    it('should reject non-finite logits', () => {
      expect(() => stableSoftmax([0, Number.POSITIVE_INFINITY])).toThrow();
    });
  });

  describe('stableSoftmax parity (CPU vs WebGPU)', () => {
    const hasWebGPU =
      typeof globalThis.navigator !== 'undefined' &&
      'gpu' in globalThis.navigator &&
      Boolean((globalThis.navigator as Navigator & { gpu?: GPU }).gpu);

    const parityTest = hasWebGPU ? it : it.skip;

    parityTest('matches CPU implementation near exponent extremes', async () => {
      const logits = new Float32Array([80, 79.5, -79.5, -80]);
      const backend = await WebGPUBackend.create();
      const tensor = await backend.createTensor(logits, [logits.length]);
      const gpuSoftmax = await backend.softmax(tensor, 1);
      const gpuProbs = await gpuSoftmax.toArray();
      const cpuProbs = stableSoftmax(Array.from(logits));

      for (let i = 0; i < gpuProbs.length; i++) {
        expect(gpuProbs[i]).toBeCloseTo(cpuProbs[i], 5);
      }

      tensor.dispose();
      gpuSoftmax.dispose();
    });
  });

  describe('mean', () => {
    it('should return 0 for empty array', () => {
      expect(mean([])).toBe(0);
    });

    it('should return the value for single element', () => {
      expect(mean([5])).toBe(5);
    });

    it('should compute arithmetic mean correctly', () => {
      expect(mean([1, 2, 3, 4, 5])).toBe(3);
      expect(mean([2, 4, 6])).toBe(4);
    });
  });

  describe('variance', () => {
    it('should return 0 for single element', () => {
      expect(variance([5])).toBe(0);
    });

    it('should compute sample variance correctly', () => {
      // Sample variance of [1, 2, 3] = 1
      expect(variance([1, 2, 3])).toBeCloseTo(1, 10);
    });

    it('should use ddof parameter', () => {
      // Population variance (ddof=0)
      expect(variance([1, 2, 3], 0)).toBeCloseTo(2 / 3, 10);
      // Sample variance (ddof=1)
      expect(variance([1, 2, 3], 1)).toBeCloseTo(1, 10);
    });
  });

  describe('stdDev', () => {
    it('should be sqrt of variance', () => {
      const values = [1, 2, 3, 4, 5];
      expect(stdDev(values)).toBeCloseTo(Math.sqrt(variance(values)), 10);
    });
  });

  describe('covariance', () => {
    it('should return 0 for unequal lengths', () => {
      expect(covariance([1, 2], [1])).toBe(0);
    });

    it('should compute covariance correctly', () => {
      // Perfect positive correlation
      const x = [1, 2, 3];
      const y = [2, 4, 6];
      expect(covariance(x, y)).toBeGreaterThan(0);
    });

    it('should return 0 for uncorrelated data', () => {
      const x = [1, 2, 3, 4];
      const y = [1, -1, 1, -1];
      // This is approximately uncorrelated
      const cov = covariance(x, y);
      expect(Math.abs(cov)).toBeLessThan(2);
    });
  });
});

// ============================================================================
// Propensity Score Estimation
// ============================================================================

describe('Propensity Score Estimation', () => {
  describe('estimatePropensityScores', () => {
    it('should return empty results for empty input', () => {
      const result = estimatePropensityScores([], []);
      expect(result.coefficients).toEqual([]);
      expect(result.propensities).toEqual([]);
    });

    it('should fit logistic regression', () => {
      // Create simple linearly separable data
      const features = [
        [1, 0],
        [2, 0],
        [3, 0],
        [4, 0],
        [5, 0]
      ];
      const treatments = [0, 0, 1, 1, 1];

      const result = estimatePropensityScores(features, treatments);

      expect(result.coefficients.length).toBeGreaterThan(0);
      expect(result.propensities.length).toBe(5);
    });

    it('should return propensities in valid range', () => {
      const features = Array.from({ length: 20 }, () => [Math.random(), Math.random()]);
      const treatments = Array.from({ length: 20 }, () => (Math.random() > 0.5 ? 1 : 0));

      const result = estimatePropensityScores(features, treatments);

      for (const p of result.propensities) {
        expect(p).toBeGreaterThanOrEqual(0.01);
        expect(p).toBeLessThanOrEqual(0.99);
      }
    });

    it('should converge for reasonable data', () => {
      const features = Array.from({ length: 100 }, () => [
        Math.random() * 2 - 1,
        Math.random() * 2 - 1
      ]);
      const treatments = features.map((f) => (f[0] > 0 ? 1 : 0));

      const result = estimatePropensityScores(features, treatments, {
        maxIterations: 500
      });

      expect(result.converged).toBe(true);
    });
  });

  describe('predictPropensity', () => {
    it('should predict propensities for new data', () => {
      const features = [
        [-1, 0],
        [1, 0]
      ];
      const treatments = [0, 1];

      const { coefficients } = estimatePropensityScores(features, treatments);
      const newFeatures = [[0, 0]];

      const predictions = predictPropensity(newFeatures, coefficients);

      expect(predictions.length).toBe(1);
      expect(predictions[0]).toBeGreaterThanOrEqual(0.01);
      expect(predictions[0]).toBeLessThanOrEqual(0.99);
    });
  });
});

// ============================================================================
// AIPW Estimator
// ============================================================================

describe('AIPW Estimator', () => {
  describe('computeAIPWEstimate', () => {
    it('should handle empty input', () => {
      const result = computeAIPWEstimate([], [], [], [], []);

      expect(result.ate.estimate).toBe(0);
      expect(result.ate.numObservations).toBe(0);
    });

    it('should compute ATE with known effect', () => {
      // Simple case: treatment adds 1 to outcome
      const outcomes = [0, 1, 0, 1, 1, 2, 1, 2];
      const treatments = [0, 1, 0, 1, 0, 1, 0, 1];
      const propensities = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
      const muA = [0, 0, 0, 0, 1, 1, 1, 1];
      const muB = [1, 1, 1, 1, 2, 2, 2, 2];

      const result = computeAIPWEstimate(outcomes, treatments, propensities, muA, muB);

      expect(result.ate.method).toBe('aipw');
      expect(result.ate.numObservations).toBe(8);
      // Estimate should be around 1 (the true effect)
      expect(Math.abs(result.ate.estimate - 1)).toBeLessThan(0.5);
    });

    it('should compute standard error', () => {
      const n = 50;
      const outcomes = Array.from({ length: n }, () => Math.random());
      const treatments = Array.from({ length: n }, () => (Math.random() > 0.5 ? 1 : 0));
      const propensities = Array.from({ length: n }, () => 0.5);
      const muA = Array.from({ length: n }, () => 0);
      const muB = Array.from({ length: n }, () => 0);

      const result = computeAIPWEstimate(outcomes, treatments, propensities, muA, muB);

      expect(result.ate.standardError).toBeGreaterThan(0);
    });

    it('should return influence function components', () => {
      const outcomes = [0, 1, 0, 1];
      const treatments = [0, 1, 0, 1];
      const propensities = [0.5, 0.5, 0.5, 0.5];
      const muA = [0, 0, 0, 0];
      const muB = [1, 1, 1, 1];

      const result = computeAIPWEstimate(outcomes, treatments, propensities, muA, muB);

      expect(result.components.length).toBe(4);
      expect(result.components[0]).toHaveProperty('ipwTerm');
      expect(result.components[0]).toHaveProperty('augmentationTerm');
      expect(result.components[0]).toHaveProperty('influenceFunction');
    });
  });

  describe('computeIPWEstimate', () => {
    it('should compute IPW estimate', () => {
      const outcomes = [0, 1, 0, 1];
      const treatments = [0, 1, 0, 1];
      const propensities = [0.5, 0.5, 0.5, 0.5];

      const result = computeIPWEstimate(outcomes, treatments, propensities);

      expect(result.method).toBe('ipw');
      expect(typeof result.estimate).toBe('number');
    });
  });
});

// ============================================================================
// Outcome Model
// ============================================================================

describe('Outcome Model', () => {
  describe('fitOutcomeModel', () => {
    it('should fit linear regression', () => {
      const outcomes = [1, 2, 3, 4, 5];
      const treatments = [0, 0, 1, 1, 1];
      const features = [
        [0],
        [1],
        [2],
        [3],
        [4]
      ];

      const model = fitOutcomeModel(outcomes, treatments, features);

      expect(typeof model.intercept).toBe('number');
      expect(typeof model.treatmentEffect).toBe('number');
      expect(model.featureCoefficients.length).toBe(1);
    });

    it('should estimate treatment effect', () => {
      // Create data where treatment adds 2
      const n = 100;
      const features: number[][] = [];
      const treatments: number[] = [];
      const outcomes: number[] = [];

      for (let i = 0; i < n; i++) {
        const x = Math.random();
        const t = Math.random() > 0.5 ? 1 : 0;
        const y = x + 2 * t + Math.random() * 0.1;

        features.push([x]);
        treatments.push(t);
        outcomes.push(y);
      }

      const model = fitOutcomeModel(outcomes, treatments, features);

      // Treatment effect should be close to 2
      expect(Math.abs(model.treatmentEffect - 2)).toBeLessThan(0.5);
    });
  });

  describe('predictOutcome', () => {
    it('should predict outcomes', () => {
      const model = {
        intercept: 1,
        treatmentEffect: 2,
        featureCoefficients: [0.5]
      };

      const features = [[1], [2]];
      const predictions = predictOutcome(features, 1, model);

      // y = 1 + 2*1 + 0.5*x
      expect(predictions[0]).toBeCloseTo(3.5, 10); // 1 + 2 + 0.5
      expect(predictions[1]).toBeCloseTo(4, 10); // 1 + 2 + 1
    });
  });
});

// ============================================================================
// Quantization
// ============================================================================

describe('Quantization', () => {
  describe('createUniformQuantization', () => {
    it('should create uniform bins', () => {
      const params = createUniformQuantization(5, [-2, 2], false);

      expect(params.numBins).toBe(5);
      expect(params.boundaries.length).toBe(4);
      expect(params.method).toBe('uniform');
    });

    it('should create symmetric bins when requested', () => {
      const params = createUniformQuantization(5, [-2, 2], true);

      expect(params.symmetric).toBe(true);
      // Check symmetry around 0
      expect(params.boundaries[0]).toBeCloseTo(-params.boundaries[params.boundaries.length - 1], 5);
    });
  });

  describe('createEntropyQuantization', () => {
    it('should create quantile-based bins', () => {
      const values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
      const params = createEntropyQuantization(values, 5);

      expect(params.numBins).toBe(5);
      expect(params.boundaries.length).toBe(4);
      expect(params.method).toBe('entropy');
    });

    it('should handle empty values', () => {
      const params = createEntropyQuantization([], 5);

      expect(params.numBins).toBe(5);
    });
  });

  describe('quantize', () => {
    it('should quantize values to correct bins', () => {
      const params = {
        numBins: 3,
        boundaries: [0, 1],
        method: 'uniform' as const,
        symmetric: false
      };

      expect(quantize(-1, params)).toBe(0);
      expect(quantize(0.5, params)).toBe(1);
      expect(quantize(2, params)).toBe(2);
    });

    it('should handle boundary values', () => {
      const params = {
        numBins: 3,
        boundaries: [0, 1],
        method: 'uniform' as const,
        symmetric: false
      };

      expect(quantize(0, params)).toBe(0);
      expect(quantize(1, params)).toBe(1);
    });
  });

  describe('batchQuantize', () => {
    it('should quantize multiple values', () => {
      const params = {
        numBins: 3,
        boundaries: [0, 1],
        method: 'uniform' as const,
        symmetric: false
      };

      const values = [-1, 0.5, 2];
      const bins = batchQuantize(values, params);

      expect(bins).toEqual([0, 1, 2]);
    });
  });

  describe('learnDequantizationMappings', () => {
    it('should learn mean for each bin', () => {
      const continuous = [0.1, 0.2, 0.9, 0.8, 1.5, 1.6];
      const quantized = [0, 0, 1, 1, 2, 2];

      const mappings = learnDequantizationMappings(continuous, quantized, 3);

      expect(mappings.length).toBe(3);
      expect(mappings[0].expectedValue).toBeCloseTo(0.15, 5);
      expect(mappings[1].expectedValue).toBeCloseTo(0.85, 5);
      expect(mappings[2].expectedValue).toBeCloseTo(1.55, 5);
    });

    it('should compute variance for each bin', () => {
      const continuous = [0, 0.2, 1, 1.2];
      const quantized = [0, 0, 1, 1];

      const mappings = learnDequantizationMappings(continuous, quantized, 2);

      expect(mappings[0].variance).toBeGreaterThan(0);
      expect(mappings[0].sampleCount).toBe(2);
    });
  });

  describe('dequantize', () => {
    it('should return expected value for bin', () => {
      const mappings = [
        { binIndex: 0, expectedValue: 0.5, variance: 0.1, sampleCount: 10 },
        { binIndex: 1, expectedValue: 1.5, variance: 0.1, sampleCount: 10 }
      ];

      expect(dequantize(0, mappings)).toBe(0.5);
      expect(dequantize(1, mappings)).toBe(1.5);
    });

    it('should return 0 for missing bin', () => {
      const mappings = [{ binIndex: 0, expectedValue: 0.5, variance: 0.1, sampleCount: 10 }];

      expect(dequantize(5, mappings)).toBe(0);
    });
  });

  describe('adaptQuantization', () => {
    it('should adapt boundaries based on observations', () => {
      const params = {
        timeStep: 0,
        numBins: 3,
        boundaries: [0, 1],
        method: 'uniform' as const,
        symmetric: false
      };

      const observations = Array.from({ length: 100 }, () => ({
        continuous: Math.random() * 2 - 0.5,
        quantized: Math.floor(Math.random() * 3)
      }));

      const adapted = adaptQuantization(params, observations, 0.1);

      expect(adapted.boundaries.length).toBe(2);
      expect(adapted.method).toBe('adaptive');
    });

    it('should not adapt with too few observations', () => {
      const params = {
        timeStep: 0,
        numBins: 3,
        boundaries: [0, 1],
        method: 'uniform' as const,
        symmetric: false
      };

      const observations = [{ continuous: 0.5, quantized: 1 }];

      const adapted = adaptQuantization(params, observations, 0.1);

      expect(adapted.boundaries).toEqual(params.boundaries);
    });
  });
});

// ============================================================================
// Statistical Testing
// ============================================================================

describe('Statistical Testing', () => {
  describe('testATESignificance', () => {
    it('should reject for large effect', () => {
      const ate = {
        estimate: 2,
        standardError: 0.1,
        confidenceInterval: [1.8, 2.2] as [number, number],
        numObservations: 100,
        method: 'aipw' as const
      };

      const result = testATESignificance(ate, 0.05);

      expect(result.reject).toBe(true);
      expect(result.pValue).toBeLessThan(0.05);
    });

    it('should not reject for small effect', () => {
      const ate = {
        estimate: 0.1,
        standardError: 1,
        confidenceInterval: [-1.9, 2.1] as [number, number],
        numObservations: 100,
        method: 'aipw' as const
      };

      const result = testATESignificance(ate, 0.05);

      expect(result.reject).toBe(false);
      expect(result.pValue).toBeGreaterThan(0.05);
    });
  });

  describe('bootstrapATEConfidenceInterval', () => {
    it('should compute confidence interval', () => {
      const n = 50;
      const outcomes = Array.from({ length: n }, () => Math.random());
      const treatments = Array.from({ length: n }, () => (Math.random() > 0.5 ? 1 : 0));
      const propensities = Array.from({ length: n }, () => 0.5);
      const muA = Array.from({ length: n }, () => 0);
      const muB = Array.from({ length: n }, () => 0);

      const ci = bootstrapATEConfidenceInterval(
        outcomes,
        treatments,
        propensities,
        muA,
        muB,
        100,
        0.95
      );

      expect(ci.length).toBe(2);
      expect(ci[0]).toBeLessThan(ci[1]);
    });
  });

  describe('estimatePower', () => {
    it('should estimate high power for large effect', () => {
      const power = estimatePower(2, 0.1, 0.05);

      expect(power).toBeGreaterThan(0.9);
    });

    it('should estimate low power for small effect', () => {
      const power = estimatePower(0.1, 1, 0.05);

      expect(power).toBeLessThan(0.2);
    });
  });

  describe('minimumDetectableEffect', () => {
    it('should compute MDE', () => {
      const mde = minimumDetectableEffect(0.8, 0.5, 0.05);

      expect(mde).toBeGreaterThan(0);
    });

    it('should increase with higher power requirement', () => {
      const mde80 = minimumDetectableEffect(0.8, 0.5, 0.05);
      const mde90 = minimumDetectableEffect(0.9, 0.5, 0.05);

      expect(mde90).toBeGreaterThan(mde80);
    });
  });
});

// ============================================================================
// Random Number Generation
// ============================================================================

describe('Random Number Generation', () => {
  describe('createSeededRandom', () => {
    it('should be reproducible', () => {
      const rng1 = createSeededRandom(12345);
      const rng2 = createSeededRandom(12345);

      const values1 = Array.from({ length: 10 }, () => rng1());
      const values2 = Array.from({ length: 10 }, () => rng2());

      expect(values1).toEqual(values2);
    });

    it('should produce different sequences for different seeds', () => {
      const rng1 = createSeededRandom(12345);
      const rng2 = createSeededRandom(54321);

      expect(rng1()).not.toEqual(rng2());
    });

    it('should produce values in [0, 1)', () => {
      const rng = createSeededRandom(42);

      for (let i = 0; i < 100; i++) {
        const val = rng();
        expect(val).toBeGreaterThanOrEqual(0);
        expect(val).toBeLessThan(1);
      }
    });
  });

  describe('randomNormal', () => {
    it('should generate normal samples', () => {
      const samples = randomNormal(0, 1, 1000);

      const m = mean(samples);
      const s = stdDev(samples);

      expect(Math.abs(m)).toBeLessThan(0.2);
      expect(Math.abs(s - 1)).toBeLessThan(0.2);
    });

    it('should respect mean and std parameters', () => {
      const samples = randomNormal(5, 2, 1000);

      const m = mean(samples);
      const s = stdDev(samples);

      expect(Math.abs(m - 5)).toBeLessThan(0.3);
      expect(Math.abs(s - 2)).toBeLessThan(0.3);
    });
  });

  describe('randomMultivariateNormal', () => {
    it('should generate multivariate normal samples', () => {
      const meanVec = [1, 2];
      const cov = [
        [1, 0.5],
        [0.5, 1]
      ];

      const samples = randomMultivariateNormal(meanVec, cov, 100);

      expect(samples.length).toBe(100);
      expect(samples[0].length).toBe(2);
    });

    it('should respect mean vector', () => {
      const meanVec = [5, 10];
      const cov = [
        [1, 0],
        [0, 1]
      ];

      const samples = randomMultivariateNormal(meanVec, cov, 1000);

      const dim0Mean = mean(samples.map((s) => s[0]));
      const dim1Mean = mean(samples.map((s) => s[1]));

      expect(Math.abs(dim0Mean - 5)).toBeLessThan(0.3);
      expect(Math.abs(dim1Mean - 10)).toBeLessThan(0.3);
    });
  });
});

// ============================================================================
// Matrix Operations
// ============================================================================

describe('Matrix Operations', () => {
  describe('dotProduct', () => {
    it('should compute dot product', () => {
      expect(dotProduct([1, 2, 3], [4, 5, 6])).toBe(32);
    });
  });

  describe('transpose', () => {
    it('should transpose matrix', () => {
      const A = [
        [1, 2],
        [3, 4],
        [5, 6]
      ];

      const AT = transpose(A);

      expect(AT).toEqual([
        [1, 3, 5],
        [2, 4, 6]
      ]);
    });
  });

  describe('matrixVectorMultiply', () => {
    it('should multiply matrix by vector', () => {
      const A = [
        [1, 2],
        [3, 4]
      ];
      const v = [1, 2];

      const result = matrixVectorMultiply(A, v);

      expect(result).toEqual([5, 11]);
    });
  });

  describe('matrixMultiply', () => {
    it('should multiply matrices', () => {
      const A = [
        [1, 2],
        [3, 4]
      ];
      const B = [
        [5, 6],
        [7, 8]
      ];

      const result = matrixMultiply(A, B);

      expect(result).toEqual([
        [19, 22],
        [43, 50]
      ]);
    });
  });

  describe('invertMatrix', () => {
    it('should invert 2x2 matrix', () => {
      const A = [
        [4, 7],
        [2, 6]
      ];

      const Ainv = invertMatrix(A);

      expect(Ainv).not.toBeNull();

      // A * A^{-1} should be identity
      const product = matrixMultiply(A, Ainv!);

      expect(product[0][0]).toBeCloseTo(1, 10);
      expect(product[0][1]).toBeCloseTo(0, 10);
      expect(product[1][0]).toBeCloseTo(0, 10);
      expect(product[1][1]).toBeCloseTo(1, 10);
    });

    it('should return null for singular matrix', () => {
      const A = [
        [1, 2],
        [2, 4]
      ];

      const Ainv = invertMatrix(A);

      expect(Ainv).toBeNull();
    });
  });
});

// ============================================================================
// Distribution Functions
// ============================================================================

describe('Distribution Functions', () => {
  describe('normalCDF', () => {
    it('should return 0.5 for z=0', () => {
      expect(normalCDF(0)).toBeCloseTo(0.5, 5);
    });

    it('should return approximately 0.975 for z=1.96', () => {
      expect(normalCDF(1.96)).toBeCloseTo(0.975, 2);
    });

    it('should be monotonically increasing', () => {
      const values = [-3, -1, 0, 1, 3];
      for (let i = 1; i < values.length; i++) {
        expect(normalCDF(values[i])).toBeGreaterThan(normalCDF(values[i - 1]));
      }
    });
  });

  describe('normalQuantile', () => {
    it('should return 0 for p=0.5', () => {
      expect(normalQuantile(0.5)).toBeCloseTo(0, 5);
    });

    it('should return approximately 1.96 for p=0.975', () => {
      expect(normalQuantile(0.975)).toBeCloseTo(1.96, 1);
    });

    it('should be inverse of normalCDF', () => {
      const zValues = [-2, -1, 0, 1, 2];
      for (const z of zValues) {
        const p = normalCDF(z);
        const zBack = normalQuantile(p);
        expect(zBack).toBeCloseTo(z, 3);
      }
    });
  });
});
