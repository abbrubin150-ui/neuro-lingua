import { describe, expect, it } from 'vitest';
import {
  heInit,
  xavierInit,
  lecunInit,
  leakyRelu,
  leakyReluDerivative,
  elu,
  eluDerivative,
  gelu,
  swish,
  cosineAnnealingLR,
  exponentialDecayLR,
  stepDecayLR,
  warmupCosineAnnealingLR,
  logSumExp,
  stableSoftmax,
  logSoftmax,
  clip,
  safeDivide,
  layerNorm,
  layerNormBackward,
  getTopKIndices,
  nucleusSampling
} from '../src/lib/MathUtils';

describe('Weight Initialization', () => {
  it('He initialization scales correctly with fan-in', () => {
    const samples: number[] = [];
    const fanIn = 100;

    // Generate multiple samples
    for (let i = 0; i < 1000; i++) {
      samples.push(heInit(fanIn, Math.random));
    }

    // Check variance is approximately 2/fan_in
    const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
    const variance = samples.reduce((a, b) => a + (b - mean) ** 2, 0) / samples.length;

    const expectedVariance = 2.0 / fanIn;

    // Allow 20% tolerance
    expect(Math.abs(variance - expectedVariance) / expectedVariance).toBeLessThan(0.2);
  });

  it('Xavier initialization scales correctly with fan-in and fan-out', () => {
    const samples: number[] = [];
    const fanIn = 50;
    const fanOut = 100;

    for (let i = 0; i < 1000; i++) {
      samples.push(xavierInit(fanIn, fanOut, Math.random));
    }

    const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
    const variance = samples.reduce((a, b) => a + (b - mean) ** 2, 0) / samples.length;

    const expectedVariance = 2.0 / (fanIn + fanOut);

    expect(Math.abs(variance - expectedVariance) / expectedVariance).toBeLessThan(0.2);
  });

  it('LeCun initialization scales correctly', () => {
    const samples: number[] = [];
    const fanIn = 100;

    for (let i = 0; i < 1000; i++) {
      samples.push(lecunInit(fanIn, Math.random));
    }

    const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
    const variance = samples.reduce((a, b) => a + (b - mean) ** 2, 0) / samples.length;

    const expectedVariance = 1.0 / fanIn;

    expect(Math.abs(variance - expectedVariance) / expectedVariance).toBeLessThan(0.2);
  });
});

describe('Activation Functions', () => {
  describe('Leaky ReLU', () => {
    it('returns x for positive values', () => {
      expect(leakyRelu(5)).toBe(5);
      expect(leakyRelu(0.1)).toBe(0.1);
    });

    it('returns alpha * x for negative values', () => {
      const alpha = 0.01;
      expect(leakyRelu(-1, alpha)).toBe(-0.01);
      expect(leakyRelu(-10, alpha)).toBe(-0.1);
    });

    it('derivative is correct', () => {
      const alpha = 0.01;
      expect(leakyReluDerivative(5, alpha)).toBe(1);
      expect(leakyReluDerivative(-1, alpha)).toBe(alpha);
    });
  });

  describe('ELU', () => {
    it('returns x for positive values', () => {
      expect(elu(5)).toBe(5);
      expect(elu(0.1)).toBe(0.1);
    });

    it('returns alpha * (e^x - 1) for negative values', () => {
      const alpha = 1.0;
      const result = elu(-1, alpha);
      const expected = alpha * (Math.exp(-1) - 1);
      expect(Math.abs(result - expected)).toBeLessThan(1e-10);
    });

    it('derivative is correct', () => {
      expect(eluDerivative(5, 1.0)).toBe(1);

      const x = -1;
      const alpha = 1.0;
      const result = eluDerivative(x, alpha);
      const expected = alpha * Math.exp(x);
      expect(Math.abs(result - expected)).toBeLessThan(1e-10);
    });

    it('is continuous at zero', () => {
      const left = elu(-0.001, 1.0);
      const right = elu(0.001, 1.0);
      expect(Math.abs(left - right)).toBeLessThan(0.01);
    });
  });

  describe('GELU', () => {
    it('is approximately zero at zero', () => {
      expect(Math.abs(gelu(0))).toBeLessThan(0.01);
    });

    it('is approximately linear for large positive values', () => {
      const x = 10;
      const result = gelu(x);
      expect(Math.abs(result - x) / x).toBeLessThan(0.01);
    });

    it('is smooth and continuous', () => {
      const x1 = 1.0;
      const x2 = 1.001;
      const diff = Math.abs(gelu(x2) - gelu(x1));
      expect(diff).toBeLessThan(0.01);
    });
  });

  describe('Swish', () => {
    it('is approximately zero at large negative values', () => {
      expect(Math.abs(swish(-10))).toBeLessThan(0.01);
    });

    it('is approximately linear for large positive values', () => {
      const x = 10;
      const result = swish(x);
      expect(Math.abs(result - x) / x).toBeLessThan(0.05);
    });
  });
});

describe('Learning Rate Scheduling', () => {
  describe('Cosine Annealing', () => {
    it('starts at maximum learning rate', () => {
      const lr = cosineAnnealingLR(0, 100, 0.1, 0);
      expect(Math.abs(lr - 0.1)).toBeLessThan(1e-6);
    });

    it('ends at minimum learning rate', () => {
      const lr = cosineAnnealingLR(100, 100, 0.1, 0.001);
      expect(Math.abs(lr - 0.001)).toBeLessThan(1e-6);
    });

    it('is at midpoint halfway through', () => {
      const lrMax = 0.1;
      const lrMin = 0.001;
      const lr = cosineAnnealingLR(50, 100, lrMax, lrMin);
      const expected = (lrMax + lrMin) / 2;
      expect(Math.abs(lr - expected)).toBeLessThan(0.001);
    });

    it('is monotonically decreasing', () => {
      const lrs = [];
      for (let i = 0; i <= 100; i++) {
        lrs.push(cosineAnnealingLR(i, 100, 0.1, 0.001));
      }

      for (let i = 1; i < lrs.length; i++) {
        expect(lrs[i]).toBeLessThanOrEqual(lrs[i - 1]);
      }
    });
  });

  describe('Exponential Decay', () => {
    it('starts at initial learning rate', () => {
      const lr = exponentialDecayLR(0, 0.1, 0.9);
      expect(lr).toBe(0.1);
    });

    it('decays exponentially', () => {
      const lrInitial = 0.1;
      const decayRate = 0.9;
      const epoch = 10;
      const lr = exponentialDecayLR(epoch, lrInitial, decayRate);
      const expected = lrInitial * Math.pow(decayRate, epoch);
      expect(Math.abs(lr - expected)).toBeLessThan(1e-10);
    });
  });

  describe('Step Decay', () => {
    it('stays constant within a step', () => {
      const lr1 = stepDecayLR(5, 0.1, 0.5, 10);
      const lr2 = stepDecayLR(8, 0.1, 0.5, 10);
      expect(lr1).toBe(lr2);
    });

    it('drops at step boundaries', () => {
      const lr1 = stepDecayLR(9, 0.1, 0.5, 10);
      const lr2 = stepDecayLR(10, 0.1, 0.5, 10);
      expect(lr2).toBe(lr1 * 0.5);
    });
  });

  describe('Warmup + Cosine Annealing', () => {
    it('linear warmup for first few epochs', () => {
      const warmup = 10;
      const lr5 = warmupCosineAnnealingLR(5, 100, 0.1, warmup);
      const lr10 = warmupCosineAnnealingLR(10, 100, 0.1, warmup);

      expect(lr5).toBeLessThan(0.1);
      expect(lr10).toBeCloseTo(0.1, 6);
    });

    it('cosine annealing after warmup', () => {
      const warmup = 10;
      const lr15 = warmupCosineAnnealingLR(15, 100, 0.1, warmup);
      const lr100 = warmupCosineAnnealingLR(100, 100, 0.1, warmup);

      expect(lr15).toBeLessThan(0.1);
      expect(lr100).toBeLessThan(lr15);
    });
  });
});

describe('Numerical Stability', () => {
  describe('logSumExp', () => {
    it('handles large positive values without overflow', () => {
      const values = [1000, 1001, 1002];
      const result = logSumExp(values);
      expect(isFinite(result)).toBe(true);
      expect(result).toBeGreaterThan(1002);
    });

    it('handles large negative values without underflow', () => {
      const values = [-1000, -1001, -1002];
      const result = logSumExp(values);
      expect(isFinite(result)).toBe(true);
      // Result should be close to max value (log-sum-exp of very negative values ≈ max)
      // But with 3 terms, we get log(1 + e^-1 + e^-2) ≈ 0.407 added to -1000
      expect(result).toBeGreaterThan(-1001);
      expect(result).toBeLessThan(-999);
    });

    it('equals max for well-separated values', () => {
      const values = [10, -100, -200];
      const result = logSumExp(values);
      expect(result).toBeCloseTo(10, 5);
    });
  });

  describe('stableSoftmax', () => {
    it('outputs sum to 1', () => {
      const logits = [2.0, 1.0, 0.1];
      const probs = stableSoftmax(logits);
      const sum = probs.reduce((a, b) => a + b, 0);
      expect(Math.abs(sum - 1.0)).toBeLessThan(1e-6);
    });

    it('handles large values without overflow', () => {
      const logits = [1000, 1001, 1002];
      const probs = stableSoftmax(logits);
      expect(probs.every((p) => isFinite(p))).toBe(true);
      const sum = probs.reduce((a, b) => a + b, 0);
      expect(Math.abs(sum - 1.0)).toBeLessThan(1e-6);
    });

    it('temperature increases entropy', () => {
      const logits = [3.0, 2.0, 1.0];
      const probs1 = stableSoftmax(logits, 0.1);
      const probs2 = stableSoftmax(logits, 10.0);

      // High temperature should be more uniform
      const max1 = Math.max(...probs1);
      const max2 = Math.max(...probs2);
      expect(max2).toBeLessThan(max1);
    });
  });

  describe('logSoftmax', () => {
    it('matches log of softmax', () => {
      const logits = [2.0, 1.0, 0.1];
      const logProbs = logSoftmax(logits);
      const probs = stableSoftmax(logits);

      for (let i = 0; i < logits.length; i++) {
        expect(Math.abs(logProbs[i] - Math.log(probs[i]))).toBeLessThan(1e-10);
      }
    });
  });

  describe('clip', () => {
    it('clips to minimum', () => {
      expect(clip(-10, 0, 10)).toBe(0);
    });

    it('clips to maximum', () => {
      expect(clip(20, 0, 10)).toBe(10);
    });

    it('passes through values in range', () => {
      expect(clip(5, 0, 10)).toBe(5);
    });
  });

  describe('safeDivide', () => {
    it('handles zero denominator', () => {
      const result = safeDivide(10, 0);
      expect(isFinite(result)).toBe(true);
    });

    it('handles normal division', () => {
      const result = safeDivide(10, 2);
      expect(result).toBeCloseTo(5, 6);
    });
  });
});

describe('Layer Normalization', () => {
  it('normalizes to zero mean and unit variance', () => {
    const x = [1.0, 2.0, 3.0, 4.0, 5.0];
    const gamma = new Array(5).fill(1.0);
    const beta = new Array(5).fill(0.0);

    const normalized = layerNorm(x, gamma, beta);

    // Check mean is close to zero
    const mean = normalized.reduce((a, b) => a + b, 0) / normalized.length;
    expect(Math.abs(mean)).toBeLessThan(1e-6);

    // Check variance is close to 1
    const variance = normalized.reduce((a, b) => a + (b - mean) ** 2, 0) / normalized.length;
    expect(Math.abs(variance - 1.0)).toBeLessThan(1e-5);
  });

  it('applies gamma and beta correctly', () => {
    const x = [1.0, 2.0, 3.0];
    const gamma = [2.0, 2.0, 2.0];
    const beta = [1.0, 1.0, 1.0];

    const normalized = layerNorm(x, gamma, beta);

    // Mean should be shifted by beta
    const mean = normalized.reduce((a, b) => a + b, 0) / normalized.length;
    expect(Math.abs(mean - 1.0)).toBeLessThan(1e-5);
  });

  it('backward pass produces gradients', () => {
    const x = [1.0, 2.0, 3.0, 4.0];
    const gamma = new Array(4).fill(1.0);
    const dOut = [0.1, 0.2, 0.3, 0.4];

    const { dx, dGamma, dBeta } = layerNormBackward(dOut, x, gamma);

    expect(dx.length).toBe(x.length);
    expect(dGamma.length).toBe(gamma.length);
    expect(dBeta.length).toBe(dOut.length);

    // Gradients should be non-zero
    expect(dx.some((g) => Math.abs(g) > 1e-6)).toBe(true);
    expect(dGamma.some((g) => Math.abs(g) > 1e-6)).toBe(true);
    expect(dBeta.every((g, i) => Math.abs(g - dOut[i]) < 1e-10)).toBe(true);
  });
});

describe('Advanced Sampling', () => {
  describe('getTopKIndices', () => {
    it('returns indices of top k values', () => {
      const arr = [0.1, 0.5, 0.2, 0.9, 0.3];
      const topK = getTopKIndices(arr, 3);

      expect(topK.length).toBe(3);
      expect(topK[0]).toBe(3); // 0.9
      expect(topK[1]).toBe(1); // 0.5
      expect(topK[2]).toBe(4); // 0.3
    });

    it('handles k larger than array length', () => {
      const arr = [0.1, 0.2, 0.3];
      const topK = getTopKIndices(arr, 10);
      expect(topK.length).toBe(3);
    });
  });

  describe('nucleusSampling', () => {
    it('samples from nucleus', () => {
      const probs = [0.5, 0.3, 0.1, 0.05, 0.05];
      const p = 0.9;

      const samples = new Set<number>();
      for (let i = 0; i < 100; i++) {
        const sample = nucleusSampling(probs, p, Math.random);
        samples.add(sample);
      }

      // Should mostly sample from top 3 tokens (0.5 + 0.3 + 0.1 = 0.9)
      expect(samples.has(0)).toBe(true);
      expect(samples.has(1)).toBe(true);
    });

    it('always returns valid index', () => {
      const probs = [0.25, 0.25, 0.25, 0.25];
      for (let i = 0; i < 50; i++) {
        const sample = nucleusSampling(probs, 0.6, Math.random);
        expect(sample).toBeGreaterThanOrEqual(0);
        expect(sample).toBeLessThan(4);
      }
    });
  });
});
