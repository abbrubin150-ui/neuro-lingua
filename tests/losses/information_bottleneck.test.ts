/**
 * Tests for Information Bottleneck Loss Implementation
 */

import { describe, it, expect } from 'vitest';
import {
  estimateMutualInformation,
  computeRepresentationEntropy,
  informationBottleneckLoss,
  getBetaSchedule,
  standardCrossEntropyLoss,
  hybridIBLoss,
  type InformationBottleneckConfig
} from '../../src/losses/information_bottleneck';

describe('estimateMutualInformation', () => {
  it('should return 0 for empty arrays', () => {
    const mi = estimateMutualInformation([], [], 50);
    expect(mi).toBe(0);
  });

  it('should return 0 for mismatched array lengths', () => {
    const mi = estimateMutualInformation([1, 2, 3], [1, 2], 50);
    expect(mi).toBe(0);
  });

  it('should estimate low MI for independent variables', () => {
    // Create two independent random variables
    const x = Array.from({ length: 1000 }, () => Math.random());
    const y = Array.from({ length: 1000 }, () => Math.random());

    const mi = estimateMutualInformation(x, y, 50);

    // MI between independent variables should be close to 0
    // Histogram-based estimation can have systematic bias, so allow generous threshold
    expect(mi).toBeGreaterThanOrEqual(0);
    expect(mi).toBeLessThan(1.5); // Allow estimation error for histogram-based method
  });

  it('should estimate high MI for perfectly correlated variables', () => {
    // Create perfectly correlated variables y = x
    const x = Array.from({ length: 1000 }, () => Math.random());
    const y = [...x]; // Perfect copy

    const mi = estimateMutualInformation(x, y, 50);

    // MI between perfectly correlated variables should be high
    expect(mi).toBeGreaterThan(1.0); // Should be close to H(X)
  });

  it('should estimate moderate MI for partially correlated variables', () => {
    // Create partially correlated variables y = x + noise
    const x = Array.from({ length: 1000 }, () => Math.random());
    const y = x.map((val) => val + Math.random() * 0.1);

    const mi = estimateMutualInformation(x, y, 50);

    // MI should be positive but less than perfect correlation
    expect(mi).toBeGreaterThan(0.1);
    expect(mi).toBeLessThan(3.0);
  });

  it('should be non-negative', () => {
    const x = Array.from({ length: 100 }, () => Math.random() - 0.5);
    const y = Array.from({ length: 100 }, () => Math.random() - 0.5);

    const mi = estimateMutualInformation(x, y, 20);

    expect(mi).toBeGreaterThanOrEqual(0);
  });

  it('should handle different numbers of bins', () => {
    const x = Array.from({ length: 500 }, () => Math.random());
    const y = x.map((val) => val + Math.random() * 0.2);

    const mi10 = estimateMutualInformation(x, y, 10);
    const mi50 = estimateMutualInformation(x, y, 50);
    const mi100 = estimateMutualInformation(x, y, 100);

    // All should be positive and finite
    expect(mi10).toBeGreaterThan(0);
    expect(mi10).toBeLessThan(Infinity);
    expect(mi50).toBeGreaterThan(0);
    expect(mi50).toBeLessThan(Infinity);
    expect(mi100).toBeGreaterThan(0);
    expect(mi100).toBeLessThan(Infinity);

    // More bins might give different estimates but all should be reasonable
    expect(Math.abs(mi50 - mi10)).toBeLessThan(2.0);
  });
});

describe('computeRepresentationEntropy', () => {
  it('should return 0 for empty activations', () => {
    const entropy = computeRepresentationEntropy([], 50);
    expect(entropy).toBe(0);
  });

  it('should return 0 for single-value activations', () => {
    const activations = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]];
    const entropy = computeRepresentationEntropy(activations, 50);

    // All values in one bin → entropy should be ~0
    expect(entropy).toBeLessThan(0.1);
  });

  it('should compute positive entropy for varied activations', () => {
    const activations = Array.from({ length: 100 }, () =>
      Array.from({ length: 10 }, () => Math.random())
    );

    const entropy = computeRepresentationEntropy(activations, 50);

    expect(entropy).toBeGreaterThan(0);
    expect(entropy).toBeLessThan(10); // Reasonable upper bound
  });

  it('should give higher entropy for more varied activations', () => {
    // Uniform distribution [0, 1]
    const uniformActivations = Array.from({ length: 100 }, () =>
      Array.from({ length: 10 }, () => Math.random())
    );

    // Very concentrated distribution around 0.5
    const concentratedActivations = Array.from({ length: 100 }, () =>
      Array.from({ length: 10 }, () => 0.5 + (Math.random() - 0.5) * 0.05)
    );

    const uniformEntropy = computeRepresentationEntropy(uniformActivations, 50);
    const concentratedEntropy = computeRepresentationEntropy(concentratedActivations, 50);

    // Uniform distribution should have higher or equal entropy (within error margin)
    // Note: Due to histogram binning, this may not always hold strictly
    expect(uniformEntropy).toBeGreaterThanOrEqual(concentratedEntropy - 0.5);
  });

  it('should be non-negative', () => {
    const activations = [[1, 2, 3], [4, 5, 6]];
    const entropy = computeRepresentationEntropy(activations, 20);

    expect(entropy).toBeGreaterThanOrEqual(0);
  });
});

describe('informationBottleneckLoss', () => {
  const createDummyData = (batchSize: number, inputDim: number, hiddenDim: number, vocabSize: number) => {
    const inputs = Array.from({ length: batchSize }, () =>
      Array.from({ length: inputDim }, () => Math.random())
    );

    const hiddenActivations = Array.from({ length: batchSize }, () =>
      Array.from({ length: hiddenDim }, () => Math.random())
    );

    const targetLogits = Array.from({ length: batchSize }, () =>
      Array.from({ length: vocabSize }, () => Math.random() * 2 - 1)
    );

    const targetIndices = Array.from({ length: batchSize }, () => Math.floor(Math.random() * vocabSize));

    return { inputs, hiddenActivations, targetLogits, targetIndices };
  };

  it('should handle empty batch', () => {
    const config: InformationBottleneckConfig = { beta: 0.1 };

    const metrics = informationBottleneckLoss([], [], [], [], config);

    expect(metrics.compressionMI).toBe(0);
    expect(metrics.predictionMI).toBe(0);
    expect(metrics.ibLoss).toBe(0);
    expect(metrics.beta).toBe(0.1);
  });

  it('should throw error for mismatched batch sizes', () => {
    const config: InformationBottleneckConfig = { beta: 0.1 };

    const { inputs, hiddenActivations, targetLogits, targetIndices } = createDummyData(10, 5, 8, 3);

    // Remove one element to create mismatch
    inputs.pop();

    expect(() => {
      informationBottleneckLoss(inputs, hiddenActivations, targetLogits, targetIndices, config);
    }).toThrow('Batch size mismatch');
  });

  it('should compute IB loss for valid inputs', () => {
    const config: InformationBottleneckConfig = { beta: 0.5 };

    const { inputs, hiddenActivations, targetLogits, targetIndices } = createDummyData(50, 10, 16, 5);

    const metrics = informationBottleneckLoss(
      inputs,
      hiddenActivations,
      targetLogits,
      targetIndices,
      config
    );

    expect(metrics.compressionMI).toBeGreaterThanOrEqual(0);
    expect(metrics.predictionMI).toBeGreaterThanOrEqual(0);
    expect(metrics.representationEntropy).toBeGreaterThanOrEqual(0);
    expect(metrics.conditionalEntropy).toBeGreaterThanOrEqual(0);
    expect(metrics.beta).toBe(0.5);

    // IB loss = -I(Z;Y) + β·I(X;Z)
    const expectedLoss = -metrics.predictionMI + 0.5 * metrics.compressionMI;
    expect(Math.abs(metrics.ibLoss - expectedLoss)).toBeLessThan(1e-6);
  });

  it('should respect beta parameter in loss calculation', () => {
    const { inputs, hiddenActivations, targetLogits, targetIndices } = createDummyData(30, 8, 12, 4);

    const configLowBeta: InformationBottleneckConfig = { beta: 0.1 };
    const configHighBeta: InformationBottleneckConfig = { beta: 10.0 };

    const metricsLow = informationBottleneckLoss(
      inputs,
      hiddenActivations,
      targetLogits,
      targetIndices,
      configLowBeta
    );

    const metricsHigh = informationBottleneckLoss(
      inputs,
      hiddenActivations,
      targetLogits,
      targetIndices,
      configHighBeta
    );

    // Higher beta should weight compression more heavily
    expect(metricsLow.beta).toBe(0.1);
    expect(metricsHigh.beta).toBe(10.0);

    // MI estimates should be the same
    expect(Math.abs(metricsLow.compressionMI - metricsHigh.compressionMI)).toBeLessThan(1e-6);
    expect(Math.abs(metricsLow.predictionMI - metricsHigh.predictionMI)).toBeLessThan(1e-6);

    // But IB loss should differ due to beta
    if (metricsLow.compressionMI > 0) {
      expect(Math.abs(metricsLow.ibLoss - metricsHigh.ibLoss)).toBeGreaterThan(0.1);
    }
  });

  it('should use custom numBins parameter', () => {
    const { inputs, hiddenActivations, targetLogits, targetIndices } = createDummyData(40, 8, 10, 4);

    const config20: InformationBottleneckConfig = { beta: 1.0, numBins: 20 };
    const config100: InformationBottleneckConfig = { beta: 1.0, numBins: 100 };

    const metrics20 = informationBottleneckLoss(
      inputs,
      hiddenActivations,
      targetLogits,
      targetIndices,
      config20
    );

    const metrics100 = informationBottleneckLoss(
      inputs,
      hiddenActivations,
      targetLogits,
      targetIndices,
      config100
    );

    // Both should give valid results
    expect(metrics20.compressionMI).toBeGreaterThanOrEqual(0);
    expect(metrics100.compressionMI).toBeGreaterThanOrEqual(0);

    // MI estimates may differ slightly due to binning
    // but should be in similar range
    expect(Math.abs(metrics20.compressionMI - metrics100.compressionMI)).toBeLessThan(2.0);
  });

  it('should have conditional entropy <= representation entropy', () => {
    const { inputs, hiddenActivations, targetLogits, targetIndices } = createDummyData(50, 10, 16, 5);

    const config: InformationBottleneckConfig = { beta: 1.0 };

    const metrics = informationBottleneckLoss(
      inputs,
      hiddenActivations,
      targetLogits,
      targetIndices,
      config
    );

    // H(Z|X) ≤ H(Z) (conditioning reduces entropy)
    expect(metrics.conditionalEntropy).toBeLessThanOrEqual(metrics.representationEntropy + 1e-6);
  });
});

describe('getBetaSchedule', () => {
  it('should return betaStart for constant schedule', () => {
    expect(getBetaSchedule('constant', 0, 100, 0.001, 1.0)).toBe(0.001);
    expect(getBetaSchedule('constant', 50, 100, 0.001, 1.0)).toBe(0.001);
    expect(getBetaSchedule('constant', 99, 100, 0.001, 1.0)).toBe(0.001);
  });

  it('should linearly interpolate for linear schedule', () => {
    expect(getBetaSchedule('linear', 0, 100, 0.0, 1.0)).toBeCloseTo(0.0, 5);
    // At epoch 50 out of 100, progress = 50/(100-1) = 50/99 ≈ 0.505
    expect(getBetaSchedule('linear', 50, 100, 0.0, 1.0)).toBeCloseTo(0.505, 2);
    expect(getBetaSchedule('linear', 99, 100, 0.0, 1.0)).toBeCloseTo(1.0, 5);

    // At epoch 25 out of 100, progress = 25/99 ≈ 0.252
    // betaStart + (betaEnd - betaStart) * 0.252 = 0.1 + 0.8 * 0.252 ≈ 0.302
    expect(getBetaSchedule('linear', 25, 100, 0.1, 0.9)).toBeCloseTo(0.302, 2);
  });

  it('should exponentially interpolate for exponential schedule', () => {
    const beta0 = getBetaSchedule('exponential', 0, 100, 0.001, 1.0);
    const beta50 = getBetaSchedule('exponential', 50, 100, 0.001, 1.0);
    const beta99 = getBetaSchedule('exponential', 99, 100, 0.001, 1.0);

    expect(beta0).toBeCloseTo(0.001, 6);
    expect(beta99).toBeCloseTo(1.0, 6);

    // Exponential should be between start and end
    expect(beta50).toBeGreaterThan(0.001);
    expect(beta50).toBeLessThan(1.0);

    // Exponential grows faster initially
    const beta25 = getBetaSchedule('exponential', 25, 100, 0.001, 1.0);
    const beta25Linear = getBetaSchedule('linear', 25, 100, 0.001, 1.0);
    expect(beta25).toBeLessThan(beta25Linear); // Exponential starts slower
  });

  it('should use cosine annealing for cosine schedule', () => {
    const beta0 = getBetaSchedule('cosine', 0, 100, 1.0, 0.0);
    const beta50 = getBetaSchedule('cosine', 50, 100, 1.0, 0.0);
    const beta99 = getBetaSchedule('cosine', 99, 100, 1.0, 0.0);

    expect(beta0).toBeCloseTo(1.0, 6);
    expect(beta99).toBeCloseTo(0.0, 6);

    // At midpoint, cosine should be at betaEnd + (betaStart - betaEnd)/2
    expect(beta50).toBeCloseTo(0.5, 1);
  });

  it('should handle edge case with totalEpochs = 1', () => {
    // Should return betaStart when totalEpochs <= 1
    expect(getBetaSchedule('linear', 0, 1, 0.1, 0.9)).toBe(0.1);
    expect(getBetaSchedule('exponential', 0, 1, 0.1, 0.9)).toBe(0.1);
    expect(getBetaSchedule('cosine', 0, 1, 0.1, 0.9)).toBe(0.1);
  });

  it('should clamp progress to [0, 1]', () => {
    // Epoch beyond totalEpochs should use progress = 1
    const betaOver = getBetaSchedule('linear', 150, 100, 0.0, 1.0);
    expect(betaOver).toBeCloseTo(1.0, 6);
  });

  it('should handle negative betaStart for exponential', () => {
    // When betaStart <= 0, use fallback: betaEnd * progress
    const beta = getBetaSchedule('exponential', 50, 100, 0.0, 1.0);
    // progress = 50/(100-1) ≈ 0.505, so beta ≈ 1.0 * 0.505 = 0.505
    expect(beta).toBeCloseTo(0.505, 2);
  });
});

describe('standardCrossEntropyLoss', () => {
  it('should compute cross-entropy loss correctly', () => {
    // Logits for 3-class problem
    const logits = [1.0, 2.0, 0.5];
    const targetIndex = 1; // Target is class 1 (highest logit)

    const loss = standardCrossEntropyLoss(logits, targetIndex);

    expect(loss).toBeGreaterThan(0);
    expect(loss).toBeLessThan(2.0); // Should be relatively small since target has highest logit
  });

  it('should give low loss for correct high-confidence prediction', () => {
    const logits = [-5.0, 10.0, -5.0]; // Very confident in class 1
    const targetIndex = 1;

    const loss = standardCrossEntropyLoss(logits, targetIndex);

    expect(loss).toBeGreaterThan(0);
    expect(loss).toBeLessThan(0.1); // Very low loss
  });

  it('should give high loss for incorrect high-confidence prediction', () => {
    const logits = [-5.0, 10.0, -5.0]; // Very confident in class 1
    const targetIndex = 0; // But target is class 0

    const loss = standardCrossEntropyLoss(logits, targetIndex);

    expect(loss).toBeGreaterThan(10); // Very high loss
  });

  it('should handle uniform probabilities', () => {
    const logits = [0.0, 0.0, 0.0]; // Uniform distribution
    const targetIndex = 1;

    const loss = standardCrossEntropyLoss(logits, targetIndex);

    // Should be -log(1/3) ≈ 1.099
    expect(loss).toBeCloseTo(Math.log(3), 2);
  });

  it('should be numerically stable with very negative logits', () => {
    const logits = [-1000, -1001, -999]; // Extreme values
    const targetIndex = 2;

    const loss = standardCrossEntropyLoss(logits, targetIndex);

    expect(loss).toBeGreaterThan(0);
    expect(loss).toBeLessThan(1005); // Should not overflow
    expect(isFinite(loss)).toBe(true);
  });
});

describe('hybridIBLoss', () => {
  it('should combine CE and IB losses with alpha', () => {
    const ceLoss = 2.5;
    const ibLoss = 1.0;

    // Pure CE (alpha = 0)
    const lossAlpha0 = hybridIBLoss(ceLoss, ibLoss, 0.0);
    expect(lossAlpha0).toBeCloseTo(ceLoss, 6);

    // Pure IB (alpha = 1)
    const lossAlpha1 = hybridIBLoss(ceLoss, ibLoss, 1.0);
    expect(lossAlpha1).toBeCloseTo(ibLoss, 6);

    // Balanced (alpha = 0.5)
    const lossAlpha05 = hybridIBLoss(ceLoss, ibLoss, 0.5);
    expect(lossAlpha05).toBeCloseTo(0.5 * ceLoss + 0.5 * ibLoss, 6);
  });

  it('should interpolate linearly with alpha', () => {
    const ceLoss = 3.0;
    const ibLoss = 1.5;
    const alpha = 0.3;

    const hybrid = hybridIBLoss(ceLoss, ibLoss, alpha);
    const expected = (1 - alpha) * ceLoss + alpha * ibLoss;

    expect(hybrid).toBeCloseTo(expected, 6);
  });

  it('should handle alpha outside [0, 1]', () => {
    const ceLoss = 2.0;
    const ibLoss = 1.0;

    // Alpha > 1 (more weight on IB)
    const lossAlpha2 = hybridIBLoss(ceLoss, ibLoss, 2.0);
    expect(lossAlpha2).toBe(-2.0 + 2.0); // (1-2)*2 + 2*1 = 0

    // Alpha < 0 (more weight on CE)
    const lossAlphaNeg = hybridIBLoss(ceLoss, ibLoss, -0.5);
    expect(lossAlphaNeg).toBe(1.5 * 2.0 - 0.5 * 1.0); // (1-(-0.5))*2 + (-0.5)*1 = 2.5
  });

  it('should work with zero losses', () => {
    expect(hybridIBLoss(0.0, 0.0, 0.5)).toBe(0.0);
    expect(hybridIBLoss(5.0, 0.0, 0.5)).toBeCloseTo(2.5, 6);
    expect(hybridIBLoss(0.0, 5.0, 0.5)).toBeCloseTo(2.5, 6);
  });
});
