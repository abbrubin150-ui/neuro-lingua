import { describe, expect, it } from 'vitest';
import {
  aggregateLogEvidence,
  bayesianWeightSamples,
  bootstrapWeightSamples,
  monteCarloExperiment,
  posteriorMonteCarlo,
  type WeightSnapshot
} from '../../src/experiments/bayesian';

describe('bayesian experiments', () => {
  const snapshot: WeightSnapshot = {
    embedding: [
      [0.5, -0.1],
      [0.2, 0.3]
    ],
    bias: [0.01, -0.02]
  };

  it('draws bootstrap samples preserving shapes', () => {
    const samples = bootstrapWeightSamples(snapshot, { numSamples: 5, rng: () => 0.25 });
    expect(samples).toHaveLength(5);
    for (const sample of samples) {
      expect(sample.weights.embedding.length).toBe(snapshot.embedding.length);
      expect(sample.weights.bias.length).toBe(snapshot.bias.length);
      expect(sample.source).toBe('bootstrap');
    }
  });

  it('performs gaussian Bayesian sampling with posterior weights', () => {
    const samples = bayesianWeightSamples(snapshot, {
      numSamples: 4,
      priorStd: 0.5,
      likelihoodStd: 0.3,
      rng: () => 0.75
    });
    expect(samples).toHaveLength(4);
    for (const sample of samples) {
      expect(sample.source).toBe('bayesian');
      expect(sample.weights.embedding[0].length).toBe(snapshot.embedding[0].length);
    }
  });

  it('estimates Monte Carlo expectations with normalized weights', () => {
    const samples = bayesianWeightSamples(snapshot, { numSamples: 3, rng: () => 0.5 });
    const result = monteCarloExperiment(samples, (weights) => weights.bias.reduce((a, b) => a + b, 0));
    expect(result.weights.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
    expect(result.effectiveSampleSize).toBeGreaterThan(0);
  });

  it('combines bootstrap and bayesian samples for posterior Monte Carlo', () => {
    const result = posteriorMonteCarlo(
      snapshot,
      (weights) => weights.embedding.flat().reduce((a, b) => a + b, 0),
      { numSamples: 3, bootstrapSamples: 2, rng: () => 0.1 }
    );
    expect(result.weights.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
    expect(Number.isFinite(result.mean)).toBe(true);
  });

  it('computes aggregate log evidence via log-sum-exp', () => {
    const logWeights = [-10, -11, -9.5];
    const evidence = aggregateLogEvidence(logWeights);
    const expected = Math.log(
      logWeights.map((w) => Math.exp(w)).reduce((a, b) => a + b, 0) / logWeights.length
    );
    expect(evidence).toBeCloseTo(expected, 6);
  });
});
