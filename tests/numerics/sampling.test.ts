import { describe, expect, it } from 'vitest';
import {
  beamSearch,
  nucleusSample,
  normalizeLogWeights,
  sampleFromLogits,
  temperatureSample,
  topKSample
} from '../../src/generation/sampling';
import { stableSoftmax } from '../../src/lib/MathUtils';

describe('sampling utilities', () => {
  it('stableSoftmax handles extreme logits', () => {
    const logits = [1000, 1001, 999];
    const probs = stableSoftmax(logits);
    const sum = probs.reduce((acc, value) => acc + value, 0);
    expect(sum).toBeCloseTo(1, 6);
    expect(probs[1]).toBeGreaterThan(probs[0]);
  });

  it('normalizes log weights using log-sum-exp', () => {
    const weights = normalizeLogWeights([-1000, -1001, -999]);
    const sum = weights.reduce((acc, value) => acc + value, 0);
    expect(sum).toBeCloseTo(1, 6);
    expect(Math.max(...weights)).toBeGreaterThan(Math.min(...weights));
  });

  it('samples with temperature, top-k, and nucleus strategies', () => {
    const logits = [0.1, 1.2, 2.5, -4.2];
    const rng = () => 0.5;
    const tempIndex = temperatureSample(logits, 0.7, rng);
    const topKIndex = topKSample(logits, 2, { rng });
    const nucleusIndex = nucleusSample(logits, 0.9, { rng });
    expect(tempIndex).toBeGreaterThanOrEqual(0);
    expect(topKIndex).toBeLessThan(logits.length);
    expect(nucleusIndex).toBeLessThan(logits.length);
  });

  it('supports beam search decoding', () => {
    const logitsTable: Record<string, number[]> = {
      '': [0.5, 1.0, 0.0],
      '0': [0.4, 0.6, 0.1],
      '1': [0.2, 0.3, 1.2]
    };
    const step = (prefix: number[]) => {
      const key = prefix.slice(-1).join('');
      return logitsTable[key as keyof typeof logitsTable] ?? logitsTable[''];
    };
    const results = beamSearch(step, [], { beamWidth: 2, maxLength: 2, temperature: 1 });
    expect(results.length).toBeGreaterThan(0);
    const weightSum = results.reduce((acc, r) => acc + r.probability, 0);
    expect(weightSum).toBeCloseTo(1, 6);
    expect(results[0].tokens.length).toBeLessThanOrEqual(2);
  });

  it('falls back to deterministic sampling when distribution collapses', () => {
    const logits = [10, -1000, -2000];
    const index = sampleFromLogits(logits, { minProbability: 0.9, rng: () => 0.1 });
    expect(index).toBe(0);
  });
});
