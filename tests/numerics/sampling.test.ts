import { describe, expect, it } from 'vitest';
import {
  beamSearch,
  contrastiveSearch,
  greedySample,
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

  it('greedy sampling always selects highest probability token', () => {
    const logits = [0.1, 2.5, 1.2, -4.2];
    const index = greedySample(logits);
    expect(index).toBe(1); // Index 1 has the highest logit (2.5)
  });

  it('greedy sampling handles negative logits', () => {
    const logits = [-5.0, -1.0, -3.0];
    const index = greedySample(logits);
    expect(index).toBe(1); // Index 1 has the highest logit (-1.0)
  });

  it('greedy sampling throws error on empty logits', () => {
    expect(() => greedySample([])).toThrow('Cannot perform greedy sampling on empty logits.');
  });

  it('contrastive search balances probability and diversity', () => {
    // Simple test case
    const logitsTable: Record<string, number[]> = {
      '0': [1.0, 0.5, 0.3],
      '1': [0.4, 1.0, 0.2],
      '2': [0.3, 0.2, 1.0]
    };

    const step = (prefix: number[]) => {
      const lastToken = prefix[prefix.length - 1];
      return logitsTable[lastToken.toString()] ?? [1.0, 0.5, 0.3];
    };

    // Simple embedding function - tokens have different embeddings
    const embeddingFn = (tokenIds: number[]) => {
      if (tokenIds.length === 0) return [];
      const tokenId = tokenIds[0];
      // Create distinct embeddings for each token
      if (tokenId === 0) return [1.0, 0.0, 0.0];
      if (tokenId === 1) return [0.0, 1.0, 0.0];
      return [0.0, 0.0, 1.0];
    };

    const result = contrastiveSearch(step, embeddingFn, [0], {
      topK: 3,
      alpha: 0.6,
      maxLength: 5
    });

    expect(result.tokens.length).toBeGreaterThan(0);
    expect(result.tokens.length).toBeLessThanOrEqual(5);
    expect(result.score).toBeDefined();
  });

  it('contrastive search respects EOS token', () => {
    const step = (prefix: number[]) => {
      if (prefix.length >= 3) return [0.1, 0.1, 5.0]; // Make token 2 (EOS) highly likely
      return [1.0, 0.5, 0.1];
    };

    const embeddingFn = (tokenIds: number[]) => {
      const tokenId = tokenIds[0] || 0;
      return [tokenId, tokenId * 0.5, tokenId * 0.2];
    };

    const result = contrastiveSearch(step, embeddingFn, [0], {
      topK: 3,
      alpha: 0.5,
      maxLength: 10,
      eosToken: 2
    });

    expect(result.tokens.length).toBeLessThan(10);
  });

  it('contrastive search validates parameters', () => {
    const dummyStep = () => [1.0, 0.5];
    const dummyEmbed = () => [0.5, 0.5];

    expect(() =>
      contrastiveSearch(dummyStep, dummyEmbed, [], {
        topK: 0,
        alpha: 0.5,
        maxLength: 10
      })
    ).toThrow('topK must be positive');

    expect(() =>
      contrastiveSearch(dummyStep, dummyEmbed, [], {
        topK: 3,
        alpha: 1.5,
        maxLength: 10
      })
    ).toThrow('alpha must be between 0 and 1');

    expect(() =>
      contrastiveSearch(dummyStep, dummyEmbed, [], {
        topK: 3,
        alpha: 0.5,
        maxLength: 0
      })
    ).toThrow('maxLength must be positive');
  });
});
