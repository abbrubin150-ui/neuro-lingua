import { describe, expect, it } from 'vitest';
import { mirostatV2Sample } from '../../src/generation/sampling';

function sequenceRng(values: number[]) {
  let idx = 0;
  return () => {
    const value = values[idx % values.length];
    idx += 1;
    return value;
  };
}

describe('mirostat v2 sampler', () => {
  it('enforces parameter bounds', () => {
    expect(() => mirostatV2Sample([0, 0], { targetEntropy: 0.0 })).toThrow();
    expect(() => mirostatV2Sample([0, 0], { learningRate: 0 })).toThrow();
    expect(() => mirostatV2Sample([0, 0], { learningRate: 2 })).toThrow();
  });

  it('converges surprise toward the target entropy', () => {
    const logits = Array.from({ length: 50 }, (_, i) => Math.log(50 - i));
    const rng = sequenceRng([0.12, 0.37, 0.73, 0.91]);

    const targetEntropy = 3.0;
    const learningRate = 0.3;
    let state = { mu: targetEntropy * 2 };
    const surprises: number[] = [];

    for (let i = 0; i < 100; i++) {
      const result = mirostatV2Sample(logits, {
        targetEntropy,
        learningRate,
        rng,
        state
      });
      state = result.state;
      surprises.push(result.surprise);
    }

    const tail = surprises.slice(-40);
    const averageSurprise = tail.reduce((acc, value) => acc + value, 0) / tail.length;

    expect(averageSurprise).toBeGreaterThan(targetEntropy - 0.35);
    expect(averageSurprise).toBeLessThan(targetEntropy + 0.35);
    expect(state.mu).toBeGreaterThan(0);
  });

  it('honors probability filters before truncation', () => {
    const logits = [5, 1, 0, -5];
    const { index } = mirostatV2Sample(logits, {
      targetEntropy: 2,
      learningRate: 0.3,
      topP: 0.2,
      rng: () => 0
    });

    expect(index).toBe(0);
  });
});
