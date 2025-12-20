import { describe, expect, it } from 'vitest';
import {
  FLOAT32_EXP_OVERFLOW_GUARD,
  FLOAT32_EXP_UNDERFLOW_GUARD,
  logSumExp,
  stableSoftmax
} from '../../src/lib/MathUtils';

function isFiniteArray(values: number[]): boolean {
  return values.every((v) => Number.isFinite(v));
}

describe('numerical stability edge cases', () => {
  it('handles overflow-prone logits while remaining normalized', () => {
    const logits = [FLOAT32_EXP_OVERFLOW_GUARD + 5, FLOAT32_EXP_OVERFLOW_GUARD + 1, -10];
    const probs = stableSoftmax(logits);

    expect(isFiniteArray(probs)).toBe(true);
    expect(probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 8);
    expect(probs[0]).toBeGreaterThan(0.97);
    expect(probs[2]).toBeGreaterThanOrEqual(0);
  });

  it('avoids underflow collapse for extremely negative logits', () => {
    const logits = new Array(1024).fill(FLOAT32_EXP_UNDERFLOW_GUARD - 50);
    logits[0] = -1; // a single higher logit to anchor normalization

    const probs = stableSoftmax(logits);
    expect(probs[0]).toBeGreaterThan(0.99);
    expect(probs.slice(1).every((p) => p >= 0)).toBe(true);
    expect(probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
  });

  it('preserves ordering on very long sequences', () => {
    const logits = Array.from({ length: 4096 }, (_, i) => (i % 2 === 0 ? i / 10 : -i / 20));
    const probs = stableSoftmax(logits);

    expect(probs.length).toBe(logits.length);
    expect(probs[10]).toBeGreaterThan(probs[11]);
    expect(probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
  });

  it('computes log-sum-exp without overflow near guard rails', () => {
    const values = [FLOAT32_EXP_OVERFLOW_GUARD + 4, FLOAT32_EXP_OVERFLOW_GUARD + 2, 0];
    const result = logSumExp(values);

    const maxVal = Math.max(...values);
    expect(Number.isFinite(result)).toBe(true);
    expect(result).toBeGreaterThan(maxVal - 1);
  });

  it('returns finite values for heavily underflowed inputs', () => {
    const values = new Array(2048).fill(FLOAT32_EXP_UNDERFLOW_GUARD * 2);
    const result = logSumExp(values);

    expect(Number.isFinite(result)).toBe(true);
    expect(result).toBeGreaterThan(FLOAT32_EXP_UNDERFLOW_GUARD * 2 - 25);
  });
});
