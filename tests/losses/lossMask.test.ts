/**
 * Tests for Loss Mask Utilities
 *
 * @module tests/losses/lossMask
 */

import { describe, it, expect } from 'vitest';
import {
  buildAnswerLossMask,
  crossEntropyMasked,
  crossEntropySingleMasked,
  applyGradientMask,
  countActiveMaskPositions,
  createSimpleCharTokenizer
} from '../../src/losses/lossMask';

describe('buildAnswerLossMask', () => {
  const tokenizer = createSimpleCharTokenizer();

  it('returns all 1s when mode is none', () => {
    const inputIds = [65, 66, 67, 68, 69]; // ABCDE
    const mask = buildAnswerLossMask(inputIds, tokenizer, 'none');

    expect(mask).toHaveLength(inputIds.length);
    expect(mask.every((v) => v === 1)).toBe(true);
  });

  it('masks before equals sign in afterEquals mode', () => {
    // "20+7=27" as char codes
    const text = '20+7=27';
    const inputIds = Array.from(text).map((c) => c.charCodeAt(0));
    const mask = buildAnswerLossMask(inputIds, tokenizer, 'afterEquals');

    // Position of '=' is 4 (0-indexed)
    // We want loss only AFTER '=', so positions 5 and 6 should be 1
    expect(mask).toHaveLength(inputIds.length);
    expect(mask.slice(0, 5).every((v) => v === 0)).toBe(true); // "20+7=" masked
    expect(mask.slice(5).every((v) => v === 1)).toBe(true); // "27" unmasked
  });

  it('masks before answer tag in afterAnswerTag mode', () => {
    // "Q:20+7A:27" as char codes
    const text = 'Q:20+7A:27';
    const inputIds = Array.from(text).map((c) => c.charCodeAt(0));
    const mask = buildAnswerLossMask(inputIds, tokenizer, 'afterAnswerTag', 'A:');

    // Position of 'A:' starts at 6
    // We want loss only AFTER 'A:', so positions 8 and 9 should be 1
    expect(mask).toHaveLength(inputIds.length);
    expect(mask.slice(0, 8).every((v) => v === 0)).toBe(true); // "Q:20+7A:" masked
    expect(mask.slice(8).every((v) => v === 1)).toBe(true); // "27" unmasked
  });

  it('returns all 0s when delimiter not found', () => {
    const text = 'hello world';
    const inputIds = Array.from(text).map((c) => c.charCodeAt(0));
    const mask = buildAnswerLossMask(inputIds, tokenizer, 'afterEquals');

    expect(mask.every((v) => v === 0)).toBe(true);
  });

  it('handles empty input', () => {
    const mask = buildAnswerLossMask([], tokenizer, 'afterEquals');
    expect(mask).toHaveLength(0);
  });

  it('handles delimiter at end of string', () => {
    const text = '20+7=';
    const inputIds = Array.from(text).map((c) => c.charCodeAt(0));
    const mask = buildAnswerLossMask(inputIds, tokenizer, 'afterEquals');

    // All positions should be masked (nothing after '=')
    expect(mask.every((v) => v === 0)).toBe(true);
  });

  it('uses custom answer tag', () => {
    const text = 'Q:test=>answer';
    const inputIds = Array.from(text).map((c) => c.charCodeAt(0));
    const mask = buildAnswerLossMask(inputIds, tokenizer, 'afterAnswerTag', '=>');

    // '=>' starts at position 6, so positions 8+ should be 1
    const delimPos = text.indexOf('=>');
    expect(mask.slice(0, delimPos + 2).every((v) => v === 0)).toBe(true);
    expect(mask.slice(delimPos + 2).every((v) => v === 1)).toBe(true);
  });
});

describe('crossEntropyMasked', () => {
  it('computes loss only for masked positions', () => {
    // Simple 3-class problem
    const logits = [
      [2.0, 1.0, 0.5], // Strongly predicts class 0
      [1.0, 2.0, 0.5], // Strongly predicts class 1
      [0.5, 1.0, 2.0] // Strongly predicts class 2
    ];
    const targets = [0, 1, 2]; // All correct
    const mask = [1, 0, 1]; // Skip position 1

    const loss = crossEntropyMasked(logits, targets, mask);

    // Loss should only be computed for positions 0 and 2
    expect(loss).toBeGreaterThan(0);
    expect(loss).toBeLessThan(1); // Low loss since predictions are correct
  });

  it('returns 0 when all positions masked', () => {
    const logits = [
      [2.0, 1.0, 0.5],
      [1.0, 2.0, 0.5]
    ];
    const targets = [0, 1];
    const mask = [0, 0];

    const loss = crossEntropyMasked(logits, targets, mask);
    expect(loss).toBe(0);
  });

  it('skips padding tokens when padId provided', () => {
    const logits = [
      [2.0, 1.0, 0.5],
      [1.0, 2.0, 0.5],
      [0.5, 1.0, 2.0]
    ];
    const targets = [0, 1, 0]; // Last target is padding
    const mask = [1, 1, 1];
    const padId = 0;

    const loss = crossEntropyMasked(logits, targets, mask, padId);

    // Should skip last position because target === padId
    expect(loss).toBeGreaterThan(0);
  });

  it('computes correct loss for single position', () => {
    // Uniform distribution logits
    const logits = [[0, 0, 0]];
    const targets = [0];
    const mask = [1];

    const loss = crossEntropyMasked(logits, targets, mask);

    // -log(1/3) = log(3) ≈ 1.0986
    expect(loss).toBeCloseTo(Math.log(3), 4);
  });
});

describe('crossEntropySingleMasked', () => {
  it('returns loss when mask is 1', () => {
    const probs = [0.7, 0.2, 0.1];
    const target = 0;

    const loss = crossEntropySingleMasked(probs, target, 1);

    expect(loss).toBeCloseTo(-Math.log(0.7), 4);
  });

  it('returns 0 when mask is 0', () => {
    const probs = [0.7, 0.2, 0.1];
    const target = 0;

    const loss = crossEntropySingleMasked(probs, target, 0);

    expect(loss).toBe(0);
  });
});

describe('applyGradientMask', () => {
  it('returns zeros when mask is 0', () => {
    const dLogits = [0.5, -0.3, 0.2];

    const masked = applyGradientMask(dLogits, 0);

    expect(masked).toEqual([0, 0, 0]);
  });

  it('returns original when mask is 1', () => {
    const dLogits = [0.5, -0.3, 0.2];

    const masked = applyGradientMask(dLogits, 1);

    expect(masked).toEqual(dLogits);
  });
});

describe('countActiveMaskPositions', () => {
  it('counts positions with value 1', () => {
    expect(countActiveMaskPositions([1, 0, 1, 1, 0])).toBe(3);
    expect(countActiveMaskPositions([0, 0, 0])).toBe(0);
    expect(countActiveMaskPositions([1, 1, 1])).toBe(3);
    expect(countActiveMaskPositions([])).toBe(0);
  });
});

describe('createSimpleCharTokenizer', () => {
  it('creates a working char tokenizer', () => {
    const tokenizer = createSimpleCharTokenizer();

    expect(tokenizer.encode('ABC')).toEqual([65, 66, 67]);
    expect(tokenizer.encode('')).toEqual([]);
    expect(tokenizer.encode('=')).toEqual([61]);
  });
});
