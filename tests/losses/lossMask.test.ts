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
  createSimpleCharTokenizer,
  validateRegExpPattern,
  findRegExpMatches,
  buildCharMaskFromMatches,
  buildRegExpLossMask
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

// New tests for RegExp masking functionality

describe('validateRegExpPattern', () => {
  it('validates a valid pattern', () => {
    expect(validateRegExpPattern('Answer:\\s*')).toEqual({ valid: true });
    expect(validateRegExpPattern('^A:')).toEqual({ valid: true });
    expect(validateRegExpPattern('\\[RESPONSE\\]')).toEqual({ valid: true });
    expect(validateRegExpPattern('(?:Output|Result):')).toEqual({ valid: true });
  });

  it('returns error for empty pattern', () => {
    const result = validateRegExpPattern('');
    expect(result.valid).toBe(false);
    expect(result.error).toBe('Pattern cannot be empty');
  });

  it('returns error for whitespace-only pattern', () => {
    const result = validateRegExpPattern('   ');
    expect(result.valid).toBe(false);
    expect(result.error).toBe('Pattern cannot be empty');
  });

  it('returns error for invalid pattern', () => {
    const result = validateRegExpPattern('[invalid');
    expect(result.valid).toBe(false);
    expect(result.error).toBeDefined();
  });

  it('returns error for unbalanced parentheses', () => {
    const result = validateRegExpPattern('(unclosed');
    expect(result.valid).toBe(false);
  });
});

describe('findRegExpMatches', () => {
  it('finds all matches with global flag', () => {
    const matches = findRegExpMatches('Answer: 42 Answer: 17', 'Answer:\\s*');

    expect(matches).toHaveLength(2);
    expect(matches[0]).toEqual({ start: 0, end: 8, match: 'Answer: ' });
    expect(matches[1]).toEqual({ start: 11, end: 19, match: 'Answer: ' });
  });

  it('finds first match without global flag', () => {
    const matches = findRegExpMatches('A: 42 A: 17', 'A:\\s*', { global: false });

    expect(matches).toHaveLength(1);
    expect(matches[0]).toEqual({ start: 0, end: 3, match: 'A: ' });
  });

  it('returns empty array for no matches', () => {
    const matches = findRegExpMatches('hello world', 'Answer:');

    expect(matches).toHaveLength(0);
  });

  it('handles case insensitive matching', () => {
    const matches = findRegExpMatches('answer: 42 ANSWER: 17', 'answer:', {
      caseInsensitive: true
    });

    expect(matches).toHaveLength(2);
  });

  it('returns empty array for invalid pattern', () => {
    const matches = findRegExpMatches('hello', '[invalid');

    expect(matches).toHaveLength(0);
  });

  it('handles zero-length matches without infinite loop', () => {
    const matches = findRegExpMatches('abc', '');

    // Should handle gracefully without hanging
    expect(Array.isArray(matches)).toBe(true);
  });
});

describe('buildCharMaskFromMatches', () => {
  it('builds mask for "after" position mode', () => {
    const text = 'Q: test A: answer';
    const matches = [{ start: 8, end: 11, match: 'A: ' }];
    const mask = buildCharMaskFromMatches(text, matches, 'after');

    // Characters 0-10 should be 0 (before and including match)
    // Characters 11-16 should be 1 (after match)
    expect(mask.slice(0, 11).every((v) => v === 0)).toBe(true);
    expect(mask.slice(11).every((v) => v === 1)).toBe(true);
  });

  it('builds mask for "before" position mode', () => {
    const text = 'question A: answer';
    const matches = [{ start: 9, end: 12, match: 'A: ' }];
    const mask = buildCharMaskFromMatches(text, matches, 'before');

    // Characters 0-8 should be 1 (before match)
    // Characters 9+ should be 0 (match and after)
    expect(mask.slice(0, 9).every((v) => v === 1)).toBe(true);
    expect(mask.slice(9).every((v) => v === 0)).toBe(true);
  });

  it('builds mask for "match" position mode', () => {
    const text = 'hello [TAG] world [TAG] end';
    const matches = [
      { start: 6, end: 11, match: '[TAG]' },
      { start: 18, end: 23, match: '[TAG]' }
    ];
    const mask = buildCharMaskFromMatches(text, matches, 'match');

    // Only positions 6-10 and 18-22 should be 1
    expect(mask[5]).toBe(0);
    expect(mask[6]).toBe(1);
    expect(mask[10]).toBe(1);
    expect(mask[11]).toBe(0);
    expect(mask[17]).toBe(0);
    expect(mask[18]).toBe(1);
    expect(mask[22]).toBe(1);
    expect(mask[23]).toBe(0);
  });

  it('builds mask for "exclude" position mode', () => {
    const text = 'hello [TAG] world';
    const matches = [{ start: 6, end: 11, match: '[TAG]' }];
    const mask = buildCharMaskFromMatches(text, matches, 'exclude');

    // Positions 6-10 should be 0, all others should be 1
    expect(mask.slice(0, 6).every((v) => v === 1)).toBe(true);
    expect(mask.slice(6, 11).every((v) => v === 0)).toBe(true);
    expect(mask.slice(11).every((v) => v === 1)).toBe(true);
  });

  it('returns all zeros when no matches (except exclude mode)', () => {
    const text = 'hello world';

    expect(buildCharMaskFromMatches(text, [], 'after').every((v) => v === 0)).toBe(true);
    expect(buildCharMaskFromMatches(text, [], 'before').every((v) => v === 0)).toBe(true);
    expect(buildCharMaskFromMatches(text, [], 'match').every((v) => v === 0)).toBe(true);
  });

  it('returns all ones for exclude mode with no matches', () => {
    const text = 'hello world';
    const mask = buildCharMaskFromMatches(text, [], 'exclude');

    expect(mask.every((v) => v === 1)).toBe(true);
  });
});

describe('buildRegExpLossMask', () => {
  const tokenizer = createSimpleCharTokenizer();

  it('builds mask for "after" position with regex', () => {
    const text = 'Q: test A: answer';
    const mask = buildRegExpLossMask(text, tokenizer, {
      pattern: 'A:\\s*',
      position: 'after'
    });

    // Should mask tokens before 'A: '
    expect(mask).toHaveLength(text.length);
    expect(countActiveMaskPositions(mask)).toBe(6); // 'answer' = 6 chars
  });

  it('builds mask for "before" position with regex', () => {
    const text = 'question A: answer';
    const mask = buildRegExpLossMask(text, tokenizer, {
      pattern: 'A:',
      position: 'before'
    });

    expect(mask).toHaveLength(text.length);
    // 'question ' = 9 chars before 'A:'
    expect(countActiveMaskPositions(mask)).toBe(9);
  });

  it('builds mask for "match" position with regex', () => {
    const text = 'hello [TAG] world';
    const mask = buildRegExpLossMask(text, tokenizer, {
      pattern: '\\[TAG\\]',
      position: 'match'
    });

    expect(mask).toHaveLength(text.length);
    expect(countActiveMaskPositions(mask)).toBe(5); // '[TAG]' = 5 chars
  });

  it('builds mask for "exclude" position with regex', () => {
    const text = 'hello [TAG] world';
    const mask = buildRegExpLossMask(text, tokenizer, {
      pattern: '\\[TAG\\]',
      position: 'exclude'
    });

    expect(mask).toHaveLength(text.length);
    expect(countActiveMaskPositions(mask)).toBe(12); // 17 - 5 = 12 chars
  });

  it('returns all zeros for invalid pattern', () => {
    const text = 'hello world';
    const mask = buildRegExpLossMask(text, tokenizer, {
      pattern: '[invalid',
      position: 'after'
    });

    expect(mask.every((v) => v === 0)).toBe(true);
  });

  it('returns all zeros for no match (except exclude)', () => {
    const text = 'hello world';
    const mask = buildRegExpLossMask(text, tokenizer, {
      pattern: 'NOTFOUND',
      position: 'after'
    });

    expect(mask.every((v) => v === 0)).toBe(true);
  });

  it('supports case insensitive matching', () => {
    const text = 'answer: 42';
    const mask = buildRegExpLossMask(text, tokenizer, {
      pattern: 'ANSWER:',
      position: 'after',
      caseInsensitive: true
    });

    expect(countActiveMaskPositions(mask)).toBeGreaterThan(0);
  });
});

describe('buildAnswerLossMask with customRegExp mode', () => {
  const tokenizer = createSimpleCharTokenizer();

  it('handles customRegExp mode with after position', () => {
    const text = 'Q: test A: answer';
    const inputIds = Array.from(text).map((c) => c.charCodeAt(0));
    const mask = buildAnswerLossMask(inputIds, tokenizer, 'customRegExp', 'A:', {
      pattern: 'A:\\s*',
      position: 'after'
    });

    expect(mask).toHaveLength(inputIds.length);
    // Should have 1s only for 'answer' portion
    expect(countActiveMaskPositions(mask)).toBe(6);
  });

  it('handles customRegExp mode with match position', () => {
    const text = 'Q: [ANS] A: 42';
    const inputIds = Array.from(text).map((c) => c.charCodeAt(0));
    const mask = buildAnswerLossMask(inputIds, tokenizer, 'customRegExp', 'A:', {
      pattern: '\\[ANS\\]',
      position: 'match'
    });

    expect(mask).toHaveLength(inputIds.length);
    expect(countActiveMaskPositions(mask)).toBe(5); // '[ANS]'
  });

  it('handles customRegExp mode with exclude position', () => {
    const text = 'Q: [SKIP] answer';
    const inputIds = Array.from(text).map((c) => c.charCodeAt(0));
    const mask = buildAnswerLossMask(inputIds, tokenizer, 'customRegExp', 'A:', {
      pattern: '\\[SKIP\\]',
      position: 'exclude'
    });

    expect(mask).toHaveLength(inputIds.length);
    // Should have 1s for everything except '[SKIP]'
    expect(countActiveMaskPositions(mask)).toBe(10); // 16 - 6 = 10
  });

  it('returns all zeros when no pattern provided', () => {
    const text = 'hello world';
    const inputIds = Array.from(text).map((c) => c.charCodeAt(0));
    const mask = buildAnswerLossMask(inputIds, tokenizer, 'customRegExp', 'A:', {
      pattern: '',
      position: 'after'
    });

    expect(mask.every((v) => v === 0)).toBe(true);
  });

  it('returns all zeros when pattern not found (except exclude)', () => {
    const text = 'hello world';
    const inputIds = Array.from(text).map((c) => c.charCodeAt(0));
    const mask = buildAnswerLossMask(inputIds, tokenizer, 'customRegExp', 'A:', {
      pattern: 'NOTFOUND',
      position: 'after'
    });

    expect(mask.every((v) => v === 0)).toBe(true);
  });

  it('defaults to "after" position when not specified', () => {
    const text = 'Q: test A: answer';
    const inputIds = Array.from(text).map((c) => c.charCodeAt(0));
    const mask = buildAnswerLossMask(inputIds, tokenizer, 'customRegExp', 'A:', {
      pattern: 'A:\\s*'
      // position not specified, should default to 'after'
    });

    expect(mask).toHaveLength(inputIds.length);
    expect(countActiveMaskPositions(mask)).toBe(6); // 'answer'
  });
});
