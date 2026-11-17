import { describe, expect, it } from 'vitest';
import {
  EarlyStopping,
  bitsPerCharacter,
  splitForValidation
} from '../../src/lib/validationSplit';

const sentences = (n: number, prefix = 'Sentence number') =>
  Array.from({ length: n }, (_, i) => `${prefix} ${i} has a few words in it.`).join(' ');

describe('splitForValidation', () => {
  it('holds out every 10th sentence and keeps the rest for training', () => {
    const split = splitForValidation(sentences(100))!;
    expect(split.unit).toBe('sentence');
    expect(split.units).toBe(100);
    expect(split.every).toBe(10);
    expect(split.validation).toContain('Sentence number 9 ');
    expect(split.validation).toContain('Sentence number 99 ');
    expect(split.train).toContain('Sentence number 0 ');
    expect(split.train).not.toContain('Sentence number 9 has');
    // nothing lost, nothing shared
    const count = (t: string) => t.split('Sentence number').length - 1;
    expect(count(split.train)).toBe(90);
    expect(count(split.validation)).toBe(10);
  });

  it('is deterministic', () => {
    const text = sentences(60);
    expect(splitForValidation(text)).toEqual(splitForValidation(text));
  });

  it('splits by paragraph when there are enough paragraphs', () => {
    const text = Array.from({ length: 40 }, (_, i) => `Paragraph ${i}. It has two sentences, and the second one is longer than the first.`).join(
      '\n\n'
    );
    const split = splitForValidation(text)!;
    expect(split.unit).toBe('paragraph');
    expect(split.validation.split('\n\n')).toHaveLength(4);
    expect(split.validation).toContain('Paragraph 9. It has two sentences');
  });

  it('splits unpunctuated text by line (Hebrew verses)', () => {
    const text = Array.from(
      { length: 50 },
      (_, i) => `ויאמר אלהים יהי אור ויהי אור וירא אלהים את האור כי טוב פסוק ${i}`
    ).join('\n');
    const split = splitForValidation(text)!;
    expect(split.unit).toBe('sentence');
    expect(split.units).toBe(50);
    expect(split.validation).toContain('פסוק 9');
    expect(split.train).not.toContain('פסוק 9\n');
  });

  it('returns null for texts too small to spare a validation part', () => {
    expect(splitForValidation('One short sentence only.')).toBeNull();
    expect(splitForValidation(sentences(10))).toBeNull(); // fewer than 20 units
    expect(splitForValidation(Array.from({ length: 30 }, () => 'Hi.').join(' '))).toBeNull();
    expect(splitForValidation('')).toBeNull();
  });

  it('follows the requested fraction', () => {
    const split = splitForValidation(sentences(100), { fraction: 0.2 })!;
    expect(split.every).toBe(5);
    expect(split.validation.split('Sentence number').length - 1).toBe(20);
  });
});

describe('bitsPerCharacter', () => {
  it('converts a per-token loss in nats to bits per character', () => {
    // 10 tokens at ln(2) nats each over 5 characters = 2 bits per character
    expect(bitsPerCharacter(Math.LN2, 10, 5)).toBeCloseTo(2, 12);
    expect(bitsPerCharacter(1, 0, 5)).toBeNaN();
  });
});

describe('EarlyStopping', () => {
  it('stops after `patience` epochs without improvement and remembers the best epoch', () => {
    const stop = new EarlyStopping(3);
    expect(stop.best).toBeNull();
    expect(stop.record(1, 2.0)).toBe(true);
    expect(stop.record(2, 1.5)).toBe(true);
    expect(stop.record(3, 1.6)).toBe(false);
    expect(stop.shouldStop()).toBe(false);
    expect(stop.record(4, 1.7)).toBe(false);
    expect(stop.shouldStop()).toBe(false);
    expect(stop.record(5, 1.55)).toBe(false);
    expect(stop.shouldStop()).toBe(true);
    expect(stop.best).toEqual({ epoch: 2, loss: 1.5 });
  });

  it('resets the wait when the loss improves again', () => {
    const stop = new EarlyStopping(2);
    stop.record(1, 2.0);
    stop.record(2, 2.1);
    stop.record(3, 1.9);
    stop.record(4, 2.0);
    expect(stop.shouldStop()).toBe(false);
  });

  it('never stops with patience 0 and ignores non-finite losses', () => {
    const stop = new EarlyStopping(0);
    stop.record(1, 1);
    for (let e = 2; e < 20; e++) stop.record(e, 2);
    expect(stop.shouldStop()).toBe(false);
    const nan = new EarlyStopping(1);
    expect(nan.record(1, NaN)).toBe(false);
    expect(nan.shouldStop()).toBe(false);
  });
});
