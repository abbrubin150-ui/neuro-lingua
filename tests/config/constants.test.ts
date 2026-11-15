import { describe, expect, it } from 'vitest';
import {
  DEFAULT_ARCHITECTURE,
  DEFAULT_TOKENIZER_CONFIG,
  defaultContextSize,
  defaultTrainingStride
} from '../../src/config/constants';

describe('UI defaults', () => {
  it('start from the Transformer with subword tokens', () => {
    expect(DEFAULT_ARCHITECTURE).toBe('transformer');
    expect(DEFAULT_TOKENIZER_CONFIG.mode).toBe('bpe');
    expect(DEFAULT_TOKENIZER_CONFIG.bpeVocabSize).toBe(2000);
  });

  it('give the Transformer a longer context than the order-blind MLPs', () => {
    expect(defaultContextSize('transformer')).toBe(32);
    expect(defaultContextSize('feedforward')).toBe(4);
    expect(defaultContextSize('advanced')).toBe(4);
  });

  it('train every token 4 times per epoch (stride = context / 4, at least 1)', () => {
    expect(defaultTrainingStride(32)).toBe(8);
    expect(defaultTrainingStride(128)).toBe(32);
    expect(defaultTrainingStride(4)).toBe(1);
    expect(defaultTrainingStride(2)).toBe(1);
  });
});
