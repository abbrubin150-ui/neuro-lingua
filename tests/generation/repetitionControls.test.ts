import { describe, expect, it } from 'vitest';
import {
  applyRepetitionControls,
  bannedNgramTokens,
  greedySample,
  recommendedNoRepeatNgram
} from '../../src/generation/sampling';
import { ProNeuralLM } from '../../src/lib/ProNeuralLM';

describe('bannedNgramTokens', () => {
  it('bans the token that would repeat an earlier n-gram', () => {
    // "a b c a b" + c would repeat the trigram "a b c"
    expect([...bannedNgramTokens([0, 1, 2, 0, 1], 3)]).toEqual([2]);
  });

  it('collects every continuation of the current suffix', () => {
    // suffix "a": followed earlier by b and by c
    expect([...bannedNgramTokens([0, 1, 0, 2, 0], 2)].sort()).toEqual([1, 2]);
  });

  it('bans nothing when the suffix never occurred before', () => {
    expect(bannedNgramTokens([0, 1, 2, 3], 3).size).toBe(0);
    expect(bannedNgramTokens([0], 3).size).toBe(0);
    expect(bannedNgramTokens([0, 1, 0, 1], 0).size).toBe(0);
  });
});

describe('applyRepetitionControls', () => {
  it('returns the same array when every control is off', () => {
    const logits = [1, 2, 3];
    expect(applyRepetitionControls(logits, [0, 1], {})).toBe(logits);
    expect(applyRepetitionControls(logits, [], { repetitionPenalty: 1.3 })).toBe(logits);
  });

  it('divides positive and multiplies negative logits of recent tokens (CTRL)', () => {
    const out = applyRepetitionControls([2, -1, 3], [0, 1], { repetitionPenalty: 2 });
    expect(out).toEqual([1, -2, 3]);
  });

  it('only looks at the last repetitionWindow tokens', () => {
    const out = applyRepetitionControls([2, 2, 2], [0, 1], {
      repetitionPenalty: 2,
      repetitionWindow: 1
    });
    expect(out).toEqual([2, 1, 2]);
  });

  it('makes greedy decoding leave a repeated n-gram', () => {
    const logits = [0, 0, 5, 1];
    expect(greedySample(logits)).toBe(2);
    const out = applyRepetitionControls(logits, [0, 1, 2, 0, 1], { noRepeatNgramSize: 3 });
    expect(out.every(Number.isFinite)).toBe(true);
    expect(greedySample(out)).toBe(3);
  });

  it('skips the n-gram rule when it would ban every token', () => {
    const logits = [1, 2];
    // suffix "0" was followed by both 0 and 1
    const out = applyRepetitionControls(logits, [0, 0, 1, 0], { noRepeatNgramSize: 2 });
    expect(out).toEqual(logits);
  });
});

describe('generate() with repetition controls', () => {
  it('breaks a greedy loop on a model trained on a repetitive corpus', async () => {
    const corpus = 'the cat sat on the mat . '.repeat(30);
    const vocab = ProNeuralLM.buildVocabulary(corpus, { mode: 'unicode' });
    const model = new ProNeuralLM(vocab, 32, 0.05, 2, 'adam', 0.9, 0, 7, { mode: 'unicode' });
    await model.train(corpus, 15);

    const words = (t: string) => t.split(/\s+/).filter(Boolean);
    const repeatedTrigrams = (t: string) => {
      const w = words(t);
      const seen = new Set<string>();
      let r = 0;
      for (let i = 0; i + 3 <= w.length; i++) {
        const g = w.slice(i, i + 3).join(' ');
        if (seen.has(g)) r++;
        seen.add(g);
      }
      return r;
    };

    const plain = await model.generate('the cat', 30, 1, 1, 0);
    expect(repeatedTrigrams(plain)).toBeGreaterThan(0);

    const blocked = await model.generate('the cat', 30, 1, 1, 0, 0, 0, 0, 0, 0.1, {
      noRepeatNgramSize: 3
    });
    expect(repeatedTrigrams('the cat ' + blocked)).toBe(0);
  });
});

describe('recommendedNoRepeatNgram', () => {
  it('spans about two words for each tokenizer', () => {
    expect(recommendedNoRepeatNgram('character')).toBe(16);
    expect(recommendedNoRepeatNgram('bpe')).toBe(3);
    expect(recommendedNoRepeatNgram('unicode')).toBe(3);
  });

  it('spans about 16 characters for subword models', () => {
    const specials = ['<PAD>', '<BOS>', '<EOS>', '<UNK>'];
    // mean subword length 4 -> floor(16 / 4)
    expect(recommendedNoRepeatNgram('bpe', [...specials, 'the ', 'king', ' and', 'lord'])).toBe(4);
    // close to single characters (small corpus, few merges): behave like a character model
    expect(recommendedNoRepeatNgram('bpe', [...specials, 'a', 'b', 'c', 'th'])).toBe(12);
    expect(recommendedNoRepeatNgram('bpe', [...specials, 'a', 'b', 'c'])).toBe(16);
    // long subwords never go below the word-trigram block
    expect(recommendedNoRepeatNgram('bpe', [...specials, 'something', 'everything'])).toBe(3);
    expect(recommendedNoRepeatNgram('bpe', specials)).toBe(3);
    // the vocabulary does not matter for other tokenizers
    expect(recommendedNoRepeatNgram('character', ['a', 'b'])).toBe(16);
    expect(recommendedNoRepeatNgram('unicode', ['a', 'b'])).toBe(3);
  });
});
