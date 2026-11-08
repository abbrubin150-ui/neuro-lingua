import { describe, expect, it } from 'vitest';
import {
  WORD_MARKER,
  decodeSubword,
  encodeSubword,
  splitSymbols,
  subwordVocabulary,
  trainSubwordModel
} from '../../src/tokenizer/subword';
import { ProNeuralLM } from '../../src/lib/ProNeuralLM';
import { TransformerLM } from '../../src/lib/TransformerLM';

const CORPUS =
  'the cat sat on the mat. the dog sat on the log! the cat saw the dog; ' +
  'a cat sat on a mat, then the cats sat together. שָׁלוֹם עולם שלום 42 4242 snake_case';

/** Naive reference BPE: recount all pairs every iteration (same pre-tokenisation and ties). */
function naiveMerges(words: Map<string, number>, numMerges: number, minFrequency = 2) {
  const splits = [...words.entries()]
    .sort((a, b) => (a[0] < b[0] ? -1 : 1))
    .map(([w, f]) => ({ s: splitSymbols(w), f }));
  const merges: [string, string][] = [];
  while (merges.length < numMerges) {
    const counts = new Map<string, number>();
    for (const { s, f } of splits) {
      for (let i = 0; i < s.length - 1; i++) {
        const k = s[i] + '\u0000' + s[i + 1];
        counts.set(k, (counts.get(k) ?? 0) + f);
      }
    }
    let best: [string, number] | null = null;
    for (const [k, c] of counts) {
      if (!best || c > best[1] || (c === best[1] && k < best[0])) best = [k, c];
    }
    if (!best || best[1] < minFrequency) break;
    const [a, b] = best[0].split('\u0000');
    merges.push([a, b]);
    for (const w of splits) {
      const next: string[] = [];
      for (let i = 0; i < w.s.length; i++) {
        if (i < w.s.length - 1 && w.s[i] === a && w.s[i + 1] === b) {
          next.push(a + b);
          i++;
        } else next.push(w.s[i]);
      }
      w.s = next;
    }
  }
  return merges;
}

describe('subword tokenizer', () => {
  it('incremental training produces exactly the merges of a naive BPE', () => {
    const model = trainSubwordModel(CORPUS, { vocabSize: 80 });
    // Reconstruct the piece frequencies through the public encoder at the character level.
    const pieces = new Map<string, number>();
    const charModel = { alphabet: model.alphabet, merges: [], lowercase: false };
    let current = '';
    for (const span of encodeSubword(CORPUS, charModel)) {
      if (span.token.startsWith(WORD_MARKER) || /[^\p{L}\p{M}\p{N}]/u.test(span.token)) {
        if (current) pieces.set(current, (pieces.get(current) ?? 0) + 1);
        current = span.token;
      } else {
        current += span.token;
      }
    }
    if (current) pieces.set(current, (pieces.get(current) ?? 0) + 1);
    expect(model.merges.length).toBeGreaterThan(10);
    expect(naiveMerges(pieces, model.merges.length)).toEqual(model.merges);
  });

  it('is deterministic', () => {
    expect(trainSubwordModel(CORPUS, { vocabSize: 120 })).toEqual(
      trainSubwordModel(CORPUS, { vocabSize: 120 })
    );
  });

  it('decodes back to the original text (single spaces) and gives exact offsets', () => {
    const model = trainSubwordModel(CORPUS, { vocabSize: 150 });
    const text = 'The cats sat, then שָׁלוֹם 4242 snake_case 🙂!';
    const spans = encodeSubword(text, model);
    expect(decodeSubword(spans.map((s) => s.token))).toBe(text);
    for (const s of spans) {
      expect(text.slice(s.start, s.end)).toBe(s.token.replace(WORD_MARKER, ''));
    }
  });

  it('keeps Hebrew niqqud attached to its letter', () => {
    const model = trainSubwordModel(CORPUS, { vocabSize: 60 });
    const tokens = encodeSubword('שָׁלוֹם', model).map((s) => s.token);
    for (const t of tokens) expect(/^[֑-ׇ]/.test(t.replace(WORD_MARKER, ''))).toBe(false);
  });

  it('vocabulary = alphabet + merged symbols', () => {
    const model = trainSubwordModel(CORPUS, { vocabSize: 100 });
    const vocab = subwordVocabulary(model);
    expect(vocab.length).toBeLessThanOrEqual(100);
    expect(new Set(vocab).size).toBe(vocab.length);
    for (const s of encodeSubword(CORPUS, model)) expect(vocab).toContain(s.token);
  });
});

describe('models with a subword tokenizer', () => {
  const bpe = trainSubwordModel(CORPUS, { vocabSize: 120 });
  const config = { mode: 'bpe' as const, bpe };

  it('builds an out-of-vocabulary-free vocabulary and generates decoded text', async () => {
    const vocab = ProNeuralLM.buildVocabulary(CORPUS, config);
    expect(vocab.slice(0, 4)).toEqual(['<PAD>', '<BOS>', '<EOS>', '<UNK>']);
    const model = new TransformerLM(vocab, 16, 0.01, 6, 'adam', 0.9, 0, 3, config, {
      numLayers: 1,
      numHeads: 2
    });
    await model.train(CORPUS, 3);
    // Unseen words made of known characters are still in vocabulary.
    const held = await model.evaluate('the dogs sat on the cats');
    expect(held.oovRate).toBe(0);
    const out = await model.generate('the cat', 8, 0.1);
    expect(out).not.toContain(WORD_MARKER);
  });

  it('round-trips the tokenizer through toJSON / fromJSON', async () => {
    const vocab = ProNeuralLM.buildVocabulary(CORPUS, config);
    const model = new ProNeuralLM(vocab, 12, 0.01, 4, 'adam', 0.9, 0, 5, config);
    await model.train(CORPUS, 2);
    const restored = ProNeuralLM.fromJSON(JSON.parse(JSON.stringify(model.toJSON())));
    expect(restored.getTokenizerConfig().mode).toBe('bpe');
    const ctx = model.getContextWindow('the cat sat');
    expect(restored.getContextWindow('the cat sat')).toEqual(ctx);
    expect(await restored.getLogitsForContext(ctx)).toEqual(await model.getLogitsForContext(ctx));
  });
});
