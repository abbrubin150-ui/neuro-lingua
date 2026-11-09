import { afterEach, describe, expect, it, vi } from 'vitest';
import { ProNeuralLM } from '../../src/lib/ProNeuralLM';
import { AdvancedNeuralLM } from '../../src/lib/AdvancedNeuralLM';
import { TransformerLM } from '../../src/lib/TransformerLM';
import { isModelPackage, parseStoredModel, packModel } from '../../src/lib/modelPackage';
import { buildVocab, parseTokenizerConfig } from '../../src/lib/utils';
import { trainSubwordModel } from '../../src/tokenizer/subword';

const TEXT =
  'the cat sat on the mat. the dog sat on the log. the cat saw the dog. a cat sat on a mat.';
const KEY = 'neuro-lingua-storage-test';

/** Make localStorage reject values longer than `limit` characters, as a full quota does. */
function limitStorage(limit: number) {
  const original = Storage.prototype.setItem;
  return vi.spyOn(Storage.prototype, 'setItem').mockImplementation(function (
    this: Storage,
    key: string,
    value: string
  ) {
    if (value.length > limit) throw new DOMException('The quota has been exceeded.', 'QuotaExceededError');
    original.call(this, key, value);
  });
}

async function trainedTransformer() {
  const model = new TransformerLM(buildVocab(TEXT, { mode: 'unicode' }), 16, 0.01, 4, 'adam', 0.9, 0, 11, undefined, {
    numLayers: 1,
    numHeads: 2
  });
  await model.train(TEXT, 2);
  return model;
}

function sizes(model: ProNeuralLM) {
  const json = model.toJSON() as unknown as Record<string, unknown>;
  return {
    json: JSON.stringify(json).length,
    float32: JSON.stringify(packModel(json, { precision: 'float32' })).length,
    float16: JSON.stringify(packModel(json, { precision: 'float16' })).length
  };
}

async function maxLogitDifference(a: ProNeuralLM, b: ProNeuralLM) {
  const ctx = a.getContextWindow('the cat sat');
  const [x, y] = [await a.getLogitsForContext(ctx), await b.getLogitsForContext(ctx)];
  return Math.max(...x.map((v, i) => Math.abs(v - y[i])));
}

afterEach(() => {
  vi.restoreAllMocks();
  localStorage.clear();
});

describe('saving models to localStorage', () => {
  it('stores plain JSON when it fits and restores the model exactly', async () => {
    const model = await trainedTransformer();
    expect(model.saveToLocalStorage(KEY)).toBe('json');
    expect(isModelPackage(JSON.parse(localStorage.getItem(KEY)!))).toBe(false);
    const restored = TransformerLM.loadFromLocalStorage(KEY)!;
    expect(await maxLogitDifference(model, restored)).toBe(0);
  });

  it('falls back to a float32 package when the plain JSON exceeds the quota', async () => {
    const model = await trainedTransformer();
    const size = sizes(model);
    expect(size.float32).toBeLessThan(size.json / 2);
    limitStorage(size.float32 + 10);

    expect(model.saveToLocalStorage(KEY)).toBe('float32');
    expect(isModelPackage(JSON.parse(localStorage.getItem(KEY)!))).toBe(true);
    const restored = TransformerLM.loadFromLocalStorage(KEY)!;
    expect(restored).toBeInstanceOf(TransformerLM);
    expect(await maxLogitDifference(model, restored)).toBeLessThan(1e-4);
  });

  it('falls back to a float16 package, then reports failure without touching the old copy', async () => {
    const model = await trainedTransformer();
    const size = sizes(model);
    const spy = limitStorage(size.float16 + 10);
    expect(model.saveToLocalStorage(KEY)).toBe('float16');
    const restored = TransformerLM.loadFromLocalStorage(KEY)!;
    expect(await maxLogitDifference(model, restored)).toBeLessThan(0.05);

    const stored = localStorage.getItem(KEY);
    spy.mockRestore();
    limitStorage(100);
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    expect(model.saveToLocalStorage(KEY)).toBeNull();
    expect(warn).toHaveBeenCalled();
    expect(localStorage.getItem(KEY)).toBe(stored);
  });

  it('restores packaged MLP and advanced models with their own class', async () => {
    const vocab = buildVocab(TEXT, { mode: 'unicode' });
    const mlp = new ProNeuralLM(vocab, 12, 0.01, 3, 'adam', 0.9, 0, 3);
    const advanced = new AdvancedNeuralLM(vocab, 12, 0.01, 3, 'adam', 0.9, 0, 3, undefined, {
      activation: 'gelu',
      useLayerNorm: true
    });
    await mlp.train(TEXT, 1);
    await advanced.train(TEXT, 1);
    for (const model of [mlp, advanced]) {
      const size = sizes(model);
      const spy = limitStorage(size.float32 + 10);
      expect(model.saveToLocalStorage(KEY)).toBe('float32');
      spy.mockRestore();
      const restored =
        model instanceof AdvancedNeuralLM
          ? AdvancedNeuralLM.loadFromLocalStorage(KEY)!
          : ProNeuralLM.loadFromLocalStorage(KEY)!;
      expect(restored.constructor).toBe(model.constructor);
      expect(await maxLogitDifference(model, restored)).toBeLessThan(1e-4);
    }
  });

  it('reads both storage encodings', () => {
    expect(parseStoredModel(null)).toBeNull();
    expect(parseStoredModel(JSON.stringify({ vocab: ['a'] }))).toEqual({ vocab: ['a'] });
  });
});

describe('subword tokenizer settings', () => {
  it('parses stored bpe settings and drops invalid merges', () => {
    const bpe = trainSubwordModel(TEXT, { vocabSize: 50 });
    expect(parseTokenizerConfig({ mode: 'bpe', bpe, bpeVocabSize: 50 })).toEqual({
      mode: 'bpe',
      bpe,
      bpeVocabSize: 50
    });
    expect(parseTokenizerConfig({ mode: 'bpe', bpe: { merges: 'nope' }, bpeVocabSize: -1 })).toEqual(
      { mode: 'bpe' }
    );
  });

  it('builds a model vocabulary from the subword tokenizer', () => {
    const bpe = trainSubwordModel(TEXT, { vocabSize: 50 });
    const vocab = buildVocab(TEXT, { mode: 'bpe', bpe });
    expect(vocab.slice(0, 4)).toEqual(['<PAD>', '<BOS>', '<EOS>', '<UNK>']);
    expect(vocab).toContain('▁cat');
  });
});
